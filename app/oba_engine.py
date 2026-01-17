from __future__ import annotations

from typing import List, Dict, Tuple, Optional, Set
from itertools import combinations

from .models import Order, Courier, RoutePlan, Stop
from .geo import haversine_km
from .config import SimConfig
from .router import Router


def _rest_key(o: Order) -> str:
    return (o.pickup_restaurant or "").strip().lower()


def _same_restaurant(cfg: SimConfig, a: Order, b: Order) -> bool:
    """
    Hackathon-friendly:
    - If names match -> same restaurant
    - Else if pickups are within restaurant_match_radius_m -> treat as same pickup 'hub'
      (your dataset uses unique restaurant names, so this is required for batching)
    """
    ka = _rest_key(a)
    kb = _rest_key(b)
    if ka and kb and ka == kb:
        return True

    rad_km = max(1.0, cfg.restaurant_match_radius_m / 1000.0)  # safety min 1km if mis-set
    d = haversine_km((a.pickup_lat, a.pickup_lng), (b.pickup_lat, b.pickup_lng))
    return d <= rad_km


def _compatible_for_batch(cfg: SimConfig, a: Order, b: Order) -> bool:
    if cfg.same_restaurant_only and not _same_restaurant(cfg, a, b):
        return False

    if abs(a.created_sec - b.created_sec) > cfg.batch_time_window_sec:
        return False

    dp = haversine_km((a.pickup_lat, a.pickup_lng), (b.pickup_lat, b.pickup_lng))
    if dp > (cfg.pickup_cluster_radius_m / 1000.0):
        return False

    dd = haversine_km((a.dropoff_lat, a.dropoff_lng), (b.dropoff_lat, b.dropoff_lng))
    if dd > (cfg.dropoff_cluster_radius_m / 1000.0):
        return False

    return True


def _solo_time_estimate_sec(cfg: SimConfig, router: Router, courier: Courier, order: Order, now: int) -> int:
    m1 = router.metric((courier.lat, courier.lng), (order.pickup_lat, order.pickup_lng), courier.vehicle_type)  # type: ignore[arg-type]
    arrive_pick = now + m1.travel_sec
    wait = max(0, order.ready_sec - arrive_pick)
    m2 = router.metric((order.pickup_lat, order.pickup_lng), (order.dropoff_lat, order.dropoff_lng), courier.vehicle_type)  # type: ignore[arg-type]
    return m1.travel_sec + wait + cfg.pickup_service_sec + m2.travel_sec + cfg.dropoff_service_sec


# ----------------------------
# Route building (insertion) -> produces arrive_sec/depart_sec
# ----------------------------

def build_route_by_insertion(
    cfg: SimConfig,
    router: Router,
    courier: Courier,
    orders: List[Order],
    now: int,
) -> Optional[RoutePlan]:
    if not orders:
        return None

    cap = int(getattr(courier, "bundle_capacity", 1) or 1)
    if len(orders) > min(cfg.max_bundle_size, cap):
        return None

    route: List[Tuple[str, Order]] = [("pickup", orders[0]), ("dropoff", orders[0])]

    for o in orders[1:]:
        best: Optional[Tuple[float, List[Tuple[str, Order]]]] = None

        for i in range(len(route) + 1):
            for j in range(i + 1, len(route) + 2):
                cand = route.copy()
                cand.insert(i, ("pickup", o))
                cand.insert(j, ("dropoff", o))
                rp = _simulate_route(cfg, router, courier, cand, now)
                if rp is None:
                    continue
                if best is None or rp.total_distance_km < best[0]:
                    best = (rp.total_distance_km, cand)

        if best is None:
            return None
        route = best[1]

    return _simulate_route(cfg, router, courier, route, now)


def _simulate_route(
    cfg: SimConfig,
    router: Router,
    courier: Courier,
    route: List[Tuple[str, Order]],
    now: int,
) -> Optional[RoutePlan]:
    load = 0
    pickup_depart: Dict[str, int] = {}
    dropoff_arrive: Dict[str, int] = {}

    t = int(now)
    total_dist_km = 0.0
    total_wait = 0
    total_late = 0

    cur_lat, cur_lng = float(courier.lat), float(courier.lng)
    stops: List[Stop] = []
    assigned_ids: List[str] = []

    for kind, o in route:
        nxt = (o.pickup_lat, o.pickup_lng) if kind == "pickup" else (o.dropoff_lat, o.dropoff_lng)

        m = router.metric((cur_lat, cur_lng), nxt, courier.vehicle_type)  # type: ignore[arg-type]
        total_dist_km += float(m.distance_km)
        arrive = t + int(m.travel_sec)

        if kind == "pickup":
            wait = max(0, int(o.ready_sec) - int(arrive))
            total_wait += wait
            depart = arrive + wait + int(cfg.pickup_service_sec)

            load += 1
            if load > int(courier.bundle_capacity):
                return None

            pickup_depart[o.order_id] = depart
            if o.order_id not in assigned_ids:
                assigned_ids.append(o.order_id)

            stops.append(
                Stop(
                    stop_type="pickup",
                    order_id=o.order_id,
                    lat=float(o.pickup_lat),
                    lng=float(o.pickup_lng),
                    arrive_sec=int(arrive),
                    depart_sec=int(depart),
                    restaurant_or_area=o.pickup_restaurant,
                )
            )
        else:
            depart = arrive + int(cfg.dropoff_service_sec)

            load -= 1
            if load < 0:
                return None

            dropoff_arrive[o.order_id] = int(arrive)
            total_late += max(0, int(depart) - int(o.promised_sec))

            if o.order_id in pickup_depart:
                tib = int(arrive) - int(pickup_depart[o.order_id])
                if tib > int(cfg.max_time_in_bag_sec):
                    return None

            stops.append(
                Stop(
                    stop_type="dropoff",
                    order_id=o.order_id,
                    lat=float(o.dropoff_lat),
                    lng=float(o.dropoff_lng),
                    arrive_sec=int(arrive),
                    depart_sec=int(depart),
                    restaurant_or_area=o.dropoff_area,
                )
            )

        t = int(depart)
        cur_lat, cur_lng = float(nxt[0]), float(nxt[1])

    # detour constraint vs solo (looser by config now)
    for oid in assigned_ids:
        order_obj = next(x for x in (x[1] for x in route) if x.order_id == oid)
        solo = _solo_time_estimate_sec(cfg, router, courier, order_obj, now)
        actual = (dropoff_arrive.get(oid, now + 10**9) - now)
        if actual > int(solo * (1.0 + float(cfg.max_detour_ratio))):
            return None

    return RoutePlan(
        courier_id=courier.courier_id,
        assigned_order_ids=assigned_ids,
        stops=stops,
        total_distance_km=float(total_dist_km),
        total_wait_sec=int(total_wait),
        total_late_sec=int(total_late),
        finish_sec=int(t),
    )


# ----------------------------
# Clustering + bundle generation
# ----------------------------

def _cluster_orders(cfg: SimConfig, orders: List[Order]) -> List[List[Order]]:
    if not orders:
        return []

    n = len(orders)
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _compatible_for_batch(cfg, orders[i], orders[j]):
                adj[i].append(j)
                adj[j].append(i)

    seen = [False] * n
    clusters: List[List[Order]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp: List[Order] = []
        while stack:
            u = stack.pop()
            comp.append(orders[u])
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        clusters.append(comp)

    clusters.sort(key=len, reverse=True)
    return clusters[: cfg.max_clusters_considered]


def _bundle_similarity(a: Order, b: Order) -> float:
    dp = haversine_km((a.pickup_lat, a.pickup_lng), (b.pickup_lat, b.pickup_lng))
    dd = haversine_km((a.dropoff_lat, a.dropoff_lng), (b.dropoff_lat, b.dropoff_lng))
    return dp + 0.7 * dd


def _generate_candidate_bundles(cfg: SimConfig, cluster: List[Order], cap_limit: int) -> List[List[Order]]:
    if not cluster:
        return []

    max_k = min(cfg.max_bundle_size, cap_limit, len(cluster))

    bundles: List[List[Order]] = [[o] for o in cluster]

    pairs: List[Tuple[float, Order, Order]] = []
    for a, b in combinations(cluster, 2):
        if _compatible_for_batch(cfg, a, b):
            pairs.append((_bundle_similarity(a, b), a, b))
    pairs.sort(key=lambda x: x[0])

    for _, a, b in pairs[: cfg.max_bundles_per_cluster]:
        bundles.append([a, b])

    if max_k >= 3:
        for _, a, b in pairs[: min(40, len(pairs))]:
            remaining = [o for o in cluster if o.order_id not in {a.order_id, b.order_id}]
            remaining.sort(key=lambda o: _bundle_similarity(a, o) + _bundle_similarity(b, o))
            for o3 in remaining[:3]:
                if len(bundles) >= cfg.max_bundles_per_cluster:
                    break
                if all(_compatible_for_batch(cfg, o3, x) for x in [a, b]):
                    bundles.append([a, b, o3])

    if max_k >= 4:
        tris = [b for b in bundles if len(b) == 3]
        for tri in tris[: min(25, len(tris))]:
            ids = {o.order_id for o in tri}
            remaining = [o for o in cluster if o.order_id not in ids]
            remaining.sort(key=lambda o: sum(_bundle_similarity(x, o) for x in tri))
            for o4 in remaining[:2]:
                if len(bundles) >= cfg.max_bundles_per_cluster:
                    break
                if all(_compatible_for_batch(cfg, o4, x) for x in tri):
                    bundles.append(tri + [o4])

    # dedup
    seen: Set[Tuple[str, ...]] = set()
    uniq: List[List[Order]] = []
    for b in bundles:
        key = tuple(sorted(o.order_id for o in b))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(b)

    # strongly keep multi-order candidates
    singles = [b for b in uniq if len(b) == 1]
    multis = [b for b in uniq if len(b) >= 2]

    multis.sort(key=lambda b: (-len(b), sum(o.created_sec for o in b)))
    singles.sort(key=lambda b: b[0].created_sec)

    limit = cfg.max_bundles_per_cluster
    if multis:
        keep_multis = min(len(multis), max(12, int(limit * 0.75)))
        keep_singles = min(len(singles), max(6, limit - keep_multis))
        return multis[:keep_multis] + singles[:keep_singles]
    return singles[:limit]


def score_route(cfg: SimConfig, rp: RoutePlan) -> float:
    late_min = rp.total_late_sec / 60.0
    wait_min = rp.total_wait_sec / 60.0
    size = max(1, len(rp.assigned_order_ids))
    return (
        cfg.w_distance * rp.total_distance_km
        + cfg.w_late_min * late_min
        + cfg.w_wait_min * wait_min
        - cfg.bundle_bonus * (size - 1)
    )


# ----------------------------
# DISPATCH (key change): prefer batch routes if any exist
# ----------------------------

def dispatch_oba(
    cfg: SimConfig,
    router: Router,
    now: int,
    pending_orders: List[Order],
    idle_couriers: List[Courier],
) -> List[Tuple[Courier, RoutePlan]]:
    if not pending_orders or not idle_couriers:
        return []

    remaining_orders: Dict[str, Order] = {o.order_id: o for o in pending_orders}
    remaining_couriers: Dict[str, Courier] = {c.courier_id: c for c in idle_couriers}
    assignments: List[Tuple[Courier, RoutePlan]] = []

    def _one_pass() -> bool:
        nonlocal assignments, remaining_orders, remaining_couriers

        orders_list = list(remaining_orders.values())
        couriers_list = list(remaining_couriers.values())
        if not orders_list or not couriers_list:
            return False

        # micro-batch window + starvation guardrail
        recent = [o for o in orders_list if o.created_sec >= now - cfg.batch_time_window_sec]
        if recent:
            oldest = sorted(orders_list, key=lambda o: o.created_sec)[: max(0, cfg.oldest_orders_guardrail)]
            seen: Set[str] = set()
            candidate_orders: List[Order] = []
            for o in recent + oldest:
                if o.order_id in seen:
                    continue
                seen.add(o.order_id)
                candidate_orders.append(o)
        else:
            candidate_orders = orders_list

        clusters = _cluster_orders(cfg, candidate_orders)

        best_any = None    # (score_per_order, courier_id, bundle_ids, rp)
        best_multi = None  # same but only size>=2

        for cluster in clusters:
            if not cluster:
                continue

            clat = sum(o.pickup_lat for o in cluster) / len(cluster)
            clng = sum(o.pickup_lng for o in cluster) / len(cluster)

            couriers_sorted = sorted(
                couriers_list,
                key=lambda c: haversine_km((c.lat, c.lng), (clat, clng)),
            )[: cfg.max_couriers_per_cluster]

            cap_limit = max((c.bundle_capacity for c in couriers_sorted), default=1)
            bundles = _generate_candidate_bundles(cfg, cluster, cap_limit)

            for c in couriers_sorted:
                for b in bundles:
                    if len(b) > c.bundle_capacity:
                        continue

                    rp = build_route_by_insertion(cfg, router, c, b, now)
                    if rp is None:
                        continue

                    size = len(rp.assigned_order_ids)
                    sc = score_route(cfg, rp) / max(1, size)

                    cand = (sc, c.courier_id, tuple(rp.assigned_order_ids), rp)

                    if best_any is None or sc < best_any[0] or (abs(sc - best_any[0]) < 1e-9 and size > len(best_any[2])):
                        best_any = cand

                    if size >= 2:
                        if best_multi is None or sc < best_multi[0] or (abs(sc - best_multi[0]) < 1e-9 and size > len(best_multi[2])):
                            best_multi = cand

        chosen = best_multi if best_multi is not None else best_any
        if chosen is None:
            return False

        _, cid, bundle_ids, rp = chosen
        courier = remaining_couriers.get(cid)
        if courier is None:
            return False

        assignments.append((courier, rp))
        remaining_couriers.pop(cid, None)
        for oid in bundle_ids:
            remaining_orders.pop(oid, None)

        return True

    progressed = _one_pass()
    while progressed and remaining_orders and remaining_couriers:
        progressed = _one_pass()

    return assignments
