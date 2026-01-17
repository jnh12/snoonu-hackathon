from __future__ import annotations
from typing import Dict, List, Optional, Literal, Tuple, Any
import random
import math

from .models import Order, Courier, RoutePlan
from .config import SimConfig
from .geo import haversine_km
from .router import Router
from .oba_engine import dispatch_oba, build_route_by_insertion


Policy = Literal["baseline", "oba"]


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(ys[int(k)])
    return float(ys[f] * (c - k) + ys[c] * (k - f))


class SimEngine:
    def __init__(self, cfg: SimConfig, router: Router):
        self.cfg = cfg
        self.router = router

        self.loaded = False
        self.policy: Policy = "baseline"
        self.sim_t: int = 0
        self.start_t: int = 0
        self.last_dispatch_t: int = 0

        self.orders: Dict[str, Order] = {}
        self.couriers: Dict[str, Courier] = {}

        # metrics aggregates (simple)
        self.total_distance_km: float = 0.0
        self.total_late_sec: int = 0
        self.total_wait_sec: int = 0
        self.delivered_count: int = 0

        # assignment log (for batching/efficiency KPIs)
        self.assignment_log: List[Dict[str, Any]] = []

        # for random spawns
        self._bounds: Optional[Tuple[float, float, float, float]] = None  # (min_lat,max_lat,min_lng,max_lng)

    def _compute_bounds(self):
        lats = [o.pickup_lat for o in self.orders.values()] + [o.dropoff_lat for o in self.orders.values()] + [c.lat for c in self.couriers.values()]
        lngs = [o.pickup_lng for o in self.orders.values()] + [o.dropoff_lng for o in self.orders.values()] + [c.lng for c in self.couriers.values()]
        if not lats or not lngs:
            self._bounds = (25.24, 25.45, 51.43, 51.58)
            return
        min_lat, max_lat = min(lats), max(lats)
        min_lng, max_lng = min(lngs), max(lngs)
        self._bounds = (min_lat - 0.01, max_lat + 0.01, min_lng - 0.01, max_lng + 0.01)

    def load(self, orders: List[Order], couriers: List[Courier]):
        self.orders = {o.order_id: o for o in orders}
        self.couriers = {c.courier_id: c for c in couriers}

        for o in self.orders.values():
            o.ready_sec = o.created_sec + self.cfg.default_prep_time_sec
            o.promised_sec = o.created_sec + o.estimated_delivery_time_min * 60

        self.loaded = True
        self._compute_bounds()
        self.reset(policy="baseline", start_time=None)

    def reset(self, policy: Policy = "baseline", start_time: Optional[int] = None):
        self.policy = policy
        self.total_distance_km = 0.0
        self.total_late_sec = 0
        self.total_wait_sec = 0
        self.delivered_count = 0
        self.assignment_log = []

        for o in self.orders.values():
            o.status = "PENDING"
            o.assigned_courier_id = None
            o.assigned_sec = None
            o.pickup_arrive_sec = None
            o.pickup_depart_sec = None
            o.dropoff_arrive_sec = None
            o.delivered_sec = None
            o.assigned_bundle_size = None

        for c in self.couriers.values():
            c.status = "OFFLINE"
            c.busy_until_sec = 0
            c.route = None
            c.route_idx = 0
            c.busy_start_sec = 0
            c.total_busy_sec = 0

        if start_time is None:
            min_t = min([o.created_sec for o in self.orders.values()] + [c.available_sec for c in self.couriers.values()])
            self.sim_t = min_t
        else:
            self.sim_t = start_time

        self.start_t = self.sim_t

        self._release_agents()
        self.last_dispatch_t = self.sim_t - self.cfg.dispatch_interval_sec

    def _release_agents(self):
        for c in self.couriers.values():
            if c.status == "OFFLINE" and c.available_sec <= self.sim_t:
                c.status = "IDLE"

    def _pending_orders(self) -> List[Order]:
        return [o for o in self.orders.values() if o.status == "PENDING" and o.created_sec <= self.sim_t]

    def _idle_couriers(self) -> List[Courier]:
        return [c for c in self.couriers.values() if c.status == "IDLE" and c.available_sec <= self.sim_t]

    def _progress_routes(self):
        for c in self.couriers.values():
            if c.status != "BUSY" or c.route is None:
                continue

            stops = c.route.stops
            while c.route_idx < len(stops) and stops[c.route_idx].depart_sec <= self.sim_t:
                s = stops[c.route_idx]
                c.lat, c.lng = s.lat, s.lng

                o = self.orders.get(s.order_id)
                if o:
                    if s.stop_type == "pickup":
                        o.status = "PICKED_UP"
                        o.pickup_arrive_sec = s.arrive_sec
                        o.pickup_depart_sec = s.depart_sec
                    else:
                        if o.status != "DELIVERED":
                            o.status = "DELIVERED"
                            o.dropoff_arrive_sec = s.arrive_sec
                            o.delivered_sec = s.depart_sec
                            self.delivered_count += 1

                c.route_idx += 1

            if c.route_idx >= len(stops):
                finish = c.route.finish_sec
                c.total_busy_sec += max(0, finish - c.busy_start_sec)

                c.status = "IDLE"
                c.route = None
                c.route_idx = 0
                c.busy_start_sec = 0
                c.busy_until_sec = 0

    def step(self, dt_sec: int = 10):
        if not self.loaded:
            raise RuntimeError("Dataset not loaded")

        self.sim_t += max(1, int(dt_sec))

        self._release_agents()
        self._progress_routes()

        if (self.sim_t - self.last_dispatch_t) >= self.cfg.dispatch_interval_sec:
            self.dispatch()
            self.last_dispatch_t = self.sim_t

    def run(self, seconds: int = 300, dt_sec: int = 10):
        steps = max(1, seconds // dt_sec)
        for _ in range(steps):
            self.step(dt_sec=dt_sec)

    def dispatch(self):
        pending = self._pending_orders()
        idle = self._idle_couriers()
        if not pending or not idle:
            return

        if self.policy == "baseline":
            self._dispatch_baseline(pending, idle)
        else:
            self._dispatch_oba(pending, idle)

    def _dispatch_baseline(self, pending: List[Order], idle: List[Courier]):
        pending_sorted = sorted(pending, key=lambda o: o.created_sec)
        idle_pool = idle[:]

        for o in pending_sorted:
            if not idle_pool:
                break

            best = min(
                idle_pool,
                key=lambda c: haversine_km((c.lat, c.lng), (o.pickup_lat, o.pickup_lng)),
            )

            rp = build_route_by_insertion(self.cfg, self.router, best, [o], self.sim_t)
            if rp is None:
                idle_pool.remove(best)
                continue

            o.status = "ASSIGNED"
            self._apply_assignment(best, rp)
            idle_pool.remove(best)

    def _dispatch_oba(self, pending: List[Order], idle: List[Courier]):
        assignments = dispatch_oba(self.cfg, self.router, self.sim_t, pending, idle)
        for c, rp in assignments:
            for oid in rp.assigned_order_ids:
                o = self.orders.get(oid)
                if o and o.status == "PENDING":
                    o.status = "ASSIGNED"
            self._apply_assignment(c, rp)

    def _apply_assignment(self, c: Courier, rp: RoutePlan):
        # courier state
        c.status = "BUSY"
        c.route = rp
        c.route_idx = 0
        c.busy_start_sec = self.sim_t
        c.busy_until_sec = rp.finish_sec

        bundle_size = len(rp.assigned_order_ids)

        # order assignment timestamps + batching info
        for oid in rp.assigned_order_ids:
            o = self.orders.get(oid)
            if not o:
                continue
            if o.assigned_sec is None:
                o.assigned_sec = self.sim_t
            if o.assigned_courier_id is None:
                o.assigned_courier_id = c.courier_id
            if o.assigned_bundle_size is None:
                o.assigned_bundle_size = bundle_size

        # assignment log (efficiency KPIs)
        first_stop = rp.stops[0] if rp.stops else None
        first_mile_km = 0.0
        if first_stop:
            first_mile_km = haversine_km((c.lat, c.lng), (first_stop.lat, first_stop.lng))

        self.assignment_log.append({
            "t": self.sim_t,
            "courier_id": c.courier_id,
            "bundle_size": bundle_size,
            "total_distance_km": rp.total_distance_km,
            "first_mile_km": first_mile_km,
        })

        # aggregates
        self.total_distance_km += rp.total_distance_km
        self.total_late_sec += rp.total_late_sec
        self.total_wait_sec += rp.total_wait_sec

    # ---------- interactive additions ----------
    def add_order(self, o: Order):
        self.orders[o.order_id] = o
        o.ready_sec = o.created_sec + self.cfg.default_prep_time_sec
        o.promised_sec = o.created_sec + o.estimated_delivery_time_min * 60

    def add_courier(self, c: Courier):
        self.couriers[c.courier_id] = c
        if c.available_sec <= self.sim_t:
            c.status = "IDLE"

    def spawn_random_orders(self, n: int, created_sec: Optional[int] = None):
        if not self._bounds:
            self._compute_bounds()
        min_lat, max_lat, min_lng, max_lng = self._bounds or (25.24, 25.45, 51.43, 51.58)

        base_t = created_sec if created_sec is not None else self.sim_t
        for i in range(n):
            oid = f"RND-{base_t}-{i}"
            plat = random.uniform(min_lat, max_lat)
            plng = random.uniform(min_lng, max_lng)
            dlat = random.uniform(min_lat, max_lat)
            dlng = random.uniform(min_lng, max_lng)
            o = Order(
                order_id=oid,
                created_sec=base_t,
                pickup_restaurant="Random Restaurant",
                pickup_lat=plat,
                pickup_lng=plng,
                dropoff_area="Random Dropoff",
                dropoff_lat=dlat,
                dropoff_lng=dlng,
                estimated_delivery_time_min=random.randint(15, 30),
            )
            self.add_order(o)

    def spawn_random_couriers(self, n: int, available_sec: Optional[int] = None):
        if not self._bounds:
            self._compute_bounds()
        min_lat, max_lat, min_lng, max_lng = self._bounds or (25.24, 25.45, 51.43, 51.58)

        base_t = available_sec if available_sec is not None else self.sim_t
        vehicles = ["bike", "motorbike", "car"]
        for i in range(n):
            cid = f"RND-C-{base_t}-{i}"
            lat = random.uniform(min_lat, max_lat)
            lng = random.uniform(min_lng, max_lng)
            v = random.choice(vehicles)
            c = Courier(
                courier_id=cid,
                available_sec=base_t,
                starting_area_hint="Random",
                lat=lat,
                lng=lng,
                vehicle_type=v,  # type: ignore
                bundle_capacity=random.randint(2, 4),
            )
            self.add_courier(c)

    def kpis(self) -> Dict[str, Any]:
        delivered = [o for o in self.orders.values() if o.status == "DELIVERED" and o.delivered_sec is not None]
        pending = [o for o in self.orders.values() if o.status == "PENDING" and o.created_sec <= self.sim_t]

        deliv_sec = [(o.delivered_sec - o.created_sec) for o in delivered]  # type: ignore[operator]
        late_sec = [(o.delivered_sec - o.promised_sec) for o in delivered]  # type: ignore[operator]
        late_pos = [max(0, x) for x in late_sec]

        # food quality
        in_bag = []
        food_wait_after_ready = []
        cold_risk = 0
        food_sit_risk = 0

        for o in delivered:
            tib = None
            if o.pickup_depart_sec is not None and o.dropoff_arrive_sec is not None:
                tib = o.dropoff_arrive_sec - o.pickup_depart_sec
                in_bag.append(tib)
                if tib > self.cfg.cold_in_bag_threshold_sec:
                    cold_risk += 1

            fw = None
            if o.pickup_arrive_sec is not None:
                fw = max(0, o.pickup_arrive_sec - o.ready_sec)
                food_wait_after_ready.append(fw)
                if fw > self.cfg.max_food_wait_after_ready_sec:
                    food_sit_risk += 1

        # backlog
        pending_age = [(self.sim_t - o.created_sec) for o in pending]
        starvation = sum(1 for x in pending_age if x > self.cfg.backlog_starvation_sec)

        # utilization
        utils: List[float] = []
        horizon_end = self.sim_t
        for c in self.couriers.values():
            if horizon_end <= c.available_sec:
                continue
            horizon_start = max(self.start_t, c.available_sec)
            horizon = horizon_end - horizon_start
            if horizon <= 0:
                continue
            busy = c.total_busy_sec
            if c.status == "BUSY":
                busy += max(0, horizon_end - c.busy_start_sec)
            utils.append(busy / horizon)

        # batching
        bundles = [x["bundle_size"] for x in self.assignment_log]
        avg_bundle_assignment = (sum(bundles) / len(bundles)) if bundles else 0.0
        bundled_assignment_rate = (sum(1 for b in bundles if b >= 2) / len(bundles)) if bundles else 0.0

        # order-level bundled rate (using assigned_bundle_size)
        delivered_bundle_sizes = [o.assigned_bundle_size or 1 for o in delivered]
        bundled_order_rate = (sum(1 for b in delivered_bundle_sizes if b >= 2) / max(1, len(delivered_bundle_sizes)))

        first_miles = [x["first_mile_km"] for x in self.assignment_log]
        avg_first_mile = (sum(first_miles) / len(first_miles)) if first_miles else 0.0

        on_time = sum(1 for x in late_sec if x <= 0)
        extreme_late = sum(1 for x in late_pos if x > self.cfg.extreme_late_sec)

        # support-risk proxy: late OR cold-risk OR food-sit-risk
        support_risk = 0
        for o in delivered:
            late = (o.delivered_sec - o.promised_sec) if o.delivered_sec is not None else 0
            tib = (o.dropoff_arrive_sec - o.pickup_depart_sec) if (o.dropoff_arrive_sec and o.pickup_depart_sec) else 0
            fw = (o.pickup_arrive_sec - o.ready_sec) if o.pickup_arrive_sec else 0
            if late > 0 or tib > self.cfg.cold_in_bag_threshold_sec or fw > self.cfg.max_food_wait_after_ready_sec:
                support_risk += 1

        return {
            "delivered": len(delivered),
            "pending": len(pending),

            "delivery_time_min": {
                "avg": round((sum(deliv_sec) / 60.0) / max(1, len(deliv_sec)), 3),
                "p90": round(_pct([x / 60.0 for x in deliv_sec], 0.90), 3),
                "p95": round(_pct([x / 60.0 for x in deliv_sec], 0.95), 3),
                "max": round((max(deliv_sec) / 60.0) if deliv_sec else 0.0, 3),
            },

            "lateness_min": {
                "avg_pos": round((sum(late_pos) / 60.0) / max(1, len(late_pos)), 3),
                "p95_pos": round(_pct([x / 60.0 for x in late_pos], 0.95), 3),
                "max_pos": round((max(late_pos) / 60.0) if late_pos else 0.0, 3),
                "on_time_rate": round(on_time / max(1, len(late_sec)), 3),
                "extreme_late_rate": round(extreme_late / max(1, len(late_pos)), 3),
            },

            "food_quality": {
                "avg_time_in_bag_min": round((sum(in_bag) / 60.0) / max(1, len(in_bag)), 3),
                "cold_risk_rate": round(cold_risk / max(1, len(in_bag)), 3),
                "avg_food_wait_after_ready_min": round((sum(food_wait_after_ready) / 60.0) / max(1, len(food_wait_after_ready)), 3),
                "food_sit_risk_rate": round(food_sit_risk / max(1, len(food_wait_after_ready)), 3),
            },

            "efficiency": {
                "total_distance_km": round(self.total_distance_km, 3),
                "km_per_delivered": round(self.total_distance_km / max(1, len(delivered)), 3),
                "avg_first_mile_km": round(avg_first_mile, 3),
            },

            "fleet": {
                "avg_utilization": round((sum(utils) / max(1, len(utils))), 3),
                "p90_utilization": round(_pct(utils, 0.90), 3) if utils else 0.0,
            },

            "batching": {
                "avg_bundle_size_per_assignment": round(avg_bundle_assignment, 3),
                "bundled_assignment_rate": round(bundled_assignment_rate, 3),
                "bundled_order_rate": round(bundled_order_rate, 3),
            },

            "backlog": {
                "max_pending_age_min": round((max(pending_age) / 60.0) if pending_age else 0.0, 3),
                "starvation_rate": round(starvation / max(1, len(pending_age)), 3),
            },

            "support_proxy": {
                "support_risk_rate": round(support_risk / max(1, len(delivered)), 3),
            },
        }

    def snapshot(self) -> Dict[str, Any]:
        pending = self._pending_orders()
        idle = self._idle_couriers()
        busy = [c for c in self.couriers.values() if c.status == "BUSY"]

        def _order_flags(o: Order) -> Dict[str, Any]:
            time_in_bag_sec = None
            food_wait_after_ready_sec = None
            lateness_sec = None

            if o.pickup_depart_sec is not None and o.dropoff_arrive_sec is not None:
                time_in_bag_sec = o.dropoff_arrive_sec - o.pickup_depart_sec

            if o.pickup_arrive_sec is not None:
                food_wait_after_ready_sec = max(0, o.pickup_arrive_sec - o.ready_sec)

            if o.delivered_sec is not None:
                lateness_sec = o.delivered_sec - o.promised_sec

            cold_risk = bool(time_in_bag_sec is not None and time_in_bag_sec > self.cfg.cold_in_bag_threshold_sec)
            food_sit_risk = bool(food_wait_after_ready_sec is not None and food_wait_after_ready_sec > self.cfg.max_food_wait_after_ready_sec)
            extreme_late = bool(lateness_sec is not None and lateness_sec > self.cfg.extreme_late_sec)

            return {
                "time_in_bag_sec": time_in_bag_sec,
                "food_wait_after_ready_sec": food_wait_after_ready_sec,
                "lateness_sec": lateness_sec,
                "cold_risk": cold_risk,
                "food_sit_risk": food_sit_risk,
                "extreme_late": extreme_late,
            }

        return {
            "sim_t": self.sim_t,
            "policy": self.policy,
            "counts": {
                "pending": len(pending),
                "idle": len(idle),
                "busy": len(busy),
                "delivered": self.delivered_count,
            },
            "kpis": self.kpis(),
            "orders": [
                {
                    "order_id": o.order_id,
                    "status": o.status,
                    "created_sec": o.created_sec,
                    "ready_sec": o.ready_sec,
                    "promised_sec": o.promised_sec,
                    "assigned_sec": o.assigned_sec,
                    "pickup_arrive_sec": o.pickup_arrive_sec,
                    "pickup_depart_sec": o.pickup_depart_sec,
                    "dropoff_arrive_sec": o.dropoff_arrive_sec,
                    "delivered_sec": o.delivered_sec,
                    "pickup_lat": o.pickup_lat,
                    "pickup_lng": o.pickup_lng,
                    "dropoff_lat": o.dropoff_lat,
                    "dropoff_lng": o.dropoff_lng,
                    "pickup_label": o.pickup_restaurant,
                    "dropoff_label": o.dropoff_area,
                    "assigned_courier_id": o.assigned_courier_id,
                    "assigned_bundle_size": o.assigned_bundle_size,
                    **_order_flags(o),
                }
                for o in self.orders.values()
            ],
            "couriers": [
                {
                    "courier_id": c.courier_id,
                    "status": c.status,
                    "lat": c.lat,
                    "lng": c.lng,
                    "vehicle_type": c.vehicle_type,
                    "bundle_capacity": c.bundle_capacity,
                    "busy_until_sec": c.busy_until_sec,
                    "route_idx": c.route_idx,
                    "route": None
                    if c.route is None
                    else {
                        "assigned_order_ids": c.route.assigned_order_ids,
                        "finish_sec": c.route.finish_sec,
                        "total_distance_km": c.route.total_distance_km,
                        "stops": [
                            {
                                "type": s.stop_type,
                                "order_id": s.order_id,
                                "lat": s.lat,
                                "lng": s.lng,
                                "arrive_sec": s.arrive_sec,
                                "depart_sec": s.depart_sec,
                                "label": s.restaurant_or_area,
                            }
                            for s in c.route.stops
                        ],
                    },
                }
                for c in self.couriers.values()
            ],
        }
