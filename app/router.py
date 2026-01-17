from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Literal
import json
from urllib.request import urlopen
from urllib.parse import urlencode

from .config import SimConfig
from .geo import haversine_km, km_to_travel_sec


Vehicle = Literal["bike", "motorbike", "car"]


def _speed_kmh(cfg: SimConfig, vehicle: Vehicle) -> float:
    if vehicle == "bike":
        return cfg.speed_bike_kmh
    if vehicle == "motorbike":
        return cfg.speed_motorbike_kmh
    return cfg.speed_car_kmh


@dataclass
class RouteMetric:
    distance_km: float
    travel_sec: int


class Router:
    def metric(self, a: Tuple[float, float], b: Tuple[float, float], vehicle: Vehicle) -> RouteMetric:
        raise NotImplementedError


class HaversineRouter(Router):
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    def metric(self, a: Tuple[float, float], b: Tuple[float, float], vehicle: Vehicle) -> RouteMetric:
        d = haversine_km(a, b)
        t = km_to_travel_sec(d, _speed_kmh(self.cfg, vehicle))
        return RouteMetric(distance_km=d, travel_sec=t)


class OsrmRouter(Router):
    """
    Optional real-road routing via local OSRM.
    Fallbacks to haversine if OSRM is down.
    """
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self._cache: Dict[Tuple, RouteMetric] = {}

    def _profile(self, vehicle: Vehicle) -> str:
        if vehicle == "bike":
            return "cycling"
        return "driving"

    def _key(self, a: Tuple[float, float], b: Tuple[float, float], vehicle: Vehicle) -> Tuple:
        r = self.cfg.cache_round_decimals
        return (round(a[0], r), round(a[1], r), round(b[0], r), round(b[1], r), vehicle)

    def metric(self, a: Tuple[float, float], b: Tuple[float, float], vehicle: Vehicle) -> RouteMetric:
        if self.cfg.cache_routes:
            k = self._key(a, b, vehicle)
            if k in self._cache:
                return self._cache[k]

        prof = self._profile(vehicle)
        coords = f"{a[1]},{a[0]};{b[1]},{b[0]}"  # lon,lat;lon,lat
        qs = urlencode({"overview": "false"})
        url = f"{self.cfg.osrm_base_url}/route/v1/{prof}/{coords}?{qs}"

        try:
            with urlopen(url, timeout=3) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            routes = payload.get("routes") or []
            if not routes:
                raise RuntimeError("OSRM returned no routes")
            r0 = routes[0]
            dist_km = float(r0["distance"]) / 1000.0
            travel_sec = int(float(r0["duration"]))
            out = RouteMetric(distance_km=dist_km, travel_sec=travel_sec)
        except Exception:
            d = haversine_km(a, b)
            t = km_to_travel_sec(d, _speed_kmh(self.cfg, vehicle))
            out = RouteMetric(distance_km=d, travel_sec=t)

        if self.cfg.cache_routes:
            self._cache[self._key(a, b, vehicle)] = out
        return out


def make_router(cfg: SimConfig) -> Router:
    if cfg.routing_mode == "osrm":
        return OsrmRouter(cfg)
    return HaversineRouter(cfg)
