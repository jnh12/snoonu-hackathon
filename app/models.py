from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, List


OrderStatus = Literal["PENDING", "ASSIGNED", "PICKED_UP", "DELIVERED"]
VehicleType = Literal["bike", "motorbike", "car"]
StopType = Literal["pickup", "dropoff"]


@dataclass
class Order:
    order_id: str
    created_sec: int
    pickup_restaurant: str
    pickup_lat: float
    pickup_lng: float
    dropoff_area: str
    dropoff_lat: float
    dropoff_lng: float
    estimated_delivery_time_min: int

    # computed by SimEngine.load()
    ready_sec: int = 0
    promised_sec: int = 0

    # lifecycle + telemetry (used by SimEngine)
    status: OrderStatus = "PENDING"
    assigned_courier_id: Optional[str] = None
    assigned_sec: Optional[int] = None

    pickup_arrive_sec: Optional[int] = None
    pickup_depart_sec: Optional[int] = None
    dropoff_arrive_sec: Optional[int] = None
    delivered_sec: Optional[int] = None

    assigned_bundle_size: Optional[int] = None


@dataclass
class Stop:
    stop_type: StopType
    order_id: str
    lat: float
    lng: float
    arrive_sec: int
    depart_sec: int
    restaurant_or_area: str


@dataclass
class RoutePlan:
    courier_id: str
    assigned_order_ids: List[str]
    stops: List[Stop] = field(default_factory=list)

    total_distance_km: float = 0.0
    total_wait_sec: int = 0
    total_late_sec: int = 0
    finish_sec: int = 0


@dataclass
class Courier:
    courier_id: str
    available_sec: int
    starting_area_hint: str
    lat: float
    lng: float
    vehicle_type: VehicleType
    bundle_capacity: int

    status: Literal["OFFLINE", "IDLE", "BUSY"] = "OFFLINE"
    busy_until_sec: int = 0
    route: Optional[RoutePlan] = None

    # used by SimEngine._progress_routes()
    route_idx: int = 0
    busy_start_sec: int = 0
    total_busy_sec: int = 0
