from pydantic import BaseModel
from typing import Literal


class SimConfig(BaseModel):
    # Simulation tick/dispatch
    dispatch_interval_sec: int = 60

    # Speeds (km/h)
    speed_bike_kmh: float = 15.0
    speed_motorbike_kmh: float = 25.0
    speed_car_kmh: float = 35.0

    # Service times
    pickup_service_sec: int = 45
    dropoff_service_sec: int = 45

    # ---- OBA controls ----
    same_restaurant_only: bool = True   # keep True, but "restaurant" is geo-grouped now
    oldest_orders_guardrail: int = 10

    # IMPORTANT FIX: dataset has unique restaurant names -> match by pickup proximity
    restaurant_match_radius_m: int = 1000  # <-- was too small before

    # Prep time model
    default_prep_time_sec: int = 8 * 60  # 8 minutes

    # OBA micro-batching / clustering
    batch_time_window_sec: int = 6 * 60
    pickup_cluster_radius_m: int = 1000
    dropoff_cluster_radius_m: int = 1600
    max_bundle_size: int = 4

    # Feasibility constraints (loosened so bundles actually pass)
    max_time_in_bag_sec: int = 35 * 60
    max_detour_ratio: float = 0.90  # <-- was too strict; killed bundles

    # Candidate pruning
    max_clusters_considered: int = 30
    max_bundles_per_cluster: int = 80
    max_couriers_per_cluster: int = 20

    # Scoring weights (slightly stronger bundle preference)
    w_distance: float = 1.0
    w_late_min: float = 2.0
    w_wait_min: float = 0.5
    bundle_bonus: float = 2.5  # <-- stronger push toward batching

    # ---- Routing ----
    routing_mode: Literal["haversine", "osrm"] = "haversine"
    osrm_base_url: str = "http://127.0.0.1:5000"
    cache_routes: bool = True
    cache_round_decimals: int = 4

    # ---- KPI thresholds (used by your SimEngine.kpis()) ----
    cold_in_bag_threshold_sec: int = 25 * 60
    max_food_wait_after_ready_sec: int = 8 * 60
    backlog_starvation_sec: int = 12 * 60
    extreme_late_sec: int = 12 * 60
