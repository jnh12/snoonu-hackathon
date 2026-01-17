import csv
from typing import List
from .models import Order, Courier


def _time_to_sec(hms: str) -> int:
    h, m, s = hms.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def load_orders(path: str) -> List[Order]:
    out: List[Order] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(
                Order(
                    order_id=row["order_id"],
                    created_sec=_time_to_sec(row["created_time"]),
                    pickup_restaurant=row["pickup_restaurant"],
                    pickup_lat=float(row["pickup_lat"]),
                    pickup_lng=float(row["pickup_lng"]),
                    dropoff_area=row["dropoff_area"],
                    dropoff_lat=float(row["dropoff_lat"]),
                    dropoff_lng=float(row["dropoff_lng"]),
                    estimated_delivery_time_min=int(row["estimated_delivery_time_min"]),
                )
            )
    return out


def load_couriers(path: str) -> List[Courier]:
    out: List[Courier] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(
                Courier(
                    courier_id=row["courier_id"],
                    available_sec=_time_to_sec(row["available_from"]),
                    starting_area_hint=row["starting_area_hint"],
                    lat=float(row["courier_lat"]),
                    lng=float(row["courier_lng"]),
                    vehicle_type=row["vehicle_type"],
                    bundle_capacity=int(row["bundle_capacity"]),
                )
            )
    return out
