"""
Nearest Jan Aushadhi Kendra search.

Every Kendra sells at the same Jan Aushadhi MRP, so the nearest Kendra is the
cheapest place to buy the generic. We don't know live stock, so answers tell
the user to check availability before going.

Locations are pincode centres, so distances are approximate.
"""
import re
from functools import lru_cache
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

KENDRAS_PATH = "data/kendras_clean.csv"
PINCODES_PATH = "data/pincode_centroids.csv"
EARTH_RADIUS_KM = 6371.0
DEFAULT_LIMIT = 5
MAX_LIMIT = 10
MAX_DISTANCE_KM = 50          # beyond this, "nearest" is not useful; say so instead


@lru_cache(maxsize=1)
def _kendras():
    """ ~22 hazaar Kendra: chhoti table, ek baar memory mein (database se ya CSV se) """
    import db
    if db.enabled():
        data = db.fetch_frame("SELECT code, name, state, district, pincode, address, lat, lon FROM kendras "
                              "WHERE lat IS NOT NULL AND lon IS NOT NULL")
    else:
        data = pd.read_csv(KENDRAS_PATH, dtype={"pincode": str, "code": str})
    return data.dropna(subset=["lat", "lon"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def _pincodes():
    import db
    if db.enabled():
        data = db.fetch_frame("SELECT pincode, lat, lon, district, state, precision FROM pincodes")
    else:
        data = pd.read_csv(PINCODES_PATH, dtype={"pincode": str})
    return data.set_index("pincode")


def _distance_km(lat, lon, lats, lons):
    """Haversine distance from one point to many points."""
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2, lon2 = np.radians(lats), np.radians(lons)
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _title(text):
    """'PUNE' -> 'Pune', but keep short codes like 'NTR'."""
    return text.title() if text.isupper() and len(text) > 3 else text


def _maps_links(row):
    place = f"Jan Aushadhi Kendra {row['address']}, {row['district']}, {row['state']} {row['pincode']}"
    return {
        "maps_url": "https://www.google.com/maps/search/?api=1&query=" + quote_plus(place),
        "directions_url": "https://www.google.com/maps/dir/?api=1&destination=" + quote_plus(place),
    }


def _result(row, distance):
    item = {                                     # owner's personal name is left out on purpose
        "kendra_code": row["code"],
        "address": row["address"],
        "district": _title(str(row["district"])),
        "state": row["state"],
        "pincode": row["pincode"],
        "distance_km": round(float(distance), 1),
    }
    item.update(_maps_links(row))
    return item


def nearest_kendras(lat, lon, limit=DEFAULT_LIMIT):
    """Nearest Kendras to a point (browser location or a pincode centre)."""
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return {"found": False, "reason": "location samajh nahi aayi"}
    if not (6 <= lat <= 38 and 68 <= lon <= 98):
        return {"found": False, "reason": "yeh location Bharat ke bahar lag rahi hai"}

    limit = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
    data = _kendras()
    distances = _distance_km(lat, lon, data["lat"].to_numpy(), data["lon"].to_numpy())
    order = np.argsort(distances)[:limit]
    kendras = [_result(data.iloc[i], distances[i]) for i in order if distances[i] <= MAX_DISTANCE_KM]
    if not kendras:
        return {"found": False, "reason": f"{MAX_DISTANCE_KM} km ke andar koi Kendra nahi mila"}
    return {
        "found": True,
        "kendras": kendras,
        "note": "Doori andaazan hai. Jaane se pehle Kendra pe dawai ka stock pooch lein.",
    }


def kendras_near_pincode(pincode, limit=DEFAULT_LIMIT):
    """Nearest Kendras to a 6-digit pincode."""
    match = re.search(r"\b(\d{6})\b", str(pincode or ""))
    if not match:
        return {"found": False, "reason": "6 ankon ka pincode chahiye, jaise 411001"}
    pin = match.group(1)
    pins = _pincodes()
    if pin not in pins.index:
        return {"found": False, "reason": f"pincode {pin} hamari list mein nahi mila"}
    centre = pins.loc[pin]
    answer = nearest_kendras(centre["lat"], centre["lon"], limit)
    if answer["found"]:
        answer["searched_near"] = f"{pin} ({_title(str(centre['district']))})"
    return answer


def kendras_near_place(place, limit=DEFAULT_LIMIT):
    """Nearest Kendras to whatever the user typed: a 6-digit pincode, or an area/address."""
    text = re.sub(r"\s+", " ", str(place or "")).strip()
    if re.search(r"\b\d{6}\b", text):
        return kendras_near_pincode(text, limit)
    if len(text) < 3:
        return {"found": False, "reason": "area ka naam ya 6 ankon ka pincode chahiye"}
    from geocode import find_place
    import requests
    try:
        spot = find_place(text)
    except requests.RequestException:
        return {"found": False, "reason": "abhi area ka naam dhoondh nahi paaye, pincode bata dijiye"}
    if not spot:
        return {"found": False, "reason": f"'{text}' jagah nahi mili; shahar ke saath area ka naam ya pincode batayein"}
    answer = nearest_kendras(spot["lat"], spot["lon"], limit)
    if answer["found"]:
        answer["searched_near"] = spot["label"]
        answer["note"] = "Doori andaazan hai. Jaane se pehle Kendra pe dawai ka stock pooch lein."
    return answer


if __name__ == "__main__":
    import json
    print(json.dumps(kendras_near_pincode("411001", 3), indent=2, ensure_ascii=False))
    print(json.dumps(nearest_kendras(18.5204, 73.8567, 2), indent=2, ensure_ascii=False))
