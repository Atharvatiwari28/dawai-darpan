"""
Turn a typed place ("Kalyani Nagar", "near Phoenix mall, Viman Nagar, Pune") into lat/lon.

1. A map search service (address in the GEOCODER_URL setting), limited to India.
   Its rules: at most 1 request per second, a real User-Agent, cache results.
2. If that is not set, is down, or can't read the text (e.g. Devanagari): our own list of
   post office names, matched by sound so spelling mistakes still work.

We only search when a user types an area name (not for pincodes or browser location),
and we never store what the user typed.
"""
import os
import re
import threading
import time
from functools import lru_cache

import requests

URL = os.getenv("GEOCODER_URL")                 # not set: only the post office list is used
HEADERS = {"User-Agent": "DawaiDarpan/1.0 (student project)"}
TIMEOUT_SECONDS = 8
MIN_GAP_SECONDS = 1.1
MAX_TRIES = 4

# Words people add around a place name that confuse the map search
FILLER_WORDS = {
    "near", "nearby", "opp", "opposite", "behind", "beside", "next", "to", "the", "my", "address", "is", "in",
    "area", "location", "home", "house", "ghar", "mera", "meri", "mere", "hamara", "paas", "pass", "ke", "ki",
    "ka", "mein", "me", "hai", "hoon", "hu", "rehta", "rehti", "raha", "rahi", "se", "kendra", "jan", "aushadhi",
    "medical", "store", "shop", "find", "nearest", "closest", "where", "kahan", "batao", "dikhao", "please",
    "pls", "maza", "majha", "jawal", "jawalcha", "javal", "aahe", "ahe", "kuthe",
    # the same words as they come out of Devanagari -> English letters
    "main", "men", "mem", "rahata", "rahati", "rahate", "rahta", "hun", "hu", "hoon", "pasa", "kauna", "kaun",
    "sa", "si", "kya", "aur", "ji", "mala", "majhe", "maajhe", "jaa", "ja", "aapka", "apna", "apana",
}

POST_OFFICES_PATH = "data/post_offices.csv"     # offline mode only; the live app reads Postgres
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
OFFICE_SUFFIX = re.compile(r"[\s.]*\b(s\.?\s?o|b\.?\s?o|h\.?\s?o|p\.?\s?o)\.?$", re.I)
MATCH_WITH_CITY = 0.80        # how close the spelling must be when the user also named the city/district
MATCH_WITHOUT_CITY = 0.90     # stricter: many villages across India share similar names

_lock = threading.Lock()
_last_call = [0.0]


def to_roman(text):
    """Devanagari -> rough English letters ("वडगाव शेरी" -> "vadagava sheri"). Latin text is returned as is."""
    if not DEVANAGARI.search(text or ""):
        return text
    from indic_transliteration import sanscript
    roman = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    return roman.replace("M", "n").replace(".N", "n").lower()


def sound_key(text):
    """Spelling-proof key: consonants only, so Wadgaon / Vadgaon / vadagava all look alike."""
    text = to_roman(str(text)).lower()
    text = text.replace("w", "v").replace("ph", "f").replace("z", "j").replace("q", "k")
    text = re.sub(r"[^a-z]", "", text)
    text = text.replace("h", "")
    text = re.sub(r"[aeiouy]", "", text)
    return re.sub(r"(.)\1+", r"\1", text)                      # "tt" -> "t"


@lru_cache(maxsize=1)
def _offices_in_memory():
    import pandas as pd
    return pd.read_csv(POST_OFFICES_PATH, dtype={"pincode": str})


@lru_cache(maxsize=1)
def _districts():
    import db
    if db.enabled():
        return {r["district"] for r in db.fetch_all("SELECT DISTINCT district FROM post_offices")}
    return set(_offices_in_memory()["district"])


def _office_candidates(first_letter, min_len, max_len, district):
    """Post offices whose sound key starts with the same letter and has a similar length."""
    import db
    if db.enabled():
        sql = ("SELECT name, district_name AS \"District\", district, key, town, lat, lon FROM post_offices "
               "WHERE left(key, 1) = %s AND length(key) BETWEEN %s AND %s")
        params = [first_letter, min_len, max_len]
        if district:
            sql += " AND district = %s"
            params.append(district)
        return db.fetch_frame(sql, params)
    offices = _offices_in_memory()
    pool = offices[offices["district"] == district] if district else offices
    return pool[(pool["key"].str[0] == first_letter) & (pool["key"].str.len().between(min_len, max_len))]


def _find_in_post_offices(text):
    """Fallback: match the area name against post office names (e.g. "Vadgaon Sheri")."""
    import difflib
    words = [w for w in re.split(r"[\s,]+", to_roman(text).lower()) if w and w not in FILLER_WORDS]
    districts = _districts()
    city = next((w for w in words if w in districts), None)
    core = " ".join(w for w in words if w != city)
    key = sound_key(core)
    if len(key) < 3:
        return None
    pool = _office_candidates(key[0], len(key) - 2, len(key) + 2, city)
    if pool.empty:
        return None
    pool = pool.reset_index(drop=True)
    scores = pool["key"].map(lambda k: difflib.SequenceMatcher(None, key, k).ratio())
    best = scores.max()
    if best < (MATCH_WITH_CITY if city else MATCH_WITHOUT_CITY):
        return None
    top = pool[scores >= best - 0.001]
    if top["district"].nunique() > 1 and top["town"].any():
        top = top[top["town"]]                                 # a town's post office beats a same-sounding village
    if top["district"].nunique() > 1:
        # tie-break on the full spelling (vowels too): "hadapasara" is close to "Hadapsar", not "Deepa Sarai"
        spelled = re.sub(r"[^a-z]", "", to_roman(core).lower())
        close = top["name"].map(lambda n: difflib.SequenceMatcher(None, spelled, re.sub(r"[^a-z]", "", n.lower())).ratio())
        winner = close.idxmax()
        rivals = close[top["district"] != top.at[winner, "district"]]
        if rivals.empty or close[winner] - rivals.max() < 0.15:
            return None                                        # truly several places: ask for the city
        top = top.loc[[winner]]
    row = top.iloc[0]
    return {"lat": float(row["lat"]), "lon": float(row["lon"]),
            "label": f"{row['name'].title()}, {row['District'].strip().title()}"}


def _search(query):
    with _lock:                                    # one request at a time, 1 per second
        wait = MIN_GAP_SECONDS - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()
        params = {"q": query, "countrycodes": "in", "format": "jsonv2", "limit": 5,
                  "addressdetails": 1, "accept-language": "en"}
        response = requests.get(URL, params=params, headers=HEADERS, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return _best_match(response.json(), query)


# Parts of an address that name the place itself (not the district or state around it)
OWN_NAME_PARTS = ("suburb", "neighbourhood", "quarter", "city_district", "city", "town", "village",
                  "hamlet", "road", "amenity", "shop", "building", "mall", "railway", "leisure")
CITY_OR_SMALLER = 13          # map "place rank": 12 and below = district / state / country


def _mentions(result, words):
    """Does the result's OWN name contain what the user said? ('Navi Mumbai' is not 'Thane')"""
    import difflib
    address = result.get("address", {})
    names = [result.get("name") or ""] + [address.get(k) or "" for k in OWN_NAME_PARTS]
    keys = [sound_key(n) for n in names if n]
    for word in words:
        wk = sound_key(word)
        if len(wk) >= 2 and any(wk in k or difflib.SequenceMatcher(None, wk, k).ratio() >= 0.8 for k in keys):
            return True
    return False


def _best_match(results, query):
    """Pick the result that IS the place the user named, preferring a city/town/area over a whole district."""
    first_part = query.split(",")[0]
    words = [w for w in re.split(r"\s+", first_part.lower()) if len(w) >= 3 and w not in FILLER_WORDS]
    relevant = [r for r in results if _mentions(r, words)] if words else results
    if not relevant:
        return None                                   # nothing is really that place: try the next spelling
    specific = [r for r in relevant if int(r.get("place_rank") or 30) >= CITY_OR_SMALLER]
    return (specific or relevant)[0]


def _label(result, query=""):
    """Short readable name, e.g. 'Kalyani Nagar, Pune'.
    If the map matched a landmark (a flyover, a shop), name it with the user's own words instead."""
    address = result.get("address", {})
    if result.get("category") in ("place", "boundary"):
        place = result.get("name") or ""
    else:
        place = (query.split(",")[0].strip().title() if query else "") or result.get("name") or ""
    city = ""
    for key in ("city", "town", "village", "state_district", "county", "state"):
        value = re.sub(r"\s+(City\s+)?(Subdistrict|District|Tahsil|Taluka)$", "", address.get(key) or "").strip()
        if value and value.lower() != place.lower():
            city = value
            break
    parts = [p for p in (place, city) if p]
    if len(parts) == 2 and parts[0].lower() == parts[1].lower():
        parts = parts[:1]
    return ", ".join(parts) or result.get("display_name", "").split(",")[0]


def _candidates(text):
    """Full text first, then without filler words, then the most specific comma part."""
    text = re.sub(r"\s+", " ", str(text or "")).strip(" ,.")
    tries = [text]
    words = [w for w in re.split(r"\s+", re.sub(r"[^\w\s,]", " ", text)) if w]
    cleaned = " ".join(w for w in words if w.lower().strip(",") not in FILLER_WORDS).strip(" ,")
    if cleaned and cleaned.lower() != text.lower():
        tries.append(cleaned)
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if len(parts) > 1:
        tries.append(parts[0])                       # "Thane, Mumbai" -> "Thane" (the city hint can mislead)
        tries.append(", ".join(parts[-2:]))
    unique = []
    for t in tries:
        if t and t.lower() not in [u.lower() for u in unique]:
            unique.append(t)
    return unique[:MAX_TRIES]


@lru_cache(maxsize=512)
def _find_place_cached(key):
    devanagari = bool(DEVANAGARI.search(key))
    if devanagari:                                   # the map mostly can't read Devanagari: try post offices first
        local = _find_in_post_offices(key)
        if local:
            return local
    if not URL:
        return None if devanagari else _find_in_post_offices(key)
    try:
        for query in _candidates(to_roman(key) if devanagari else key):
            result = _search(query)
            if result:
                return {"lat": float(result["lat"]), "lon": float(result["lon"]), "label": _label(result, query)}
    except requests.RequestException:
        local = None if devanagari else _find_in_post_offices(key)
        if local:
            return local                             # map service down: post offices still work
        raise
    return None if devanagari else _find_in_post_offices(key)


def find_place(text):
    """Returns {"lat", "lon", "label"}, None if not found. Raises requests.RequestException if offline."""
    key = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if len(key) < 3:
        return None
    return _find_place_cached(key)


if __name__ == "__main__":
    for q in ["Kalyani Nagar", "Vadgaon Sheri", "वडगाव शेरी पुणे", "वर्डगाव शेरी", "मैं वडगाव शेरी पूने में रहता हूँ"]:
        print(q, "->", find_place(q))
