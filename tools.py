"""Dawai Darpan ke TOOLS: chhote, saaf functions jinhe AI agent khud chunega.
Har tool ek dict lautata hai (JSON ban sake), aur galti ko bhi dict mein batata hai."""
import json
from pathlib import Path
from functools import lru_cache
import pandas as pd
from typing import Annotated
from langgraph.types import interrupt
from langgraph.prebuilt import InjectedState
from clean import find_salts, find_strengths, make_key, find_form, find_release
import difflib
from search import find_brand, find_generic, saving_percent, simple_name, brand_names_starting_with
import db

DISCLAIMER = "Dawai badalne se pehle doctor ya pharmacist se zaroor poochein."


def _number(x):
    """ numpy/pandas ka number → normal Python float (ya None). JSON sirf normal types samajhta hai. """
    if x is None or pd.isna(x):
        return None
    return round(float(x), 2)


def _generic_answer(key, form, release_group, brand_price, price_source, brand_release="normal"):
    """ Dono tools ka common hissa: key se Jan Aushadhi generic + bachat """
    generic = find_generic(key, form, release_group)
    if generic is None:
        return {"found": False, "key": key,
                "reason": "Jan Aushadhi mein exact generic (same salt + strength + form) nahi mila"}
    answer = {
        "found": True,
        "key": key,
        "generic_name": generic["Generic Name"],
        "generic_pack": generic["Unit Size"],
        "generic_mrp": _number(generic["MRP"]),
        "generic_per_unit": _number(generic["per_unit"]),
        "brand_per_unit": _number(brand_price),
        "brand_price_source": price_source,
        "saving_percent": None,
        "saving_per_unit": None,
        "disclaimer": DISCLAIMER,
    }
    if answer["brand_per_unit"] and answer["generic_per_unit"]:
        answer["saving_percent"] = round(saving_percent(answer["brand_per_unit"],
                                                        answer["generic_per_unit"]))
        answer["saving_per_unit"] = round(answer["brand_per_unit"] - answer["generic_per_unit"], 2)
    warnings = []
    if form is None:
        warnings.append("Form pakka nahi pata, tablet/capsule maana hai")
    if generic["release"] != brand_release:
        warnings.append(f"Coating alag ho sakti hai (brand: {brand_release}, generic: {generic['release']})")
    if warnings:
        answer["warnings"] = warnings
    return answer


# ---------- "Kya aapka matlab ...?" ----------
# Whisper/typing se naam galat aa sakta hai ("Augmentn"). Milte-julte naam SIRF SUJHAAV hain:
# user khud chunega. Apne-aap kabhi match nahi karte (dawai mein galat match = khatra).
@lru_cache(maxsize=256)
def _names_by_prefix(prefix):
    """ Pehle 2 akshar same wale naam: 2.4 lakh mein se kuch hazaar (har prefix ek hi baar) """
    index = {}
    for full in dict.fromkeys(brand_names_starting_with(prefix)):
        index.setdefault(" ".join(simple_name(full).split()[:2]), full)
    return index


def suggest_brands(query, n=3):
    q = simple_name(query)
    if len(q) < 2:
        return []
    pool = _names_by_prefix(q[:2])
    best = difflib.get_close_matches(q, list(pool), n=n, cutoff=0.6)
    return [pool[b] for b in best]


def search_by_brand(brand_name: str, composition: str | None = None,
                    mrp: float | None = None, pack_count: int | None = None) -> dict:
    """Brand ke naam se (jaise 'Dolo 650', 'Pan-D') uska saasta Jan Aushadhi generic aur bachat dhoondho.
    Agar photo/user se composition (salt + strength) pata hai toh woh bhi do: wahi sabse bharosemand hai.
    Agar user ne strip ka MRP aur goliyon ki ginti batayi ho, toh woh bhi do: wahi asli daam hai."""
    brand = find_brand(brand_name)
    photo_key = None
    if composition:
        photo_key = make_key(find_salts(composition), find_strengths(composition))

    if brand is None or pd.isna(brand["key"]):
        if photo_key:                                    # brand nahi mila, lekin salt pata hai
            return search_by_salt(composition, mrp=mrp, pack_count=pack_count)
        return {"found": False, "reason": f"'{brand_name}' brand data mein nahi mila. "
                                          "Salt/composition pata ho toh search_by_salt use karo.",
                "did_you_mean": suggest_brands(brand_name)}

    # 🔍 Photo ka salt aur dataset ka salt alag? Photo jeetegi (dataset kabhi 2 se zyada salt nahi rakhta)
    if photo_key and photo_key != brand["key"]:
        answer = search_by_salt(composition, mrp=mrp, pack_count=pack_count)
        # User ke liye seedhi baat: doosri dawai ke salt ya "dataset" ka zikr nahi
        answer.setdefault("warnings", []).append(
            "Is naam se ek aur dawai bhi milti hai jisme alag salt hain. "
            "Humne strip pe likhe salt ke hisaab se hi dhoondha hai.")
        return answer

    if mrp and pack_count:
        price, source = mrp / pack_count, "strip pe chhapa MRP"
        strip_price, strip_pack = mrp, f"{pack_count} goli"
    else:
        price, source = brand["per_unit"], "approx (asli daam alag ho sakta hai)"
        strip_price, strip_pack = brand["price(₹)"], brand["pack_size_label"]
    answer = _generic_answer(brand["key"], brand["form"], brand["release_group"], price, source,
                             brand_release=brand["release"])
    answer["brand_name"] = brand["name"]
    answer["manufacturer"] = brand["manufacturer_name"]
    # AI ko poore number do, taaki use khud calculate na karna pade 🧮
    answer["brand_strip_price"] = _number(strip_price)
    answer["brand_strip_pack"] = strip_pack
    return answer


def search_by_salt(composition: str, mrp: float | None = None, pack_count: int | None = None,
                   salt_guessed_by_ai: bool = False) -> dict:
    """Salt aur strength se (jaise 'Azithromycin 500 mg' ya 'Paracetamol 650mg') Jan Aushadhi generic dhoondho.
    Strength zaroori hai. MRP aur goliyon ki ginti pata ho toh do, taaki bachat nikle.
    salt_guessed_by_ai=True tab do jab salt user/photo ne nahi bataya, balki tumne apni jaankari se pehchana."""
    key = make_key(find_salts(composition), find_strengths(composition))
    if key is None:
        return {"found": False, "reason": "Salt ya strength samajh nahi aaya. "
                                          "Salt ka naam aur strength (mg) dono chahiye."}
    form = find_form(composition)
    release = find_release(composition) or "normal"
    release_group = "SR" if release == "SR" else "normal"
    price = mrp / pack_count if (mrp and pack_count) else None
    source = "strip pe chhapa MRP" if price else None
    answer = _generic_answer(key, form, release_group, price, source, brand_release=release)
    if salt_guessed_by_ai:          # code se pakka: AI ka andaaza hai, yeh user ko zaroor bataya jaaye
        answer.setdefault("warnings", []).append(
            "Is dawai ka salt hamare database mein nahi tha; maine apni aam jaankari se pehchana hai. "
            "Strip ya dabbe pe likha salt aur strength zaroor milayein.")
    if price:
        answer["brand_strip_price"] = _number(mrp)
        answer["brand_strip_pack"] = f"{pack_count} goli"
    return answer


# ---------- Dawai kis kaam aati hai (dataset se, AI ki yaaddasht se NAHI) ----------
INFO_FILE = Path("data/medicine_uses.csv")


@lru_cache(maxsize=1)
def _load_info():
    """ 2.48 lakh dawaiyon ka "kis kaam aati hai" data (offline mode), sirf pehli zaroorat pe load karo """
    if not INFO_FILE.is_file():
        return None
    df = pd.read_csv(INFO_FILE, low_memory=False)
    df["simple_name"] = df["name"].apply(simple_name)
    return df


def _non_empty(row, prefix):
    """ use0, use1, ... jaise columns mein se khaali hata ke list banao """
    cols = [c for c in row.index if c.lower().startswith(prefix.lower())]
    return [str(row[c]).strip() for c in cols if pd.notna(row[c]) and str(row[c]).strip()]


def _find_info(brand_name):
    """ Naam se dawai ki jaankari: {name, uses, side_effects, therapeutic_class, habit_forming} ya None """
    import re
    query = simple_name(brand_name)
    if not query:
        return None
    pattern = r"^" + re.escape(query) + r"\b"
    if db.enabled():
        return db.fetch_one("SELECT name, uses, side_effects, therapeutic_class, habit_forming FROM medicine_uses "
                            "WHERE simple_name LIKE %s AND simple_name ~ %s ORDER BY id LIMIT 1",
                            (query[0].replace("%", r"\%").replace("_", r"\_") + "%", db.to_pg_regex(pattern)))
    df = _load_info()
    if df is None:
        return None
    matches = df[df["simple_name"].str.contains(pattern, regex=True)]
    if len(matches) == 0:
        return None
    row = matches.iloc[0]
    return {"name": row["name"], "uses": _non_empty(row, "use"), "side_effects": _non_empty(row, "sideEffect"),
            "therapeutic_class": row.get("Therapeutic Class"), "habit_forming": row.get("Habit Forming")}


def medicine_info(brand_name: str, include_side_effects: bool = False) -> dict:
    """Brand dawai AAM TAUR PE kis kaam aati hai (uses) batao.
    include_side_effects=True SIRF tab do jab user ne KHUD side effects ke baare mein poocha ho.
    SIRF tab chalao jab user ne KHUD poocha ho ki dawai kis kaam aati hai ya side effects kya hain.
    Saste generic ya daam ke sawaal pe yeh tool MAT chalao.
    Kisi insaan ko kaunsi dawai ya kitni dose leni chahiye, yeh is tool se KABHI mat batao."""
    if not db.enabled() and not INFO_FILE.is_file():
        return {"found": False, "reason": "Dawai ki jaankari wala data abhi available nahi hai"}
    info = _find_info(brand_name)
    if info is None:
        return {"found": False, "reason": f"'{brand_name}' ki jaankari data mein nahi mili",
                "did_you_mean": suggest_brands(brand_name)}
    clean = lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else v
    return {
        "found": True,
        "name": info["name"],
        "used_for": list(info["uses"]),
        # Side effects sirf maangne pe: bina pooche dikhana user ko darata hai, aur woh poocha bhi nahi
        "common_side_effects": list(info["side_effects"])[:6] if include_side_effects else None,
        "therapeutic_class": clean(info["therapeutic_class"]),
        "habit_forming": clean(info["habit_forming"]),
        "disclaimer": "Yeh sirf aam jaankari hai. Yeh dawai aapke liye sahi hai ya nahi, aur kitni leni hai, "
                      "yeh sirf doctor bata sakte hain.",
    }


@lru_cache(maxsize=32)
def _read_photo_once(photo_path):
    """ Gemini ko ek photo ke liye sirf EK baar bulao (yaad rakho, interrupt ke baad tool dobara chalta hai) """
    from read_strip import read_strip           # yahan import: Gemini sirf zaroorat pe load ho
    return read_strip(photo_path)


# Photo confirm karne ka sawaal, user ki chuni hui bhasha mein
CONFIRM_QUESTION = {
    "Hinglish": "Maine photo mein '{name}' padha (MRP: {mrp}, goliyan: {pack}). Kya yeh sahi hai? 'haan' bolo, ya sahi naam likho.",
    "Hindi":    "मैंने फोटो में '{name}' पढ़ा (MRP: {mrp}, गोलियाँ: {pack})। क्या यह सही है? 'हाँ' लिखें, या सही नाम लिखें।",
    "English":  "I read '{name}' in the photo (MRP: {mrp}, tablets: {pack}). Is that right? Type 'yes', or type the correct name.",
    "Marathi":  "मी फोटोमध्ये '{name}' वाचले (MRP: {mrp}, गोळ्या: {pack}). हे बरोबर आहे का? 'हो' लिहा, किंवा बरोबर नाव लिहा.",
}


def read_attached_photo(state: Annotated[dict, InjectedState]) -> dict:
    """User ne jo dawai ki photo ATTACH ki hai, use padho (brand naam, salt, MRP, goliyon ki ginti).
    Padhne ke baad user se confirm karwaya jaata hai. Uske baad search_by_brand ya search_by_salt use karo."""
    photo_path = state.get("photo")             # UI ne photo save karke uska raasta State mein rakha
    if not photo_path or not Path(photo_path).is_file():
        return {"read": False, "reason": "User ne koi photo attach nahi ki. Unse photo attach karne ko kaho."}

    from read_strip import NotAnImage
    try:
        info = _read_photo_once(photo_path)
    except NotAnImage:
        return {"read": False, "reason": "Yeh file photo nahi hai (ya kharab hai). "
                                          "Dawai ki photo (JPG, PNG, HEIC, WEBP...) attach karne ko kaho."}
    if info is None:
        return {"read": False, "reason": "Photo padhne wala AI abhi busy hai, thodi der baad try karo"}
    if not info.is_medicine:
        return {"read": False, "reason": "Photo mein dawai nahi dikh rahi"}

    reading = info.model_dump()                 # Pydantic object → dict
    shown = info.brand_name or info.composition or "kuch saaf nahi"
    # ⏸️ Graph yahan RUKTA hai. User ka jawab aane pe yahin se aage chalta hai.
    language = state.get("language") or "Hinglish"
    template = CONFIRM_QUESTION.get(language, CONFIRM_QUESTION["Hinglish"])
    user_answer = interrupt({
        "question": template.format(name=shown, mrp=info.mrp, pack=info.pack_count),
    })
    return {"read": True, "photo_reading": reading, "user_confirmation": user_answer}


if __name__ == "__main__":
    tests = [
        search_by_brand("Dolo 650"),
        search_by_brand("Pan 40"),
        search_by_brand("Pan-D", mrp=210.0, pack_count=15),
        search_by_brand("Bio-AZ 500"),
        search_by_salt("Azithromycin Anhydrous 500 mg", mrp=75.5, pack_count=3),
        search_by_salt("Hydrochloride Tablets"),
    ]
    for result in tests:
        print(json.dumps(result, indent=2, ensure_ascii=False))   # JSON ban gaya = AI ko bhej sakte hain ✅
        print("-" * 50)


def find_kendra(place: str) -> dict:
    """User ke paas ke Jan Aushadhi Kendra (dukaan) dhoondho: pata aur doori.
    Tab chalao jab user pooche ki saasti generic dawai KAHAN milegi / paas ka Kendra / kahan se khareedein.
    place = user ki batayi jagah, ENGLISH akshar mein jaisa map pe likha hota hai: 6 ankon ka pincode, ya
    area/pata + shahar (jaise "Wadgaon Sheri, Pune"). Awaaz ki galat spelling sudhaar sakte ho, jagah wahi rakho.
    Khud se pincode ya shahar KABHI mat banao. User ne koi jagah nahi batayi toh yeh tool MAT chalao, pehle poochho."""
    from kendra import kendras_near_place
    answer = kendras_near_place(place, limit=5)
    if answer.get("found"):
        answer["same_price_everywhere"] = ("Jan Aushadhi ka daam har Kendra pe ek jaisa hai, isliye sabse paas "
                                           "wala Kendra hi sabse sasta padega.")
    return answer
