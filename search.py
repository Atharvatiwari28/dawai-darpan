import re
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.max_columns", 10)

# "Pan-D" aur "PAN D" ek hi naam hain: chhote akshar, dash → space
def simple_name(text):
    return re.sub(r"[-\s]+", " ", str(text).lower()).strip()


# ---------- Data: Postgres (live server) ya CSV (offline/dev) ----------
# Live server pe DATABASE_URL hota hai: sirf zaroori rows database se aati hain (kam memory).
# Bina DATABASE_URL ke: CSV files, pehli zaroorat pe ek hi baar load.
import db
from functools import lru_cache

BRAND_COLUMNS = ('name, manufacturer_name, price AS "price(₹)", pack_size_label, composition, key, form, '
                 'release, release_group, pack, per_unit')
GENERIC_COLUMNS = ('generic_name AS "Generic Name", unit_size AS "Unit Size", mrp AS "MRP", '
                   'group_name AS "Group Name", pack, per_unit, form, release, release_group, key')


def _as_row(record):
    """Database row → pandas Series jaisa (khaali = NaN), taaki baaki code pehle jaisa hi chale."""
    return pd.Series({k: (float("nan") if v is None else v) for k, v in record.items()})


@lru_cache(maxsize=1)
def _brands_csv():
    brands = pd.read_csv("data/brands_clean.csv")
    # NaN == NaN Python mein False hota hai, isliye "khaali" ko ek naam de do
    brands["release"] = brands["release"].fillna("normal")
    # Matching ke liye sirf ek sawaal zaroori hai: dheere ghulti hai (SR) ya nahi?
    # GR (coating) brand data mein aksar likha hi nahi hota, isliye use "normal" group mein rakho
    brands["release_group"] = brands["release"].replace("GR", "normal")
    brands["simple_name"] = brands["name"].apply(simple_name)
    return brands


@lru_cache(maxsize=1)
def _generics_csv():
    generics = pd.read_csv("data/medicines_clean.csv")
    generics["release"] = generics["release"].fillna("normal")
    generics["release_group"] = generics["release"].replace("GR", "normal")
    return generics


def _first_brand_matching(pattern, first_letter):
    if db.enabled():
        record = db.fetch_one(f"SELECT {BRAND_COLUMNS} FROM brands WHERE simple_name LIKE %s "
                              "AND simple_name ~ %s ORDER BY id LIMIT 1",
                              (first_letter.replace("%", r"\%").replace("_", r"\_") + "%", db.to_pg_regex(pattern)))
        return None if record is None else _as_row(record)
    brands = _brands_csv()
    matches = brands[brands["simple_name"].str.contains(pattern, regex=True)]
    return None if len(matches) == 0 else matches.iloc[0]


def find_brand(brand_name):
    """ "Pan-D" → brand ki row (Series), na mile toh None """
    query = simple_name(brand_name)
    if not query:
        return None
    # \b: "pan d" se "pan dsr" match na ho, sirf "pan d ..." ho
    brand = _first_brand_matching(r"^" + re.escape(query) + r"\b", query[0])
    if brand is None:
        # Dusri koshish: space ka farak maaf ("SolvinCold" = "Solvin Cold")
        letters = query.replace(" ", "")
        brand = _first_brand_matching(r"^" + r"\s?".join(re.escape(ch) for ch in letters) + r"\b", letters[0])
    return brand


def brand_names_starting_with(prefix):
    """ Same pehle 2 akshar wale saare brand naam (file ke order mein): "Kya aapka matlab...?" ke liye """
    if db.enabled():
        rows = db.fetch_all("SELECT name FROM brands WHERE simple_name LIKE %s ORDER BY id",
                            (prefix.replace("%", r"\%").replace("_", r"\_") + "%",))
        return [r["name"] for r in rows]
    brands = _brands_csv()
    return list(brands.loc[brands["simple_name"].str.startswith(prefix), "name"])


def find_generic(key, form, release_group):
    """ key + form + release_group → Jan Aushadhi ki row, na mile toh None.
        form None ho (photo se pata nahi chala) toh tablet/capsule maan lo. """
    if form is not None and pd.isna(form):
        return None                       # brand ka form data mein khaali: pehle jaisa, koi match nahi
    if db.enabled():
        if form is None:
            record = db.fetch_one(f"SELECT {GENERIC_COLUMNS} FROM generics WHERE key = %s AND release_group = %s "
                                  "AND form IN ('tablet', 'capsule') ORDER BY id LIMIT 1", (key, release_group))
        else:
            record = db.fetch_one(f"SELECT {GENERIC_COLUMNS} FROM generics WHERE key = %s AND release_group = %s "
                                  "AND form = %s ORDER BY id LIMIT 1", (key, release_group, form))
        return None if record is None else _as_row(record)
    generics = _generics_csv()
    found = generics[(generics["key"] == key) &
                     (generics["release_group"] == release_group)]
    if form is None:
        found = found[found["form"].isin(["tablet", "capsule"])]
    else:
        found = found[found["form"] == form]
    if len(found) == 0:
        return None
    return found.iloc[0]


def saving_percent(brand_price, generic_price):
    return (brand_price - generic_price) / brand_price * 100


if __name__ == "__main__":
    for name in ["Dolo 650", "Augmentin 625", "Pan 40", "Pan-D", "Telma 40"]:
        brand = find_brand(name)
        if brand is None:
            print(f"❌ {name}: brand nahi mila")
            continue
        generic = find_generic(brand["key"], brand["form"], brand["release_group"])
        if generic is None:
            print(f"😕 {brand['name']}: generic nahi mila")
            continue
        s = saving_percent(brand["per_unit"], generic["per_unit"])
        print(f"✅ {brand['name']} ₹{brand['per_unit']} → {generic['Generic Name']} "
              f"₹{generic['per_unit']}  ({s:.0f}% bachat)")
