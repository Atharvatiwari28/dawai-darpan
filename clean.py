import re
import pandas as pd
import numpy as np

pd.set_option("display.width", 200)          # terminal mein columns mat chhupao
pd.set_option("display.max_colwidth", 45)    # lambe naam chhote dikhao


# ---------- 1. Pack mein kitni goliyan ----------
def pack_count(unit_size):
    """Pack mein kitni goliyan: "10's" → 10,  "100 ml" → None"""
    unit_size = unit_size.replace("’", "'")
    match = re.match(r"^\s*(\d+)\s*'?\s*s\b", unit_size, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def brand_pack_count(label):
    """Brand pack mein kitni goliyan: "strip of 15 tablets" → 15,  "bottle of 100 ml syrup" → None"""
    label = str(label).lower()
    match = re.search(r"\bof\s+(\d+)\s+[a-z ]*?(tablet|capsule)", label)
    if match:
        return int(match.group(1))
    return None


# ---------- 2. Dawai ka form ----------
# shabd jo naam mein dikhta hai  →  asli form
FORM_WORDS = {
    "injection": "injection", "infusion": "injection", "vaccine": "injection",
    "tablet": "tablet",
    "capsule": "capsule",
    "rotacap": "inhaler", "respule": "inhaler", "inhaler": "inhaler", "inhalation": "inhaler",
    "syrup": "syrup",
    "suspension": "suspension",
    "solution": "solution", "mouthwash": "solution",
    "drop": "drops",
    "spray": "spray",
    "cream": "cream",
    "gel": "gel",
    "ointment": "ointment",
    "lotion": "lotion",
    "shampoo": "shampoo",
    "powder": "powder",
    "lozenge": "lozenge",
    "sachet": "sachet",
    "strip": "strip",
    "patch": "patch",
    "oil": "oil",
}

def find_form(name):
    """ "Metronidazole Infusion IP 500 mg" → "injection" """
    name = name.lower()
    for word, form in FORM_WORDS.items():
        if re.search(r"\b" + word, name):
            return form
    return None


# ---------- 3. Release type (dawai body mein kaise ghulti hai) ----------
RELEASE_WORDS = {
    "sustained": "SR", "prolonged": "SR", "extended": "SR",
    "modified": "SR", "controlled": "SR", "long acting": "SR",
    "sr": "SR", "er": "SR", "xr": "SR", "pr": "SR", "cr": "SR",   # brand labels ke chhote roop
    "gastro-resistant": "GR", "gastro resistant": "GR",
    "enteric": "GR", "delayed": "GR",
}

def find_release(name):
    """ "Diclofenac Sodium Prolonged Release Tablets" → "SR",  normal → None """
    name = name.lower()
    for word, release in RELEASE_WORDS.items():
        if re.search(r"\b" + word + r"\b", name):      # dono taraf \b: "er" ≠ "erythromycin"
            return release
    return None


# ---------- 4. Strength (kitni taakat) ----------
def expand_slash(name):
    """ "Amlodipine + Enalapril Tablet (5/5 Mg)" → "Amlodipine + Enalapril Tablet 5mg, 5mg" """
    match = re.search(r"\(([\d./]+)\s*(mg|mcg)\)", name, re.IGNORECASE)
    if match:
        numbers = match.group(1).split("/")          # "5/5" → ["5", "5"]
        unit = match.group(2)                        # "Mg"
        new_text = ", ".join(n + unit for n in numbers)   # "5Mg, 5Mg"
        name = name.replace(match.group(0), new_text)
    return name

# number (dashamlav ke saath bhi)  +  space (ho ya na ho)  +  unit
STRENGTH_PATTERN = r"(\d+\.?\d*)\s*(mcg|mg|gm|g|iu|%)"

def find_strengths(name):
    """ "Telmisartan 80mg and Cilnidipine 10mg" → [(80.0, "mg"), (10.0, "mg")] """
    name = expand_slash(name)
    matches = re.findall(STRENGTH_PATTERN, name, re.IGNORECASE)
    result = []
    for number, unit in matches:
        unit = unit.lower()
        if unit == "gm":            # "gm" aur "g" ek hi cheez
            unit = "g"
        result.append((float(number), unit))
    return result


# ---------- 5. Salts (dawai ke asli ingredients) ----------
# ek hi salt, alag spelling → ek standard spelling
SPELLINGS = {
    "amoxycillin": "amoxicillin",
    "guaiphenesin": "guaifenesin",
    "cetrizine": "cetirizine",
    "levocetrizine": "levocetirizine",
    "sulphate": "sulfate",
    "hcl": "hydrochloride",
    # brand vs generic data ki tulna se mile (har ek haath se check kiya ✋)
    "tazobactum": "tazobactam",
    "beclometasone": "beclomethasone",
    "sulphadiazine": "sulfadiazine",
    "picosulphate": "picosulfate",
    "sulphide": "sulfide",
    "torasemide": "torsemide",
    "linceed": "linseed",
    "cyanocobalamine": "cyanocobalamin",
    "ethamsylate": "etamsylate",
    "nimesulid": "nimesulide",
    "dithylamine": "diethylamine",
    "flupenthixol": "flupentixol",
    "cefalexin": "cephalexin",
    "oxelate": "oxalate",
    "lipolic": "lipoic",
    "ciclosporin": "cyclosporin",
    "cyclosporine": "cyclosporin",
    "trihexiphenidyl": "trihexyphenidyl",
    "flurouracil": "fluorouracil",
    "carboxymethlycellulose": "carboxymethylcellulose",
    "thiocholchicoside": "thiocolchicoside",
    "nortryptyline": "nortriptyline",
    "megaldrate": "magaldrate",
}

# poore salt ke doosre naam (ek se zyada shabd) → ek standard naam
SALT_NAMES = {
    "potassium clavulanate": "clavulanic acid",
    "povidone-iodine": "povidone iodine",
    "methyl ergometrine": "methylergometrine",
    "ethinylestradiol": "ethinyl estradiol",
    "isosorbidemononitrate": "isosorbide mononitrate",
    "levo-carnitine": "levocarnitine",
    "l-carnitine": "levocarnitine",
    "di-cloxacillin": "dicloxacillin",
    "s - amlodipine": "s-amlodipine",
    "methylsalicylate": "methyl salicylate",
    "acetylsalicylic acid aspirin": "aspirin",
    "acetylsalicylic acid": "aspirin",
}

# salt ka "saathi" hissa: dawai wahi rehti hai, isliye match ke liye hata do
SALT_SUFFIXES = {"sodium", "potassium", "hydrochloride", "dihydrochloride",
                 "sulfate", "maleate", "besylate", "besilate", "hydrobromide",
                 "trihydrate", "monohydrate", "dihydrate", "anhydrous", "calcium"}

def normalize_salt(salt):
    """ "Amoxycillin Trihydrate" → "amoxicillin",  "Sodium Chloride" → "sodium chloride" """
    words = salt.lower().split()
    words = [SPELLINGS.get(w, w) for w in words]         # spelling theek karo
    first, rest = words[:1], words[1:]
    rest = [w for w in rest if w not in SALT_SUFFIXES]   # pehla shabd kabhi mat hatao
    salt = " ".join(first + rest)
    return SALT_NAMES.get(salt, salt)                    # poora naam badlo (agar list mein hai)


# yeh shabd salt ka hissa nahi hain, inhe hata do
JUNK_WORDS = (r"\b(ip|bp|usp|tablets?|capsules?|injection|infusion|syrup|oral|orally|suspension|"
              r"solution|cream|gel|ointment|eye|ear|nasal|ophthalmic|drops?|lotion|powder|"
              r"spray|inhalation|rotacaps?|respules?|strips?|dispersible|disintegrating|"
              r"for|sustained|prolonged|extended|controlled|modified|delayed|release|relase|"
              r"gastro|resistant|resistance|enteric|coated|film|uncoated|"
              r"paediatric|dry|w/w|w/v|v/v|per|ml|lozenges?|pellets?|each|contains)\b")

def find_salts(name):
    """ "Telmisartan 80mg and Cilnidipine 10mg Tablets" → ["telmisartan", "cilnidipine"] """
    name = expand_slash(name).lower()                   # 1. slash kholo, chhote akshar
    name = re.sub(r"\(\s*as\b[^)]*\)", " ", name)        #    "(as Azithromycin dihydrate IP)" hatao
    name = re.sub(r"\bequivalent to\b", " ", name)       #    "... equivalent to Pantoprazole" 
    name = re.sub(STRENGTH_PATTERN, " ", name)          # 2. "80mg" hatao
    name = re.sub(r"\d+\.?\d*\s*ml\b", " ", name)        #    "/5ml" bhi hatao
    name = re.sub(JUNK_WORDS, " ", name)                # 3. "tablets", "ip" hatao
    parts = re.split(r"\band\b|\+|,|&", name)          # 4. jodne wale shabdon pe todo
    salts = []
    for part in parts:
        part = re.sub(r"[^a-z\- ]", " ", part)          # 5. bracket, number, chinh hatao
        part = " ".join(part.split()).strip(" -")       #    faltu space/dash hatao
        part = normalize_salt(part)                     # 6. ek salt = ek naam
        if part:                                        #    khaali tukda chhod do
            salts.append(part)
    return salts


# ---------- 6. Salt + strength jodo → matching key ----------
def make_key(salts, strengths):
    """ ["paracetamol", "aceclofenac"], [(325.0,"mg"), (100.0,"mg")]
        → "aceclofenac 100mg + paracetamol 325mg" """
    if len(salts) != len(strengths) or len(salts) == 0:
        return None                                  # jod nahi sakte, toh guess mat karo
    pieces = []
    for salt, (number, unit) in zip(salts, strengths):
        pieces.append(f"{salt} {number:g}{unit}")    # 100.0 → "100"
    pieces.sort()                                    # order se fark na pade
    return " + ".join(pieces)


# ---------- Data saaf karo ----------
def main():
    df = pd.read_csv("data/jan_aushadhi_products.csv")
    df = df[df["Group Name"] != "Surgical & Medical Consumables"]
    df.loc[df["MRP"] == 0, "MRP"] = np.nan

    df["pack"] = df["Unit Size"].apply(pack_count)
    df["per_unit"] = (df["MRP"] / df["pack"]).round(2)
    df["form"] = df["Generic Name"].apply(find_form)
    df["release"] = df["Generic Name"].apply(find_release)
    df["strengths"] = df["Generic Name"].apply(find_strengths)
    df["salts"] = df["Generic Name"].apply(find_salts)
    df["key"] = df.apply(lambda row: make_key(row["salts"], row["strengths"]), axis=1)

    print(df[["Generic Name", "key"]].head(10))
    print("Dawaiyan:", len(df))
    print("Per-goli daam mila:", df["per_unit"].notna().sum())
    print("Form mila:", df["form"].notna().sum())
    print(df["release"].value_counts())
    print("Strength mili:", (df["strengths"].str.len() > 0).sum())
    same_count = df["salts"].str.len() == df["strengths"].str.len()
    print("Salt aur strength ki ginti barabar:", same_count.sum())
    print("Key bani:", df["key"].notna().sum())
    all_salts = set(s for salt_list in df["salts"] for s in salt_list)
    print("Alag-alag salt naam:", len(all_salts))

    df.to_csv("data/medicines_clean.csv", index=False)
    print("✅ data/medicines_clean.csv save ho gaya")


    # ---------- 🎯 Demo: Dolo 650 ka generic dhoondho ----------
    wanted_key = "paracetamol 650mg"
    wanted_form = "tablet"
    found = df[(df["key"] == wanted_key) & (df["form"] == wanted_form)]
    print("\n🔍 Dhoondh rahe hain:", wanted_key, wanted_form)
    print(found[["Generic Name", "Unit Size", "MRP", "per_unit"]])


# Sirf tab chalao jab yeh file SEEDHI chalayi gayi ho (python clean.py),
# import hone pe nahi (from clean import ...)
if __name__ == "__main__":
    main()
