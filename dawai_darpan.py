import sys
import pandas as pd
from read_strip import read_strip
from clean import find_salts, find_strengths, make_key, find_form, find_release
from search import find_brand, find_generic, saving_percent


def key_from_photo(info):
    """ Photo ke composition se key banao. Code AI ke jawab ko JAANCHTA hai:
        strength na mile ya ginti na mile → None (kachra pakda gaya) """
    if not info.composition:
        return None
    return make_key(find_salts(info.composition), find_strengths(info.composition))


def run(photo_path):
    # 1. AI photo padhe 👁️
    info = read_strip(photo_path)
    if info is None:
        print("❌ Abhi AI busy hai, thodi der baad try karo")
        return
    if not info.is_medicine:
        print("🤔 Is photo mein dawai nahi dikh rahi. Strip ki photo bhejo.")
        return
    print(f"📸 Padha: naam={info.brand_name}, salt={info.composition}, "
          f"MRP={info.mrp}, pack={info.pack_count}")

    photo_text = f"{info.brand_name or ''} {info.composition or ''}"
    brand = find_brand(info.brand_name) if info.brand_name else None

    # 2. Raasta 1: brand data mein mila
    if brand is not None and pd.notna(brand["key"]):
        print(f"📚 Brand data mein mila: {brand['name']} ({brand['manufacturer_name']})")
        key, form, release_group = brand["key"], brand["form"], brand["release_group"]
        data_price = brand["per_unit"]
    # 3. Raasta 2: photo ke composition se (code se jaanch ke)
    else:
        key = key_from_photo(info)
        if key is None:
            print("📸 Naam ya salt saaf nahi padh paaye. Dawai ke naam aur salt wali "
                  "side ki saaf photo bhejo.")
            return
        print(f"🧪 Brand data mein nahi mila, photo ke salt se dhoondh rahe hain")
        form = find_form(photo_text)
        release_group = "SR" if find_release(photo_text) == "SR" else "normal"
        data_price = None
    print(f"🔑 Key: {key} | {form or 'tablet/capsule?'} | {release_group}")

    # 4. Jan Aushadhi generic
    generic = find_generic(key, form, release_group)
    if generic is None:
        print("😕 Jan Aushadhi mein exact generic nahi mila")
        return
    print(f"✅ Generic: {generic['Generic Name']}  (₹{generic['MRP']} / {generic['Unit Size']})")
    if form is None:
        print("   ⚠️ Photo se form pakka nahi pata, tablet/capsule maana hai")

    # 5. Bachat: photo ka MRP sabse bharosemand, warna dataset ka daam
    if info.mrp and info.pack_count:
        brand_price, source = round(info.mrp / info.pack_count, 2), "photo ka MRP"
    elif data_price is not None and pd.notna(data_price):
        brand_price, source = data_price, "dataset, approx"
    else:
        brand_price = None
    if brand_price and pd.notna(generic["per_unit"]):
        s = saving_percent(brand_price, generic["per_unit"])
        print(f"💰 Per goli: brand ₹{brand_price} ({source})  vs  generic "
              f"₹{generic['per_unit']}  →  {s:.0f}% bachat")
    print("⚕️ Dawai badalne se pehle doctor ya pharmacist se zaroor poochein.")


if __name__ == "__main__":
    run(sys.argv[1])