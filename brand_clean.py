import pandas as pd
from clean import (brand_pack_count, find_form, find_release,
                   find_strengths, find_salts, make_key)

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 45)
pd.set_option("display.max_columns", 10)


def main():
    df = pd.read_csv("data/brand_medicines.csv")
    print("Kul brands:", len(df))

    # 1. Band ho chuki dawaiyan hatao
    df = df[df["Is_discontinued"] == False]
    print("Chalu brands:", len(df))

    # 2. Dono composition columns ko jodo
    comp1 = df["short_composition1"].fillna("")
    comp2 = df["short_composition2"].fillna("")
    df["composition"] = comp1 + " + " + comp2

    # 3. Wahi purane auzaar, naya data ♻️
    df["salts"] = df["composition"].apply(find_salts)
    df["strengths"] = df["composition"].apply(find_strengths)
    df["key"] = df.apply(lambda row: make_key(row["salts"], row["strengths"]), axis=1)

    label = df["name"] + " " + df["pack_size_label"].fillna("")
    df["form"] = label.apply(find_form)
    df["release"] = label.apply(find_release)

    # 4. Per-goli daam
    df["pack"] = df["pack_size_label"].apply(brand_pack_count)
    df["per_unit"] = (df["price(₹)"] / df["pack"]).round(2)

    print(df[["name", "key", "per_unit"]].head(8))
    print("Key bani:", df["key"].notna().sum())
    print("Per-goli daam mila:", df["per_unit"].notna().sum())

    keep = ["name", "manufacturer_name", "price(₹)", "pack_size_label",
            "composition", "key", "form", "release", "pack", "per_unit"]
    df[keep].to_csv("data/brands_clean.csv", index=False)
    print("✅ data/brands_clean.csv save ho gaya")


if __name__ == "__main__":
    main()