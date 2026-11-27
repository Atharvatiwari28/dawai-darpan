import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 40)

df = pd.read_csv("data/brand_medicines.csv")

# 1. Kitna bada hai, kaunse columns hain?
print("Rows:", len(df))
print(df.columns.tolist())

# 2. Pehli 5 rows
print(df.head())

# 3. Kitni dawaiyan band (discontinued) ho chuki hain?
print(df["Is_discontinued"].value_counts())

# 4. Kitni dawaiyon mein doosra salt bhi hai?
print("2nd salt wali:", df["short_composition2"].notna().sum())

# 5. Dolo 650 dhoondho 🔍
dolo = df[df["name"].str.contains("Dolo 650", na=False)]
print(dolo[["name", "price(₹)", "pack_size_label", "short_composition1"]])