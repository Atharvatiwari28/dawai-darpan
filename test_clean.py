from clean import (pack_count, brand_pack_count, find_form, find_release,
                   find_strengths, find_salts, make_key)


tests = [
    ("10's", 10),
    ("10 's", 10),
    ("15's", 15),
    ("7’s", 7),
    ("60 s in container", 60),
    ("1's", 1),
    ("100 ml", None),
    ("30 gm", None),
    ("10 sachet in monocarton", None),
    ("Pack of 4 Sanitary Napkins", None),
    ("5 ml", None),
    ("200 ml bottle", None),
]
passed = 0
for value, expected in tests:
    got = pack_count(value)
    if got == expected:
        passed += 1
    else:
        print("❌", value, "→", got, "| chahiye tha:", expected)
print(f"pack_count: {passed}/{len(tests)} pass")

brand_pack_tests = [
    ("strip of 15 tablets", 15),
    ("strip of 10 capsules", 10),
    ("strip of 10 tablet sr", 10),
    ("bottle of 30 tablets", 30),
    ("strip of 10 soft gelatin capsules", 10),
    ("Strip Of 4 Chewable Tablets", 4),
    ("bottle of 100 ml syrup", None),
    ("vial of 1 injection", None),
    ("tube of 15 gm cream", None),
]
passed = 0
for label, expected in brand_pack_tests:
    got = brand_pack_count(label)
    if got == expected:
        passed += 1
    else:
        print("❌", label, "→", got, "| chahiye tha:", expected)
print(f"brand_pack_count: {passed}/{len(brand_pack_tests)} pass")

form_tests = [
    ("Paracetamol Tablets IP 650 mg", "tablet"),
    ("Venlafaxine Prolonged Release Capsules 150mg", "capsule"),
    ("Ondansetron Injection IP 2mg per ml", "injection"),
    ("Terbinafine Hydrochloride Dusting Powder 1% w/w", "powder"),
    ("Latanoprost Eye Drops IP 0.005% w/v", "drops"),
    ("Telmisartan 80mg and Cilnidipine 10mg Tablets", "tablet"),
    ("Metronidazole  Infusion IP 500 mg", "injection"),
    ("Formoterol 6mcg and Budesonide 200mcg Rotacaps", "inhaler"),
    ("Gemcitabine Injection IP 1000mg", "injection"),
    ("Vitamin E Softgel Capsules 400 mg", "capsule"),
]
passed = 0
for name, expected in form_tests:
    got = find_form(name)
    if got == expected:
        passed += 1
    else:
        print("❌", name, "→", got, "| chahiye tha:", expected)
print(f"find_form: {passed}/{len(form_tests)} pass")

release_tests = [
    ("Diclofenac Sodium Prolonged Release Tablets IP 100mg", "SR"),
    ("Metformin Hydrochloride (sustained-relase) 500mg", "SR"),
    ("Venlafaxine Prolonged Release Capsules 150mg", "SR"),
    ("Aspirin Gastro-resistant Tablets IP 150 mg", "GR"),
    ("Paracetamol Tablets IP 650 mg", None),
    ("Telmisartan 80mg and Cilnidipine 10mg Tablets", None),
    ("strip of 10 tablet sr", "SR"),
    ("Erythromycin Tablets IP 250 mg", None),
]
passed = 0
for name, expected in release_tests:
    got = find_release(name)
    if got == expected:
        passed += 1
    else:
        print("❌", name, "→", got, "| chahiye tha:", expected)
print(f"find_release: {passed}/{len(release_tests)} pass")

strength_tests = [
    ("Paracetamol Tablets IP 650 mg", [(650, "mg")]),
    ("Telmisartan 80mg and Cilnidipine 10mg Tablets", [(80, "mg"), (10, "mg")]),
    ("Terbutaline 1.25mg, Ambroxol 15mg, Guaiphenesin 50mg", [(1.25, "mg"), (15, "mg"), (50, "mg")]),
    ("Vitamin B12 tabs 1500mcg", [(1500, "mcg")]),
    ("Cholecalciferol Granules 60000 IU per gm", [(60000, "iu")]),
    ("Lactulose Solution IP 10gm per 15ml", [(10, "g")]),
    ("Mometasone Furoate Cream IP 0.1%w/w", [(0.1, "%")]),
    ("Calamine Lotion IP", []),
    ("Atenolol + Chlorthalidone Tablet (25/12.5 Mg)", [(25, "mg"), (12.5, "mg")]),
]
passed = 0
for name, expected in strength_tests:
    got = find_strengths(name)
    if got == expected:
        passed += 1
    else:
        print("❌", name, "→", got, "| chahiye tha:", expected)
print(f"find_strengths: {passed}/{len(strength_tests)} pass")

salt_tests = [
    ("Paracetamol Tablets IP 650 mg", ["paracetamol"]),
    ("Telmisartan 80mg and Cilnidipine 10mg Tablets", ["telmisartan", "cilnidipine"]),
    ("Amlodipine + Enalapril Tablet (5/5 Mg)", ["amlodipine", "enalapril"]),
    ("Atorvastatin (10mg) + Aspirin (75mg) + Ramipril (5mg) Capsule", ["atorvastatin", "aspirin", "ramipril"]),
    ("Terbutaline 1.25mg, Ambroxol 15mg, Guaiphenesin 50mg", ["terbutaline", "ambroxol", "guaifenesin"]),
    ("Diclofenac Sodium Prolonged Release Tablets IP 100mg", ["diclofenac"]),
    ("Metformin Hydrochloride (sustained-relase) 500mg", ["metformin"]),
    ("Amoxycillin Trihydrate Dispersible Tablets 250mg", ["amoxicillin"]),
    ("Sodium Chloride Injection IP 0.9%W/V", ["sodium chloride"]),
    ("Amoxycillin 500mg and Potassium Clavulanate 125mg Tablets", ["amoxicillin", "clavulanic acid"]),
    ("Amoxycillin  (500mg) +   Clavulanic Acid (125mg)", ["amoxicillin", "clavulanic acid"]),
    ("Ambroxol (30mg/5ml) + Levosalbutamol (1mg/5ml)", ["ambroxol", "levosalbutamol"]),
    ("Pantoprazole Gastro Resistant Tablets IP 40 mg", ["pantoprazole"]),
    ("Azithromycin Anhydrous 500 mg", ["azithromycin"]),
    ("Azithromycin (anhydrous) 500 mg (as Azithromycin dihydrate IP)", ["azithromycin"]),
    ("Pantoprazole 40 mg (As gastro-resistant pellets), Domperidone 30 mg (As Prolonged-release pellets)", ["pantoprazole", "domperidone"]),
    ("2,4-Dichlorobenzyl Alcohol 1.2 mg, Amylmetacresol 0.6 mg Lozenges", ["dichlorobenzyl alcohol", "amylmetacresol"]),
    ("Aspirin Enteric Coated Tablets IP 75mg", ["aspirin"]),
    ("Acetylsalicylic Acid (Aspirin) Gastro-resistant Tablets IP 325mg", ["aspirin"]),
]
passed = 0
for name, expected in salt_tests:
    got = find_salts(name)
    if got == expected:
        passed += 1
    else:
        print("❌", name, "→", got, "| chahiye tha:", expected)
print(f"find_salts: {passed}/{len(salt_tests)} pass")

key_tests = [
    (["paracetamol"], [(650.0, "mg")], "paracetamol 650mg"),
    (["paracetamol", "aceclofenac"], [(325.0, "mg"), (100.0, "mg")], "aceclofenac 100mg + paracetamol 325mg"),
    (["terbutaline"], [(1.25, "mg")], "terbutaline 1.25mg"),
    (["calamine"], [], None),
]
passed = 0
for salts, strengths, expected in key_tests:
    got = make_key(salts, strengths)
    if got == expected:
        passed += 1
    else:
        print("❌", salts, strengths, "→", got, "| chahiye tha:", expected)
print(f"make_key: {passed}/{len(key_tests)} pass")
