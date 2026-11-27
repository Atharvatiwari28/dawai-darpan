"""Bina-AI fallback ke tests: koi API nahi chahiye, hamesha same result (pakka)."""
from fallback import answer_without_ai, find_brand_in_text

cases = [
    # (sawaal, bhasha, jawab mein yeh hona chahiye, yeh NAHI hona chahiye)
    ("Dolo 650 ka sasta option hai kya?", "Hinglish", ["Paracetamol Tablets IP 650 mg", "1.03"], []),
    ("Aur Telma 40 ka?", "English", ["Telmisartan", "1.24", "doctor"], []),
    ("Pan 40 ka sasta option?", "Hindi", ["Pantoprazole", "1.25", "कोटिंग"], []),
    ("Paracetamol 650 mg ka generic?", "Hinglish", ["Tablets IP 650 mg"], ["Suspension"]),   # brand-jaal
    ("SolvinCold: Paracetamol 500 mg, Phenylephrine 10 mg aur Chlorpheniramine 2 mg ka sasta option?",
     "Hinglish", ["1.31"], []),
    ("Mujhe bukhaar hai, kaunsi dawai lun?", "Hinglish", ["naam aur strength"], ["₹"]),        # salah nahi
    ("what is python?", "English", ["medicine name"], ["₹"]),
]

passed = 0
for question, language, must_have, must_not in cases:
    answer = answer_without_ai(question, language=language)
    ok = all(m in answer for m in must_have) and not any(m in answer for m in must_not)
    passed += ok
    if not ok:
        print("❌", question, "\n", answer, "\n")
print(f"answer_without_ai: {passed}/{len(cases)} pass")

assert find_brand_in_text("Augmentin 625 Duo ka sasta?") == "Augmentin 625 Duo"
print("find_brand_in_text: ✅")
