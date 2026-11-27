"""🛟 Graceful degradation: saare AI models fail ho jaayein, tab bhi user ko jawab mile.
Yahan koi LLM nahi: sirf humara code (search + templates). Chat jaisa smart nahi, lekin kabhi band nahi."""
import re
import pandas as pd
from search import find_brand
from clean import find_strengths
from tools import search_by_brand, search_by_salt, medicine_info, _read_photo_once

# Har bhasha ke liye seedhe, pehle se likhe vaakya (AI nahi, isliye galti ka chance nahi)
TEXT = {
    "Hinglish": {
        "intro": "🤖 AI abhi thoda busy hai, lekin yeh rahi jaankari:",
        "found": "💊 {brand} → saasta generic: {generic}",
        "price": "💰 Per goli: brand ₹{brand_price} vs generic ₹{generic_price} ({saving}% bachat)",
        "generic_only": "💰 Generic ka daam: ₹{generic_price} per goli",
        "approx": "   (brand ka daam andaazan hai)",
        "not_found": "😕 Iska Jan Aushadhi generic nahi mila.",
        "uses": "💊 {name} aam taur pe in kaamon mein aati hai: {uses}",
        "ask": "🤖 AI abhi busy hai. Kripya dawai ka naam aur strength likhiye (jaise 'Dolo 650'), "
               "ya thodi der baad try karein.",
        "disclaimer": "⚕️ Dawai badalne se pehle doctor ya pharmacist se zaroor poochein.",
    },
    "Hindi": {
        "intro": "🤖 AI अभी थोड़ा व्यस्त है, लेकिन यह रही जानकारी:",
        "found": "💊 {brand} → सस्ती जेनेरिक दवा: {generic}",
        "price": "💰 प्रति गोली: ब्रांड ₹{brand_price}, जेनेरिक ₹{generic_price} ({saving}% बचत)",
        "generic_only": "💰 जेनेरिक का दाम: ₹{generic_price} प्रति गोली",
        "approx": "   (ब्रांड का दाम अनुमानित है)",
        "not_found": "😕 इसकी जन औषधि जेनेरिक दवा नहीं मिली।",
        "uses": "💊 {name} आमतौर पर इन कामों में आती है: {uses}",
        "ask": "🤖 AI अभी व्यस्त है। कृपया दवा का नाम और ताकत लिखें (जैसे 'Dolo 650'), या थोड़ी देर बाद कोशिश करें।",
        "disclaimer": "⚕️ दवा बदलने से पहले डॉक्टर या फार्मासिस्ट से ज़रूर पूछें।",
    },
    "English": {
        "intro": "🤖 The AI is busy right now, but here is the information:",
        "found": "💊 {brand} → cheaper generic: {generic}",
        "price": "💰 Per tablet: brand ₹{brand_price} vs generic ₹{generic_price} ({saving}% saving)",
        "generic_only": "💰 Generic price: ₹{generic_price} per tablet",
        "approx": "   (brand price is approximate)",
        "not_found": "😕 No Jan Aushadhi generic found for this.",
        "uses": "💊 {name} is generally used for: {uses}",
        "ask": "🤖 The AI is busy. Please type the medicine name and strength (like 'Dolo 650'), "
               "or try again in a little while.",
        "disclaimer": "⚕️ Please ask a doctor or pharmacist before switching medicines.",
    },
    "Marathi": {
        "intro": "🤖 AI सध्या व्यस्त आहे, पण ही माहिती:",
        "found": "💊 {brand} → स्वस्त जेनेरिक औषध: {generic}",
        "price": "💰 प्रति गोळी: ब्रँड ₹{brand_price}, जेनेरिक ₹{generic_price} ({saving}% बचत)",
        "generic_only": "💰 जेनेरिकची किंमत: ₹{generic_price} प्रति गोळी",
        "approx": "   (ब्रँडची किंमत अंदाजे आहे)",
        "not_found": "😕 याचे जन औषधी जेनेरिक औषध मिळाले नाही.",
        "uses": "💊 {name} साधारणपणे यासाठी वापरले जाते: {uses}",
        "ask": "🤖 AI सध्या व्यस्त आहे. कृपया औषधाचे नाव आणि ताकद लिहा (उदा. 'Dolo 650'), किंवा थोड्या वेळाने प्रयत्न करा.",
        "disclaimer": "⚕️ औषध बदलण्यापूर्वी डॉक्टर किंवा फार्मासिस्टला नक्की विचारा.",
    },
}

# Sawaal ke yeh shabd brand ka naam nahi ho sakte
STOPWORDS = {"ka", "ki", "ke", "kya", "hai", "hain", "sasta", "saste", "option", "generic", "batao",
             "bataiye", "dawai", "dawa", "tablet", "tablets", "capsule", "the", "for", "what", "is",
             "cheaper", "cheap", "price", "aur", "iska", "iski", "mujhe", "chahiye", "of", "me",
             "mein", "kis", "kaam", "aati", "used", "use", "side", "effects", "please", "tell"}
USES_WORDS = re.compile(r"kis kaam|kaam aati|used for|use of|side effect|upyog|किस काम|उपयोग", re.IGNORECASE)


def find_brand_in_text(text):
    """ "Dolo 650 ka sasta option hai kya?" → "Dolo 650"  (brand data mein jo mile) """
    words = [w for w in re.findall(r"[\w\-]+", text) if w.lower() not in STOPWORDS]
    candidates = []
    for size in (3, 2, 1):                         # lambe naam pehle: "Augmentin 625 Duo" > "Augmentin"
        for i in range(len(words) - size + 1):
            phrase = " ".join(words[i:i + size])
            if len(phrase) >= 3:
                candidates.append(phrase)
    # Number wale pehle ("Dolo 650"), kyunki brand naam mein aksar strength hoti hai
    candidates.sort(key=lambda p: not re.search(r"\d", p))
    for phrase in candidates:
        brand = find_brand(phrase)
        if brand is not None and pd.notna(brand["key"]):
            return phrase
    return None


# Tool ki chetavniyan Hinglish mein hoti hain. Template wale jawab (bina AI) mein inhe bhi sahi bhasha mein do
WARNINGS = {
    "Form pakka nahi pata": {
        "Hindi": "दवा का रूप पक्का नहीं पता, गोली/कैप्सूल माना है",
        "English": "Form not certain; assumed tablet/capsule",
        "Marathi": "औषधाचा प्रकार नक्की माहीत नाही; गोळी/कॅप्सूल मानले आहे"},
    "Coating alag ho sakti hai": {
        "Hindi": "कोटिंग अलग हो सकती है",
        "English": "The coating may be different",
        "Marathi": "कोटिंग वेगळे असू शकते"},
    "Is naam se ek aur dawai": {
        "Hindi": "इस नाम से एक और दवा भी मिलती है जिसमें अलग साल्ट हैं। हमने स्ट्रिप पर लिखे साल्ट के हिसाब से ढूँढा है।",
        "English": "Another medicine with this name has different salts. We searched by the salt printed on the strip.",
        "Marathi": "या नावाचे दुसरे औषधही आहे ज्यात वेगळे क्षार आहेत. आम्ही स्ट्रिपवर लिहिलेल्या क्षारानुसार शोधले आहे."},
    "Is dawai ka salt hamare database": {
        "Hindi": "इस दवा का साल्ट हमारे डेटाबेस में नहीं था; मैंने अपनी सामान्य जानकारी से पहचाना है। स्ट्रिप पर लिखा साल्ट ज़रूर मिलाएँ।",
        "English": "This medicine's salt wasn't in our database; I identified it from my general knowledge. Please match it with the salt printed on the strip.",
        "Marathi": "या औषधाचा क्षार आमच्या डेटाबेसमध्ये नव्हता; मी माझ्या सामान्य माहितीवरून ओळखला आहे. स्ट्रिपवर लिहिलेला क्षार नक्की तपासा."},
}


def translate_warning(warning, text):
    language = next((lang for lang, t in TEXT.items() if t is text), "Hinglish")
    for start, versions in WARNINGS.items():
        if warning.startswith(start):
            return versions.get(language, warning)
    return warning


def _price_answer(result, text, label):
    if not result.get("found"):
        return [text["not_found"]]
    lines = [text["found"].format(brand=label, generic=result["generic_name"])]
    if result.get("brand_per_unit") and result.get("generic_per_unit"):
        lines.append(text["price"].format(brand_price=result["brand_per_unit"],
                                          generic_price=result["generic_per_unit"],
                                          saving=result["saving_percent"]))
        if "approx" in str(result.get("brand_price_source")):
            lines.append(text["approx"])
    elif result.get("generic_per_unit"):
        lines.append(text["generic_only"].format(generic_price=result["generic_per_unit"]))
    lines += [f"⚠️ {translate_warning(w, text)}" for w in result.get("warnings", [])]
    return lines


def price_reply(result, language="Hinglish"):
    """ Tool ka result → seedha, pakka jawab (koi LLM nahi). Normal flow mein bhi speed ke liye use hota hai """
    text = TEXT.get(language, TEXT["Hinglish"])
    label = result.get("brand_name") or result.get("key", "")
    return "\n".join(_price_answer(result, text, label) + [text["disclaimer"]])


def answer_without_ai(question, photo=None, language="Hinglish"):
    """ AI ke bina jawab. Kuch samajh na aaye toh imaandari se naam likhne ko kaho. """
    text = TEXT.get(language, TEXT["Hinglish"])
    lines = None

    if photo:                                       # photo: Gemini vision chal jaaye toh achha
        try:
            info = _read_photo_once(photo)
        except Exception:
            info = None
        if info and info.is_medicine and (info.brand_name or info.composition):
            if info.brand_name:
                result = search_by_brand(info.brand_name, composition=info.composition,
                                         mrp=info.mrp, pack_count=info.pack_count)
            else:
                result = search_by_salt(info.composition, mrp=info.mrp, pack_count=info.pack_count)
            lines = _price_answer(result, text, info.brand_name or info.composition)

    # Salt + unit likha hai ("Paracetamol 650 mg")? Pehle salt se dhoondho, warna "Paracetamol" naam ka
    # koi brand (jaise syrup) galti se mil sakta hai
    if lines is None and find_strengths(question):
        salt_text = question.split(":")[-1]                     # "SolvinCold: Paracetamol..." → naam hatao
        salt_text = re.sub(r"\baur\b|\bor\b", ",", salt_text, flags=re.IGNORECASE)   # "aur" = jodne wala shabd
        cleaned = " ".join(w for w in re.findall(r"[\w\-.,+]+", salt_text)
                           if w.lower().strip("?.,") not in STOPWORDS)
        result = search_by_salt(cleaned)
        if result.get("key"):                       # salt samajh aaya: mila ya nahi, wahi jawab (brand pe mat jao)
            lines = _price_answer(result, text, cleaned)

    if lines is None:
        brand_name = find_brand_in_text(question)
        if brand_name and USES_WORDS.search(question):
            info = medicine_info(brand_name)
            if info.get("found") and info["used_for"]:
                lines = [text["uses"].format(name=info["name"], uses=", ".join(info["used_for"]))]
        if lines is None and brand_name:
            lines = _price_answer(search_by_brand(brand_name), text, brand_name)

    if lines is None:
        return text["ask"]
    return "\n".join([text["intro"]] + lines + [text["disclaimer"]])
