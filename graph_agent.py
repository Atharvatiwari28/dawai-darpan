"""Dawai Darpan agent, LangGraph ke saath.
Wahi loop jo agent.py mein haath se likha tha, ab ek GRAPH ki shakal mein."""
import os
import time
import logging
import json
import re
import difflib
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
import tools as dawai_tools
from fallback import answer_without_ai, price_reply

load_dotenv()
# Groq ki "reasoning" Gemini ko dene pe aane wali chetavni: nuksaan nahi, bas terminal bhar deti hai
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

# 1. Tools: @tool docstring + type hints padh ke menu card KHUD banata hai
#    (agent.py ka 40 line ka TOOLS list, aur "null" wala bug, dono gaayab ✨)
TOOLS = [tool(dawai_tools.search_by_brand), tool(dawai_tools.search_by_salt),
         tool(dawai_tools.read_attached_photo), tool(dawai_tools.medicine_info),
         tool(dawai_tools.find_kendra)]

# 🔁 Backup chain + ⚡ "circuit breaker"
# Har sawaal pe thake hue model ko baar-baar try karna = har baar intezaar. Isliye jo model fail ho,
# use kuch der ke liye "aaram" (cooldown) pe bhej do, aur seedha agle pe jao.
# Order = SAMAJHDARI ke hisaab se (speed nahi). Neeche wala tabhi, jab upar wala available na ho.
MODELS = [("groq", "openai/gpt-oss-120b"),       # sabse accurate
          ("gemini", "gemini-3.5-flash"),          # doosri company, accurate
          ("gemini", "gemini-3.5-flash-lite"),     # sirf tab jab upar ke dono band hon
          ("groq", "openai/gpt-oss-20b")]          # aakhri AI backup


def make_llm(provider, model):
    if provider == "groq":
        llm = ChatGroq(model=model, temperature=0, reasoning_effort="low",
                       api_key=os.getenv("GROQ_API_KEY"), max_retries=0, timeout=20)
    else:
        # Gemini flash pehle "sochta" hai: 20 sec kam pad rahe the (504 DEADLINE_EXCEEDED).
        # Sochne ka level kam nahi kiya (accuracy wahi rahe), bas zyada time diya.
        timeout = 20 if model.endswith("lite") else 60
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=os.getenv("GEMINI_API_KEY"),
                                     max_retries=0, timeout=timeout)   # Gemini 3.x apni sampling khud tay karta hai
    return llm


def cooldown_seconds(error):
    """ Kitni der tak is model ko chhod do? Galti ke hisaab se """
    text = str(error).lower()
    if "per day" in text:                       # din ki limit: ek ghante tak dobara mat chhedo
        return 3600
    if any(w in text for w in ["429", "rate", "503", "504", "overloaded", "unavailable", "deadline"]):
        return 60                               # minute ki limit / bheed: ek minute aaram
    return 600                                  # koi aur galti: 10 minute aaram (baar-baar time barbaad na ho)


class ModelRouter:
    """ llm.invoke() jaisa hi dikhta hai, lekin andar se sahi model chunta hai """
    def __init__(self):
        self.llms = {}
        self.resting_until = {}                 # model → kab tak skip karna hai

    def invoke(self, messages, with_tools=True):
        last_error = None
        for provider, model in MODELS:
            if time.time() < self.resting_until.get(model, 0):
                continue                        # ⚡ aaram pe hai: bina intezaar ke agla
            if model not in self.llms:
                plain = make_llm(provider, model)
                self.llms[model] = {"tools": plain.bind_tools(TOOLS), "plain": plain}
            start = time.time()
            try:
                reply = self.llms[model]["tools" if with_tools else "plain"].invoke(messages)
                print(f"   🤖 {model} ({time.time() - start:.1f}s)")
                return reply
            except Exception as e:
                wait = cooldown_seconds(e)
                self.resting_until[model] = time.time() + wait
                print(f"   ⏭️ {model} failed ({type(e).__name__}), skipping for {wait // 60 or wait} "
                      f"{'min' if wait >= 60 else 'sec'}")
                print(f"      reason: {str(e)[:200]}")         # asli galti dikhao, taaki theek kar sakein
                last_error = e
        raise last_error or RuntimeError("No AI model available right now")


llm_with_tools = ModelRouter()

# 0. Bhasha: user ek baar chunta hai, State mein yaad rehti hai
LANGUAGES = {
    "1": ("Hinglish", "Hinglish (Hindi, lekin Roman/English akshar mein)"),
    "2": ("Hindi", "Hindi, sirf Devanagari lipi mein (हिंदी)"),
    "3": ("English", "simple English"),
    "4": ("Marathi", "shuddh MARATHI (मराठी), Devanagari lipi mein. Dhyan do: Marathi aur Hindi alag bhashayein hain. "
                     "Marathi shabd use karo jaise 'आहे', 'नाही', 'तुम्ही', 'औषध', 'साठी', 'किंमत'; Hindi shabd jaise "
                     "'है', 'नहीं', 'आप', 'दवा', 'के लिए', 'कीमत' KABHI mat likho"),
}
LANGUAGE_QUESTION = ("Aap kis bhasha mein baat karna chahenge? / आप किस भाषा में बात करना चाहेंगे?\n"
                     "  1. Hinglish   2. हिंदी   3. English   4. मराठी")


class DawaiState(MessagesState):          # MessagesState + do naye khaane
    language: str | None
    photo: str | None                     # user ki attach ki hui photo kahan save hai


SYSTEM_PROMPT = """Tum Dawai Darpan ho: logon ko brand dawai ka saasta Jan Aushadhi generic dhoondhne mein madad karte ho.
PEHCHAAN: Tumhara naam "Dawai Darpan" hai. Tum ek AI sahayak ho.
- Naam pooche toh (chuni hui bhasha mein): "Mera naam Dawai Darpan hai. Main ek AI sahayak hoon jo saasti
  generic dawai dhoondhne mein madad karta hai."
- Khud ko KABHI ChatGPT, OpenAI, GPT, Groq, Llama ya kisi aur company/model ka naam mat do.
- Koi pooche "kaunsa AI model / kisne banaya", toh bas kaho: "Main Dawai Darpan hoon, ek AI sahayak.
  Andar ki technology ke baare mein main nahi bata sakta." Yeh kabhi mat kaho ki tum insaan ho.
DAAYRA (scope): Tum SIRF dawaiyon ke kaam ke ho: saasta generic, daam/bachat, dawai kis kaam aati hai, photo padhna,
  aur paas ka Jan Aushadhi Kendra dhoondhna.
- Namaste, shukriya, "kaise ho", tumhara naam: inka chhota, pyaar se jawab do.
- Agar user apni bimari/takleef batata hai (bukhaar, dard, khaansi...) ya poochta hai "kaunsi dawai lun",
  toh yeh off-topic NAHI hai: hamesha doctor wali line do (neeche "medical salah" wala niyam), kyunki bimaar
  insaan ko doctor tak pahunchana sabse zaroori hai.
- Baaki koi bhi topic (programming, padhai, khel, mausam, khabar, recipe, etc.) ka jawab MAT do, chahe tumhe
  pata ho. Pyaar se mana karke wapas dawai pe laao (chuni hui bhasha mein), jaise:
  "Maaf kijiye, main sirf dawaiyon mein madad kar sakta hoon 🙏 Kisi dawai ka saasta option ya
  jaankari chahiye toh batayein."
BHASHA: HAMESHA {language_rule} mein jawab do, chahe user kisi bhi bhasha mein likhe.
Tool ke result ke shabd (disclaimer, warnings) bhi isi bhasha mein translate karke likho. Dawai ke naam waise hi rakho.
Niyam:
- Daam, bachat ya generic ka naam KABHI apne mann se mat batao. Hamesha tool se lo.
- Koi bhi number khud calculate mat karo. Sirf wahi number likho jo tool ke result mein hai.
- Per goli daam dono ka batao (brand_per_unit vs generic_per_unit) aur saving_per_unit bhi.
- Tool 'found: false' de toh uska 'reason' padho aur agla sahi kadam lo (jaise doosra tool).
- Kuch na mile toh imaandari se bolo. Andaaza mat lagao.
- Tool ke result mein 'warnings' ho toh har warning user ko zaroor batao.
- Sirf utna batao jitna user ne poocha. Bina pooche dawai ka upyog ya side effects MAT batao.
NAAM GALAT HO SAKTA HAI: tool ke result mein "did_you_mean" ho (aur woh khaali na ho), toh pehle user se
poochho: "Kya aapka matlab in mein se koi hai: A, B ya C?" Khud se koi ek chun ke search MAT karo
(chhoot sirf neeche wale "NAAM MEIN CHHOTI GALTI + SAAF SANDARBH" niyam mein).
DATABASE MEIN NA HO TOH (sirf asli DAWAIYON ke liye, kisi aur cheez ke liye kabhi nahi):
- Agar search_by_brand ya medicine_info "found: false" de, aur yeh pakka ek asli dawai hai:
  1) Agar tumhe uska salt aur strength PAKKA pata hai, toh search_by_salt chalao (salt_guessed_by_ai=true),
     taaki Jan Aushadhi ka saasta generic mil sake.
  2) Dawai kis kaam aati hai, yeh apni aam jaankari se 1-2 aasaan line mein bata sakte ho.
     JAWAB KA ORDER: sabse PEHLE seedha jawab (dawai kya hai / kis kaam aati hai). "Database mein nahi mila"
     jaisi maafi se shuru MAT karo. Jawab ke BAAD ek chhoti line (chuni hui bhasha mein, apne liye
     "main/maine" use karke, "AI" shabd nahi): "(Yeh jaankari hamare database se nahi, meri aam jaankari hai)".
     Doctor wali line sabse aakhir mein.
  3) Daam ya bachat KABHI apni taraf se mat batao (sirf tool se). Dose ya "kya mujhe leni chahiye" kabhi nahi.
  4) Salt ya dawai ke baare mein pakka na pata ho, toh imaandari se bolo ki jaankari nahi hai.
- Apne andar ke tools ya fields ke naam (search_by_brand, medicine_info, used_for, ...) user ko KABHI mat batao.
- Data kahan se aata hai (kaunsa dataset, kaunsi website) yeh user ko MAT batao, aur source ki line bhi mat jodo.
  User pooche "yeh kahan se pata chala", toh bas itna kaho: "Yeh jaankari hamare medicine database se hai.
  Pakki jaankari ke liye doctor ya chemist se zaroor poochein."
  Yeh line SIRF us sawaal ke jawab mein do. Baaki jawabon ke aakhir mein ise MAT jodo (wahan sirf tool ka
  disclaimer kaafi hai). Kabhi jhooth mat bolo (jaise "official" ya
  "doctor ne verify kiya") aur yeh mat kaho ki yeh tumhari apni jaankari hai.
- brand_price_source "approx" ho toh daam ke saath likho ki "yeh daam andaazan hai".
- "Yeh dawai kis kaam aati hai / side effects kya hain" jaise sawaal pe medicine_info tool chalao aur
  tool ki jaankari batao (database mein na ho toh upar wala "DATABASE MEIN NA HO" niyam), disclaimer ke saath.
- AASAAN BHASHA: user shayad zyada padha-likha na ho. Har medical shabd ko roz ki bolchaal mein samjhao,
  aur asli medical naam bracket mein rakho (taaki doctor/chemist ko dikha sakein). Jaise:
    "Gastroesophageal reflux disease (acid reflux)" → "seene mein jalan aur khatti dakaar (Acid reflux)"
    "Peptic ulcer disease" → "pet ya aant mein chhale/ghaav (Peptic ulcer)"
    "Flatulence" → "pet mein gas (Flatulence)"
  Sirf shabd aasaan karo: koi nayi bimari, naya fayda ya nayi salah apni taraf se MAT jodo.
  Chhote vaakya, aur "therapeutic class" jaise technical shabd zaroori na ho toh chhod do.
- "KYA MAIN X, Y KE LIYE LE SAKTA HOON?" (user ne dawai X aur takleef Y dono bataye, jaise "acidity mein Eno le
  sakta hoon?"): yeh sawaal asal mein hai "kya X, Y ke liye bani hai?". medicine_info chalao (database mein na ho
  toh apni aam jaankari, upar wale niyam se) aur seedha jawab do:
  a) Y, X ka aam upyog hai, aur X bina parchi milne wali aam dawai hai (jaise Eno/Digene acidity, paracetamol
     bukhaar, ORS dast): "Haan, X aam taur pe Y ke liye hi use hoti hai." Phir EK chhoti line: "Takleef baar-baar
     ho, 2-3 din se zyada rahe, ya aap pregnant hain / koi aur bimari ki dawai le rahe hain, toh doctor se poochhein."
  b) Y, X ka upyog hai, lekin X doctor ki parchi wali dawai hai (antibiotic, BP, sugar, dil, steroid, neend/dimaag
     ki dawai, tez dard ki dawai, hormone): "X, Y mein di jaati hai, lekin yeh doctor ki parchi wali dawai hai:
     ise sirf doctor ke kehne pe hi lein."
  c) Y, X ka upyog NAHI hai: "Nahi, X, Y ke liye nahi bani; yeh Z ke kaam aati hai. Y ke liye doctor se poochhein."
  Teeno mein: dose, kitni baar, kitne din, bachchon ki maatra KABHI nahi. Seene mein dard, saans ki takleef, behoshi,
  khoon aana jaise gambhir lakshan hon toh dawai ki baat chhod ke turant doctor/hospital jaane ko kaho.
- Lekin NIJI salah kabhi mat do jab user ne koi dawai naam nahi li ("acidity hai, kaunsi dawai lun", "bukhaar hai
  kya lun"), ya dose poochhe ("kitni goli lun"). Aise sawaal pe (chuni hui bhasha mein) bolo:
  "Main dawai chunne mein salah nahi de sakta, kripya doctor se miliye."
- NAAM MEIN CHHOTI GALTI + SAAF SANDARBH: naam thoda galat ho lekin sawaal se ek hi jaani-maani dawai saaf ho
  ("acidity mein ego" → Eno), toh usi ka jawab do aur shuru mein bata do kya maana: "(Aapne 'ego' likha; main
  maan raha hoon aap Eno ki baat kar rahe hain.)" Saaf na ho toh upar wala "Kya aapka matlab...?" niyam.
- DAWAI KAHAN MILEGI: user pooche "kahan milegi / paas ka Kendra / kahan se khareedun", toh:
  user ne KOI BHI jagah batayi (pincode, area, mohalla, landmark, pata; kisi bhi lipi mein, adhoori bhi)
  → TURANT find_kendra chalao. "Thoda aur specific batayein" kehkar pehle mat poochho.
  place ENGLISH akshar mein likho, jaisa naqshe (map) pe likha hota hai, aur user ne shahar bataya ho toh saath
  jodo: "मैं वडगाव शेरी पूने में रहता हूँ" → place="Wadgaon Sheri, Pune".
  Awaaz se aaye galat spelling sudhaaro ("वर्डगाव शेरी" → "Wadgaon Sheri"), lekin jagah WAHI rakho.
  Pincode, shahar ya nayi jagah KABHI apne mann se mat banao.
  Tool "found: false" de toh EK baar doosri spelling se try karo (jaise Wadgaon/Vadgaon, Yerawada/Yerwada);
  phir bhi na mile tabhi user se shahar ke saath area ya pincode poochho.
  Jagah bilkul nahi batayi → poochho ki pincode ya area ka naam batayein (ya app mein "📍 Paas ka Kendra"
  button daba ke location de sakte hain).
  Jawab CHHOTA rakho, kyunki app neeche Kendra ki poori list (pata, doori, raasta button) khud dikhata hai:
  1) searched_near likho ("X ke paas yeh Kendra mile"), 2) sabse paas wala EK Kendra: area aur lagbhag doori,
  3) same_price_everywhere wali baat, 4) tool ka note (stock pehle pooch lein). Poori list dobara mat likho.
  Website links ya URL mat likho. Kendra mein stock hai ya nahi, yeh tumhe nahi pata: kabhi mat kaho ki dawai
  wahan pakka milegi.
- Agar user ke message mein "[📎 photo attached]" ho, toh pehle read_attached_photo chalao.
  Agar user_confirmation 'haan' hai toh photo_reading use karo; agar user ne naya naam likha hai toh WOH naam use karo.
  Photo ka composition, MRP aur pack_count search tool ko zaroor do (search_by_brand ko bhi composition do).
- Aakhir mein tool ka disclaimer zaroor likho (agar tool chala ho)."""


# 2. Node: "agent" (AI sochta hai: jawab doon ya tool maangoon?)
# Har bhasha ke kai naam: user jo bhi likhe
LANGUAGE_ALIASES = {
    "1": "Hinglish", "hinglish": "Hinglish", "roman": "Hinglish",
    "2": "Hindi", "hindi": "Hindi", "हिंदी": "Hindi", "हिन्दी": "Hindi",
    "3": "English", "english": "English", "angrezi": "English",
    "4": "Marathi", "marathi": "Marathi", "मराठी": "Marathi",
}


def match_language(answer):
    """ "hinglishh", "Englsh", " 3 " → sahi bhasha.  Kuch samajh na aaye → None """
    answer = answer.strip().lower()
    if answer in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[answer]
    # Typo maaf: sabse milta-julta naam (sirf 4 bhashaon mein se, isliye yahan fuzzy match safe hai)
    close = difflib.get_close_matches(answer, LANGUAGE_ALIASES.keys(), n=1, cutoff=0.6)
    return LANGUAGE_ALIASES[close[0]] if close else None


def choose_language_node(state: DawaiState):
    """ ⏸️ Bhasha pata nahi? User se poochho (sirf ek baar) """
    question = LANGUAGE_QUESTION
    while True:
        language = match_language(str(interrupt({"question": question})))
        if language:
            return {"language": language}
        question = "Samajh nahi aaya 🙂 Bas number likhiye: 1, 2, 3 ya 4\n" + LANGUAGE_QUESTION


def route_start(state: DawaiState):
    """ 🔀 START ke baad: bhasha chuni hui hai toh seedha agent, warna pehle bhasha """
    return "agent" if state.get("language") else "choose_language"


def agent_node(state: DawaiState):
    rule = next(rule for name, rule in LANGUAGES.values() if name == state["language"])
    messages = [SystemMessage(SYSTEM_PROMPT.format(language_rule=rule))] + state["messages"]
    reply = llm_with_tools.invoke(messages)
    return {"messages": [reply]}                 # nayi message history mein JUD jaati hai


PRICE_TOOLS = {"search_by_brand", "search_by_salt"}


def route_after_tools(state: DawaiState):
    """ ⚡ Daam mil gaya? Doosri baar AI ko bulane ki zaroorat nahi: seedha pakka jawab.
        Kuch nahi mila / kai tools chale → AI ke paas wapas (woh agla kadam soche) """
    last_ai = next(m for m in reversed(state["messages"]) if m.type == "ai")
    calls = last_ai.tool_calls or []
    if calls and calls[0]["args"].get("salt_guessed_by_ai"):
        return "agent"          # database mein nahi tha: AI ko upyog wali line + chetavni samjhaani hai
    if len(calls) == 1 and calls[0]["name"] in PRICE_TOOLS:
        try:
            result = json.loads(state["messages"][-1].content)
        except (json.JSONDecodeError, TypeError):
            return "agent"
        if result.get("found"):
            return "quick_answer"
    return "agent"


def quick_answer_node(state: DawaiState):
    result = json.loads(state["messages"][-1].content)
    return {"messages": [AIMessage(price_reply(result, state.get("language") or "Hinglish"))]}


# ---------- Bhasha ki pakki jaanch (prompt ke bharose nahi) ----------
# AI kabhi Hindi/Marathi chunne pe bhi Roman akshar mein, ya Marathi ki jagah Hindi mein likh deta hai.
# Code jaanchta hai; galat ho toh EK baar sahi bhasha mein dobara likhwata hai, aur tabhi rakhta hai
# jab saare number (₹, %, km) bilkul wahi rahein. Warna purana jawab hi rehta hai (accuracy pehle).
DEVANAGARI_CHAR = re.compile(r"[\u0900-\u097F]")
MARATHI_WORDS = re.compile(r"(आहे|आहेत|नाही|तुम्ही|तुमच्या|साठी|औषध|किंमत|आणि|मध्ये|करा|घ्या|जवळ)")
HINDI_WORDS = re.compile(r"(है|हैं|नहीं|आप|आपके|के लिए|दवा|कीमत|और|में|करें|लें|पास)")
TARGET = {
    "Hindi": "Hindi, written ONLY in Devanagari script (हिंदी)",
    "Marathi": "pure Marathi (मराठी, not Hindi), written ONLY in Devanagari script",
    "Hinglish": "Hinglish: Hindi words written in Roman/English letters",
    "English": "simple English",
}


def _devanagari_share(text):
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if DEVANAGARI_CHAR.match(c)) / len(letters) if letters else 1.0


# Hindi/Marathi ke aam shabd jab Roman akshar mein likhe hon (pate aur dawai ke naam mein yeh nahi hote)
ROMAN_HINDI_MARATHI = {
    "hai", "hain", "ke", "ka", "ki", "ko", "mein", "se", "yeh", "ye", "aur", "nahi", "paas", "liye", "aap",
    "aapke", "aapko", "jaankari", "dawai", "daam", "hota", "hoti", "pe", "par", "sirf", "zaroor", "lagbhag",
    "aahe", "aani", "ani", "je", "aushadh", "sathi", "kimat", "tumhi", "nahi", "aahet", "madhye",
}


def _roman_hindi_words(text):
    return sum(1 for w in re.findall(r"[a-z]+", text.lower()) if w in ROMAN_HINDI_MARATHI)


def wrong_language(answer, language):
    share = _devanagari_share(answer)
    if language in ("Hindi", "Marathi"):
        if _roman_hindi_words(answer) >= 3:
            return True                                    # Roman akshar mein likh diya
        if language == "Marathi":
            return len(HINDI_WORDS.findall(answer)) > len(MARATHI_WORDS.findall(answer)) + 2   # Hindi likh di
        return False
    return share > 0.4                                     # Hinglish/English chuni, Devanagari mein likh diya


def _numbers(text):
    digits = text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    return sorted(re.findall(r"\d+(?:\.\d+)?", digits))


def check_language_node(state: DawaiState):
    language = state.get("language") or "Hinglish"
    last = state["messages"][-1]
    answer = last.text
    if not answer.strip() or not wrong_language(answer, language):
        return {}
    instruction = (f"Rewrite the answer below in {TARGET[language]}. Keep the same meaning, order and length. "
                   "Keep every number, ₹ amount, % and km EXACTLY as written. Keep medicine names, salt names "
                   "and addresses as they are. Do not add or remove anything. Reply with only the rewritten answer.")
    try:
        reply = llm_with_tools.invoke([SystemMessage(instruction), HumanMessage(answer)], with_tools=False)
        fixed = reply.text.strip()
    except Exception as error:
        print(f"   ⚠️ language rewrite failed ({type(error).__name__}); keeping the original answer")
        return {}
    if not fixed or _numbers(fixed) != _numbers(answer) or wrong_language(fixed, language):
        print("   ⚠️ language rewrite changed numbers or is still off; keeping the original answer")
        return {}
    print(f"   🔤 answer rewritten in {language}")
    return {"messages": [AIMessage(content=fixed, id=last.id)]}       # same id: purane jawab ki jagah


def route_after_agent(state: DawaiState):
    """ 🔀 Tool maanga → tools; jawab taiyar → bhasha jaanch """
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "check_language"


# 3. Graph banao: dabbe (nodes) + teer (edges)
builder = StateGraph(DawaiState)
builder.add_node("choose_language", choose_language_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(TOOLS))      # tool chalane wala ready-made node
builder.add_conditional_edges(START, route_start, ["choose_language", "agent"])
builder.add_edge("choose_language", "agent")
builder.add_node("check_language", check_language_node)
builder.add_conditional_edges("agent", route_after_agent, ["tools", "check_language"])
builder.add_edge("check_language", END)
builder.add_node("quick_answer", quick_answer_node)
builder.add_conditional_edges("tools", route_after_tools, ["agent", "quick_answer"])   # ⚡ shortcut
builder.add_edge("quick_answer", END)

# 4. Yaaddasht: har thread_id ki baatcheet alag yaad rehti hai
CHAT_KEEP_DAYS = 3        # baatcheet (sehat ki jaankari) itne din baad mita do


def make_checkpointer():
    """ Baatcheet ki yaaddasht: server pe Postgres mein (restart pe bhi bachi rahe), offline pe memory mein """
    import db
    if db.enabled():
        from langgraph.checkpoint.postgres import PostgresSaver
        saver = PostgresSaver(db.pool())
        saver.setup()                       # tables pehli baar banao (dobara chalane pe kuch nahi karta)
        return saver
    return InMemorySaver()


checkpointer = make_checkpointer()
graph = builder.compile(checkpointer=checkpointer)


def cleanup_old_chats(days=CHAT_KEEP_DAYS):
    """ CHAT_KEEP_DAYS se purani baatcheet database se mita do. Kitni mitayi, woh lautao. """
    import db
    if not db.enabled():
        return 0
    rows = db.fetch_all("SELECT thread_id FROM checkpoints GROUP BY thread_id "
                        "HAVING max((checkpoint->>'ts')::timestamptz) < now() - make_interval(days => %s)", (days,))
    for row in rows:
        checkpointer.delete_thread(row["thread_id"])
    return len(rows)


# 🛡️ Output guardrail: AI ke jawab ko user tak pahunchne se PEHLE code se saaf karo
SOURCE_LINE = re.compile(r"\*?Yeh jaankari hamare medicine database se hai[^\n]*", re.IGNORECASE)
ASKS_SOURCE = re.compile(r"kahan se|kaha se|kaise pata|kaise pta|source|where did|where do|how did you know|"
                         r"how do you know|kahaan se|कहाँ से|कहां से|कुठून", re.IGNORECASE)


def guard_output(question, answer):
    """ Prompt ne mana kiya phir bhi AI source line jod de, toh code hata dega (sawaal source ka na ho toh) """
    if not ASKS_SOURCE.search(question):
        answer = SOURCE_LINE.sub("", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer).strip()     # bachi hui khaali lines saaf
    return answer


def _terminal_answer(question_for_user):
    """ Graph ka sawaal (bhasha / photo confirm) terminal pe poochho """
    answer = input(f"🙋 {question_for_user}\n👤 ")
    if answer.strip().lower() in ["bye", "exit", "quit"]:   # beech mein bhi band kar sake
        raise SystemExit("👋 Bye! Apna dhyan rakhiye.")
    return answer


def run_turn(question, photo=None, thread_id="user-1", answer_fn=_terminal_answer):
    """ Ek sawaal ka poora safar. Lautata hai: jawab + kaunse tools chale + unke results.
        answer_fn: graph ruke toh jawab kaun dega (terminal pe insaan, eval mein script) """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}   # MAX_STEPS jaisa 🛑
    before = len(graph.get_state(config).values.get("messages", []))   # pehle se kitni baatein yaad hain
    update = {"messages": [("user", question)], "photo": None}   # purani photo har naye sawaal pe hatao
    if photo:
        update["messages"] = [("user", f"{question}\n[📎 photo attached]")]   # AI ko bas itna pata chale
        update["photo"] = photo                                              # asli photo State mein
    try:
        result = graph.invoke(update, config)
        # ⏸️ Graph ruka hai? Jawab maango, phir wahin se chalu karo ▶️
        while "__interrupt__" in result:
            answer = answer_fn(result["__interrupt__"][0].value["question"])
            result = graph.invoke(Command(resume=answer), config)
    except Exception as e:
        # 🛟 Saare AI models fail (limit / outage)? App band nahi hoga: bina AI ke jawab
        print(f"   ⚠️ All AI models failed ({type(e).__name__}), answering without AI")
        language = graph.get_state(config).values.get("language") or "Hinglish"
        return {"answer": answer_without_ai(question, photo=photo, language=language),
                "tool_calls": [], "tool_results": [], "fallback": True, "error": type(e).__name__}

    tool_calls, tool_results = [], []
    for msg in result["messages"][before:]:                              # sirf IS sawaal ke naye steps
        for call in getattr(msg, "tool_calls", []) or []:
            tool_calls.append({"name": call["name"], "args": call["args"]})
        if msg.type == "tool":
            try:
                tool_results.append(json.loads(msg.content))
            except (json.JSONDecodeError, TypeError):
                tool_results.append({"raw": str(msg.content)})
    # .text: Groq seedha string deta hai, Gemini "parts" ki list. .text dono ko saaf text bana deta hai
    return {"answer": guard_output(question, result["messages"][-1].text),
            "tool_calls": tool_calls, "tool_results": tool_results}


def _collect_turn(result, question):
    """ Graph ke result se: jawab + is sawaal ke tools aur unke results """
    messages = result["messages"]
    last_human = max(i for i, m in enumerate(messages) if m.type == "human")
    tool_calls, tool_results = [], []
    for msg in messages[last_human + 1:]:
        for call in getattr(msg, "tool_calls", []) or []:
            tool_calls.append({"name": call["name"], "args": call["args"]})
        if msg.type == "tool":
            try:
                tool_results.append(json.loads(msg.content))
            except (json.JSONDecodeError, TypeError):
                tool_results.append({"raw": str(msg.content)})
    return {"answer": guard_output(question, messages[-1].text),
            "tool_calls": tool_calls, "tool_results": tool_results}


def web_turn(thread_id, question=None, photo=None, language=None, resume=None):
    """ Web/API ke liye: graph ruke toh sawaal LAUTAO (browser user se poochega),
        aur agli request mein resume=jawab ke saath wahin se chalu karo.
        Lautata hai: {"status": "needs_input", "question": ...} ya {"status": "done", "answer": ...} """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
    try:
        if resume is not None:
            result = graph.invoke(Command(resume=resume), config)
        else:
            update = {"messages": [("user", question)], "photo": None}
            if language:
                update["language"] = language            # web page pe pehle hi chun li: dobara mat poochho
            if photo:
                update["messages"] = [("user", f"{question}\n[📎 photo attached]")]
                update["photo"] = photo
            result = graph.invoke(update, config)
        if "__interrupt__" in result:
            return {"status": "needs_input", "question": result["__interrupt__"][0].value["question"]}
        state_question = question or ""
        return {"status": "done", **_collect_turn(result, state_question)}
    except Exception as e:
        print(f"   ⚠️ All AI models failed ({type(e).__name__}), answering without AI")
        lang = language or graph.get_state(config).values.get("language") or "Hinglish"
        return {"status": "done", "answer": answer_without_ai(question or "", photo=photo, language=lang),
                "tool_calls": [], "tool_results": [], "fallback": True}


def ask(question, photo=None, thread_id="user-1"):
    """ Terminal chat ke liye: tools print karo, jawab lautao """
    turn = run_turn(question, photo=photo, thread_id=thread_id)
    for call in turn["tool_calls"]:
        print(f"🔧 {call['name']}({call['args']})")
    return turn["answer"]


if __name__ == "__main__":
    print("Dawai Darpan 💊  (brand naam likho | photo attach: '/photo strip2.jpg' + sawaal)")
    print("   'bhasha' = bhasha badlo | 'bye' = band karo")
    config = {"configurable": {"thread_id": "user-1"}}
    print(f"🤖 {ask('Namaste')}")          # shuru mein hi bhasha poochho + swagat
    while True:
        q = input("\n👤 ").strip()
        if q.lower() in ["bye", "exit", "quit"]:
            break
        if q.lower() in ["bhasha", "language", "भाषा"]:
            graph.update_state(config, {"language": None})     # State mein se bhasha mita do
            q = input("👤 Ab apna sawaal likho: ").strip()
        photo = None
        match = re.search(r"/photo\s+(\S+)", q)     # 📎 UI ke "attach" button ki nakal (terminal mein)
        if match:                                    # "/photo" kahin bhi ho (aage 👤 ho tab bhi)
            name = match.group(1)
            photo = name if "/" in name else f"photos/{name}"
            q = q[match.end():].strip() or "Is dawai ka sasta option batao"
            print(f"📎 Photo attach hui: {photo}")
        print(f"🤖 {ask(q, photo=photo)}")
