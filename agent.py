"""Dawai Darpan ka pehla agent: BINA framework ke, haath se likha loop.
AI tool chunta hai → Python chalata hai → result AI ko → AI jawab deta hai."""
import os
import json
from dotenv import load_dotenv
from groq import Groq
from tools import search_by_brand, search_by_salt

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"
MAX_STEPS = 5          # AI loop mein phans jaaye toh rok do 🛑

# 1. Tools ki "menu card": AI ko sirf yeh dikhta hai (code nahi, sirf naam + kaam + inputs)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_by_brand",
            "description": search_by_brand.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "brand_name": {"type": "string", "description": "Brand ka naam, jaise 'Dolo 650'"},
                    "mrp": {"type": ["number", "null"], "description": "Strip ka MRP rupees mein (agar user ne bataya)"},
                    "pack_count": {"type": ["integer", "null"], "description": "Strip mein kitni goliyan (agar pata ho)"},
                },
                "required": ["brand_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_salt",
            "description": search_by_salt.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "composition": {"type": "string", "description": "Salt + strength, jaise 'Azithromycin 500 mg'"},
                    "mrp": {"type": ["number", "null"], "description": "Strip ka MRP rupees mein (agar user ne bataya)"},
                    "pack_count": {"type": ["integer", "null"], "description": "Strip mein kitni goliyan (agar pata ho)"},
                },
                "required": ["composition"],
            },
        },
    },
]

# 2. Naam (text) → asli Python function
AVAILABLE = {"search_by_brand": search_by_brand, "search_by_salt": search_by_salt}

SYSTEM_PROMPT = """Tum Dawai Darpan ho: logon ko brand dawai ka saasta Jan Aushadhi generic dhoondhne mein madad karte ho.
Niyam:
- Daam, bachat ya generic ka naam KABHI apne mann se mat batao. Hamesha tool se lo.
- Koi bhi number khud calculate mat karo. Sirf wahi number likho jo tool ke result mein hai.
- Per goli daam dono ka batao (brand_per_unit vs generic_per_unit), taaki tulna saaf ho.
- Tool 'found: false' de toh uska 'reason' padho aur agla sahi kadam lo (jaise doosra tool).
- Kuch na mile toh imaandari se bolo. Andaaza mat lagao.
- Jawab Hinglish mein, chhota aur saaf. Aakhir mein tool ka disclaimer zaroor likho.
- Dawai se jude medical salah (kaunsi dawai leni chahiye, dose) mat do: doctor ke paas bhejo."""


def ask(user_message):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}]

    for step in range(1, MAX_STEPS + 1):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS,
            temperature=0, reasoning_effort="low",
        )
        reply = response.choices[0].message

        # 3a. AI ne tool nahi maanga → yahi final jawab hai
        if not reply.tool_calls:
            return reply.content

        # 3b. AI ne tool maanga → chalao, result wapas do
        # AI ki "request" bhi history mein rakho (saaf dict bana ke)
        messages.append({
            "role": "assistant",
            "content": reply.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments}}
                           for c in reply.tool_calls],
        })
        for call in reply.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)    # AI JSON text bhejta hai → dict
            print(f"🔧 Step {step}: {name}({args})")
            func = AVAILABLE.get(name)
            if func is None:
                result = {"found": False, "reason": f"'{name}' naam ka koi tool nahi hai"}
            else:
                try:
                    result = func(**args)                 # **args: dict ko keyword arguments bana do
                except TypeError as e:                    # AI ne galat input bheja
                    result = {"found": False, "reason": f"Tool ko galat input mila: {e}"}
            print(f"   ↳ found={result.get('found')}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,                  # kis request ka jawab hai
                "content": json.dumps(result, ensure_ascii=False),
            })

    return "Maaf kijiye, jawab dhoondhne mein bahut steps lag gaye. Thoda alag tarah poochiye."


if __name__ == "__main__":
    questions = [
        "Dolo 650 ka sasta option hai kya?",
        "Bio-AZ 500 ka generic batao, isme Azithromycin 500mg hai. Strip 75.50 ki hai, 3 goli.",
        "Mujhe bukhaar hai, kaunsi dawai lun?",
    ]
    for q in questions:
        print(f"\n👤 {q}")
        print(f"🤖 {ask(q)}")