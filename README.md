# दवाई दर्पण · Dawai Darpan

**Find the cheaper Jan Aushadhi generic for a branded medicine, and the nearest Kendra that sells it.**

Type a medicine name, ask by voice, or send a photo of the strip. Dawai Darpan finds the Jan Aushadhi
generic with the same salt, strength and form, shows how much you save, and lists the nearest
Pradhan Mantri Bhartiya Janaushadhi Kendras. It speaks Hinglish, Hindi, English and Marathi.

## What it does

- **Brand → generic:** matches on salt + strength + form + release type (SR is never matched with a
  normal tablet). The composition printed on the strip always takes priority.
- **Strip photo:** reads the brand, composition, MRP and pack size with a vision model, then asks the
  user to confirm before searching (human-in-the-loop).
- **Voice:** speech-to-text, and a spoken answer in the user's language.
- **Nearest Kendra:** by browser location, pincode, or an area name ("Kalyani Nagar, Pune",
  "वडगाव शेरी पुणे"), with Google Maps directions.
- **Guard-rails:** numbers come only from tools (never from the model), medicine-only scope, no
  personal medical advice, side effects only when asked, works in degraded mode if the AI is down.

## How it works

```
Browser (frontend/index.html) ──> FastAPI (api.py) ──> LangGraph agent (graph_agent.py)
                                                         │  tools.py: search_by_brand, search_by_salt,
                                                         │            medicine_info, read_attached_photo,
                                                         │            find_kendra
                                                         ▼
                                          Postgres: app data and chat memory
```

- **Agent:** LangGraph with tool calling, a model router with fallbacks (Groq → Gemini), and a fast
  path that answers price questions without a second model call.
- **Storage:** Postgres in production (`db.py`). Chats are deleted after 3 days.
- **Locations:** area names in Latin or Devanagari script, with a spelling-tolerant fallback.
- **MCP server:** `dawai_mcp_server.py` exposes the same tools to any MCP client.

## Run it locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then fill in your own keys
python -m uvicorn api:app --reload --port 8002
```

Open http://127.0.0.1:8002.

The app's data is not included in this repository.

## Tech

Python · FastAPI · LangGraph / LangChain · Groq · Google Gemini · Postgres (Neon) · psycopg ·
pandas · edge-tts · vanilla HTML/CSS/JS · MCP · Render
