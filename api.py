"""Dawai Darpan ka web backend (FastAPI) 🌐
Browser (frontend/index.html) yahan request bhejta hai: text, photo, ya awaaz.

Chalao:   python -m uvicorn api:app --reload --port 8002
Kholo:    http://127.0.0.1:8002
"""
import os
import threading
import time
import uuid
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from graph_agent import web_turn, cleanup_old_chats
from voice import transcribe, clean_for_speech, edge_speech, gemini_tts

CLEANUP_EVERY_HOURS = 6


def _cleanup_loop():
    """ Har 6 ghante: purani baatcheet mitao (sehat ki jaankari zaroorat se zyada din na rahe) """
    while True:
        try:
            deleted = cleanup_old_chats()
            if deleted:
                print(f"Deleted {deleted} old chats")
        except Exception as error:                      # cleanup fail ho toh bhi app chalti rahe
            print(f"Old-chat cleanup failed: {error}")
        time.sleep(CLEANUP_EVERY_HOURS * 3600)


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=_cleanup_loop, daemon=True, name="chat-cleanup").start()
    yield


app = FastAPI(title="Dawai Darpan API", lifespan=lifespan)

UPLOADS = Path("uploads")
UPLOADS.mkdir(exist_ok=True)
MAX_PHOTO_MB = 10
MAX_AUDIO_MB = 10
LANGUAGES = {"Hinglish", "Hindi", "English", "Marathi"}

# thread_id → photo ka raasta (photo confirm hone tak rakhni padti hai, phir mita dete hain)
pending_photos = {}


def _save_upload(upload: UploadFile, folder: Path, max_mb: int, default_ext: str) -> str:
    data = upload.file.read()
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"File {max_mb} MB se badi hai")
    ext = Path(upload.filename or "").suffix.lower() or default_ext
    path = folder / f"{uuid.uuid4().hex}{ext}"           # random naam: user ka asli naam kabhi nahi
    path.write_bytes(data)
    return str(path)


def _cleanup_photo(thread_id: str):
    path = pending_photos.pop(thread_id, None)
    if path and os.path.exists(path):
        os.remove(path)                                   # 🔒 dawai ki photo = health data, turant mitao


def _suggestions(result: dict):
    """ "Kya aapka matlab...?" wale naam: browser inhe tap karne wale buttons banayega """
    names = []
    for r in result.get("tool_results", []):
        for n in r.get("did_you_mean") or []:
            if n not in names:
                names.append(n)
    return names[:3]


def _cards(result: dict):
    """ Browser mein "bachat card" dikhane ke liye: sirf mile hue generic results """
    return [r for r in result.get("tool_results", []) if r.get("found") and r.get("generic_name")]


def _kendras(result: dict):
    """ Agent ne Kendra dhoondhe hon toh browser unka card banayega """
    for r in result.get("tool_results", []):
        if r.get("found") and r.get("kendras"):
            return {"list": r["kendras"], "near": r.get("searched_near")}
    return None


@app.get("/kendras")
def kendras(lat: float | None = None, lon: float | None = None, place: str | None = None):
    """ 📍 button: browser ki location, ya likha hua pincode/area, se paas ke Kendra.
    Location kahin save nahi hoti, na log hoti hai. """
    from kendra import kendras_near_place, nearest_kendras
    if place:
        return kendras_near_place(place[:120], limit=5)
    if lat is None or lon is None:
        return {"found": False, "reason": "location, pincode ya area chahiye"}
    return nearest_kendras(round(lat, 3), round(lon, 3), limit=5)    # ~100 m tak gol: zyada sateek nahi chahiye


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(thread_id: str = Form(...),
         language: str = Form("Hinglish"),
         message: str = Form(""),
         photo: UploadFile | None = File(None),
         audio: UploadFile | None = File(None)):
    if language not in LANGUAGES:
        raise HTTPException(422, "Language galat hai")

    transcript = None
    if audio is not None:                                 # 🎙️ awaaz → text
        path = _save_upload(audio, Path(tempfile.gettempdir()), MAX_AUDIO_MB, ".webm")
        try:
            transcript = transcribe(path, language)
        finally:
            os.remove(path)                               # 🔒 awaaz bhi turant mitao
        message = transcript

    photo_path = None
    if photo is not None:                                 # 📎 photo
        _cleanup_photo(thread_id)
        photo_path = _save_upload(photo, UPLOADS, MAX_PHOTO_MB, ".jpg")
        pending_photos[thread_id] = photo_path
        message = message or "Is dawai ka sasta option batao"

    if not message.strip():
        raise HTTPException(422, "Kuch likho, bolo, ya photo bhejo")

    result = web_turn(thread_id, question=message, photo=photo_path, language=language)
    if result["status"] == "done":
        _cleanup_photo(thread_id)
    return {**result, "transcript": transcript, "cards": _cards(result), "suggestions": _suggestions(result),
            "kendras": _kendras(result)}


@app.post("/resume")
def resume(thread_id: str = Form(...), answer: str = Form(...)):
    """ Agent ne kuch poocha tha (jaise "photo mein Pan-D padha, sahi hai?"): user ka jawab yahan aata hai """
    result = web_turn(thread_id, resume=answer)
    if result["status"] == "done":
        _cleanup_photo(thread_id)
    return {**result, "transcript": None, "cards": _cards(result), "suggestions": _suggestions(result),
            "kendras": _kendras(result)}


@app.post("/tts")
def tts(text: str = Form(...), language: str = Form("Hinglish")):
    """ Jawab → awaaz (MP3/WAV). Kuch na chale toh 204: browser apni awaaz use karega """
    spoken = clean_for_speech(text)[:1500]
    path = edge_speech(spoken, language)
    media = "audio/mpeg"
    if not path:
        path, media = gemini_tts(spoken, language), "audio/wav"
    if not path or not os.path.exists(path):
        return Response(status_code=204)
    return FileResponse(path, media_type=media)


# Web page: same server se, taaki CORS ki jhanjhat hi na ho
@app.get("/chat", include_in_schema=False)
def chat_page():
    # The chat is its own page (/chat); the same HTML decides which page to show from the URL
    return FileResponse("frontend/index.html")


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
