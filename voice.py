"""🎙️ Awaaz: bolo → text (Groq Whisper) → agent → jawab → suno (Mac ki awaaz).

Chalao:  python voice.py
         Enter dabao → bolo → Enter dabao (ruk jao) → jawab suno
"""
import os
import re
import base64
from pathlib import Path
import time
import wave
import tempfile
import subprocess
from dotenv import load_dotenv
from functools import lru_cache
from groq import Groq
import asyncio
import edge_tts
from google import genai
from google.genai import types

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 🔊 Gemini TTS: insaan jaisi awaaz, Hinglish/Hindi/Marathi achhe se bolti hai.
# Naam badalte rehte hain, isliye kai try karte hain; koi na chale toh Mac ki awaaz.
TTS_MODELS = ["gemini-3.8-flash-tts", "gemini-3.1-flash-tts-preview", "gemini-2.5-flash-preview-tts"]
TTS_VOICE = "Charon"          # shaant, saaf, bharosemand mard ki awaaz (Gemini ki prebuilt voice)
TTS_STYLE = {
    "Hinglish": "Read this in Hindi with a warm, calm, clear Indian accent, like a caring pharmacist",
    "Hindi": "Read this in Hindi with a warm, calm, clear voice, like a caring pharmacist",
    "English": "Read this with a warm, calm, clear Indian English accent, like a caring pharmacist",
    "Marathi": "Read this in Marathi with a warm, calm, clear voice, like a caring pharmacist",
}

SAMPLE_RATE = 16000          # Whisper 16 kHz pe kaam karta hai; isse zyada = bas badi file

# Whisper ko "hint": in shabdon ki spelling aise hi likhna (dawai ke naam galat na sune)
WHISPER_HINT = ("Dolo 650, Pan-D, Pan 40, Telma 40, Augmentin 625, Azithromycin, Paracetamol, "
                "Jan Aushadhi, generic, sasta option, dawai")

# Whisper ki bhasha (ISO code). Hinglish: code nahi do, taaki Roman (English akshar) mein likhe
WHISPER_LANGUAGE = {"Hinglish": None, "Hindi": "hi", "English": "en", "Marathi": "mr"}

# Mac ki awaazein ("say -v '?'" se dekh sakte ho).
# Lekha = Hindi awaaz, Devanagari padhti hai. Roman akshar ("Aap") use naye lagte hain → "A A P" padhti hai.
# Veena/Rishi = Indian English awaazein, Roman akshar shabd ki tarah padhti hain → Hinglish ke liye sahi.



# ---------- 1. Record ----------
def record_until_enter():
    """ Enter dabao tak mic se suno. Lautata hai: WAV file ka raasta """
    import numpy as np                     # yahan import: server (bina mic) pe voice.py ka baaki hissa chale
    import sounddevice as sd
    chunks = []

    def on_audio(data, frames, time_info, status):
        chunks.append(data.copy())               # mic se aaya har chhota tukda jama karo

    input("🎙️  Enter dabao aur bolo...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=on_audio):
        input("🔴 Sun raha hoon... bolne ke baad Enter dabao")
    if not chunks:
        return None

    audio = np.concatenate(chunks)
    if len(audio) < SAMPLE_RATE * 0.5:           # aadhe second se kam = kuch bola hi nahi
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:   # kaam ke baad mita denge
        path = tmp.name
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)                        # int16 = 2 bytes
        f.setframerate(SAMPLE_RATE)
        f.writeframes(audio.tobytes())
    return path


# ---------- 2. Speech → Text ----------
def transcribe(path, language="Hinglish"):
    with open(path, "rb") as audio:
        result = groq_client.audio.transcriptions.create(
            file=audio,
            model="whisper-large-v3-turbo",      # tez aur multilingual
            language=WHISPER_LANGUAGE.get(language),
            prompt=WHISPER_HINT,
            temperature=0.0,
        )
    return result.text.strip()


# ---------- 3. Text → Speech ----------
def clean_for_speech(text):
    """ Bolne layak banao: emoji, markdown (**, |, #) hatao; ₹ → rupaye; har line = alag vaakya """
    text = re.sub(r"₹\s?(\d+(?:\.\d+)?)", r"\1 rupaye", text)          # "₹1.03" → "1.03 rupaye"
    text = text.replace("→", ",").replace("\ufe0f", "")
    # Shabd jo awaaz galat padhti hai → bolne wala roop
    text = re.sub(r"(\d)\s?mg\b", r"\1 milligram", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d)\s?mcg\b", r"\1 microgram", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d)\s?%", r"\1 percent", text)
    text = re.sub(r"\bvs\.?\b", "aur", text, flags=re.IGNORECASE)
    text = re.sub(r"\bIP\b", "", text)                                  # "Tablets IP" → "Tablets"
    text = re.sub(r"[*_#|`>~]", " ", text)
    text = re.sub(r"[\U0001F000-\U0001FFFF\u2600-\u27BF]", " ", text)   # emoji
    text = re.sub(r"-{3,}", " ", text)
    # Har nayi line ek alag vaakya hai: aakhir mein full stop lagao, warna awaaz bina ruke bolti jaati hai
    lines = [l.strip(" ,;:") for l in text.splitlines()]
    lines = [l if re.search(r"[.!?।]$", l) else l + "." for l in lines if l]
    text = " ".join(lines)
    return re.sub(r"\s+", " ", text).strip()


def add_pauses_for_mac(text):
    """ Mac ki 'say' ko har vaakya ke baad aadha second chup rehne ko kaho """
    return re.sub(r"([.!?।])\s+", r"\1 [[slnc 450]] ", text)


# 🗣️ Microsoft Edge ki neural awaazein: free, bina API key, Indian accent, Hinglish achha bolti hain.
#    (Edge browser ki "Read aloud" wali service. Unofficial hai, isliye peeche Gemini + Mac backup)
#    Kaunsi awaaz pasand hai, sunke chuno: python voice_test.py
EDGE_VOICES = {                          # tumne sunke chuni: Neerja (Indian English, mahila)
    "Hinglish": "en-IN-NeerjaNeural",
    "English": "en-IN-NeerjaNeural",
    "Hindi": "hi-IN-SwaraNeural",         # Devanagari text ke liye Hindi mahila awaaz (Neerja jaisi)
    "Marathi": "mr-IN-AarohiNeural",      # Marathi mahila awaaz
}


@lru_cache(maxsize=64)
def edge_speech(spoken, language):
    """ Text → MP3 file (Edge neural voice). Fail ho toh None """
    voice = EDGE_VOICES.get(language, EDGE_VOICES["Hinglish"])
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        path = tmp.name
    try:
        asyncio.run(edge_tts.Communicate(spoken, voice, rate="-5%").save(path))
    except Exception as e:
        print(f"   🔇 Edge voice failed ({type(e).__name__}): {str(e)[:120]}")
        return None
    return path if os.path.getsize(path) > 0 else None


_working_tts = {"model": None}     # jo model ek baar chal gaya, agli baar seedha wahi


def _gemini_audio(model, spoken, language):
    """ Gemini se awaaz ke bytes + sample rate. Naye TTS models "interactions" API se chalte hain """
    styled = f"{TTS_STYLE.get(language, TTS_STYLE['Hinglish'])}: {spoken}"
    if hasattr(gemini_client, "interactions"):                 # google-genai ka naya SDK
        result = gemini_client.interactions.create(
            model=model, input=styled, response_format={"type": "audio"},
            generation_config={"speech_config": [{"voice": TTS_VOICE}]},
        )
        audio = result.output_audio
        raw = base64.b64decode(audio.data) if isinstance(audio.data, str) else audio.data
        return raw, (audio.sample_rate or 24000), str(audio.mime_type or "")
    response = gemini_client.models.generate_content(          # purana tareeka (purane models)
        model=model, contents=styled,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE))),
        ),
    )
    return response.candidates[0].content.parts[0].inline_data.data, 24000, ""


@lru_cache(maxsize=64)
def gemini_tts(spoken, language):
    """ Text → awaaz (WAV file ka raasta). Same jawab dobara bolna ho toh cache se (turant, quota bacha) """
    models = [_working_tts["model"]] if _working_tts["model"] else TTS_MODELS
    for model in models:
        try:
            raw, rate, mime = _gemini_audio(model, spoken, language)
        except Exception as e:
            print(f"   🔇 {model} TTS failed ({type(e).__name__}): {str(e)[:120]}")
            continue
        _working_tts["model"] = model
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        if "wav" in mime:                         # pehle se poori WAV file
            Path(path).write_bytes(raw)
        else:                                     # "raw PCM": 16-bit mono, upar se header lagao
            with wave.open(path, "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(rate)
                f.writeframes(raw)
        return path
    return None


def speak(text, language="Hinglish"):
    """ Awaaz ki chain: Edge neural → Gemini TTS → Mac ki awaaz. Koi na koi hamesha bolega """
    spoken = clean_for_speech(text)
    path = edge_speech(spoken, language)                      # 1️⃣ Edge neural (sabse natural, free)
    if path and os.path.exists(path):
        print(f"   🔊 Edge voice: {EDGE_VOICES.get(language)}")
        subprocess.run(["afplay", path])
        return
    path = gemini_tts(spoken, language)                       # 2️⃣ Gemini TTS
    if path and os.path.exists(path):
        print(f"   🔊 Gemini voice: {TTS_VOICE} ({_working_tts['model']})")
        subprocess.run(["afplay", path])          # Mac ka audio player
        return
    installed = installed_mac_voices()
    voice = next((v for v in MAC_VOICES.get(language, []) if v in installed), None)
    if voice:
        print(f"   🔊 Mac voice: {voice}")
        subprocess.run(["say", "-v", voice, add_pauses_for_mac(spoken)])
    else:
        print("   🔊 Mac default voice")
        subprocess.run(["say", add_pauses_for_mac(spoken)])


# ---------- Chat ----------
def main():
    from graph_agent import ask, graph        # yahan import: pehle mic/awaaz ki galti jaldi dikhe

    config = {"configurable": {"thread_id": "voice-user"}}
    print("Dawai Darpan 🎙️  (Enter = bolo | type karke bhi likh sakte ho | 'bye' = band)")
    print(f"🤖 {ask('Namaste', thread_id='voice-user')}")
    language = graph.get_state(config).values.get("language") or "Hinglish"

    while True:
        typed = input("\n👤 (Enter = bolo, ya likho): ").strip()
        if typed.lower() in ["bye", "exit", "quit"]:
            break
        if typed:
            question = typed
        else:
            path = record_until_enter()
            if path is None:
                print("🤔 Kuch sunai nahi diya, dobara bolo")
                continue
            start = time.time()
            try:
                question = transcribe(path, language)
            finally:
                os.remove(path)                    # 🔒 awaaz ki file turant mita do (health data!)
            print(f"   ⏱️ speech→text: {time.time() - start:.1f}s")
            if not question:
                print("🤔 Samajh nahi aaya, dobara bolo")
                continue
            print(f"🗣️ Aapne kaha: {question}")

        start = time.time()
        answer = ask(question, thread_id="voice-user")
        print(f"   ⏱️ agent: {time.time() - start:.1f}s")
        print(f"🤖 {answer}")
        start = time.time()
        speak(answer, language)
        print(f"   ⏱️ speaking: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
