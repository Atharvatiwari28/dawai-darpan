"""Kaunsi awaaz sabse achhi lagti hai? Ek hi Hinglish vaakya alag-alag awaazon mein suno."""
import asyncio
import subprocess
import tempfile
import edge_tts

SENTENCE = ("Aapki dawai ka sasta option mil gaya. Dolo 650 ka generic, Paracetamol 650 milligram, "
            "sirf 1.03 rupaye per goli. Aapko 55 percent bachat hogi. "
            "Dawai badalne se pehle doctor se zaroor poochein.")

VOICES = [
    ("hi-IN-MadhurNeural", "Hindi, mard"),
    ("hi-IN-SwaraNeural", "Hindi, mahila"),
    ("en-IN-PrabhatNeural", "Indian English, mard"),
    ("en-IN-NeerjaNeural", "Indian English, mahila"),
]


async def say(voice):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        path = tmp.name
    await edge_tts.Communicate(SENTENCE, voice, rate="-5%").save(path)
    return path


for number, (voice, label) in enumerate(VOICES, 1):
    print(f"\n{number}. {voice}  ({label})")
    try:
        path = asyncio.run(say(voice))
        subprocess.run(["afplay", path])
    except Exception as e:
        print(f"   ❌ nahi chala: {type(e).__name__}")

print("\nKaunsa number sabse achha laga? voice.py ke EDGE_VOICES mein wahi naam daal denge.")
