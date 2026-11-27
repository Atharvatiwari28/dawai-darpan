import os
import sys
import time
import io
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

register_heif_opener()      # iPhone ki HEIC/HEIF photos bhi Pillow khol sake
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
from pydantic import BaseModel, Field

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Naya model pehle; limit khatam ho toh agla (Yojana Mitra wala fallback idea ♻️)
# Pehle halke (lite) models: kam bheed, tez jawab. Phir bade models.
VISION_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
                 "gemini-3.6-flash", "gemini-3.8-flash"]
RETRIES = 1          # 503 pe sirf 1 baar aur mauka, phir agla model
WAIT_SECONDS = 2
TIMEOUT_MS = 20_000  # 20 sec se zyada koi model na le
MAX_SIZE = 1024      # photo ki lambi side zyada se zyada itne pixel


# 📋 Yeh woh "form" hai jo AI ko bharna hai
class StripInfo(BaseModel):
    is_medicine: bool = Field(description="Kya photo mein sach mein dawai ki strip/dabba/bottle hai?")
    brand_name: str | None = Field(description="Brand ka naam strength ke saath, jaise 'Dolo 650'. Na dikhe toh null.")
    composition: str | None = Field(description="Salt aur strength jaisa likha hai, jaise 'Paracetamol 650 mg'. Na dikhe toh null.")
    mrp: float | None = Field(description="MRP rupees mein, sirf number. Na dikhe toh null.")
    pack_count: int | None = Field(description="Pack mein kitni goli/capsule. Na dikhe toh null.")


PROMPT = """Yeh ek dawai ki photo hai. Sirf wahi likho jo photo mein SAAF dikh raha hai.
Agar koi cheez padhne mein nahi aa rahi, toh use null rakho. Andaaza mat lagao.
Agar photo mein dawai nahi hai, toh is_medicine false rakho."""


def shrink_photo(image_path):
    """ Phone ki 4 MB photo → ~150 KB JPEG. Text phir bhi padhne layak rehta hai. """
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)            # iPhone photo ko seedha ghumao
    img = img.convert("RGB")                      # PNG/transparent ko bhi JPEG-friendly banao
    img.thumbnail((MAX_SIZE, MAX_SIZE))           # aspect ratio wahi rehta hai
    buffer = io.BytesIO()                         # RAM mein ek "nakli file"
    img.save(buffer, format="JPEG", quality=85)
    before = os.path.getsize(image_path) // 1024
    after = buffer.tell() // 1024
    print(f"🗜️ Photo: {before} KB → {after} KB")
    return buffer.getvalue()


class NotAnImage(Exception):
    """ File photo nahi hai (PDF, video, ya kharab file) """


def read_strip(image_path):
    """ photo (JPG, PNG, WEBP, HEIC, ...) → StripInfo (ya None agar koi model kaam na kare) """
    try:
        image_bytes = shrink_photo(image_path)   # koi bhi format → chhoti JPEG
    except UnidentifiedImageError:
        raise NotAnImage(image_path)
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

    start = time.time()
    for model in VISION_MODELS:
        for attempt in range(RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[image_part, PROMPT],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=StripInfo,      # 👈 "sirf yeh form bharo"
                        temperature=0,                  # padhna hai, kahani nahi banani
                        http_options=types.HttpOptions(timeout=TIMEOUT_MS),
                    ),
                )
                print(f"🤖 Model: {model}  ({time.time() - start:.1f} sec)")
                return response.parsed                  # seedha StripInfo object milta hai
            except Exception as e:                      # timeout bhi pakdo, sirf APIError nahi
                if not isinstance(e, errors.APIError):
                    print(f"⚠️ {model} ne time liya ya error ({type(e).__name__}), agla model")
                    break
                if e.code == 503 and attempt < RETRIES:
                    print(f"⏳ {model} busy hai (503), {WAIT_SECONDS} sec ruk ke dobara...")
                    time.sleep(WAIT_SECONDS)
                    continue                            # isi model ko dobara try karo
                print(f"⚠️ {model} nahi chala ({e.code}: {e.message}), agla model")
                break                                   # agle model pe jao
    return None


if __name__ == "__main__":
    path = sys.argv[1]                              # python read_strip.py photos/dolo.jpg
    info = read_strip(path)
    if info is None:
        print("❌ Koi model kaam nahi kiya, thodi der baad try karo")
    else:
        print(info.model_dump_json(indent=2))
