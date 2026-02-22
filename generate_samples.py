"""
generate_samples.py
-------------------
Generates multilingual sample audio clips for testing the VoiceDataCurator pipeline.

Uses gTTS (Google Text-to-Speech) to create real speech in:
  English, Hindi, Marathi, French, Spanish, German, Arabic, Japanese

Saves MP3 files to data/raw/  — ready to run through main.py.

Usage:
    pip install gTTS
    python generate_samples.py
"""

import os
import time
from pathlib import Path

try:
    from gtts import gTTS
except ImportError:
    print("gTTS not found. Installing...")
    os.system("pip install gTTS")
    from gtts import gTTS

OUTPUT_DIR = Path("./data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Sample sentences per language
# Format: (language_code, tld, filename_prefix, list_of_sentences)
# ---------------------------------------------------------------------------
SAMPLES = [
    # ── English ─────────────────────────────────────────────────────────────
    ("en", "com", "en", [
        "The quick brown fox jumps over the lazy dog.",
        "Speech recognition technology has advanced significantly in recent years.",
        "Hello, my name is Alex and I am testing the voice data curator.",
        "Artificial intelligence is transforming the way we process language.",
        "Please speak clearly into the microphone for best results.",
    ]),
    # ── Hindi ────────────────────────────────────────────────────────────────
    ("hi", "com", "hi", [
        "नमस्ते, मेरा नाम अनन्या है और मैं हिंदी में बोल रही हूं।",
        "भारत एक विविधताओं से भरा देश है जहाँ अनेक भाषाएँ बोली जाती हैं।",
        "आज का मौसम बहुत सुहाना है और आसमान साफ है।",
        "कृत्रिम बुद्धिमत्ता का उपयोग आज हर क्षेत्र में हो रहा है।",
        "यह एक परीक्षण वाक्य है जिसे आवाज़ डेटासेट के लिए रिकॉर्ड किया गया है।",
    ]),
    # ── Marathi ──────────────────────────────────────────────────────────────
    ("mr", "com", "mr", [
        "नमस्कार, माझे नाव प्रिया आहे आणि मी मराठीत बोलत आहे.",
        "महाराष्ट्र हे भारतातील एक महत्त्वाचे राज्य आहे.",
        "आजचे हवामान खूप छान आहे आणि आकाश स्वच्छ आहे.",
        "हे एक चाचणी वाक्य आहे जे ध्वनी डेटासाठी रेकॉर्ड केले आहे.",
    ]),
    # ── French ───────────────────────────────────────────────────────────────
    ("fr", "fr", "fr", [
        "Bonjour, je m'appelle Marie et je parle en français.",
        "La technologie de reconnaissance vocale a fait de grands progrès.",
        "Le ciel est bleu et le soleil brille aujourd'hui.",
        "L'intelligence artificielle transforme notre façon de vivre.",
    ]),
    # ── Spanish ──────────────────────────────────────────────────────────────
    ("es", "es", "es", [
        "Hola, me llamo Carlos y estoy hablando en español.",
        "La tecnología de inteligencia artificial avanza rápidamente.",
        "El reconocimiento de voz es una herramienta muy útil hoy en día.",
        "Buenos días, ¿cómo estás tú hoy?",
    ]),
    # ── German ───────────────────────────────────────────────────────────────
    ("de", "de", "de", [
        "Guten Tag, mein Name ist Klaus und ich spreche auf Deutsch.",
        "Die Spracherkennungstechnologie hat sich in den letzten Jahren stark verbessert.",
        "Künstliche Intelligenz verändert die Art und Weise, wie wir kommunizieren.",
    ]),
    # ── Japanese ─────────────────────────────────────────────────────────────
    ("ja", "com", "ja", [
        "こんにちは、私の名前は田中です。日本語でお話しします。",
        "人工知能は私たちの生活を大きく変えています。",
        "音声認識技術は近年大きく進歩しています。",
    ]),
    # ── Arabic ───────────────────────────────────────────────────────────────
    ("ar", "com", "ar", [
        "مرحباً، اسمي أحمد وأنا أتحدث باللغة العربية.",
        "تقنية الذكاء الاصطناعي تتطور بشكل متسارع في عصرنا الحالي.",
    ]),
]

# ---------------------------------------------------------------------------
# Generate audio files
# ---------------------------------------------------------------------------
def generate_samples():
    total = sum(len(sentences) for _, _, _, sentences in SAMPLES)
    print(f"\nVoiceDataCurator -- Sample Generator")
    print(f"  Generating {total} audio clips across {len(SAMPLES)} languages...\n")

    generated = 0
    errors = 0

    for lang_code, tld, prefix, sentences in SAMPLES:
        lang_name = {
            "en": "English", "hi": "Hindi", "mr": "Marathi",
            "fr": "French", "es": "Spanish", "de": "German",
            "ja": "Japanese", "ar": "Arabic",
        }.get(lang_code, lang_code)

        print(f"  [{lang_name}]")

        for i, sentence in enumerate(sentences, start=1):
            filename = OUTPUT_DIR / f"{prefix}_{i:02d}.mp3"

            # Skip if already exists
            if filename.exists():
                print(f"    [ok] {filename.name} (already exists, skipping)")
                generated += 1
                continue

            try:
                tts = gTTS(text=sentence, lang=lang_code, tld=tld, slow=False)
                tts.save(str(filename))
                print(f"    [ok] {filename.name}")
                generated += 1
                # Small delay to avoid rate limiting
                time.sleep(0.4)
            except Exception as e:
                print(f"    [ERR] {filename.name} -- ERROR: {e}")
                errors += 1
                time.sleep(1.0)  # Back off on error

        print()

    print("=" * 50)
    print(f"  Generated : {generated} files")
    if errors:
        print(f"  Errors    : {errors} files")
    print(f"  Location  : {OUTPUT_DIR.resolve()}")
    print("=" * 50)
    print("\nReady! Now run:")
    print("  python main.py --input ./data/raw\n")


if __name__ == "__main__":
    generate_samples()
