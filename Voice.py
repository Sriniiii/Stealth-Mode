import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import openai
from dotenv import load_dotenv
import os
import sys
import threading
import time
import datetime
import textwrap
import webrtcvad

vad = webrtcvad.Vad(2)  # Mode 1: Less aggressive, Mode 3: More aggressive


# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Whisper model
model = WhisperModel("base", device="cpu", compute_type="int8")

SAMPLE_RATE = 16000
BLOCK_SIZE = int(SAMPLE_RATE * 3)  # 3 seconds

transcript_buffer = []
last_transcript = ""
MAX_TRANSCRIPT_CHARS = 10000  # Max length before truncating buffer
LINE_WRAP_WIDTH = 284  # Character limit before line break in formatted output
#Voice Activity Detection 


# ------------------ Utility Functions ------------------

def clean_text(text):
    text = text.strip()
    if len(text.split()) <= 2 and text.endswith('.'):
        text = text[:-1]
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    return text

def smart_corrections(text):
    # Reduced aggressive correction list
    corrections = {
        "choir descendant": "car's extended",
        "your choir descendant": "your car's extended warranty",
        "i was going today.": "I was going to say"
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

def is_silent(audio, threshold=0.01):
    rms = np.linalg.norm(audio) / np.sqrt(len(audio))
    return rms < threshold

def is_junk_text(text):
    words = text.strip().lower().split()
    return len(words) <= 2 and all(w in {"you", "uh", "um", "like", "..."} for w in words)

def append_to_file(text, filename="transcript_log.txt"):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {text}\n")

def wrap_formatted_text(text, width=284):
    paragraphs = text.split("\n")
    wrapped = []
    for p in paragraphs:
        if p.strip():
            wrapped.extend(textwrap.wrap(p, width=width))
        else:
            wrapped.append("")
    return "\n".join(wrapped)

# ------------------ Transcription Callback ------------------

def callback(indata, frames, time, status):
    global last_transcript, transcript_buffer

    if status:
        print("Audio stream error:", status)
        return

    audio = indata[:, 0]
    if is_silent(audio):
        return

    segments, _ = model.transcribe(audio, language="en")
    new_text = ""

    for segment in segments:
        cleaned = clean_text(segment.text)
        corrected = smart_corrections(cleaned)
        new_text += corrected + " "

    new_text = new_text.strip()
    if new_text and not is_junk_text(new_text):
        transcript_buffer.append(new_text)
        append_to_file(new_text)

        # Trim buffer if too long
        if len(" ".join(transcript_buffer)) > MAX_TRANSCRIPT_CHARS:
            transcript_buffer = [transcript_buffer[-1]]

        new_total = " ".join(transcript_buffer)
        if new_total != last_transcript:
            diff = new_total[len(last_transcript):]
            sys.stdout.write(diff)
            sys.stdout.flush()
            last_transcript = new_total

# ------------------ GPT Cleanup ------------------

def format_transcript_with_gpt(text: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who formats speech transcripts into clear, punctuated, well-paragraphed text."},
                {"role": "user", "content": f"Please clean up and format this transcript:\n{text}"}
            ],
            temperature=0.3,
        )
        formatted = response['choices'][0]['message']['content'].strip()
        return wrap_formatted_text(formatted, width=LINE_WRAP_WIDTH)
    except Exception as e:
        print("GPT error:", e)
        return text

def background_gpt_cleanup(interval=60):
    def run():
        while True:
            raw = " ".join(transcript_buffer)
            if raw:
                formatted = format_transcript_with_gpt(raw)
                timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                with open("transcript_formatted.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n\n{timestamp}\n{formatted}")
            time.sleep(interval)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

# ------------------ Main Execution ------------------

try:
    print("Listening... (Ctrl+C to stop)\n")
    background_gpt_cleanup(interval=60)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK_SIZE, callback=callback):
        while True:
            sd.sleep(100)

except KeyboardInterrupt:
    print("\n\nStopped by user.")
    raw_transcript = " ".join(transcript_buffer)
    print("\nRaw Transcript:\n")
    print(raw_transcript)

    final_formatted = format_transcript_with_gpt(raw_transcript)
    print("\nFormatted Transcript:\n")
    print(final_formatted)

    with open("transcript_final.txt", "w", encoding="utf-8") as f:
        f.write(final_formatted)
