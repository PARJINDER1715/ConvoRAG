"""
ConvoRAG Backend — Flask API (Groq Version)

Run:
python app.py

Open:
http://localhost:5000
"""

import os
import json
import re
import random

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from groq import Groq

# ──────────────────────────────────────────────────────────────────────────────
# Load .env
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Flask App
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static")
CORS(app)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.environ.get(
    "CSV_PATH",
    os.path.join(BASE_DIR, "conversations.csv")
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

MODEL = "llama-3.3-70b-versatile"

# ──────────────────────────────────────────────────────────────────────────────
# Validate API Key
# ──────────────────────────────────────────────────────────────────────────────

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found.\n"
        "Create a .env file and add:\n"
        "GROQ_API_KEY=your_key_here"
    )

client = Groq(api_key=GROQ_API_KEY)

# ──────────────────────────────────────────────────────────────────────────────
# Load CSV
# ──────────────────────────────────────────────────────────────────────────────

print(f"Loading CSV: {CSV_PATH}")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"CSV file not found at:\n{CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH, header=None, names=["conversation"])

# ──────────────────────────────────────────────────────────────────────────────
# Parse Conversations
# ──────────────────────────────────────────────────────────────────────────────

ALL_MESSAGES = []

for day_idx, row in df.iterrows():

    conv = str(row["conversation"])

    for line in conv.strip().split("\n"):

        line = line.strip()

        if line.startswith("User 1:"):
            ALL_MESSAGES.append({
                "idx": len(ALL_MESSAGES),
                "day": int(day_idx),
                "spk": "User 1",
                "text": line[8:].strip()
            })

        elif line.startswith("User 2:"):
            ALL_MESSAGES.append({
                "idx": len(ALL_MESSAGES),
                "day": int(day_idx),
                "spk": "User 2",
                "text": line[8:].strip()
            })

print(f"Parsed {len(ALL_MESSAGES)} messages across {len(df)} days")

# ──────────────────────────────────────────────────────────────────────────────
# Build Topic Segments
# ──────────────────────────────────────────────────────────────────────────────

TOPIC_SEGMENTS = []

for i in range(0, min(len(ALL_MESSAGES), 2000), 25):

    chunk = ALL_MESSAGES[i:i + 25]

    if not chunk:
        continue

    text = "\n".join(f"{m['spk']}: {m['text']}" for m in chunk)

    TOPIC_SEGMENTS.append({
        "id": i // 25,
        "start": chunk[0]["idx"],
        "end": chunk[-1]["idx"],
        "text": text[:1200]
    })

# ──────────────────────────────────────────────────────────────────────────────
# Build 100 Message Chunks
# ──────────────────────────────────────────────────────────────────────────────

CHUNKS_100 = []

for i in range(0, min(len(ALL_MESSAGES), 2000), 100):

    chunk = ALL_MESSAGES[i:i + 100]

    if not chunk:
        continue

    text = "\n".join(f"{m['spk']}: {m['text']}" for m in chunk)

    CHUNKS_100.append({
        "id": i // 100,
        "start": chunk[0]["idx"],
        "end": chunk[-1]["idx"],
        "text": text[:2500]
    })

# ──────────────────────────────────────────────────────────────────────────────
# Persona Sample
# ──────────────────────────────────────────────────────────────────────────────

random.seed(42)

u1_all = [
    m for m in ALL_MESSAGES
    if m["spk"] == "User 1" and len(m["text"]) > 15
]

U1_PERSONA = sorted(
    random.sample(u1_all, min(300, len(u1_all))),
    key=lambda x: x["idx"]
)

print(
    f"Ready: {len(TOPIC_SEGMENTS)} topic segments, "
    f"{len(CHUNKS_100)} 100-msg chunks"
)

# ──────────────────────────────────────────────────────────────────────────────
# Groq LLM Helper
# ──────────────────────────────────────────────────────────────────────────────

def llm(messages, system="", max_tokens=800):

    formatted = []

    if system:
        formatted.append({"role": "system", "content": system})

    formatted.extend(messages)

    response = client.chat.completions.create(
        model=MODEL,
        messages=formatted,
        temperature=0.7,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content

# ──────────────────────────────────────────────────────────────────────────────
# Similarity Search
# ──────────────────────────────────────────────────────────────────────────────

def tokenize(text):
    return set(re.sub(r"[^a-z0-9\s]", "", text.lower()).split())

def sim(a, b):
    ta, tb = tokenize(a), tokenize(b)
    inter = len(ta & tb)
    return inter / (len(ta | tb) ** 0.5 + 1e-9)

def top_k(query, items, k=4, key="text"):
    return sorted(items, key=lambda x: sim(query, x.get(key, "")), reverse=True)[:k]

# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/info")
def info():
    return jsonify({
        "total_messages": len(ALL_MESSAGES),
        "total_days": len(df),
        "topic_segments": len(TOPIC_SEGMENTS),
        "chunks_100": len(CHUNKS_100),
        "persona_messages": len(U1_PERSONA),
    })


@app.route("/api/segments")
def segments():
    return jsonify(TOPIC_SEGMENTS)


@app.route("/api/chunks100")
def chunks100():
    return jsonify(CHUNKS_100)


@app.route("/api/persona_texts")
def persona_texts():
    return jsonify([m["text"] for m in U1_PERSONA])


# ── NEW: Process a single topic segment ──────────────────────────────────────

@app.route("/api/process_segment", methods=["POST"])
def process_segment():
    data = request.json
    seg = data["segment"]
    prev_topic = data.get("prev_topic", "none")

    try:
        result = llm(
            [{
                "role": "user",
                "content": (
                    f'Previous topic: "{prev_topic}"\n\n'
                    f'Conversation segment (msgs {seg["start"]}-{seg["end"]}):\n'
                    f'{seg["text"]}\n\n'
                    f'Return ONLY valid JSON, no markdown, no explanation:\n'
                    f'{{"topic":"short phrase","changed":true,"summary":"2-3 sentences"}}'
                )
            }],
            system="You detect topic changes in conversations. Return strict JSON only. No markdown fences.",
            max_tokens=350
        )

        # Strip markdown fences if model adds them anyway
        clean = result.strip()
        clean = re.sub(r"^```(?:json)?", "", clean).strip()
        clean = re.sub(r"```$", "", clean).strip()

        return jsonify(json.loads(clean))

    except Exception as e:
        # Fallback so the frontend doesn't break
        return jsonify({
            "topic": "General conversation",
            "changed": False,
            "summary": f"Could not parse segment. Error: {str(e)[:100]}"
        })


# ── NEW: Summarize a 100-message chunk ───────────────────────────────────────

@app.route("/api/summarize_chunk", methods=["POST"])
def summarize_chunk():
    data = request.json
    chunk = data["chunk"]

    try:
        summary = llm(
            [{
                "role": "user",
                "content": (
                    f'Summarize in 3-4 sentences what happens in messages '
                    f'{chunk["start"]}–{chunk["end"]}:\n\n'
                    f'{chunk["text"][:2200]}'
                )
            }],
            system="Summarize conversations concisely and factually. Plain text only.",
            max_tokens=300
        )
        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── NEW: Extract persona from User 1 messages ─────────────────────────────────

@app.route("/api/extract_persona", methods=["POST"])
def extract_persona():
    texts = "\n".join([m["text"] for m in U1_PERSONA[:200]])

    try:
        result = llm(
            [{
                "role": "user",
                "content": (
                    f"These are User 1's messages across many conversations:\n\n"
                    f"{texts}\n\n"
                    f"Return ONLY valid JSON, no markdown:\n"
                    f'{{"habits":["..."],"personal_facts":["..."],'
                    f'"personality_traits":["..."],"interests":["..."],'
                    f'"communication_style":{{"tone":"...","message_length":"...",'
                    f'"emoji_usage":"...","patterns":["..."]}},'
                    f'"life_situation":"...","summary":"2-3 sentence portrait"}}'
                )
            }],
            system="Extract user persona from messages. Return strict JSON only. No markdown fences.",
            max_tokens=1000
        )

        clean = result.strip()
        clean = re.sub(r"^```(?:json)?", "", clean).strip()
        clean = re.sub(r"```$", "", clean).strip()

        return jsonify(json.loads(clean))

    except Exception as e:
        return jsonify({"_error": str(e)}), 500


# ── NEW: RAG Query ────────────────────────────────────────────────────────────

@app.route("/api/query", methods=["POST"])
def query():
    data = request.json
    q = data.get("query", "")
    topic_cps = data.get("topic_checkpoints", [])
    chunk_sums = data.get("chunk_summaries", [])

    # Retrieve relevant context using keyword similarity
    t_hits = top_k(q, topic_cps, 3, key="summary") if topic_cps else []
    c_hits = top_k(q, chunk_sums, 3, key="summary") if chunk_sums else []
    s_hits = top_k(q, TOPIC_SEGMENTS, 3)

    ctx = "\n\n".join([
        "TOPIC SUMMARIES:\n" + "\n".join(
            f"• [msgs {t['start']}-{t['end']}] {t.get('topic','')}: {t.get('summary','')}"
            for t in t_hits
        ),
        "100-MSG SUMMARIES:\n" + "\n".join(
            f"• [msgs {c['start']}-{c['end']}] {c.get('summary','')}"
            for c in c_hits
        ),
        "RAW EXCERPTS:\n" + "\n---\n".join(
            s["text"][:500] for s in s_hits
        ),
    ])

    try:
        answer = llm(
            [{"role": "user", "content": f"Question: {q}\n\nContext:\n{ctx}"}],
            system="Answer questions about conversation history using the provided context. Be specific and cite message ranges when relevant.",
            max_tokens=600
        )
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])
    persona = data.get("persona", {})
    topic_cps = data.get("topic_checkpoints", [])
    chunk_sums = data.get("chunk_summaries", [])

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    user_msg = messages[-1].get("content", "")

    # Live RAG retrieval
    s_hits = top_k(user_msg, TOPIC_SEGMENTS, 2)

    p_ctx = f"USER PERSONA:\n{json.dumps(persona, indent=2)}" if persona else "Persona not extracted yet."
    t_ctx = ("TOPICS:\n" + "\n".join(
        f"• {t.get('topic','')} (msgs {t['start']}-{t['end']}): {str(t.get('summary',''))[:120]}"
        for t in topic_cps[:6]
    )) if topic_cps else ""
    c_ctx = ("MESSAGE BLOCKS:\n" + "\n".join(
        f"• Msgs {c['start']}-{c['end']}: {str(c.get('summary',''))[:120]}"
        for c in chunk_sums[:4]
    )) if chunk_sums else ""
    raw_ctx = "RAW EXCERPTS:\n" + "\n---\n".join(s["text"][:400] for s in s_hits)

    system = f"""You are an intelligent assistant who knows this user deeply from their conversation history.

{p_ctx}

{t_ctx}

{c_ctx}

{raw_ctx}

Answer questions about who they are, their habits, personality, and communication style. Be warm, specific, and insightful."""

    try:
        reply = llm(messages, system, max_tokens=700)
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
