"""
Luvit AI Coach — FastAPI Backend
=================================
Install: pip install fastapi uvicorn transformers peft torch sqlite-utils python-dotenv
Run:     uvicorn main:app --reload --port 8000
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Models path (update after fine-tuning) ────────────────────────────────────
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./luvit-coach-adapter")
BASE_MODEL   = os.getenv("BASE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
USE_LOCAL_MODEL = Path(ADAPTER_PATH).exists()

# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Luvit AI Coach API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Database ──────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect("luvit_users.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT,
            goals TEXT,
            fitness_level TEXT,
            equipment TEXT,
            weekly_days INTEGER,
            injuries TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            coach TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT
        );
    """)
    conn.commit()
    conn.close()

init_db()

# ── Load model (lazy, only if available) ──────────────────────────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if not USE_LOCAL_MODEL:
        print("⚠️  No fine-tuned adapter found. Using mock responses for development.")
        return None

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel

    print("Loading fine-tuned model...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    _pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("✅ Model loaded!")
    return _pipeline


# ── Coach personas ────────────────────────────────────────────────────────────
PERSONAS = {
    "lucia": {
        "name": "Lucia", "emoji": "💪", "tagline": "Tough love, real results",
        "system_prompt": "You are Lucia, a high-energy tough-love personal trainer on the Luvit fitness app. You push users hard but care deeply about their success. Keep responses concise for mobile. Always ask about injuries before prescribing exercises.",
    },
    "arne": {
        "name": "Arne", "emoji": "🧘", "tagline": "Calm, sustainable, science-based",
        "system_prompt": "You are Arne, a calm science-based personal trainer on the Luvit fitness app. You explain the 'why' and focus on sustainable habits. Keep responses concise for mobile.",
    },
    "maya": {
        "name": "Maya", "emoji": "🌟", "tagline": "Cheerful, beginner-friendly",
        "system_prompt": "You are Maya, an encouraging beginner-friendly trainer on the Luvit fitness app. You celebrate every small win and make fitness fun. Keep responses concise and use emojis.",
    },
}

# Mock responses for development (before fine-tuned model is ready)
MOCK_RESPONSES = {
    "lucia": [
        "Let's GO! 💪 I'm Lucia and I don't do excuses. Tell me — what's your main goal? Lose fat, build muscle, or boost endurance?",
        "Love that energy! How many days a week can you train? And do you have gym access or working from home?",
        "Perfect. Before I build your plan — any injuries or limitations I need to know about?",
    ],
    "arne": [
        "Great to meet you! To build the right programme, I need to understand your goals. Are you focusing on strength, endurance, or a combination?",
        "Good choice. How many days per week and how long per session can you commit to?",
    ],
    "maya": [
        "Hi there! I'm Maya and I'm SO excited to be your coach! 🌟 First things first — tell me a little about yourself and what you'd love to achieve!",
    ],
}

_mock_counters = {}

def generate_response(coach: str, messages: list[dict]) -> str:
    pipe = get_pipeline()

    if pipe is None:
        # Development mock
        coach_responses = MOCK_RESPONSES.get(coach, MOCK_RESPONSES["lucia"])
        idx = _mock_counters.get(coach, 0) % len(coach_responses)
        _mock_counters[coach] = idx + 1
        return coach_responses[idx]

    persona = PERSONAS[coach]
    formatted = [{"role": "system", "content": persona["system_prompt"]}] + messages
    result = pipe(formatted, max_new_tokens=300, temperature=0.7, do_sample=True)
    return result[0]["generated_text"][-1]["content"]


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    user_id: str
    coach: str = "lucia"
    message: str
    conversation_history: list[dict] = []

class OnboardingData(BaseModel):
    user_id: str
    name: str
    goals: str
    fitness_level: str  # beginner / intermediate / advanced
    equipment: str
    weekly_days: int
    injuries: Optional[str] = None

class UserProfile(BaseModel):
    user_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Luvit AI Coach API 🏋️", "version": "1.0.0", "model_loaded": USE_LOCAL_MODEL}


@app.get("/coaches")
def list_coaches():
    return [
        {"id": k, "name": v["name"], "emoji": v["emoji"], "tagline": v["tagline"]}
        for k, v in PERSONAS.items()
    ]


@app.post("/chat")
def chat(req: ChatRequest):
    if req.coach not in PERSONAS:
        raise HTTPException(status_code=400, detail=f"Unknown coach: {req.coach}")

    # Build conversation history
    history = req.conversation_history + [{"role": "user", "content": req.message}]

    # Generate response
    response = generate_response(req.coach, history)

    # Save to DB
    conn = get_db()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO conversations (user_id, coach, role, content, created_at) VALUES (?,?,?,?,?)",
        (req.user_id, req.coach, "user", req.message, now)
    )
    conn.execute(
        "INSERT INTO conversations (user_id, coach, role, content, created_at) VALUES (?,?,?,?,?)",
        (req.user_id, req.coach, "assistant", response, now)
    )
    conn.commit()
    conn.close()

    return {
        "coach": req.coach,
        "response": response,
        "timestamp": now,
    }


@app.post("/onboarding")
def save_onboarding(data: OnboardingData):
    conn = get_db()
    conn.execute("""
        INSERT OR REPLACE INTO users 
        (id, name, goals, fitness_level, equipment, weekly_days, injuries, created_at)
        VALUES (?,?,?,?,?,?,?,?)
    """, (
        data.user_id, data.name, data.goals, data.fitness_level,
        data.equipment, data.weekly_days, data.injuries,
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

    # Generate first coach message using profile
    intro_message = f"Hi! I'm {data.name}, my goal is {data.goals}, I can train {data.weekly_days} days/week, I have: {data.equipment}"
    response = generate_response("lucia", [{"role": "user", "content": intro_message}])

    return {"message": "Profile saved", "coach_intro": response}


@app.get("/history/{user_id}")
def get_history(user_id: str, coach: str = "lucia", limit: int = 50):
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content, created_at FROM conversations WHERE user_id=? AND coach=? ORDER BY created_at DESC LIMIT ?",
        (user_id, coach, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


@app.get("/profile/{user_id}")
def get_profile(user_id: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return dict(row)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
