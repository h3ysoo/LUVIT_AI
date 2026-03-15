"""
Luvit AI Coach — FastAPI Backend v3
=====================================
Claude API entegrasyonu ile gerçek AI koç!

Install: pip install fastapi uvicorn anthropic python-dotenv
Run:     python main.py
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("❌ ANTHROPIC_API_KEY not found in .env file!")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

app = FastAPI(title="Luvit AI Coach API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Database ──────────────────────────────────────────────────────────────────
def get_db():
    import pathlib
    db_path = pathlib.Path(__file__).parent / "luvit_users.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            TEXT PRIMARY KEY,
            name          TEXT,
            goals         TEXT,
            fitness_level TEXT,
            equipment     TEXT,
            weekly_days   INTEGER,
            injuries      TEXT,
            active_coach  TEXT DEFAULT 'lucia',
            created_at    TEXT,
            updated_at    TEXT
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    TEXT NOT NULL,
            coach      TEXT NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS coach_switches (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT NOT NULL,
            from_coach  TEXT,
            to_coach    TEXT NOT NULL,
            reason      TEXT,
            switched_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

init_db()

# ── Coach Personas ────────────────────────────────────────────────────────────
PERSONAS = {
    "lucia": {
        "name": "Lucia", "emoji": "💪",
        "tagline": "Tough love, real results",
        "color": "#E63946",
        "system_prompt": """You are Lucia, a high-energy personal trainer on the Luvit fitness app.
Your style: tough love, direct, motivating, no excuses — but you genuinely care about users.
You push people beyond their comfort zone with phrases like "Let's GO!", "No excuses!", "You've got this!"

Your capabilities:
- Create personalized weekly workout programmes
- Give nutrition and meal guidance  
- Track and celebrate user progress
- Recommend Luvit training videos
- Send motivational check-ins

Rules:
- Keep responses CONCISE — this is a mobile app (max 150 words)
- Always ask about injuries before prescribing exercises
- Suggest Luvit videos when relevant: "Check out [name] in the Luvit library!"
- If user reports pain or injury, recommend rest and consulting a professional
- Use emojis sparingly but effectively 💪🔥""",
    },
    "arne": {
        "name": "Arne", "emoji": "🧘",
        "tagline": "Calm, sustainable, science-based",
        "color": "#2D6A4F",
        "system_prompt": """You are Arne, a calm and knowledgeable personal trainer on the Luvit fitness app.
Your style: patient, educational, science-based, holistic — mind and body.
You explain the 'why' behind exercises and help users build long-term habits.

Your capabilities:
- Create balanced workout programmes with proper rest
- Provide evidence-based nutrition advice
- Guide recovery and mobility routines
- Track gradual progress over time
- Recommend Luvit videos with clear explanations

Rules:
- Keep responses CONCISE — this is a mobile app (max 150 words)
- Personalize: ask about goals, level, time, equipment
- Explain benefits of exercises simply
- Never diagnose injuries — recommend professional consultation
- Suggest Luvit videos: "I recommend [name] from the Luvit library."
- Use calm, measured language""",
    },
    "maya": {
        "name": "Maya", "emoji": "🌟",
        "tagline": "Cheerful, beginner-friendly",
        "color": "#F4A261",
        "system_prompt": """You are Maya, an upbeat personal trainer on the Luvit fitness app.
Your style: warm, patient, celebratory — every small win matters!
You specialize in helping beginners feel confident and excited about fitness.

Your capabilities:
- Design beginner-friendly workout programmes
- Give simple, easy-to-follow nutrition tips
- Celebrate progress milestones enthusiastically
- Recommend beginner Luvit videos with clear instructions
- Send friendly motivational reminders

Rules:
- Keep responses CONCISE — this is a mobile app (max 150 words)
- ALWAYS check experience level and any fears or limitations first
- Break exercises into simple steps
- Celebrate every achievement no matter how small 🎉
- Avoid fitness jargon — use plain language
- Use friendly emojis naturally 🌟✨💕
- Suggest Luvit videos: "Try [name] in the Luvit library — perfect for beginners!" """,
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_user(conn, user_id: str):
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row else None

def get_conversation_history(conn, user_id: str, coach: str, limit: int = 10) -> list:
    rows = conn.execute(
        "SELECT role, content FROM conversations WHERE user_id=? AND coach=? ORDER BY created_at DESC LIMIT ?",
        (user_id, coach, limit)
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

def save_message(conn, user_id: str, coach: str, role: str, content: str):
    conn.execute(
        "INSERT INTO conversations (user_id, coach, role, content, created_at) VALUES (?,?,?,?,?)",
        (user_id, coach, role, content, datetime.utcnow().isoformat())
    )

def build_system_prompt(coach: str, user_profile: dict = None) -> str:
    system = PERSONAS[coach]["system_prompt"]
    if user_profile and user_profile.get("name"):
        system += f"""

User profile:
- Name: {user_profile.get('name')}
- Goal: {user_profile.get('goals')}
- Fitness level: {user_profile.get('fitness_level')}
- Equipment: {user_profile.get('equipment')}
- Available days/week: {user_profile.get('weekly_days')}
- Injuries/limitations: {user_profile.get('injuries') or 'none'}

Use this profile to personalize every response. Address them by name occasionally."""
    return system

def call_claude(system: str, messages: list) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        system=system,
        messages=messages,
    )
    return response.content[0].text

# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    user_id: str
    message: str
    coach: Optional[str] = None
    include_history: bool = True

class OnboardingData(BaseModel):
    user_id: str
    name: str
    goals: str
    fitness_level: str
    equipment: str
    weekly_days: int
    injuries: Optional[str] = None
    preferred_coach: Optional[str] = "lucia"

class CoachSwitchRequest(BaseModel):
    user_id: str
    new_coach: str
    reason: Optional[str] = None

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Luvit AI Coach API 🏋️",
        "version": "3.0.0",
        "powered_by": "Claude AI (Anthropic)",
        "coaches": list(PERSONAS.keys()),
    }

@app.get("/coaches")
def list_coaches():
    return [
        {"id": k, "name": v["name"], "emoji": v["emoji"], "tagline": v["tagline"], "color": v["color"]}
        for k, v in PERSONAS.items()
    ]

@app.post("/chat")
def chat(req: ChatRequest):
    conn = get_db()
    try:
        user = get_user(conn, req.user_id)
        coach = req.coach or (user["active_coach"] if user else "lucia")

        if coach not in PERSONAS:
            raise HTTPException(400, f"Unknown coach: {coach}")

        # Build history
        history = []
        if req.include_history:
            history = get_conversation_history(conn, req.user_id, coach, limit=10)
        history.append({"role": "user", "content": req.message})

        # Call Claude with coach persona
        system = build_system_prompt(coach, user)
        response = call_claude(system, history)

        # Save to DB
        save_message(conn, req.user_id, coach, "user", req.message)
        save_message(conn, req.user_id, coach, "assistant", response)
        conn.commit()

        return {
            "coach": coach,
            "coach_name": PERSONAS[coach]["name"],
            "response": response,
            "timestamp": datetime.utcnow().isoformat(),
        }
    finally:
        conn.close()

@app.post("/onboarding")
def save_onboarding(data: OnboardingData):
    if data.preferred_coach not in PERSONAS:
        raise HTTPException(400, f"Unknown coach: {data.preferred_coach}")

    conn = get_db()
    try:
        now = datetime.utcnow().isoformat()
        conn.execute("""
            INSERT OR REPLACE INTO users
            (id, name, goals, fitness_level, equipment, weekly_days, injuries, active_coach, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            data.user_id, data.name, data.goals, data.fitness_level,
            data.equipment, data.weekly_days, data.injuries,
            data.preferred_coach, now, now
        ))
        conn.commit()

        user_profile = {
            "name": data.name, "goals": data.goals,
            "fitness_level": data.fitness_level, "equipment": data.equipment,
            "weekly_days": data.weekly_days, "injuries": data.injuries,
        }
        system = build_system_prompt(data.preferred_coach, user_profile)
        intro_msg = (
            f"Hi! I'm {data.name}. My goal is {data.goals}. "
            f"I'm {data.fitness_level} level, I have {data.equipment}, "
            f"and I can train {data.weekly_days} days per week."
            + (f" I have this limitation: {data.injuries}." if data.injuries else "")
        )
        response = call_claude(system, [{"role": "user", "content": intro_msg}])

        save_message(conn, data.user_id, data.preferred_coach, "user", intro_msg)
        save_message(conn, data.user_id, data.preferred_coach, "assistant", response)
        conn.commit()

        return {
            "message": "Profile saved ✅",
            "active_coach": data.preferred_coach,
            "coach_name": PERSONAS[data.preferred_coach]["name"],
            "coach_intro": response,
        }
    finally:
        conn.close()

@app.post("/switch-coach")
def switch_coach(req: CoachSwitchRequest):
    if req.new_coach not in PERSONAS:
        raise HTTPException(400, f"Unknown coach: {req.new_coach}")

    conn = get_db()
    try:
        user = get_user(conn, req.user_id)
        if not user:
            raise HTTPException(404, "User not found. Complete onboarding first.")

        old_coach = user["active_coach"]
        if old_coach == req.new_coach:
            return {"message": f"Already using {PERSONAS[req.new_coach]['name']}!"}

        conn.execute(
            "UPDATE users SET active_coach=?, updated_at=? WHERE id=?",
            (req.new_coach, datetime.utcnow().isoformat(), req.user_id)
        )
        conn.execute(
            "INSERT INTO coach_switches (user_id, from_coach, to_coach, reason, switched_at) VALUES (?,?,?,?,?)",
            (req.user_id, old_coach, req.new_coach, req.reason, datetime.utcnow().isoformat())
        )
        conn.commit()

        system = build_system_prompt(req.new_coach, user)
        welcome = f"Hi! I just switched from {PERSONAS[old_coach]['name']} to you. Please introduce yourself!"
        response = call_claude(system, [{"role": "user", "content": welcome}])

        save_message(conn, req.user_id, req.new_coach, "user", welcome)
        save_message(conn, req.user_id, req.new_coach, "assistant", response)
        conn.commit()

        return {
            "message": f"Switched to {PERSONAS[req.new_coach]['name']} ✅",
            "previous_coach": old_coach,
            "active_coach": req.new_coach,
            "welcome_message": response,
        }
    finally:
        conn.close()

@app.get("/history/{user_id}")
def get_history(user_id: str, coach: Optional[str] = None, limit: int = 50):
    conn = get_db()
    try:
        if coach:
            rows = conn.execute(
                "SELECT coach, role, content, created_at FROM conversations WHERE user_id=? AND coach=? ORDER BY created_at DESC LIMIT ?",
                (user_id, coach, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT coach, role, content, created_at FROM conversations WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit)
            ).fetchall()
        return {"user_id": user_id, "count": len(rows), "messages": [dict(r) for r in reversed(rows)]}
    finally:
        conn.close()

@app.get("/profile/{user_id}")
def get_profile(user_id: str):
    conn = get_db()
    try:
        user = get_user(conn, user_id)
        if not user:
            raise HTTPException(404, "User not found")
        counts = {}
        for cid in PERSONAS:
            row = conn.execute(
                "SELECT COUNT(*) as c FROM conversations WHERE user_id=? AND coach=? AND role='user'",
                (user_id, cid)
            ).fetchone()
            counts[cid] = row["c"]
        return {**user, "active_coach_info": PERSONAS[user["active_coach"]], "message_counts": counts}
    finally:
        conn.close()

@app.delete("/history/{user_id}")
def clear_history(user_id: str, coach: Optional[str] = None):
    conn = get_db()
    try:
        if coach:
            conn.execute("DELETE FROM conversations WHERE user_id=? AND coach=?", (user_id, coach))
        else:
            conn.execute("DELETE FROM conversations WHERE user_id=?", (user_id,))
        conn.commit()
        return {"message": "History cleared ✅"}
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    print("🏋️  Luvit AI Coach API v3.0 — Powered by Claude")
    print("📖  Docs: http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)