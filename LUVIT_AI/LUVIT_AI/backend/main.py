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
from fastapi.responses import FileResponse
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
            id               TEXT PRIMARY KEY,
            name             TEXT,
            goals            TEXT,
            fitness_level    TEXT,
            equipment        TEXT,
            weekly_days      INTEGER,
            injuries         TEXT,
            active_coach     TEXT DEFAULT 'lucia',
            workout_location TEXT,
            last_checkin     TEXT,
            workout_count    INTEGER DEFAULT 0,
            created_at       TEXT,
            updated_at       TEXT
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
        CREATE TABLE IF NOT EXISTS onboarding_sessions (
            user_id          TEXT PRIMARY KEY,
            step             INTEGER DEFAULT 0,
            workout_location TEXT,
            fitness_goal     TEXT,
            weekly_days      TEXT,
            experience_level TEXT,
            injuries         TEXT,
            created_at       TEXT,
            updated_at       TEXT
        );
    """)
    # Migrate existing DB: add new columns if they don't exist
    for col, definition in [
        ("workout_location", "TEXT"),
        ("last_checkin", "TEXT"),
        ("workout_count", "INTEGER DEFAULT 0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col} {definition}")
        except Exception:
            pass  # Column already exists
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
- Workout location: {user_profile.get('workout_location') or 'home'}
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

class StartConversationRequest(BaseModel):
    user_id: str
    answer: Optional[str] = None  # None = start fresh

class WeeklyCheckinRequest(BaseModel):
    user_id: str
    message: str  # User's answer to "Geçen hafta nasıl gitti?"

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Luvit AI Coach API 🏋️",
        "version": "3.0.0",
        "powered_by": "Claude AI (Anthropic)",
        "coaches": list(PERSONAS.keys()),
    }

@app.get("/demo")
def demo():
    html_path = Path(__file__).parent / "demo.html"
    return FileResponse(str(html_path), media_type="text/html")

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

# ── Onboarding Questions ──────────────────────────────────────────────────────
ONBOARDING_QUESTIONS = [
    {
        "step": 1,
        "field": "workout_location",
        "question": "Nerede antrenman yapıyorsun? 🏠\n(ev / gym / dışarısı)",
    },
    {
        "step": 2,
        "field": "fitness_goal",
        "question": "Fitness hedefin ne? 🎯\n(kilo vermek / kas yapmak / sağlıklı kalmak / atletizm)",
    },
    {
        "step": 3,
        "field": "weekly_days",
        "question": "Haftada kaç gün antrenman yapabilirsin? 📅",
    },
    {
        "step": 4,
        "field": "experience_level",
        "question": "Deneyim seviyeni nasıl tanımlarsın? 💡\n(yeni başlayan / orta seviye / ileri seviye)",
    },
    {
        "step": 5,
        "field": "injuries",
        "question": "Herhangi bir sakatlığın ya da fiziksel sınırlılığın var mı? 🩺\n(varsa belirt, yoksa 'yok' yaz)",
    },
]

COACH_RECOMMENDATION_PROMPT = """Based on the user's onboarding answers, recommend the best coach from: lucia, arne, or maya.

User answers:
- Workout location: {workout_location}
- Fitness goal: {fitness_goal}
- Weekly training days: {weekly_days}
- Experience level: {experience_level}
- Injuries/limitations: {injuries}

Coach profiles:
- lucia: High-energy, tough love, best for weight loss, athletic goals, motivated intermediate/advanced users
- arne: Calm, science-based, best for muscle building, users who want to understand the 'why', sustainable long-term approach
- maya: Cheerful, beginner-friendly, best for complete beginners, confidence building, gentle start

Respond in Turkish with:
1. Which coach you recommend (just the id: lucia/arne/maya)
2. A warm, personalized 2-3 sentence explanation of WHY this coach fits them
3. A brief intro from that coach welcoming them

Format your response as JSON:
{{"recommended_coach": "...", "reason": "...", "coach_intro": "..."}}"""


@app.post("/start-conversation")
def start_conversation(req: StartConversationRequest):
    """
    Sequential onboarding: ask questions one by one, then recommend a coach.
    - First call (answer=None): resets session, returns question 1
    - Subsequent calls (with answer): saves answer, returns next question
    - After all 5 answers: recommends best coach
    """
    conn = get_db()
    try:
        now = datetime.utcnow().isoformat()

        # Start fresh
        if req.answer is None:
            conn.execute("""
                INSERT OR REPLACE INTO onboarding_sessions
                (user_id, step, workout_location, fitness_goal, weekly_days, experience_level, injuries, created_at, updated_at)
                VALUES (?, 0, NULL, NULL, NULL, NULL, NULL, ?, ?)
            """, (req.user_id, now, now))
            conn.commit()

            return {
                "step": 1,
                "total_steps": 5,
                "question": ONBOARDING_QUESTIONS[0]["question"],
                "done": False,
            }

        # Load current session
        row = conn.execute(
            "SELECT * FROM onboarding_sessions WHERE user_id=?", (req.user_id,)
        ).fetchone()

        if not row:
            raise HTTPException(400, "No active session. Call /start-conversation with answer=null first.")

        session = dict(row)
        current_step = session["step"]

        if current_step >= 5:
            raise HTTPException(400, "Onboarding already completed.")

        # Save the answer for the current step
        field = ONBOARDING_QUESTIONS[current_step]["field"]
        next_step = current_step + 1

        conn.execute(
            f"UPDATE onboarding_sessions SET {field}=?, step=?, updated_at=? WHERE user_id=?",
            (req.answer, next_step, now, req.user_id)
        )
        conn.commit()

        # If more questions remain, return next one
        if next_step < 5:
            return {
                "step": next_step + 1,
                "total_steps": 5,
                "question": ONBOARDING_QUESTIONS[next_step]["question"],
                "done": False,
            }

        # All 5 answers collected — ask Claude to recommend a coach
        updated = dict(conn.execute(
            "SELECT * FROM onboarding_sessions WHERE user_id=?", (req.user_id,)
        ).fetchone())

        prompt = COACH_RECOMMENDATION_PROMPT.format(
            workout_location=updated["workout_location"],
            fitness_goal=updated["fitness_goal"],
            weekly_days=updated["weekly_days"],
            experience_level=updated["experience_level"],
            injuries=updated["injuries"],
        )

        raw = call_claude(
            system="You are a helpful assistant that recommends fitness coaches. Always respond with valid JSON.",
            messages=[{"role": "user", "content": prompt}],
        )

        import json as _json
        try:
            result = _json.loads(raw)
        except Exception:
            # Fallback: extract JSON from response
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            result = _json.loads(match.group()) if match else {
                "recommended_coach": "maya",
                "reason": "Sana en uygun koçu önermek için bilgilerini değerlendirdim.",
                "coach_intro": "Merhaba! Seninle çalışmaya hazırım!",
            }

        recommended = result.get("recommended_coach", "maya")
        if recommended not in PERSONAS:
            recommended = "maya"

        # Clean up session
        conn.execute("DELETE FROM onboarding_sessions WHERE user_id=?", (req.user_id,))
        conn.commit()

        return {
            "step": 5,
            "total_steps": 5,
            "done": True,
            "recommended_coach": recommended,
            "coach_name": PERSONAS[recommended]["name"],
            "coach_emoji": PERSONAS[recommended]["emoji"],
            "reason": result.get("reason", ""),
            "coach_intro": result.get("coach_intro", ""),
            "collected_data": {
                "workout_location": updated["workout_location"],
                "fitness_goal": updated["fitness_goal"],
                "weekly_days": updated["weekly_days"],
                "experience_level": updated["experience_level"],
                "injuries": updated["injuries"],
            },
        }
    finally:
        conn.close()


@app.post("/weekly-checkin")
def weekly_checkin(req: WeeklyCheckinRequest):
    """
    Haftalık check-in: koç 'Geçen hafta nasıl gitti?' diye sorar,
    kullanıcı cevaplar, koç programı günceller ve motive eder.
    """
    conn = get_db()
    try:
        user = get_user(conn, req.user_id)
        if not user:
            raise HTTPException(404, "User not found. Complete onboarding first.")

        coach = user["active_coach"]
        now = datetime.utcnow().isoformat()

        # Build check-in context for the coach
        checkin_system = build_system_prompt(coach, user) + """

This is a WEEKLY CHECK-IN conversation. The user is reporting on their last week.
Your job:
1. Acknowledge their effort warmly (even if they missed sessions)
2. Evaluate what they reported — celebrate wins, address struggles without judgment
3. Suggest 1-2 specific adjustments for the coming week based on their feedback
4. End with an encouraging message for the week ahead
Keep it under 150 words."""

        history = get_conversation_history(conn, req.user_id, coach, limit=6)
        checkin_prompt = f"Haftalık check-in: {req.message}"
        messages = history + [{"role": "user", "content": checkin_prompt}]

        response = call_claude(checkin_system, messages)

        # Update last_checkin and increment workout_count if they mentioned doing workouts
        workout_keywords = ["yaptım", "antrenman", "egzersiz", "koştum", "çalıştım", "tamamladım"]
        mentioned_workout = any(kw in req.message.lower() for kw in workout_keywords)
        current_count = user.get("workout_count") or 0

        conn.execute(
            "UPDATE users SET last_checkin=?, workout_count=?, updated_at=? WHERE id=?",
            (now[:10], current_count + (1 if mentioned_workout else 0), now, req.user_id)
        )

        save_message(conn, req.user_id, coach, "user", checkin_prompt)
        save_message(conn, req.user_id, coach, "assistant", response)
        conn.commit()

        updated_user = get_user(conn, req.user_id)

        return {
            "coach": coach,
            "coach_name": PERSONAS[coach]["name"],
            "response": response,
            "last_checkin": updated_user.get("last_checkin"),
            "workout_count": updated_user.get("workout_count", 0),
            "timestamp": now,
        }
    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn
    print("Luvit AI Coach API v3.0 - Powered by Claude")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)