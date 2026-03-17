"""
Luvit AI Coach — Endpoint Unit Tests (mock Claude API)
Run: python test_endpoints.py
"""
import os
import sys
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Setup: fake API key + mock anthropic before importing main ─────────────────
os.environ["ANTHROPIC_API_KEY"] = "sk-test-mock-key"

mock_anthropic_module = MagicMock()
mock_client = MagicMock()
mock_anthropic_module.Anthropic.return_value = mock_client
sys.modules["anthropic"] = mock_anthropic_module

from fastapi.testclient import TestClient

# Patch get_db to use a temp SQLite DB for each test run
_TEST_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_TEST_DB_PATH = Path(_TEST_DB.name)
_TEST_DB.close()

import main as app_module

_original_get_db = app_module.get_db
def _test_get_db():
    conn = sqlite3.connect(str(_TEST_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

app_module.get_db = _test_get_db
app_module.init_db()  # create tables in test DB

client = TestClient(app_module.app)

# ── Mock Claude response helper ────────────────────────────────────────────────
def set_claude_response(text: str):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=text)]
    mock_client.messages.create.return_value = mock_response


# ══════════════════════════════════════════════════════════════════════════════
class TestDBMigration(unittest.TestCase):
    def test_new_columns_exist(self):
        conn = _test_get_db()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
        conn.close()
        self.assertIn("workout_location", cols, "workout_location column missing")
        self.assertIn("last_checkin", cols, "last_checkin column missing")
        self.assertIn("workout_count", cols, "workout_count column missing")
        print("  ✅ users table has new columns")

    def test_onboarding_sessions_table(self):
        conn = _test_get_db()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(onboarding_sessions)").fetchall()]
        conn.close()
        expected = ["user_id", "step", "workout_location", "fitness_goal",
                    "weekly_days", "experience_level", "injuries"]
        for col in expected:
            self.assertIn(col, cols, f"Column {col} missing from onboarding_sessions")
        print("  ✅ onboarding_sessions table structure OK")


# ══════════════════════════════════════════════════════════════════════════════
class TestStartConversation(unittest.TestCase):
    USER = "test_user_onboard"

    def test_1_start_fresh_returns_question_1(self):
        r = client.post("/start-conversation", json={"user_id": self.USER, "answer": None})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["step"], 1)
        self.assertEqual(data["total_steps"], 5)
        self.assertFalse(data["done"])
        self.assertIn("antrenman", data["question"].lower())
        print(f"  ✅ Step 1 question: {data['question'][:60]}...")

    def test_2_answer_step1_returns_question_2(self):
        r = client.post("/start-conversation", json={"user_id": self.USER, "answer": "ev"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["step"], 2)
        self.assertFalse(data["done"])
        self.assertIn("hedef", data["question"].lower())
        print(f"  ✅ Step 2 question: {data['question'][:60]}...")

    def test_3_answer_step2_returns_question_3(self):
        r = client.post("/start-conversation", json={"user_id": self.USER, "answer": "kas yapmak"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["step"], 3)
        self.assertIn("gün", data["question"].lower())
        print(f"  ✅ Step 3 question: {data['question'][:60]}...")

    def test_4_answer_step3_returns_question_4(self):
        r = client.post("/start-conversation", json={"user_id": self.USER, "answer": "4"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["step"], 4)
        self.assertIn("deneyim", data["question"].lower())
        print(f"  ✅ Step 4 question: {data['question'][:60]}...")

    def test_5_answer_step4_returns_question_5(self):
        r = client.post("/start-conversation", json={"user_id": self.USER, "answer": "orta seviye"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["step"], 5)
        self.assertIn("sakatl", data["question"].lower())
        print(f"  ✅ Step 5 question: {data['question'][:60]}...")

    def test_6_final_answer_returns_coach_recommendation(self):
        # Mock Claude returning a valid JSON recommendation
        set_claude_response(json.dumps({
            "recommended_coach": "arne",
            "reason": "Kas yapmak istiyorsun ve orta seviyedesin. Arne'nin bilimsel yaklaşımı sana çok uygun.",
            "coach_intro": "Merhaba! Ben Arne. Birlikte güçleneceğiz! 🧘"
        }))

        r = client.post("/start-conversation", json={"user_id": self.USER, "answer": "sol dizimde hafif ağrı"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["done"])
        self.assertIn(data["recommended_coach"], ["lucia", "arne", "maya"])
        self.assertIn("reason", data)
        self.assertIn("coach_intro", data)
        self.assertIn("collected_data", data)
        self.assertEqual(data["collected_data"]["workout_location"], "ev")
        self.assertEqual(data["collected_data"]["fitness_goal"], "kas yapmak")
        self.assertEqual(data["collected_data"]["weekly_days"], "4")
        print(f"  ✅ Coach recommended: {data['recommended_coach']} ({data['coach_name']})")
        print(f"     Reason: {data['reason'][:80]}...")

    def test_7_session_cleaned_up_after_completion(self):
        conn = _test_get_db()
        row = conn.execute(
            "SELECT * FROM onboarding_sessions WHERE user_id=?", (self.USER,)
        ).fetchone()
        conn.close()
        self.assertIsNone(row, "Session should be deleted after completion")
        print("  ✅ Session cleaned up from DB after completion")

    def test_8_error_on_answer_without_session(self):
        r = client.post("/start-conversation", json={"user_id": "no_session_user", "answer": "ev"})
        self.assertEqual(r.status_code, 400)
        print("  ✅ Returns 400 when answer given without active session")

    def test_9_restart_resets_session(self):
        # Start a session, answer once, then restart
        client.post("/start-conversation", json={"user_id": "restart_user", "answer": None})
        client.post("/start-conversation", json={"user_id": "restart_user", "answer": "gym"})
        # Restart
        r = client.post("/start-conversation", json={"user_id": "restart_user", "answer": None})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["step"], 1)
        # Verify step was reset in DB
        conn = _test_get_db()
        row = dict(conn.execute(
            "SELECT * FROM onboarding_sessions WHERE user_id='restart_user'"
        ).fetchone())
        conn.close()
        self.assertEqual(row["step"], 0)
        self.assertIsNone(row["workout_location"])
        print("  ✅ Restart resets session correctly")


# ══════════════════════════════════════════════════════════════════════════════
class TestWeeklyCheckin(unittest.TestCase):
    USER = "test_user_checkin"

    def setUp(self):
        # Create a user with onboarding data
        set_claude_response("Harika! İlk haftaya hoş geldin!")
        client.post("/onboarding", json={
            "user_id": self.USER,
            "name": "Ahmet",
            "goals": "kas yapmak",
            "fitness_level": "orta",
            "equipment": "dumbbell",
            "weekly_days": 4,
            "injuries": None,
            "preferred_coach": "arne"
        })

    def test_1_checkin_returns_coach_response(self):
        set_claude_response("Geçen haftayı iyi tamamladın! 💪 Bu hafta ağırlıkları biraz artıralım.")
        r = client.post("/weekly-checkin", json={
            "user_id": self.USER,
            "message": "3 antrenman yaptım, ama perşembe yorgundum ve atladım"
        })
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("response", data)
        self.assertIn("coach", data)
        self.assertIn("last_checkin", data)
        self.assertIn("workout_count", data)
        self.assertEqual(data["coach"], "arne")
        print(f"  ✅ Check-in response received from {data['coach_name']}")
        print(f"     Response: {data['response'][:80]}...")

    def test_2_last_checkin_date_updated(self):
        set_claude_response("Süper haftaydı!")
        r = client.post("/weekly-checkin", json={
            "user_id": self.USER,
            "message": "Tüm antrenmanları tamamladım yaptım"
        })
        data = r.json()
        from datetime import datetime
        self.assertIsNotNone(data["last_checkin"])
        # Should be today's date
        today = datetime.utcnow().strftime("%Y-%m-%d")
        self.assertEqual(data["last_checkin"], today)
        print(f"  ✅ last_checkin updated to: {data['last_checkin']}")

    def test_3_workout_count_increments_when_workout_mentioned(self):
        set_claude_response("Mükemmel!")
        r = client.post("/weekly-checkin", json={
            "user_id": self.USER,
            "message": "Bu hafta 4 antrenman yaptım, harika hissettim"
        })
        data = r.json()
        self.assertGreater(data["workout_count"], 0)
        print(f"  ✅ workout_count incremented to: {data['workout_count']}")

    def test_4_checkin_saved_in_conversation_history(self):
        set_claude_response("Devam et!")
        client.post("/weekly-checkin", json={
            "user_id": self.USER,
            "message": "Biraz zor bir haftaydı"
        })
        conn = _test_get_db()
        rows = conn.execute(
            "SELECT * FROM conversations WHERE user_id=? AND coach='arne' ORDER BY created_at DESC LIMIT 2",
            (self.USER,)
        ).fetchall()
        conn.close()
        self.assertGreaterEqual(len(rows), 2)
        print(f"  ✅ Check-in messages saved to conversation history ({len(rows)} messages)")

    def test_5_returns_404_for_unknown_user(self):
        r = client.post("/weekly-checkin", json={
            "user_id": "nonexistent_user_xyz",
            "message": "Merhaba"
        })
        self.assertEqual(r.status_code, 404)
        print("  ✅ Returns 404 for unknown user")


# ══════════════════════════════════════════════════════════════════════════════
class TestExistingEndpoints(unittest.TestCase):
    """Mevcut endpoint'lerin bozulmadığını doğrula."""

    def test_root(self):
        r = client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("coaches", r.json())
        print("  ✅ GET / OK")

    def test_coaches_list(self):
        r = client.get("/coaches")
        self.assertEqual(r.status_code, 200)
        coaches = r.json()
        ids = [c["id"] for c in coaches]
        self.assertIn("lucia", ids)
        self.assertIn("arne", ids)
        self.assertIn("maya", ids)
        print(f"  ✅ GET /coaches OK — {ids}")

    def test_chat(self):
        set_claude_response("Haydi, antrenman vakti! 💪")
        r = client.post("/chat", json={
            "user_id": "chat_test_user",
            "message": "Merhaba!",
            "coach": "lucia"
        })
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["coach"], "lucia")
        self.assertIn("response", data)
        print(f"  ✅ POST /chat OK")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  LUVIT AI COACH — Endpoint Tests")
    print("="*60 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    print("[1] DB Migration Tests")
    suite.addTests(loader.loadTestsFromTestCase(TestDBMigration))

    print("\n[2] /start-conversation Tests")
    suite.addTests(loader.loadTestsFromTestCase(TestStartConversation))

    print("\n[3] /weekly-checkin Tests")
    suite.addTests(loader.loadTestsFromTestCase(TestWeeklyCheckin))

    print("\n[4] Existing Endpoints (regression)")
    suite.addTests(loader.loadTestsFromTestCase(TestExistingEndpoints))

    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    print("\n" + "="*60)
    if result.wasSuccessful():
        print(f"  ✅ ALL {result.testsRun} TESTS PASSED")
    else:
        print(f"  ❌ {len(result.failures)} failures, {len(result.errors)} errors")
        for f in result.failures + result.errors:
            print(f"\n  FAIL: {f[0]}")
            print(f"  {f[1]}")
    print("="*60 + "\n")

    # Cleanup temp DB
    _TEST_DB_PATH.unlink(missing_ok=True)
