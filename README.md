# 🏋️ Luvit AI Coach

AI-powered personal trainer built on open-source LLaMA 3.1 8B, fine-tuned with QLoRA for the Luvit fitness app.

---

## 📁 Project Structure

```
luvit-ai/
├── data/
│   └── build_dataset.py      ← Generate training data
├── training/
│   └── finetune_qlora.py     ← Fine-tune on Google Colab
├── prompts/
│   └── coach_personas.py     ← Coach personalities & system prompts
├── backend/
│   └── main.py               ← FastAPI server (connect to Luvit app)
└── README.md
```

---

## 🗺️ 8-Week Roadmap

| Week | Task |
|------|------|
| 1–2  | Build dataset, design coach personas |
| 3–4  | Fine-tune with QLoRA on Colab |
| 5–6  | Build FastAPI backend, add user memory |
| 7–8  | Integrate with Luvit app, push notifications, demo |

---

## 🚀 Getting Started

### Step 1 — Install Dependencies

```bash
pip install transformers peft trl bitsandbytes accelerate datasets torch fastapi uvicorn sqlite-utils
```

### Step 2 — Build Training Dataset

```bash
cd data
python build_dataset.py
# Output: data/training_data.jsonl
```

Expand `RAW_CONVERSATIONS` in `build_dataset.py` to **500-1000 conversation pairs** for best results.

### Step 3 — Fine-tune on Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Runtime → Change runtime type → **A100 GPU** (Colab Pro recommended)
3. Upload `training_data.jsonl` and `finetune_qlora.py`
4. Install dependencies:
   ```
   !pip install transformers peft trl bitsandbytes accelerate datasets
   ```
5. Get LLaMA access at [huggingface.co/meta-llama](https://huggingface.co/meta-llama) (free, ~24h approval)
6. Run the fine-tuning script (~45 min)
7. Download the `luvit-coach-adapter/` folder (~80MB)

> **Alternative model** (no approval needed): Change `BASE_MODEL` to `"mistralai/Mistral-7B-Instruct-v0.3"`

### Step 4 — Run the Backend

```bash
cd backend
ADAPTER_PATH=../luvit-coach-adapter python main.py
# Server running at http://localhost:8000
```

If adapter isn't ready yet, the server runs in **mock mode** — great for testing the app!

### Step 5 — Test the API

```bash
# List coaches
curl http://localhost:8000/coaches

# Chat with Lucia
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "coach": "lucia", "message": "Hi! Can you make me a weekly programme?"}'
```

---

## 🤖 Coach Profiles

| Coach | Style | Best For |
|-------|-------|----------|
| **Lucia** 💪 | Tough love, high energy | Motivation, fat loss |
| **Arne** 🧘 | Calm, science-based | Muscle building, beginners |
| **Maya** 🌟 | Cheerful, encouraging | Complete beginners |

Add more coaches by adding entries to `prompts/coach_personas.py`.

---

## 📱 Luvit App Integration

In the Luvit React Native app, replace the existing chat logic with:

```javascript
// Example API call from the app
const sendMessage = async (userId, coach, message, history) => {
  const res = await fetch("http://YOUR_SERVER/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      coach: coach,
      message: message,
      conversation_history: history
    })
  });
  const data = await res.json();
  return data.response;
};
```

---

## 🚢 Deployment (Free)

Deploy the FastAPI backend to [Railway](https://railway.app) or [Render](https://render.com):

```bash
# Railway (simplest)
npm install -g @railway/cli
railway login
railway init
railway up
```

Note: Free tiers won't run the full LLaMA model — use a small model or the Anthropic API as a fallback for production.

---

## 🔧 Tech Stack

- **Base Model**: LLaMA 3.1 8B Instruct (Meta, Apache 2.0)
- **Fine-tuning**: QLoRA via HuggingFace PEFT + TRL
- **Training**: Google Colab A100 (~free)
- **Backend**: FastAPI + Python
- **Database**: SQLite (dev) / PostgreSQL (prod)
- **Deployment**: Railway / Render
- **Notifications**: Firebase Cloud Messaging

---

## 📬 Contact

Project by students at [your institution] for Luvit.  
Supervisor: Laura Saez Escudero  
Company contact: Annette Fält-Vannum (annfalt@outlook.com)
