# Pixel‑Artist + Critic — Interactive RetroDiffusion Director

An **LLM‑powered art director** that pairs an “Artist” bot with a “Critic” bot  
to iterate toward the perfect pixel‑art scene via RetroDiffusion.  
A lightweight Gradio UI lets you watch each revision, pause the workflow, and  
add your own feedback on the fly.

<img width="1486" height="1087" alt="image" src="https://github.com/user-attachments/assets/b6b2452c-d052-42b3-8629-9db4df892119" />


---

## 1 · Add your API keys

Edit the file called **`api_keys.json`** in the project root:

```json
{
  "openai": "sk‑your_OpenAI_key_here",
  "retro":  "rdpk‑your_RetroDiffusion_key_here"
}
```

* **OpenAI key** → <https://platform.openai.com/account/api-keys>  
* **RetroDiffusion key** → <https://www.retrodiffusion.ai/app/devtools>

---

## 2 · Python setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`

```
requests
pillow
gradio>=4
openai
```

> **Python 3.9 or newer** recommended.

---

## 3 · Run the playground

```bash
python artist_playground.py
```
*To change the credit cost limit per-session, edit the `CREDIT_LIMIT` constant in the script. Defailt is 30.*

Your browser opens with the chat‑style playground.

### Controls

| Button | What it does |
|--------|--------------|
| **RUN / RESUME** | Start or un‑pause the current session |
| **PAUSE** | Halt after the current frame so you can add notes |
| **Critique ▶** | While paused, send your feedback (merges into the goal and auto‑resumes) |
| **STOP** | Cancel the session entirely |

When the Critic approves, **`outputs/final.png`** (3 × nearest‑neighbour upscale for viewing ease) is saved automatically.

---

## 4 · Folder layout

```
.
├─ outputs/
│  ├─ step_000.png   # each intermediate render (up‑scaled)
│  ├─ step_001.png
│  └─ final.png      # critic‑approved frame
├─ api_keys.json     # ← YOUR KEYS LIVE HERE
├─ artist_playground.py
├─ requirements.txt
```

---

## 5 · Highlights

* **Two‑bot loop** – the Artist writes full prompts; the Critic nit‑picks.  
* **Live pause / critique / resume** – inject human art‑direction at any step.  
* **Contact‑sheet fallback** – if no approval happens, a sheet of all attempts is built and the Critic picks the best.

Happy pixelling!
