import os, sys, json, re, random, base64, pathlib, math, threading, queue, time
from typing import Dict, List, Optional, Tuple
import requests, gradio as gr
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# ──────────────────  << ALL ORIGINAL CONSTANTS / PROMPTS UNCHANGED >>  ──────────────────

CREDIT_LIMIT   = 30
MIN_APPROVED   = 1
MAX_RD_ERRORS  = 4
DEFAULT_SIZE   = 256
COLS           = 4
RD_URL         = "https://api.retrodiffusion.ai/v1/inferences"
OUT            = pathlib.Path("outputs"); OUT.mkdir(exist_ok=True)

# ---------- system prompts ----------

ARTIST_SYSTEM = """
You are PixelArtistGPT.

You always start from **the most recent critic feedback**.
• Carefully list the objects / stylistic fixes the critic demands.
• Rewrite the prompt so those items are clearly described.
• If nothing is missing, you may keep the prompt identical.
• DO NOT ignore any critic comment, no matter how small.
• ALWAYS choose a style that allows the size requested by the user!

Respond ONLY with RAW JSON:

{
  "action": "txt2img",
  "comment": "...",
  "note": "summarise in ≤20 words exactly which critic fixes you addressed, or how you are fulfilling the user request",
  "payload": {
    "prompt": "... full prompt text (no dimensions) ...",
    "prompt_style": "... rd_plus__ style ...",
    "width": (width ex. 256),
    "height": (height ex. 256),
    "num_images": 1,
    "seed": 0,               // leave 0 unless you set change_seed=true
    "change_seed": true|false
  }
}

Remember to use a style that allows the size you are asking for.

Do NOT include a seed unless you truly need a new composition.
"""

CRITIC_SYSTEM = """
You are PixelCriticGPT.
Reply ONLY with RAW JSON:

Your job: push the Artist until the image *visibly* changes and meets every
detail of the user's goal.  Be laser-specific: point out composition errors,
missing objects, palette problems, silhouette issues, lighting, and style
mismatches.  If two successive images look almost identical, insist on a NEW
SEED and clearly explain why.

If you have asked for the same fixes twice in a row and they are **still not
delivered**, or the images are not improving in their accuracy, instruct the Artist to **try a completely different approach**:
• suggest a new prompt_style or viewpoint  
• demand `"change_seed": true`  
• list *precisely* what must change.

If more work needed:
{ "status": "revise", "feedback": "harsh detailed feedback" }

Only do this if the image completely fulfills all parts of the request made by the user:
{ "status": "approve", "feedback": "praise" }

When I show you the final contact sheet, choose the image that best fulfills all criteria of the user instructions:
{ "choice": "k", "reason": "<one-sentence justification>" }
Never default to 1; justify your pick briefly.
"""

STYLE_CHEAT = """
STYLE CHEAT SHEET (rd_plus) - Align the correct style to the user intent.
    rd_plus__default        64x64-512x512, Clean pixel art, bold colors, clear outlines
    rd_plus__retro      64x64-512x512, Classic PC98 inspired palette and dithering
    rd_plus__watercolor     64x64-512x512, Pixel art blended with watercolor softness
    rd_plus__textured       64x64-512x512, Semi-realistic pixels, heavy shading and texture
    rd_plus__cartoon        64x64-512x512, Bold outlines, simple shapes, flat shading
    rd_plus__ui_element     64x64-512x512, UI boxes, buttons, interface parts
    rd_plus__item_sheet     64x64-512x512, Sheets of objects on plain backdrop
    rd_plus__character_turnaround       64x64-512x512, Character sprites from multiple angles
    rd_plus__topdown_map        64x64-512x512, 3/4 top-down video-game maps and scenes
    rd_plus__topdown_asset      64x64-512x512, Single 3/4 top-down game assets
    rd_plus__isometric      64x64-512x512, 45 deg isometric scenes, consistent outlines
    rd_plus__isometric_asset        64x64-512x512, Single isometric objects on neutral background
    rd_plus__classic        32x32-192x192, strong outlines, medium resolution
    rd_plus__low_res        16x16-128x128, tiny high-quality pixel assets
    rd_plus__mc_item        16x16-128x128, Minecraft-style items
    rd_plus__mc_texture     16x16-128x128, Minecraft-style block textures
"""

STYLE_LIMITS = {
    "rd_plus__default":            (64, 512),
    "rd_plus__retro":              (64, 512),
    "rd_plus__watercolor":         (64, 512),
    "rd_plus__textured":           (64, 512),
    "rd_plus__cartoon":            (64, 512),
    "rd_plus__ui_element":         (64, 512),
    "rd_plus__item_sheet":         (64, 512),
    "rd_plus__character_turnaround": (64, 512),
    "rd_plus__topdown_map":        (64, 512),
    "rd_plus__topdown_asset":      (64, 512),
    "rd_plus__isometric":          (64, 512),
    "rd_plus__isometric_asset":    (64, 512),
    "rd_plus__classic":            (32, 192),
    "rd_plus__low_res":            (16, 128),
    "rd_plus__mc_item":            (16, 128),
    "rd_plus__mc_texture":         (16, 128),
}

# ---------------- helpers ----------------

def need(var: str) -> str:
    val = os.getenv(var)
    if not val:
        sys.exit(f"Environment variable {var} missing")
    return val

def load_keys() -> Tuple[str, str]:
    cfg = json.load(open(pathlib.Path(__file__).with_name("api_keys.json"), "r"))
    return cfg["openai"].strip(), cfg["retro"].strip()

OPENAI_API_KEY, RD_KEY = load_keys()

client = OpenAI(api_key=OPENAI_API_KEY)

DIM_RE = re.compile(r"\b\d+\s*x\s*\d+\b", re.I)
strip_dims = lambda t: DIM_RE.sub("", t).strip()

def rd_post(payload: Dict) -> Optional[Dict]:
    try:
        res = requests.post(RD_URL, headers={"X-RD-Token": RD_KEY},
                            json=payload, timeout=120)
        res.raise_for_status(); return res.json()
    except requests.HTTPError:
        return None

OUTPUT_DIR = pathlib.Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def save_png(b64: str, filename: str) -> pathlib.Path:
    raw = base64.b64decode(b64)
    img = Image.open(BytesIO(raw)).convert("RGBA")

    # 4× nearest‑neighbour upscale
    w, h = img.size
    img = img.resize((w*3, h*3), Image.NEAREST)

    path = OUTPUT_DIR / filename
    img.save(path, format="PNG")
    return path

def openai_chat(sys_prompt: str, msgs: List[Dict]) -> str:
    all_msgs = [{"role": "system", "content": sys_prompt},
                {"role": "system", "content": STYLE_CHEAT}] + msgs
    resp = client.chat.completions.create(model="gpt-4o",
                                          messages=all_msgs, temperature=0.2)
    return resp.choices[0].message.content.strip()

# ──────────────────  BACKGROUND SESSION THREAD  ──────────────────
class Session(threading.Thread):
    """
    Background thread that runs the Artist / Critic loop and streams
    UI‑friendly events through `ui_q`.

    External control:
        session.pause_event.set()   → pause generation
        session.pause_event.clear() → resume
        session.stop_event.set()    → abort
    """
    # -- tiny prompt used to merge a single critique line into the goal ----
    GOAL_SYS  = "You rewrite pixel‑art generation goals."
    GOAL_INST = (
        "Rewrite the ORIGINAL GOAL so it *fully incorporates* the SINGLE "
        "USER CRITIQUE line.  Keep it concise, factual and self‑contained. "
        "Do not add anything else."
    )
    GOAL_REWRITE_MODEL = "gpt-4o-mini"   # fast & cheap is fine here

    def __init__(self, goal: str, ui_q: queue.Queue):
        super().__init__(daemon=True)
        self.goal_original = goal          # keep for reference
        self.goal          = goal          # mutable current goal
        self.ui_q          = ui_q

        # control flags
        self.pause_event = threading.Event()
        self.stop_event  = threading.Event()
        # legacy aliases (code outside Session may still use them)
        self.pause = self.pause_event
        self.stop  = self.stop_event

        # critique queues
        self.user_fb: queue.Queue[str] = queue.Queue()
        self.all_fb:  list[str]        = []   # persistent bullet list

    # ------------------------------------------------ UI helper
    def ui(self, sender: str, text: str = "",
           img: Optional[pathlib.Path] = None) -> None:
        """Push a UI event."""
        self.ui_q.put((sender, text, img.as_posix() if img else None))

    # ------------------------------------------------ merge helper
    def merge_goal(self, critique_line: str) -> None:
        """Call a small LLM to fold the critique into self.goal."""
        prompt = (
            f"ORIGINAL GOAL:\n{self.goal}\n\n"
            f"USER CRITIQUE:\n{critique_line}\n\n{self.GOAL_INST}"
        )
        try:
            merged = client.chat.completions.create(
                model=self.GOAL_REWRITE_MODEL,
                messages=[
                    {"role": "system", "content": self.GOAL_SYS},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.2
            ).choices[0].message.content.strip()
            if merged:
                self.goal = merged
                self.ui("System", f"Goal updated:\n{merged}")
        except Exception as e:
            self.ui("System", f"Goal‑rewrite failed ({e}); keeping old goal.")

    # ------------------------------------------------ brief helper
    def current_brief(self) -> str:
        """Return goal + accumulated bullet critique."""
        if not self.all_fb:
            return self.goal
        return self.goal + "\n\nADDITIONAL NOTES:\n• " + "\n• ".join(self.all_fb)

    # ------------------------------------------------ main loop
    def run(self) -> None:
        artist_ctx, critic_ctx = [], []
        remaining, rd_err, step = CREDIT_LIMIT, 0, -1
        seed: Optional[int] = None
        typing = False

        critic_ctx.append({"role": "user",
                           "content": json.dumps({"goal": self.goal})})

        while not self.stop_event.is_set():
            # ---------- pause gate ----------
            while self.pause_event.is_set() and not self.stop_event.is_set():
                if typing:                # clear bubble only once
                    self.ui("DONE")
                    typing = False
                time.sleep(0.25)

            # ---------- ingest new critique ----------
            while not self.user_fb.empty():
                new_fb = self.user_fb.get().strip()
                if new_fb:
                    self.all_fb.append(new_fb)
                    self.merge_goal(new_fb)   # rewrite goal here

            # ---------- Artist ----------
            self.ui("TYPING"); typing = True
            artist_ctx.append({
                "role": "user",
                "content": json.dumps({
                    "brief":             self.current_brief(),
                    "remaining_credits": remaining,
                    "seed":              seed
                })
            })

            art_raw = openai_chat(ARTIST_SYSTEM, artist_ctx)
            try:
                art_cmd = json.loads(art_raw)
            except Exception:
                self.ui("System", "Artist JSON invalid"); self.ui("DONE"); break
            artist_ctx.append({"role": "assistant", "content": art_raw})

            art_comment = art_cmd.get("comment", "")

            pay = art_cmd["payload"]
            pay["prompt"] = strip_dims(pay["prompt"])
            pay.setdefault("prompt_style", "rd_plus__default")
            pay.setdefault("width",  DEFAULT_SIZE)
            pay.setdefault("height", DEFAULT_SIZE)
            pay.setdefault("num_images", 1)

            if seed is None or pay.pop("change_seed", False):
                seed = random.randint(1, 2**31 - 1)
            pay["seed"] = seed

            lim = STYLE_LIMITS.get(pay["prompt_style"])
            if lim and not (lim[0] <= pay["width"] <= lim[1] and
                            lim[0] <= pay["height"] <= lim[1]):
                artist_ctx.append({"role": "user", "content": json.dumps({
                    "feedback": f"{pay['width']}×{pay['height']} invalid for "
                                f"{pay['prompt_style']}"
                })})
                self.ui("Artist", "(size/style mismatch – retrying)")
                self.ui("DONE"); typing = False; continue

            rd = rd_post(pay)
            if not rd:
                rd_err += 1
                if rd_err >= MAX_RD_ERRORS:
                    self.ui("System", "Too many RD errors – abort.")
                    self.ui("DONE"); break
                self.ui("DONE"); typing = False; continue
            rd_err = 0
            remaining -= rd["credit_cost"]; step += 1

            img_b64 = rd["base64_images"][0]
            img_path = save_png(img_b64, f"step_{step:03d}.png")

            self.ui("Image", f"render {step}", img_path)
            if art_comment:
                self.ui("Artist", art_comment)

            # mid‑cycle PAUSE check
            if self.pause_event.is_set():
                self.ui("DONE"); typing = False; continue

            # ---------- Critic ----------
            critic_ctx.append({"role": "user", "content": json.dumps({
                "brief": self.current_brief(), "step": step
            })})
            critic_ctx.append({"role": "user", "content": [{
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + img_b64}
            }]})

            crit_raw = openai_chat(CRITIC_SYSTEM, critic_ctx)
            try:
                crit_cmd = json.loads(crit_raw)
            except Exception:
                self.ui("System", "Critic JSON invalid"); self.ui("DONE"); break
            critic_ctx.append({"role": "assistant", "content": crit_raw})
            self.ui("Critic", crit_cmd.get("feedback", ""))

            if crit_cmd["status"] == "approve":
                save_png(img_b64, "final.png")
                self.ui("System", "Saved outputs/final.png")
                self.ui("DONE"); break

            artist_ctx.append({"role": "user", "content": json.dumps({
                "feedback": crit_cmd["feedback"]
            })})

            self.ui("DONE"); typing = False

        # ensure bubble cleared on exit
        if typing:
            self.ui("DONE")

# ----------------------------------------------------------------------
def launch_gradio() -> None:
    """
    Gradio front‑end for the Pixel‑Artist / Critic workflow.

    • RUN / RESUME  – starts a new Session or clears pause.
    • PAUSE         – toggles the Session’s pause_event.
    • STOP          – sets the Session’s stop_event.
    • CRITIQUE ▶    – pushes a critique line, *clears pause* so work resumes,
                      and immediately shows the new goal once the thread rewrites it.
    """
    ui_q: queue.Queue = queue.Queue()
    sess_holder: dict[str, Optional[Session]] = {"session": None}
    chat_history: list[tuple[str, object]] = []
    typing = False          # drives the animated “…” bubble

    # ----------------------------- helper to (re)render the entire chat as HTML
    def render_chat() -> str:
        html = (
            '<div id="chat-window" style="height:800px;overflow-y:auto;'
            'background:#2b2b2b;padding:12px;border:2px solid grey;'
            'border-radius:8px;font-family:sans-serif;scroll-behavior:smooth;">'
        )
        color_map = {
            "user":   ("#008bff", "#ffffff"),
            "artist": ("#8237ff", "#ffffff"),
            "critic": ("#2f9400", "#ffffff"),
            "system": ("#555555", "#ffffff"),
            "image":  ("#555555", "#ffffff"),
        }

        for sender, msg in chat_history:
            key = sender.lower()
            bg, fg = color_map.get(key, color_map["system"])
            align  = "flex-end" if key == "user" else "center" if key == "image" else "flex-start"

            html += f'<div style="display:flex;justify-content:{align};margin-bottom:10px;">'
            if isinstance(msg, tuple):              # (path, caption)
                img_path, caption = msg
                img_url = f"file/{img_path}"
                html += (
                    f'<div style="max-width:70%;background:{bg};border-radius:12px;'
                    f'padding:8px;box-shadow:0 1px 2px rgba(0,0,0,0.5);">'
                    f'<img src="{img_url}" style="max-width:100%;border-radius:8px;'
                    f'display:block;margin-bottom:5px;">'
                    f'<div style="font-size:14px;color:{fg};">{caption}</div>'
                    f'</div>'
                )
            else:
                label = "" if key == "user" else (
                    f'<div style="font-size:11px;margin-bottom:2px;color:#cccccc;">{sender}</div>')
                html += (
                    f'<div style="max-width:70%;background:{bg};border-radius:12px;'
                    f'padding:8px 12px;box-shadow:0 1px 2px rgba(0,0,0,0.5);'
                    f'font-size:14px;color:{fg};">{label}{msg}</div>'
                )
            html += "</div>"

        # typing indicator
        if typing:
            html += (
                '<div style="display:flex;justify-content:flex-start;margin-bottom:10px;">'
                '<div style="background:#555;border-radius:12px;padding:8px 12px;'
                'box-shadow:0 1px 2px rgba(0,0,0,0.5);color:#fff;font-size:14px;">'
                '<span class="dot" style="animation:blink 1.4s infinite;">• </span>'
                '<span class="dot" style="animation:blink 1.4s infinite .2s;">• </span>'
                '<span class="dot" style="animation:blink 1.4s infinite .4s;">• </span>'
                '</div></div>'
                '<style>@keyframes blink{0%{opacity:.2;}20%{opacity:1;}100%{opacity:.2;}}</style>'
            )

        # invisible gif forces autoscroll
        html += (
            '<img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==" '
            'style="display:none" '
            'onload="const w=document.getElementById(\'chat-window\');if(w){w.scrollTop=w.scrollHeight;}" />'
        )
        html += "</div>"
        return html

    # ----------------------------- callbacks
    def run(goal: str) -> str:
        """Start new Session OR resume if paused."""
        if sess_holder["session"] and sess_holder["session"].is_alive():
            sess_holder["session"].pause_event.clear()          # resume
        else:
            sess_holder["session"] = Session(goal, ui_q)
            sess_holder["session"].start()
            chat_history.append(("User", goal))
        return render_chat()

    def pause() -> str:
        if s := sess_holder["session"]:
            s.pause_event.set()
            chat_history.append(("System", "Paused"))
        return render_chat()

    def stop() -> str:
        if s := sess_holder["session"]:
            s.stop_event.set()
            chat_history.append(("System", "Stopped"))
        return render_chat()

    def critique(text: str) -> tuple[str, str]:
        """
        • push critique into Session.user_fb
        • resume generation immediately (clear pause)
        • echo critique in chat
        """
        if text and (s := sess_holder["session"]):
            s.user_fb.put(text)
            s.pause_event.clear()          # auto‑resume
            chat_history.append(("User", text))
        return "", render_chat()

    def poll() -> str:
        """Pull queued UI events from Session → update chat."""
        nonlocal typing
        while not ui_q.empty():
            sender, txt, img_path = ui_q.get()

            # sentinel controls for typing bubble
            if sender == "TYPING":
                typing = True
                continue
            if sender == "DONE":
                typing = False
                continue

            chat_history.append(
                (sender, (img_path, txt)) if img_path else (sender, txt)
            )
        return render_chat()

    # ----------------------------- assemble UI
    with gr.Blocks(title="Pixel Artist + Critic") as demo:
        gr.Markdown("### Pixel Artist / Critic Playground")

        goal_in = gr.Textbox(label="Your request")
        with gr.Row():
            run_btn   = gr.Button("RUN / RESUME")
            pause_btn = gr.Button("PAUSE")
            stop_btn  = gr.Button("STOP")

        chatbox = gr.HTML(render_chat(), label="Conversation", elem_id="chatbox")

        with gr.Row():
            fb_in  = gr.Textbox(label="Critique (when paused)")
            fb_btn = gr.Button("CRITIQUE ▶")

        # wiring
        run_btn.click(run,     [goal_in],        chatbox)
        pause_btn.click(pause, None,            chatbox)
        stop_btn.click(stop,   None,            chatbox)
        fb_btn.click(critique, [fb_in],         [fb_in, chatbox])
        demo.load(poll,        None,            chatbox, every=0.5)

    demo.launch(
        inbrowser=True,
        allowed_paths=[os.path.abspath(OUTPUT_DIR)]
    )

# ----------------------------------------------------------------------
if __name__ == "__main__":
    launch_gradio()