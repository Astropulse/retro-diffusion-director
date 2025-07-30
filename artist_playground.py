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
• Ensure the artwork maintains a pixel art style, not other style influences.
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

If no significant improvement has been made in a couple tries, change the seed.

If the image is ugly or does not match the style requested, tell the artist to use a different style.

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

def trim_context(ctx: List[Dict], max_items: int = 20):
    if len(ctx) > max_items:
        del ctx[:len(ctx)-max_items]

def openai_chat(sys_prompt: str, msgs: List[Dict]) -> str:
    all_msgs = [{"role": "system", "content": sys_prompt},
                {"role": "system", "content": STYLE_CHEAT}]
    
    for m in msgs:
        if isinstance(m["content"], list):
            # This allows mixing text + image
            all_msgs.append({"role": m["role"], "content": m["content"]})
        else:
            all_msgs.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
    
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=all_msgs,
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

def modify_goal_with_critique(original_goal: str, critique: str) -> str:
    prompt = f"""
    The user provided this original art goal:
    "{original_goal}"

    They now gave additional critique or adjustments:
    "{critique}"

    Rewrite the art goal as a single improved prompt that incorporates all previous requirements and the new critique.
    """
    response = openai_chat("You are a helpful assistant rewriting prompts.", [{"role":"user","content":prompt}])
    return response

def fix_json(raw: str, who: str) -> Optional[dict]:
    """
    Try to salvage an invalid‑JSON reply by asking gpt‑4o‑mini to emit
    *only* the corrected JSON.  Returns the dict or None on failure.
    """
    try:
        prompt = (
            f"The following string is a reply from {who} that SHOULD be pure "
            f'JSON but is malformed.  Fix the formatting and return ONLY the '
            f"valid JSON without any extra commentary.\n\n----\n{raw}\n----"
        )
        fixed = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0           # deterministic
        ).choices[0].message.content.strip()
        return json.loads(fixed)
    except Exception:
        return None

# ──────────────────  BACKGROUND SESSION THREAD  ──────────────────
class Session(threading.Thread):
    """
    Background thread that runs the Artist / Critic loop and streams
    UI‑friendly events through `ui_q`.

    Control flags
    -------------
    pause_event.set()   → pause generation
    pause_event.clear() → resume / continue
    stop_event.set()    → abort the thread
    """

    # ── tiny helper model used to fold 1‑line user critique into the goal ──
    GOAL_SYS  = "You rewrite pixel‑art generation goals."
    GOAL_INST = (
        "Rewrite the ORIGINAL GOAL so it fully incorporates the SINGLE "
        "USER CRITIQUE line. Keep it concise, factual and self‑contained. "
        "Do not add anything else."
    )
    GOAL_REWRITE_MODEL = "gpt-4o-mini"          # fast & cheap

    JSON_RETRY = 1                              # one auto‑retry on bad JSON

    def __init__(self, goal: str, ui_q: queue.Queue):
        super().__init__(daemon=True)
        self.last_img_b64: Optional[str] = None
        self.nudge_event = threading.Event()
        self.goal_original = goal               # for reference
        self.goal          = goal               # mutable current goal
        self.ui_q          = ui_q

        # control flags
        self.pause_event = threading.Event()
        self.stop_event  = threading.Event()
        self.pause = self.pause_event           # legacy aliases (keep)
        self.stop  = self.stop_event

        # feedback queues
        self.user_fb: queue.Queue[str] = queue.Queue()
        self.all_fb:  list[str]        = []     # persistent bullet list

        # per‑session cryptographically random generator ⇒ unique seeds
        self._rng = random.SystemRandom()

    # ────────────────────────────── helpers
    def ui(self, sender: str, txt: str = "",
           img: Optional[pathlib.Path] = None) -> None:
        """Push a UI event: (sender, text, img‑path|None)."""
        self.ui_q.put((sender, txt, img.as_posix() if img else None))
    
    def ui_multiline(self, sender: str, block: str) -> None:
        """Send one chat bubble per line of `block` (preserving blank lines)."""
        for line in block.splitlines():
            # empty string → render as a visually blank line
            self.ui(sender, line if line else " ")

    def merge_goal(self, line: str) -> None:
        """Fold one critique line into `self.goal` and show the last image."""
        prompt_txt = (
            f"ORIGINAL GOAL:\n{self.goal}\n\n"
            f"USER CRITIQUE:\n{line}\n\n{self.GOAL_INST}"
        )

        msgs = [
            {"role": "system", "content": self.GOAL_SYS},
            {"role": "user",   "content": prompt_txt},
        ]

        # If we already have an image, append it as vision context
        if self.last_img_b64:
            msgs.append({
                "role": "user",
                "content": [
                    {"type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + self.last_img_b64
                    }}
                ]
            })

        try:
            merged = client.chat.completions.create(
                model=self.GOAL_REWRITE_MODEL,
                messages=msgs,
                temperature=0.2
            ).choices[0].message.content.strip()

            if merged:
                self.goal = merged
                self.ui("System", f"Goal updated:\n{merged}")

        except Exception as e:
            self.ui("System", f"Goal‑rewrite failed ({e}); keeping old goal.")

    def current_brief(self) -> str:
        """Goal text plus any accumulated bullet‑point feedback."""
        return (self.goal if not self.all_fb else
                self.goal + "\n\nADDITIONAL NOTES:\n• " + "\n• ".join(self.all_fb))

    # ------------- contact‑sheet helpers (unchanged from original) -------------
    def create_contact_sheet(self,
                             imgs : list[pathlib.Path],
                             cols : int = 4) -> pathlib.Path:
        """Build a numbered contact sheet (placement shuffled)."""
        pil = [Image.open(p) for p in imgs]
        random.shuffle(pil)                                   # avoid bias

        w, h  = pil[0].size
        rows  = math.ceil(len(pil) / cols)
        sheet = Image.new("RGBA", (w*cols, h*rows), (30, 30, 30, 255))

        draw  = ImageDraw.Draw(sheet)
        font  = ImageFont.load_default()

        for i, im in enumerate(pil):
            x, y = (i % cols)*w, (i // cols)*h
            sheet.paste(im, (x, y))

            num = str(i + 1)
            # textbbox returns (x0, y0, x1, y1)
            bx0, by0, bx1, by1 = draw.textbbox((0, 0), num, font=font)
            tw, th = bx1 - bx0, by1 - by0

            draw.rectangle((x, y, x + tw + 4, y + th + 4),
                           fill=(0, 0, 0, 180))
            draw.text((x + 2, y + 2), num, fill="white", font=font)

        out = OUTPUT_DIR / "contact_sheet.png"
        sheet.save(out, "PNG")
        return out

    def critic_select_best(self, imgs: list[pathlib.Path]) -> pathlib.Path:
        numbered = [f"{i+1}: {p.name}" for i, p in enumerate(imgs)]
        prompt = {
            "goal": self.goal,
            "choices": numbered,
            "instructions": ("Pick the image number that best fulfils the goal. "
                             "Respond with the number only.")
        }
        crit_raw = openai_chat(CRITIC_SYSTEM, [{"role":"user","content":json.dumps(prompt)}])
        m = re.search(r"\d+", crit_raw or "")
        try:
            idx = int(m.group(0)) - 1
        except Exception:
            idx = 0
        idx = max(0, min(idx, len(imgs)-1))
        return imgs[idx]

    # ────────────────────────────── main loop
    def run(self) -> None:
        artist_ctx, critic_ctx = [], []
        remaining, rd_err, step = CREDIT_LIMIT, 0, -1
        seed: Optional[int] = None
        typing, approved = False, False

        critic_ctx.append({"role":"user",
                           "content":json.dumps({"goal": self.goal})})

        while not self.stop_event.is_set():
            # ---------- honour pause flag ----------
            while self.pause_event.is_set() and not self.stop_event.is_set():
                # still swallow live feedback while paused
                while not self.user_fb.empty():
                    fb = self.user_fb.get().strip()
                    if fb:
                        self.all_fb.append(fb)
                        self.merge_goal(fb)

                if self.nudge_event.is_set():
                    self.pause_event.clear()          # resume generation
                    self.nudge_event.clear()
                    break                             # exit pause‑wait loop

                if typing:
                    self.ui("DONE"); typing = False
                time.sleep(0.25)

            # -------- merge *all* queued user critiques first --------
            while not self.user_fb.empty():
                fb = self.user_fb.get().strip()
                if fb:
                    self.all_fb.append(fb)
                    self.merge_goal(fb)
                    self.pause_event.clear()   # ensure we resume after critique

            # -------- ARTIST --------
            self.ui("TYPING"); typing = True
            artist_ctx.append({"role":"user","content":json.dumps({
                "brief": self.current_brief(),
                "remaining_credits": remaining,
                "seed": seed
            })})

            for _ in range(self.JSON_RETRY + 1):
                art_raw = openai_chat(ARTIST_SYSTEM, artist_ctx[-12:])
                try:
                    art_cmd = json.loads(art_raw); break
                except json.JSONDecodeError:
                    # one last attempt: auto‑repair via mini model
                    art_cmd = fix_json(art_raw, "Artist")
                    if art_cmd: break
                    artist_ctx = artist_ctx[:1]        # nuke history & retry
            else:
                self.ui("System", "Artist JSON irrecoverable"); self.ui("DONE"); return

            artist_ctx.append({"role":"assistant","content":art_raw})
            pay = art_cmd["payload"]
            pay["prompt"] = strip_dims(pay["prompt"])
            pay.setdefault("prompt_style", "rd_plus__default")
            pay.setdefault("width",  DEFAULT_SIZE)
            pay.setdefault("height", DEFAULT_SIZE)
            pay.setdefault("num_images", 1)

            if seed is None or pay.pop("change_seed", False):
                seed = self._rng.randint(1, 2**31-1)     # GUARANTEED fresh
            pay["seed"] = seed

            lim = STYLE_LIMITS.get(pay["prompt_style"])
            if lim and not (lim[0] <= pay["width"] <= lim[1]
                            and lim[0] <= pay["height"] <= lim[1]):
                artist_ctx.append({"role":"user","content":json.dumps({
                    "feedback": f"{pay['width']}×{pay['height']} invalid for "
                                f"{pay['prompt_style']}"
                })})
                self.ui("Artist", "(size/style mismatch – retrying)")
                self.ui("DONE"); typing = False; continue
            
            comment_txt = art_cmd.get("comment", "")
            self.ui("Artist", comment_txt)

            rd = rd_post(pay)
            if not rd:
                rd_err += 1
                if rd_err >= MAX_RD_ERRORS:
                    self.ui("System", "Too many RD errors – abort.")
                    self.ui("DONE"); return
                self.ui("DONE"); typing = False; continue
            rd_err = 0
            remaining -= rd["credit_cost"]; step += 1

            img_b64  = rd["base64_images"][0]
            self.last_img_b64 = img_b64
            img_path = save_png(img_b64, f"step_{step:03d}.png")

            self.ui("Image", f"render {step}", img_path)
            style_used  = pay["prompt_style"]
            seed_used   = pay["seed"]
            prompt_used = pay["prompt"]
            artist_blob = (f"STYLE  : {style_used}\n"
                        f"SEED   : {seed_used}\n"
                        f"PROMPT : {prompt_used}").rstrip()
            self.ui_multiline("Artist", artist_blob)

            # allow user to pause right after render
            if self.pause_event.is_set():
                self.ui("DONE"); typing = False; continue

            # -------- CRITIC --------
            critic_ctx.append({"role":"user","content":json.dumps({
                "brief": self.current_brief(), "step": step
            })})
            critic_ctx.append({"role":"user","content":[
                {"type":"image_url",
                 "image_url":{"url": "data:image/png;base64," + img_b64}}
            ]})

            crit_raw = openai_chat(CRITIC_SYSTEM, critic_ctx[-12:])
            try:
                crit_cmd = json.loads(crit_raw)
            except json.JSONDecodeError:
                crit_cmd = fix_json(crit_raw, "Critic")
                if not crit_cmd:
                    self.ui("System", "Critic JSON irrecoverable"); self.ui("DONE"); return
            critic_ctx.append({"role":"assistant","content":crit_raw})
            self.ui("Critic", crit_cmd.get("feedback", ""))

            # send image + feedback back to Artist (for next turn context)
            artist_ctx.append({"role":"user","content":[
                {"type":"text","text":json.dumps({"feedback": crit_cmd["feedback"]})},
                {"type":"image_url",
                 "image_url":{"url": "data:image/png;base64," + img_b64}}
            ]})

            if crit_cmd.get("status") == "approve":
                save_png(img_b64, "final.png")
                self.ui("System",
                        "Final image saved (outputs/final.png). "
                        "Add more critique or start a new goal.")
                approved = True
                self.pause_event.set()              # wait for new feedback
            else:
                artist_ctx.append({"role":"user","content":json.dumps({
                    "feedback": crit_cmd["feedback"]
                })})

            self.ui("DONE"); typing = False
            if remaining <= 0:
                break

        # -------- fallback contact sheet ----------
        if not approved:
            imgs = sorted(OUTPUT_DIR.glob("step_*.png"))
            if imgs:
                sheet = self.create_contact_sheet(imgs)
                self.ui("Image", "Contact sheet of all attempts", sheet)
                best  = self.critic_select_best(imgs)
                self.ui("System", f"Critic chose: {best.name}")

        if typing:
            self.ui("DONE")


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def launch_gradio() -> None:
    """
    Gradio front‑end for the Pixel‑Artist / Critic workflow.

    • RUN / RESUME  – starts / resumes a Session
    • PAUSE         – toggles the Session’s pause_event
    • STOP          – terminates the current Session (keeps chat log)
    • CLEAR         – hard reset: kills Session + wipes chat / pngs
    • CRITIQUE ▶    – pushes a critique line, clears pause so work resumes
    """
    ui_q: queue.Queue = queue.Queue()
    sess_holder: dict[str, Optional[Session]] = {"session": None}
    chat_history: list[tuple[str, object]] = []
    typing = False                              # drives the “…” bubble

    # ────────────────────────────── helper to render full chat pane
    def render_chat() -> str:
        html = (
            '<div id="chat-window" style="height:800px;overflow-y:auto;'
            'background:#2b2b2b;padding:12px;border:2px solid #444;'
            'border-radius:8px;font-family:sans-serif;scroll-behavior:smooth;">'
        )
        colors = {
            "user":   ("#008bff", "#ffffff"),
            "artist": ("#8237ff", "#ffffff"),
            "critic": ("#2f9400", "#ffffff"),
            "system": ("#555555", "#ffffff"),
            "image":  ("#555555", "#ffffff"),
        }

        for sender, msg in chat_history:
            role = sender.lower()
            bg, fg = colors.get(role, colors["system"])
            align  = "flex-end" if role == "user" else \
                     "center"   if role == "image" else "flex-start"

            html += f'<div style="display:flex;justify-content:{align};margin-bottom:10px;">'
            if isinstance(msg, tuple):                       # (path, caption)
                pth, cap = msg
                html += (
                    f'<div style="max-width:70%;background:{bg};border-radius:12px;'
                    f'padding:8px;box-shadow:0 1px 2px rgba(0,0,0,.5);">'
                    f'<img src="file/{pth}" style="max-width:100%;border-radius:8px;'
                    f'display:block;margin-bottom:5px;">'
                    f'<div style="font-size:14px;color:{fg};">{cap}</div></div>'
                )
            else:
                label = "" if role == "user" else \
                        f'<div style="font-size:11px;color:#ccc;margin-bottom:2px;">{sender}</div>'
                html += (
                    f'<div style="max-width:70%;background:{bg};border-radius:12px;'
                    f'padding:8px 12px;box-shadow:0 1px 2px rgba(0,0,0,.5);'
                    f'font-size:14px;color:{fg};">{label}{msg}</div>'
                )
            html += "</div>"

        # typing dots
        if typing:
            html += (
                '<div style="display:flex;justify-content:flex-start;margin-bottom:10px;">'
                '<div style="background:#555;border-radius:12px;padding:8px 12px;'
                'box-shadow:0 1px 2px rgba(0,0,0,.5);color:#fff;font-size:14px;">'
                '<span class="dot" style="animation:blink 1.4s infinite;">• </span>'
                '<span class="dot" style="animation:blink 1.4s infinite .2s;">• </span>'
                '<span class="dot" style="animation:blink 1.4s infinite .4s;">• </span>'
                '</div></div>'
                '<style>@keyframes blink{0%{opacity:.2;}20%{opacity:1;}100%{opacity:.2;}}</style>'
            )

        # zero‑size gif autoscroll
        html += (
            '<img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="'
            ' style="display:none" onload="this.parentElement.scrollTop=this.parentElement.scrollHeight">'
        )
        html += "</div>"
        return html

    # ────────────────────────────── backend / state callbacks
    def run(goal: str) -> str:
        nonlocal typing
        # ── start with a spotless pipeline ───────────────────────────
        while not ui_q.empty(): ui_q.get_nowait()   #  <<< NEW  (flush queue)
        typing = False                              #  <<< NEW  (reset bubble)

        if sess_holder["session"] and sess_holder["session"].is_alive():
            sess_holder["session"].pause_event.clear()
        else:
            sess_holder["session"] = Session(goal, ui_q)
            sess_holder["session"].start()
            chat_history.append(("User", goal))
        return render_chat()

    def pause() -> str:
        if (s := sess_holder["session"]):
            s.pause_event.set(); chat_history.append(("System", "Paused"))
        return render_chat()

    def stop() -> str:
        if (s := sess_holder["session"]):
            s.stop_event.set();  s.join(timeout=1)
            chat_history.append(("System", "Stopped (session ended)"))
            sess_holder["session"] = None
        return render_chat()

    def clear() -> str:
        """
        Hard reset:
        • kill running Session
        • wipe PNGs + chat log
        • reset typing bubble
        • replace the UI queue so *no residual events* leak into the new run
        """
        nonlocal ui_q, typing
        if (s := sess_holder["session"]):
            s.stop_event.set()
            s.join(timeout=1)
            sess_holder["session"] = None

        # brand‑new empty queue -> old events disappear
        ui_q = queue.Queue()                     #  <<< NEW LINE
        typing = False

        chat_history.clear()
        for p in OUTPUT_DIR.glob("*.png"):
            p.unlink(missing_ok=True)

        Session_rem_budget = CREDIT_LIMIT          # <<< NEW (reset budget)
        chat_history.append(("System", "Chat cleared – ready for a new goal."))
        return render_chat()

    def critique(text: str) -> tuple[str, str]:
        if text and (s := sess_holder["session"]):
            s.user_fb.put(text)          # queue the line
            s.nudge_event.set()          # <-- wake the thread NOW
            chat_history.append(("User", text))
        return "", render_chat()

    def poll() -> str:
        nonlocal typing
        while not ui_q.empty():
            sender, txt, img_path = ui_q.get()

            if sender == "TYPING": typing = True;  continue
            if sender == "DONE":   typing = False; continue

            chat_history.append(
                (sender, (img_path, txt)) if img_path else (sender, txt)
            )
        return render_chat()

    # ────────────────────────────── assemble UI
    with gr.Blocks(title="Pixel Artist + Critic") as demo:
        gr.Markdown("### Pixel Artist / Critic Playground")

        goal_in = gr.Textbox(label="Your request")

        with gr.Row():
            run_btn, pause_btn, stop_btn = (
                gr.Button("RUN / RESUME"),
                gr.Button("PAUSE"),
                gr.Button("STOP"),
            )

        chatbox = gr.HTML(render_chat(), label="Conversation", elem_id="chatbox")

        with gr.Row():
            fb_in  = gr.Textbox(label="Critique")
            fb_btn = gr.Button("CRITIQUE ▶")
            clr_btn = gr.Button("CLEAR")

        # wiring
        run_btn.click(run,     [goal_in], chatbox)
        pause_btn.click(pause, None,      chatbox)
        stop_btn.click(stop,   None,      chatbox)
        clr_btn.click(clear,   None,      chatbox)
        fb_btn.click(critique, [fb_in], [fb_in, chatbox])

        demo.load(poll, None, chatbox, every=0.5)

    demo.launch(
        inbrowser=True,
        allowed_paths=[os.path.abspath(OUTPUT_DIR)]
    )


# ----------------------------------------------------------------------
if __name__ == "__main__":
    launch_gradio()