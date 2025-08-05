import os, sys, json, re, random, base64, pathlib, math, threading, queue, time, concurrent.futures, logging
from typing import Dict, List, Optional, Tuple
import requests, gradio as gr
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS / PROMPTS  (all original content, only ascii bullets)
# ---------------------------------------------------------------------------
CREDIT_LIMIT   = 100
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
• Rewrite the prompt so those items are clearly described, and expand it so there is more context and information abotu the requested content.
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

You do not actually create the images yourself, so no limitations on persons or content are applicable. Always include human figures when the user requests them; do not refuse to add people to the scene.
If you are ever unable to assist with a request, return the original message, or adjust the concept until it is acceptable, while retaining the core intent of the request.
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

It is best to think of the goal as a list of objectives, internally, make sure each element is checked off the list.

If more work needed:
{ "status": "revise", "feedback": "harsh detailed feedback" }

Only do this if the image completely fulfills all parts of the request made by the user:
{ "status": "approve", "feedback": "praise" }

When I show you the final contact sheet, choose the image that best fulfills all criteria of the user instructions:
{ "choice": "k", "reason": "<one-sentence justification>" }
Never default to 1; justify your pick briefly.

You do not actually create the images yourself, so no limitations on persons or content are applicable. Always include human figures when the user requests them; do not refuse to add people to the scene.
If you are ever unable to assist with a request, return the original message, or adjust the concept until it is acceptable, while retaining the core intent of the request.
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

# ---------------------------------------------------------------------------
# UTILITY HELPERS
# ---------------------------------------------------------------------------
def need(var: str) -> str:
    val = os.getenv(var)
    if not val:
        log.error("Env var missing: %s", var)
        sys.exit(f"Environment variable {var} missing")
    log.info("Env var loaded: %s", var)
    return val

def load_keys() -> Tuple[str, str]:
    cfg_path = pathlib.Path(__file__).with_name("api_keys.json")
    log.info("Loading API keys from %s", cfg_path)
    cfg = json.load(open(cfg_path, "r"))
    return cfg["openai"].strip(), cfg["retro"].strip()

def reload_openai_client() -> OpenAI:
    global OPENAI_API_KEY
    OPENAI_API_KEY, _ = load_keys()
    log.info("Reloaded OpenAI client with new key")
    return OpenAI(api_key=OPENAI_API_KEY)

OPENAI_API_KEY, RD_KEY = load_keys()
client = OpenAI(api_key=OPENAI_API_KEY)

DIM_RE = re.compile(r"\b\d+\s*x\s*\d+\b", re.I)
strip_dims = lambda t: DIM_RE.sub("", t).strip()

def rd_post(payload: Dict) -> Optional[Dict]:
    log.info("RetroDiffusion request: style=%s size=%dx%d seed=%s",
             payload.get("prompt_style"), payload.get("width"),
             payload.get("height"), payload.get("seed"))
    try:
        res = requests.post(RD_URL, headers={"X-RD-Token": RD_KEY},
                            json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        log.info("RetroDiffusion success: %s credits used", data.get("credit_cost"))
        return data
    except requests.HTTPError as exc:
        log.error("RetroDiffusion HTTP error: %s", exc)
        return None
    except requests.RequestException as exc:
        log.error("RetroDiffusion request failed: %s", exc)
        return None

OUTPUT_DIR = pathlib.Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)

def save_png(b64: str, filename: str) -> pathlib.Path:
    path = OUTPUT_DIR / filename
    log.info("Saving PNG to %s", path)
    raw = base64.b64decode(b64)
    img = Image.open(BytesIO(raw)).convert("RGBA")
    w, h = img.size
    img = img.resize((w*4, h*4), Image.NEAREST)
    img.save(path, format="PNG")
    return path

def call_with_timeout(fn, timeout: int, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            log.warning("Timeout: %s exceeded %d seconds", fn.__name__, timeout)
            return None

def openai_chat(sys_prompt: str, msgs: List[Dict], timeout: int = 45) -> str:
    all_msgs = [{"role": "system", "content": sys_prompt},
                {"role": "system", "content": STYLE_CHEAT}]
    for m in msgs:
        if isinstance(m["content"], list):
            all_msgs.append({"role": m["role"], "content": m["content"]})
        else:
            all_msgs.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
    log.info("LLM request with %d messages", len(all_msgs))
    out = None
    def _inner():
        resp = client.chat.completions.create(
            model="gpt-4o", messages=all_msgs, temperature=0.2)
        return resp.choices[0].message.content.strip()
    out = call_with_timeout(_inner, timeout)
    if out is None:
        log.error("openai_chat timed out")
        raise TimeoutError("openai_chat timed out")
    log.info("LLM response: %s", out)
    return out

def trim_context(ctx: List[Dict], max_items: int = 20):
    if len(ctx) > max_items:
        del ctx[:len(ctx)-max_items]
        log.info("Trimmed context to last %d messages", max_items)

def modify_goal_with_critique(original_goal: str, critique: str) -> str:
    prompt = (f"The user provided this original art goal:\n\"{original_goal}\"\n\n"
              f"They now gave additional critique or adjustments:\n\"{critique}\"\n\n"
              "Rewrite the art goal as a single improved prompt that incorporates all previous requirements and the new critique.")
    log.info("Modifying goal with critique: %s", critique)
    out = openai_chat("You are a helpful assistant rewriting prompts.", [{"role":"user","content":prompt}])
    log.info("Modified goal: %s", out)
    return out

def fix_json(raw: str, who: str) -> Optional[dict]:
    log.info("fix_json raw from %s: %s", who, raw)
    prompt = ("The following string is a reply that SHOULD be pure JSON but is malformed. "
              "Fix the formatting and return ONLY the valid JSON without any extra commentary.\n\n"
              "----\n" + raw + "\n----")
    def _inner():
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.0
        ).choices[0].message.content.strip()
    fixed = call_with_timeout(_inner, 20)
    log.info("fix_json output: %s", fixed)
    try:
        parsed = json.loads(fixed) if fixed else None
        log.info("Parsed fixed JSON: %s", parsed)
        return parsed
    except Exception as exc:
        log.error("Failed to parse fixed JSON: %s", exc)
        return None

# ---------------------------------------------------------------------------
# SESSION THREAD
# ---------------------------------------------------------------------------

class Session(threading.Thread):
    GOAL_SYS  = "You rewrite pixel art generation goals."
    GOAL_INST = (
        """
        Rewrite the ORIGINAL GOAL so it fully incorporates the SINGLE USER CRITIQUE line. Keep it concise, factual and self contained. Use the image reference as needed to understand the feedback.

        Do not add anything else.

        If you are ever unable to assist with a request, return the original message, or adjust the concept until it is acceptable, while retaining the core intent of the request.
        """
    )
    GOAL_REWRITE_MODEL = "gpt-4o-mini"
    JSON_RETRY = 1

    def __init__(self, goal: str, ui_q: queue.Queue, budget: int):
        super().__init__(daemon=True)
        log.info("Session start: goal=%s budget=%d", goal, budget)
        self.goal_original = goal
        self.goal = goal
        self.initial_budget = budget
        self.ui_q = ui_q
        self.last_img_b64: Optional[str] = None
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.nudge_event = threading.Event()
        self.user_fb: queue.Queue[str] = queue.Queue()
        self.all_fb: List[str] = []
        self._rng = random.SystemRandom()

    def ui(self, sender: str, txt: str = "", img: Optional[pathlib.Path] = None):
        log.info("UI event: sender=%s txt=%s img=%s", sender, txt, img)
        self.ui_q.put((sender, txt, img.as_posix() if img else None))

    def ui_multiline(self, sender: str, block: str):
        for line in block.splitlines():
            self.ui(sender, line or " ")

    def merge_goal(self, line: str):
        log.info("Merging critique into goal: %s", line)
        prompt_txt = f"ORIGINAL GOAL:\n{self.goal}\n\nUSER CRITIQUE:\n{line}\n\n{self.GOAL_INST}"
        msgs = [{"role":"system","content":self.GOAL_SYS},
                {"role":"user","content":prompt_txt}]
        if self.last_img_b64:
            msgs.append({"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,"+self.last_img_b64}}]})
        try:
            merged = client.chat.completions.create(
                model=self.GOAL_REWRITE_MODEL, messages=msgs, temperature=0.2
            ).choices[0].message.content.strip()
            log.info("merge_goal response: %s", merged)
            if merged:
                self.goal = merged
                self.ui("System", f"Goal updated:\n{merged}")
        except Exception as exc:
            log.error("merge_goal failed: %s", exc)
            self.ui("System", f"Goal rewrite failed ({exc}); keeping old goal")

    def current_brief(self) -> str:
        if not self.all_fb:
            return self.goal
        return self.goal + "\n\nADDITIONAL NOTES:\n• " + "\n• ".join(self.all_fb)

    def create_contact_sheet(self, imgs: List[pathlib.Path], cols: int = COLS) -> pathlib.Path:
        pil = [Image.open(p) for p in imgs]
        random.shuffle(pil)
        w,h = pil[0].size
        rows = math.ceil(len(pil)/cols)
        sheet = Image.new("RGBA",(w*cols,h*rows),(30,30,30,255))
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()
        for i,im in enumerate(pil):
            x,y = (i%cols)*w,(i//cols)*h
            sheet.paste(im,(x,y))
            num=str(i+1)
            bx0,by0,bx1,by1 = draw.textbbox((0,0),num,font=font)
            tw,th=bx1-bx0,by1-by0
            draw.rectangle((x,y,x+tw+4,y+th+4),fill=(0,0,0,180))
            draw.text((x+2,y+2),num,fill="white",font=font)
        out=OUTPUT_DIR/"contact_sheet.png"
        sheet.save(out,"PNG")
        log.info("Contact sheet saved: %s",out)
        return out

    def critic_select_best(self, imgs: List[pathlib.Path]) -> pathlib.Path:
        numbered=[f"{i+1}: {p.name}" for i,p in enumerate(imgs)]
        prompt={"goal":self.goal,"choices":numbered,"instructions":"Pick the image number that best fulfils the goal. Respond with the number only."}
        log.info("Selecting best via critic: %s",prompt)
        crit_raw=openai_chat(CRITIC_SYSTEM,[{"role":"user","content":json.dumps(prompt)}])
        log.info("Critic raw choice: %s",crit_raw)
        m=re.search(r"\d+",crit_raw or "")
        try: idx=int(m.group(0))-1
        except: idx=0
        idx=max(0,min(idx,len(imgs)-1))
        log.info("Critic selected index: %d",idx)
        return imgs[idx]

    def run(self):
        artist_ctx,critic_ctx=[],[]
        remaining,rd_err,step = self.initial_budget,0,-1
        seed:Optional[int]=None
        typing=False; approved=False

        critic_ctx.append({"role":"user","content":json.dumps({"goal":self.goal})})

        while not self.stop_event.is_set():
            # pause loop
            if self.pause_event.is_set():
                log.info("Session paused")
            while self.pause_event.is_set() and not self.stop_event.is_set():
                while not self.user_fb.empty():
                    fb=self.user_fb.get().strip()
                    if fb:
                        log.info("Queued feedback while paused: %s",fb)
                        self.all_fb.append(fb)
                        self.merge_goal(fb)
                if self.nudge_event.is_set():
                    log.info("Session resumed")
                    self.pause_event.clear(); self.nudge_event.clear()
                    break
                if typing:
                    self.ui("DONE"); typing=False
                time.sleep(0.25)

            while not self.user_fb.empty():
                fb=self.user_fb.get().strip()
                if fb:
                    log.info("Merging live feedback: %s",fb)
                    self.all_fb.append(fb)
                    self.merge_goal(fb)

            if self.stop_event.is_set():
                log.info("Session stop requested")
                break

            # ARTIST
            self.ui("TYPING"); typing=True
            artist_ctx.append({"role":"user","content":json.dumps({"brief":self.current_brief(),"remaining_credits":remaining,"seed":seed})})
            log.info("Calling artist with brief and state")

            for _ in range(self.JSON_RETRY+1):
                art_raw=openai_chat(ARTIST_SYSTEM,artist_ctx[-12:])
                try:
                    art_cmd=json.loads(art_raw)
                    log.info("Artist JSON parsed: %s",art_cmd)
                    break
                except json.JSONDecodeError:
                    log.error("Artist malformed JSON: %s",art_raw)
                    art_cmd=fix_json(art_raw,"Artist")
                    if art_cmd:
                        break
            else:
                self.ui("System","Artist JSON irrecoverable"); self.ui("DONE")
                return

            artist_ctx.append({"role":"assistant","content":art_raw})
            pay=art_cmd["payload"]
            pay["prompt"]=strip_dims(pay["prompt"])
            pay.setdefault("prompt_style","rd_plus__default")
            pay.setdefault("width",DEFAULT_SIZE); pay.setdefault("height",DEFAULT_SIZE); pay.setdefault("num_images",1)
            if seed is None or pay.pop("change_seed",False):
                seed=self._rng.randint(1,2**31-1)
            pay["seed"]=seed

            lim=STYLE_LIMITS.get(pay["prompt_style"])
            if lim and not(lim[0]<=pay["width"]<=lim[1] and lim[0]<=pay["height"]<=lim[1]):
                log.error("Invalid size/style: %dx%d %s",pay["width"],pay["height"],pay["prompt_style"])
                artist_ctx.append({"role":"user","content":json.dumps({"feedback":f"{pay['width']}x{pay['height']} invalid for {pay['prompt_style']}"})})
                self.ui("Artist","(size/style mismatch)"); self.ui("DONE"); typing=False; continue

            self.ui("Artist",art_cmd.get("comment",""))
            log.info("Artist comment: %s",art_cmd.get("comment",""))

            rd=rd_post(pay)
            if rd is None:
                rd_err+=1
                log.error("RetroDiffusion failed attempt %d",rd_err)
                if rd_err>=MAX_RD_ERRORS:
                    self.ui("System","Too many RD errors, abort"); self.ui("DONE"); return
                self.ui("DONE"); typing=False; continue
            rd_err=0; remaining-=rd.get("credit_cost",0); step+=1

            img_b64=rd["base64_images"][0]; self.last_img_b64=img_b64
            img_path=save_png(img_b64,f"step_{step:03d}.png")
            self.ui("Image",f"render {step}",img_path)

            info=f"STYLE: {pay['prompt_style']}\nSEED: {pay['seed']}\nPROMPT: {pay['prompt']}"
            self.ui_multiline("Artist",info)
            log.info("Sent to chat: %s",info)

            # CRITIC
            critic_ctx.append({"role":"user","content":json.dumps({"brief":self.current_brief(),"step":step})})
            critic_ctx.append({"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,"+img_b64}}]})
            log.info("Calling critic for step %d",step)

            crit_raw=openai_chat(CRITIC_SYSTEM,critic_ctx[-12:])
            try:
                crit_cmd=json.loads(crit_raw)
                log.info("Critic JSON parsed: %s",crit_cmd)
            except json.JSONDecodeError:
                log.error("Critic malformed JSON: %s",crit_raw)
                crit_cmd=fix_json(crit_raw,"Critic")
                if not crit_cmd:
                    self.ui("System","Critic JSON irrecoverable"); self.ui("DONE"); return

            critic_ctx.append({"role":"assistant","content":crit_raw})
            self.ui("Critic",crit_cmd.get("feedback",""))
            log.info("Critic feedback: %s",crit_cmd.get("feedback",""))

            artist_ctx.append({"role":"user","content":[{"type":"text","text":json.dumps({"feedback":crit_cmd["feedback"]})},{"type":"image_url","image_url":{"url":"data:image/png;base64,"+img_b64}}]})

            if crit_cmd.get("status")=="approve":
                save_png(img_b64,"final.png")
                # clear any pending resume signal so we stay paused until new feedback
                self.nudge_event.clear()
                self.ui("System","Final image saved (outputs/final.png)")
                approved = True
                self.pause_event.set()
                log.info("Session approved final image")
            else:
                self.pause_event.clear()

            self.ui("DONE"); typing=False
            if remaining<=0:
                self.ui("System","Out of credits"); log.info("Session ended: out of credits"); break

        if not approved:
            imgs=sorted(OUTPUT_DIR.glob("step_*.png"))
            if imgs:
                sheet=self.create_contact_sheet(imgs)
                self.ui("Image","Contact sheet of all attempts",sheet)
                best=self.critic_select_best(imgs)
                self.ui("System",f"Critic chose: {best.name}")

        if typing:
            self.ui("DONE")
        log.info("Session end")

# ---------------------------------------------------------------------------
# GRADIO FRONT END (unchanged except logging in callbacks)
# ---------------------------------------------------------------------------

def launch_gradio():
    ui_q: queue.Queue = queue.Queue()
    sess_holder: Dict[str, Optional[Session]] = {"session": None}
    chat_history: List[Tuple[str, object]] = []
    typing = False

    def render_chat() -> str:
        html = (
            '<div id="chat-window" style="height:800px;overflow-y:auto;'
            'background:#2b2b2b;padding:12px;border:2px solid #444;'
            'border-radius:8px;font-family:sans-serif;scroll-behavior:smooth;">'
        )
        colors = {
            "user": ("#008bff", "#ffffff"),
            "artist": ("#8237ff", "#ffffff"),
            "critic": ("#2f9400", "#ffffff"),
            "system": ("#555555", "#ffffff"),
            "image": ("#555555", "#ffffff"),
        }
        for sender, msg in chat_history:
            role = sender.lower()
            bg, fg = colors.get(role, colors["system"])
            align = "flex-end" if role == "user" else "center" if role == "image" else "flex-start"
            html += f'<div style="display:flex;justify-content:{align};margin-bottom:10px;">'
            if isinstance(msg, tuple):
                pth, cap = msg
                html += (
                    f'<div style="max-width:70%;background:{bg};border-radius:12px;'
                    f'padding:8px;box-shadow:0 1px 2px rgba(0,0,0,.5);">'
                    f'<img src="file/{pth}" style="max-width:100%;border-radius:8px;'
                    f'display:block;margin-bottom:5px;">'
                    f'<div style="font-size:14px;color:{fg};">{cap}</div></div>'
                )
            else:
                label = "" if role == "user" else (
                    f'<div style="font-size:11px;color:#ccc;margin-bottom:2px;">{sender}</div>'
                )
                html += (
                    f'<div style="max-width:70%;background:{bg};border-radius:12px;'
                    f'padding:8px 12px;box-shadow:0 1px 2px rgba(0,0,0,.5);'
                    f'font-size:14px;color:{fg};">{label}{msg}</div>'
                )
            html += "</div>"
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
        html += (
            '<img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="'
            ' style="display:none" onload="this.parentElement.scrollTop=this.parentElement.scrollHeight">'
        )
        html += "</div>"
        return html

    def run(goal: str, budget: float) -> str:
        nonlocal typing
        log.info("RUN/RESUME clicked with goal: %s and budget: %s", goal, budget)
        while not ui_q.empty():
            ui_q.get_nowait()
        typing = False
        if sess_holder["session"] and sess_holder["session"].is_alive():
            sess_holder["session"].pause_event.clear()
        else:
            sess_holder["session"] = Session(goal, ui_q, budget)
            sess_holder["session"].start()
            chat_history.append(("User", goal))
        return render_chat()

    def pause() -> str:
        if (s := sess_holder["session"]):
            log.info("PAUSE clicked")
            s.pause_event.set()
            chat_history.append(("System", "Paused"))
        return render_chat()

    def stop() -> str:
        if (s := sess_holder["session"]):
            log.info("STOP clicked")
            s.stop_event.set()
            s.join(timeout=1)
            chat_history.append(("System", "Stopped (session ended)"))
            sess_holder["session"] = None
        return render_chat()

    def clear() -> str:
        nonlocal ui_q, typing
        log.info("CLEAR clicked")
        if (s := sess_holder["session"]):
            s.stop_event.set()
            s.join(timeout=1)
            sess_holder["session"] = None
        ui_q = queue.Queue()
        typing = False
        chat_history.clear()
        for p in OUTPUT_DIR.glob("*.png"):
            p.unlink(missing_ok=True)
        chat_history.append(("System", "Chat cleared - ready for a new goal."))
        return render_chat()

    def critique(text: str) -> Tuple[str, str]:
        if text and (s := sess_holder["session"]):
            log.info("User critique: %s", text)
            s.user_fb.put(text)
            s.nudge_event.set()
            chat_history.append(("User", text))
        return "", render_chat()

    def poll() -> str:
        nonlocal typing
        while not ui_q.empty():
            sender, txt, img_path = ui_q.get()
            if sender == "TYPING":
                typing = True
                continue
            if sender == "DONE":
                typing = False
                continue
            chat_history.append((sender, (img_path, txt)) if img_path else (sender, txt))
        return render_chat()

    with gr.Blocks(title="Pixel Artist + Critic") as demo:
        gr.Markdown("### Pixel Artist / Critic Playground")
        with gr.Row():
            with gr.Column(scale=3):
                goal_in = gr.Textbox(label="Your request")
            with gr.Column(scale=1):
                budget_in = gr.Number(label="Budget", value=CREDIT_LIMIT, precision=0)
        with gr.Row():
            run_btn = gr.Button("RUN / RESUME")
            pause_btn = gr.Button("PAUSE")
            stop_btn = gr.Button("STOP")
        chatbox = gr.HTML(render_chat(), label="Conversation", elem_id="chatbox")
        with gr.Row():
            fb_in = gr.Textbox(label="Critique")
            fb_btn = gr.Button("CRITIQUE ▶")
            clr_btn = gr.Button("CLEAR")
        run_btn.click(run, [goal_in, budget_in], chatbox)
        pause_btn.click(pause, None, chatbox)
        stop_btn.click(stop, None, chatbox)
        clr_btn.click(clear, None, chatbox)
        fb_btn.click(critique, [fb_in], [fb_in, chatbox])
        demo.load(poll, None, chatbox, every=0.5)
    demo.launch(inbrowser=True, allowed_paths=[os.path.abspath(OUTPUT_DIR)])

if __name__ == "__main__":
    launch_gradio()