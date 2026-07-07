#!/usr/bin/env python3
"""Coding Agent Latency Monitor — Kai build 2026-07-06.

v3: REAL KILL SWITCH. All N runs are launched server-side and multiplexed over one
SSE connection (bypasses the browser 6-conn limit). /kill sets a stop flag AND slams
every upstream model connection shut, so vLLM aborts the requests instantly and the
GPUs actually stop (v1/v2 Stop only cut the browser; the server kept generating → GPUs
pinned). Also real per-node GPU%/RAM via SSH for the 3090.
"""
import json, urllib.request, urllib.parse, os, threading, queue, subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 7900
HERE = os.path.dirname(os.path.abspath(__file__))

def _load_json(name, default):
    p = os.path.join(HERE, name)
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
    return default

# Endpoints: label -> "base_url|model". Put yours in presets.json (see presets.example.json).
PRESETS = _load_json("presets.json", {"Local vLLM (:8000)": "http://localhost:8000/v1|your-model-name"})
# Optional real GPU%/RAM via SSH: {"<ip-substring-of-endpoint>": ["user@host", "~/path/to/ssh_key"]}
# See nodes.example.json. Empty -> the hardware strip just shows vLLM /metrics.
NODES = {k: (v[0], os.path.expanduser(v[1])) for k, v in _load_json("nodes.json", {}).items()}

# Persistent run history. Prefer an external drive so runs survive; fall back to local.
_RUNS_ROOT = _load_json("config.json", {}).get("runs_dir") or "/Volumes/Seagate/coding-agent-monitor-runs"
RUNS_DIR = _RUNS_ROOT if os.path.isdir(os.path.dirname(_RUNS_ROOT.rstrip("/"))) else HERE
try:
    os.makedirs(RUNS_DIR, exist_ok=True)
except Exception:
    RUNS_DIR = HERE
RUNS_FILE = os.path.join(RUNS_DIR, "runs.jsonl")

STOP = threading.Event()          # set by /kill -> workers bail
CONNS = []                        # live upstream responses (so /kill can slam them shut)
CLOCK = threading.Lock()


def stream_one(idx, ep, model, prompt, mx, temp, think, q):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": mx, "temperature": temp, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": think}}
    r = None
    try:
        if STOP.is_set():
            q.put({"run": idx, "killed": True}); return
        req = urllib.request.Request(ep.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=900)
        with CLOCK:
            CONNS.append(r)
        for raw in r:
            if STOP.is_set():
                break
            line = raw.decode("utf-8", "ignore")
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            try:
                j = json.loads(d)
            except Exception:
                continue
            ch = (j.get("choices") or [{}])[0]
            delta = (ch.get("delta") or {}).get("content") or ""
            u = (j.get("usage") or {}).get("completion_tokens")
            if delta or u:
                q.put({"run": idx, "c": delta, "u": u})
        q.put({"run": idx, "killed": True} if STOP.is_set() else {"run": idx, "done": True})
    except Exception as e:
        q.put({"run": idx, "killed": True} if STOP.is_set() else {"run": idx, "err": str(e)[:160]})
    finally:
        try:
            if r:
                r.close()
        except Exception:
            pass


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body=b""):
        self.send_response(code); self.send_header("Content-Type", ctype)
        if body:
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, "text/html; charset=utf-8", f.read())
        elif u.path == "/presets":
            self._send(200, "application/json", json.dumps(PRESETS).encode())
        elif u.path == "/kill":
            STOP.set()
            n = 0
            with CLOCK:
                for r in CONNS:
                    try:
                        r.close(); n += 1
                    except Exception:
                        pass
                CONNS.clear()
            self._send(200, "application/json", json.dumps({"ok": True, "closed": n}).encode())
        elif u.path == "/hw":
            self._hw(q.get("ep", [""])[0])
        elif u.path == "/runall":
            self._runall(q)
        elif u.path == "/artall":
            self._artall(q)
        elif u.path == "/runs":
            rows = []
            try:
                with open(RUNS_FILE) as f:
                    for ln in f:
                        ln = ln.strip()
                        if ln:
                            try:
                                rows.append(json.loads(ln))
                            except Exception:
                                pass
            except FileNotFoundError:
                pass
            rows.reverse()  # newest first
            self._send(200, "application/json", json.dumps({"runs": rows, "store": RUNS_FILE}).encode())
        else:
            self._send(404, "text/plain", b"nope")

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/save":
            try:
                ln = int(self.headers.get("Content-Length", 0))
                rec = json.loads(self.rfile.read(ln).decode("utf-8", "ignore"))
                with open(RUNS_FILE, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                self._send(200, "application/json", json.dumps({"ok": True, "store": RUNS_FILE}).encode())
            except Exception as e:
                self._send(200, "application/json", json.dumps({"ok": False, "err": str(e)[:140]}).encode())
        else:
            self._send(404, "text/plain", b"nope")

    def _hw(self, ep):
        out = {"running": None, "waiting": None, "gpu_cache": None, "gpus": [], "ram": None}
        try:
            base = ep.rsplit("/v1", 1)[0]
            raw = urllib.request.urlopen(base + "/metrics", timeout=5).read().decode("utf-8", "ignore")
            for line in raw.splitlines():
                if line.startswith("#"):
                    continue
                if "num_requests_running{" in line:
                    out["running"] = float(line.rsplit(" ", 1)[-1])
                elif "num_requests_waiting{" in line:
                    out["waiting"] = float(line.rsplit(" ", 1)[-1])
                elif "gpu_cache_usage_perc{" in line:
                    out["gpu_cache"] = round(float(line.rsplit(" ", 1)[-1]) * 100, 1)
        except Exception:
            pass
        for sub, (hostspec, key) in NODES.items():
            if sub in ep:
                try:
                    cmd = ("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu "
                           "--format=csv,noheader,nounits; echo ---; free -g | awk '/Mem:/{print $3\"/\"$2}'")
                    r = subprocess.run(["ssh", "-i", key, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                                        "-o", "ConnectTimeout=6", "-o", "StrictHostKeyChecking=no", hostspec, cmd],
                                       capture_output=True, text=True, timeout=12)
                    gsec, _, rsec = r.stdout.partition("---")
                    for ln in gsec.strip().splitlines():
                        p = [x.strip() for x in ln.split(",")]
                        if len(p) == 5:
                            out["gpus"].append({"i": p[0], "util": p[1], "used": p[2], "total": p[3], "temp": p[4]})
                    out["ram"] = rsec.strip() or None
                except Exception:
                    pass
                break
        self._send(200, "application/json", json.dumps(out).encode())

    def _runall(self, q):
        ep = q.get("ep", [""])[0]; model = q.get("model", ["?"])[0]
        prompt = q.get("prompt", ["Say hi."])[0]
        # Optional mixed-prompt mode: prompts = JSON array of strings; run i (1-indexed)
        # gets prompts[(i-1) % len(prompts)]. Falls back to `prompt` when absent/invalid.
        prompts = None
        raw_prompts = q.get("prompts", [""])[0]
        if raw_prompts:
            try:
                pl = json.loads(raw_prompts)
                if isinstance(pl, list) and pl and all(isinstance(x, str) for x in pl):
                    prompts = pl
            except Exception:
                prompts = None
        mx = int(q.get("max_tokens", ["1024"])[0]); temp = float(q.get("temp", ["0.2"])[0])
        think = q.get("think", ["1"])[0] == "1"; n = max(1, min(32, int(q.get("n", ["6"])[0])))
        STOP.clear()
        with CLOCK:
            CONNS.clear()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        evq = queue.Queue()
        for i in range(1, n + 1):
            p_i = prompts[(i - 1) % len(prompts)] if prompts else prompt
            threading.Thread(target=stream_one, args=(i, ep, model, p_i, mx, temp, think, evq), daemon=True).start()
        done = 0
        try:
            while done < n:
                ev = evq.get()
                if ev.get("done") or ev.get("err") or ev.get("killed"):
                    done += 1
                self.wfile.write(("data: " + json.dumps(ev) + "\n\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # browser vanished — slam upstream too so GPUs don't keep churning
            STOP.set()
            with CLOCK:
                for r in CONNS:
                    try:
                        r.close()
                    except Exception:
                        pass
                CONNS.clear()


def _art_prompt(i, n, w, h, shape):
    """Conductor pattern: every agent imagines the SAME full canvas, renders only its column slice.
    For math shapes (wave/sine) the conductor PRECOMPUTES each column's ink row, so slices align
    perfectly even on small models."""
    import math
    x0 = (i - 1) * w // n
    x1 = i * w // n
    sw = x1 - x0
    hint = ""
    s = shape.lower()
    if "wave" in s or "sine" in s or "sin" in s:
        amp = max(1, h // 2 - 2)
        rows = [int(round(h / 2 + amp * math.sin(2 * math.pi * x / w))) for x in range(x0, x1)]
        rows = [min(h - 1, max(0, r)) for r in rows]
        table = ", ".join(f"col {c}→row {r}" for c, r in enumerate(rows))
        hint = (f" I have PRECOMPUTED your ink positions. Your slice has {sw} local columns (0..{sw-1}). "
                f"Place a '#' at exactly these (local column → row) positions, one '#' per column: {table}. "
                f"Every other character in your slice is a space. Do not add any other ink.")
    return (
        f"You are renderer {i} of {n} in a perfectly synchronized ASCII-art grid. "
        f"The GLOBAL canvas is {w} columns wide and {h} rows tall and depicts: {shape}.{hint} "
        f"Coordinate system: column 0 is the far left of the FULL canvas, row 0 is the top. "
        f"First mentally render the ENTIRE {w}x{h} picture, then output ONLY your vertical slice: "
        f"columns {x0} through {x1-1} (slice width {sw}) for ALL {h} rows, top to bottom. "
        f"Use '#' for solid ink, '~' or '.' for soft edges, and spaces for empty background. "
        f"Every other renderer imagines the exact same full picture, so your slice must line up with theirs. "
        f"OUTPUT FORMAT (strict): exactly {h} lines, each line exactly {sw} characters. "
        f"No code fences, no commentary, no blank lines before or after, nothing else."
    )


class ArtH:  # namespace holder (methods bound onto H below)
    def _artall(self, q):
        ep = q.get("ep", [""])[0]; model = q.get("model", ["?"])[0]
        shape = q.get("shape", ["a sine wave"])[0]
        w = max(20, min(240, int(q.get("w", ["120"])[0])))
        h = max(6, min(60, int(q.get("h", ["24"])[0])))
        n = max(1, min(16, int(q.get("n", ["6"])[0])))
        temp = float(q.get("temp", ["0.2"])[0])
        think = q.get("think", ["0"])[0] == "1"
        STOP.clear()
        with CLOCK:
            CONNS.clear()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        evq = queue.Queue()
        for i in range(1, n + 1):
            threading.Thread(target=stream_one,
                             args=(i, ep, model, _art_prompt(i, n, w, h, shape), h * (w // n + 10) + 500, temp, think, evq),
                             daemon=True).start()
        done = 0
        try:
            while done < n:
                ev = evq.get()
                if ev.get("done") or ev.get("err") or ev.get("killed"):
                    done += 1
                self.wfile.write(("data: " + json.dumps(ev) + "\n\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            STOP.set()
            with CLOCK:
                for r in CONNS:
                    try:
                        r.close()
                    except Exception:
                        pass
                CONNS.clear()


H._artall = ArtH._artall

if __name__ == "__main__":
    print(f"Coding Agent Latency Monitor v3 (kill switch) on http://0.0.0.0:{PORT}/")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
