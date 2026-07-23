#!/usr/bin/env python3
"""Coding Agent Latency Monitor — Kai build 2026-07-06.

v3: REAL KILL SWITCH. All N runs are launched server-side and multiplexed over one
SSE connection (bypasses the browser 6-conn limit). /kill sets a stop flag AND slams
every upstream model connection shut, so vLLM aborts the requests instantly and the
GPUs actually stop (v1/v2 Stop only cut the browser; the server kept generating → GPUs
pinned). Also real per-node GPU%/RAM via SSH for the 3090.
"""
import json, urllib.request, urllib.parse, os, threading, queue, subprocess, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "7900"))
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
# Optional real GPU%/RAM via SSH. Each value is [user@host, ssh_key] OR
# [user@host, ssh_key, "Display Name"] (3rd element optional; defaults to the host).
# See nodes.example.json. Empty -> the hardware strip just shows vLLM /metrics.
def _parse_nodes(raw):
    """substring -> (hostspec, expanded_key, display_name). Backward-compatible with
    2-element [user@host, key] entries; ignores non-list values like a "_comment" key."""
    out = {}
    for k, v in (raw or {}).items():
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        hostspec = v[0]
        key = os.path.expanduser(v[1])
        name = (v[2] if len(v) >= 3 and v[2] else hostspec.split("@")[-1])
        out[k] = (hostspec, key, name)
    return out

NODES = _parse_nodes(_load_json("nodes.json", {}))

# Agent-mode harness profiles: label -> {sys, tools, tool_result} token sizes. These model
# the REAL prefill/context weight a model carries inside an agent harness (system prompt +
# tool schemas + injected tool results) so the felt end-to-end latency is authentic. Content
# is generic filler — prefill cost depends on token COUNT, not text — so nothing proprietary
# is needed or shipped. Override with harness-profiles.json (see harness-profiles.example.json).
HARNESS = _load_json("harness-profiles.json", {
    "Bare endpoint (control)": {"sys": 0, "tools": 0, "tool_result": 0},
    "Light agent (~4k ctx)":   {"sys": 2500, "tools": 1200, "tool_result": 300},
    "Standard agent (~12k ctx)": {"sys": 7000, "tools": 4500, "tool_result": 700},
    "Supervisor (~26k ctx)":   {"sys": 16000, "tools": 8500, "tool_result": 1400},
    "Heavy agent (~46k ctx)":  {"sys": 30000, "tools": 14000, "tool_result": 2200},
})

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

# ---- Live Fleet + Live Models (/api/fleet): every node + every up model, always moving ----
_FLEET_TTL = 2.0                  # cache window so rapid polls don't spam SSH
_FLEET_LOCK = threading.Lock()
_FLEET_CACHE = {"ts": 0.0, "data": None}
_SSH_OPTS = ["-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
             "-o", "ConnectTimeout=6", "-o", "StrictHostKeyChecking=no"]
_NVIDIA_CMD = ("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu "
               "--format=csv,noheader,nounits; echo ---; free -g | awk '/Mem:/{print $3\"/\"$2}'")


def _probe_node(sub, hostspec, key, name, out):
    """SSH one node for per-GPU util/mem/temp + RAM. Same commands/opts as _hw. Fills `out[sub]`;
    leaves it as the pre-seeded offline stub on any failure so unreachable nodes still render."""
    node = {"name": name, "host": hostspec.split("@")[-1], "gpus": [], "ram": None, "up": False}
    try:
        r = subprocess.run(["ssh", "-i", key] + _SSH_OPTS + [hostspec, _NVIDIA_CMD],
                           capture_output=True, text=True, timeout=10)
        gsec, _, rsec = r.stdout.partition("---")
        for ln in gsec.strip().splitlines():
            p = [x.strip() for x in ln.split(",")]
            if len(p) == 5:
                node["gpus"].append({"i": p[0], "util": p[1], "used": p[2], "total": p[3], "temp": p[4]})
        node["ram"] = rsec.strip() or None
        if node["gpus"] or node["ram"]:
            node["up"] = True
            out[sub] = node
    except Exception:
        pass


def _probe_model(name, ep, out):
    """Is this endpoint serving? Hit /v1/models (up check) then /metrics for load. Fills out[ep]
    only when up so callers can drop dead endpoints."""
    try:
        urllib.request.urlopen(ep.rstrip("/") + "/models", timeout=3).read()
    except Exception:
        return
    host = urllib.parse.urlparse(ep).hostname or ep
    m = {"name": name, "endpoint": ep, "host": host, "up": True,
         "running": None, "waiting": None, "kv": None}
    try:
        base = ep.rsplit("/v1", 1)[0]
        raw = urllib.request.urlopen(base + "/metrics", timeout=3).read().decode("utf-8", "ignore")
        for line in raw.splitlines():
            if line.startswith("#"):
                continue
            if "num_requests_running{" in line:
                m["running"] = float(line.rsplit(" ", 1)[-1])
            elif "num_requests_waiting{" in line:
                m["waiting"] = float(line.rsplit(" ", 1)[-1])
            elif "gpu_cache_usage_perc{" in line:
                m["kv"] = round(float(line.rsplit(" ", 1)[-1]) * 100, 1)
    except Exception:
        pass
    out[ep] = m


def _fleet_snapshot():
    """Poll ALL nodes + all unique preset endpoints in parallel (short timeouts), cached ~2s."""
    now = time.time()
    with _FLEET_LOCK:
        c = _FLEET_CACHE
        if c["data"] is not None and (now - c["ts"]) < _FLEET_TTL:
            return c["data"]
    # nodes: pre-seed offline stubs so timed-out/unreachable hosts still appear (dimmed)
    node_out = {sub: {"name": nm, "host": hs.split("@")[-1], "gpus": [], "ram": None, "up": False}
                for sub, (hs, ky, nm) in NODES.items()}
    threads = []
    for sub, (hs, ky, nm) in NODES.items():
        t = threading.Thread(target=_probe_node, args=(sub, hs, ky, nm, node_out), daemon=True)
        t.start(); threads.append(t)
    # models: one probe per UNIQUE endpoint (split preset value on '|')
    eps = {}
    for val in PRESETS.values():
        parts = str(val).split("|")
        ep = parts[0].strip()
        if ep and ep not in eps:
            eps[ep] = (parts[1].strip() if len(parts) > 1 and parts[1].strip() else ep)
    model_out = {}
    mthreads = []
    for ep, mname in eps.items():
        t = threading.Thread(target=_probe_model, args=(mname, ep, model_out), daemon=True)
        t.start(); mthreads.append(t)
    for t in threads:
        t.join(timeout=11)
    for t in mthreads:
        t.join(timeout=5)
    data = {"nodes": [node_out[s] for s in NODES], "models": list(model_out.values()),
            "ts": int(now)}
    with _FLEET_LOCK:
        _FLEET_CACHE["data"] = data
        _FLEET_CACHE["ts"] = time.time()
    return data


def stream_one(idx, ep, model, prompt, mx, temp, think, q):
    ctk = {"enable_thinking": think}
    # MiniMax-M3 uses its own thinking flag, not enable_thinking. Only send it to M3/MiniMax
    # endpoints (sending it to other models errors them out).
    if "minimax" in model.lower() or "m3" in model.lower():
        ctk["thinking_mode"] = "enabled" if think else "disabled"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": mx, "temperature": temp, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": ctk}
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
            _dd = ch.get("delta") or {}
            # reasoning models (M3, DS4, GLM) stream thinking in delta.reasoning with content=null;
            # count those tokens too or the monitor shows nothing during the think phase
            delta = (_dd.get("content") or "") or (_dd.get("reasoning") or "") or (_dd.get("reasoning_content") or "")
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


# ---- 🤖 Agent mode: measure REAL felt end-to-end latency through a harness ----
_FILLER = ("the assistant carefully analyzes each request considers the available tools and the "
           "provided context then responds with a clear correct and well structured answer ").split()


def _filler(tokens):
    """Generic text of ~`tokens` tokens. No proprietary content — prefill cost is token-count
    based, so a size-matched generic prompt measures the identical felt latency as a real one."""
    if tokens <= 0:
        return ""
    words = max(1, int(tokens * 0.75))  # ~1.33 tokens/word for English-ish text
    return " ".join(_FILLER[i % len(_FILLER)] for i in range(words))


def _stream_turn(run, turn, ep, model, messages, cap, temp, think, q):
    """One real streaming chat call. Emits live ttft/deltas; returns timing. Captures the REAL
    prefill size (usage.prompt_tokens) and decode tokens (usage.completion_tokens)."""
    ctk = {"enable_thinking": think}
    if "minimax" in model.lower() or "m3" in model.lower():
        ctk["thinking_mode"] = "enabled" if think else "disabled"
    body = {"model": model, "messages": messages, "max_tokens": cap, "temperature": temp,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": ctk}
    res = {"ttft": None, "decode_s": 0.0, "out_tokens": 0, "in_tokens": None, "text": ""}
    r = None
    t0 = time.time()
    try:
        if STOP.is_set():
            q.put({"run": run, "turn": turn, "killed": True}); return res
        req = urllib.request.Request(ep.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=900)
        with CLOCK:
            CONNS.append(r)
        first = None
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
            _dd = ch.get("delta") or {}
            delta = (_dd.get("content") or "") or (_dd.get("reasoning") or "") or (_dd.get("reasoning_content") or "")
            usage = j.get("usage") or {}
            if usage.get("completion_tokens"):
                res["out_tokens"] = usage["completion_tokens"]
            if usage.get("prompt_tokens"):
                res["in_tokens"] = usage["prompt_tokens"]
            if delta:
                if first is None:
                    first = time.time(); res["ttft"] = first - t0
                    q.put({"run": run, "turn": turn, "ttft": round(res["ttft"], 3)})
                res["text"] += delta
                q.put({"run": run, "turn": turn, "c": delta})
        if res["ttft"] is not None:
            res["decode_s"] = max(0.0, time.time() - (t0 + res["ttft"]))
        if not res["out_tokens"] and res["text"]:
            res["out_tokens"] = max(1, len(res["text"]) // 4)
    except Exception as e:
        q.put({"run": run, "turn": turn, "err": str(e)[:160]})
    finally:
        try:
            if r:
                r.close()
        except Exception:
            pass
    return res


def _agent_one(run, ep, model, prof, prompt, history, turns, tool_ms, final_max, tool_max, temp, think, q):
    """Simulate one agent servicing a request THROUGH a harness: it carries the profile's
    system+tools prefill every turn, does `turns` tool round-trips (model turn -> tool latency
    -> injected tool result), then a final answer turn. All timing is measured against the real
    endpoint, so vLLM prefix-caching applies exactly as in production. Reports true felt E2E."""
    try:
        sys_txt = _filler(prof.get("sys", 0))
        tools_txt = _filler(prof.get("tools", 0))
        system = (sys_txt + ("\n\nAVAILABLE TOOLS:\n" + tools_txt if tools_txt else "")).strip()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        # prior conversation turns (real back-and-forth) grow the prefill each follow-up
        for h in (history or []):
            if isinstance(h, dict) and h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": str(h["content"])})
        messages.append({"role": "user", "content": prompt})
        felt = 0.0; total_out = 0; decode_tps = []; final_text = ""
        total_turns = turns + 1
        for t in range(1, total_turns + 1):
            if STOP.is_set():
                q.put({"run": run, "killed": True}); return
            is_final = (t == total_turns)
            cap = final_max if is_final else tool_max
            q.put({"run": run, "turn": t, "phase": "start", "final": is_final})
            res = _stream_turn(run, t, ep, model, messages, cap, temp, think, q)
            if res["ttft"] is None and not res["text"]:
                q.put({"run": run, "killed": True} if STOP.is_set() else {"run": run, "err": "turn %d empty" % t})
                return
            turn_s = (res["ttft"] or 0) + res["decode_s"]
            felt += turn_s; total_out += res["out_tokens"]
            if res["decode_s"] > 0.05:
                decode_tps.append(res["out_tokens"] / res["decode_s"])
            q.put({"run": run, "turn": t, "phase": "turn_done", "ttft": round(res["ttft"] or 0, 3),
                   "decode_s": round(res["decode_s"], 3), "out": res["out_tokens"], "in": res["in_tokens"],
                   "tps": round(res["out_tokens"] / res["decode_s"], 1) if res["decode_s"] > 0.05 else None})
            if is_final:
                final_text = res["text"] or ""
            if not is_final:
                messages.append({"role": "assistant", "content": res["text"] or "(calling tool)"})
                if tool_ms > 0:
                    q.put({"run": run, "turn": t, "phase": "tool", "tool_ms": tool_ms})
                    time.sleep(tool_ms / 1000.0); felt += tool_ms / 1000.0
                messages.append({"role": "user", "content": "TOOL RESULT:\n" + _filler(prof.get("tool_result", 0))})
        avg_tps = round(sum(decode_tps) / len(decode_tps), 1) if decode_tps else 0
        q.put({"run": run, "phase": "final", "felt_e2e": round(felt, 2), "total_out": total_out,
               "avg_decode_tps": avg_tps, "turns": total_turns, "tool_ms": tool_ms, "answer": final_text})
    except Exception as e:
        q.put({"run": run, "err": str(e)[:160]})


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
        elif u.path == "/api/fleet":
            self._send(200, "application/json", json.dumps(_fleet_snapshot()).encode())
        elif u.path == "/runall":
            self._runall(q)
        elif u.path == "/artall":
            self._artall(q)
        elif u.path == "/harness":
            self._send(200, "application/json", json.dumps(HARNESS).encode())
        elif u.path == "/agentrun":
            self._agentrun(q)
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
        for sub, (hostspec, key, name) in NODES.items():
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
        think = q.get("think", ["1"])[0] == "1"; n = max(1, min(256, int(q.get("n", ["6"])[0])))
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


class AgentH:  # namespace holder (methods bound onto H below)
    def _agentrun(self, q):
        ep = q.get("ep", [""])[0]; model = q.get("model", ["?"])[0]
        prompt = q.get("prompt", ["Write a function to merge two sorted lists."])[0]
        prof_name = q.get("profile", ["Standard agent (~12k ctx)"])[0]
        prof = HARNESS.get(prof_name, {"sys": 0, "tools": 0, "tool_result": 0})
        turns = max(0, min(12, int(q.get("turns", ["3"])[0])))
        tool_ms = max(0, min(10000, int(q.get("tool_ms", ["300"])[0])))
        final_max = int(q.get("max_tokens", ["512"])[0])
        tool_max = max(8, min(512, int(q.get("tool_max", ["64"])[0])))
        temp = float(q.get("temp", ["0.2"])[0]); think = q.get("think", ["0"])[0] == "1"
        n = max(1, min(256, int(q.get("n", ["1"])[0])))
        # prior conversation (for follow-up messages in the chat) — JSON [{role,content},...]
        history = []
        raw_hist = q.get("history", [""])[0]
        if raw_hist:
            try:
                hl = json.loads(raw_hist)
                if isinstance(hl, list):
                    history = hl[-40:]  # cap
            except Exception:
                history = []
        STOP.clear()
        with CLOCK:
            CONNS.clear()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        evq = queue.Queue()
        for i in range(1, n + 1):
            threading.Thread(target=_agent_one,
                             args=(i, ep, model, prof, prompt, history, turns, tool_ms, final_max, tool_max, temp, think, evq),
                             daemon=True).start()
        done = 0
        try:
            while done < n:
                ev = evq.get()
                if ev.get("phase") == "final" or (("err" in ev or ev.get("killed")) and "turn" not in ev):
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


H._agentrun = AgentH._agentrun

if __name__ == "__main__":
    print(f"Coding Agent Latency Monitor v3 (kill switch) on http://0.0.0.0:{PORT}/")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
