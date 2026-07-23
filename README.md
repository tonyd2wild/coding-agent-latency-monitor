# ⚡ Coding Agent Latency Monitor

A tiny, dependency-free live dashboard that fires **N parallel streaming coding-agent runs** at any **OpenAI-compatible** endpoint (vLLM, SGLang, llama.cpp server, TGI, …) and shows real-time **TTFT · tok/s · E2E · tokens** per run, an aggregate throughput readout, a hardware strip, and a **shareable run-summary** you can copy straight into a post.

Great for stress-testing a local rig (DGX Spark, RTX box, etc.) under concurrent agent traffic and getting honest numbers.

> Inspired by [@sonoda_mj](https://x.com/sonoda_mj)'s parallel-agent DGX Spark bench. Built to make that kind of result one click to reproduce and share.

## Features

- **True parallelism** — all N runs are launched **server-side** and multiplexed back over one connection, so you're not capped by the browser's ~6-connections-per-host limit. Goes up to 32 parallel.
- **Real kill switch** — `Stop` / `🛑 KILL ALL` (and closing the tab) tell the server to slam every upstream model connection shut, so the backend **actually aborts** the requests and your GPUs stop. No runaway jobs after you hit stop.
- **Live per-run metrics** — TTFT, live tok/s, end-to-end time, token count, streaming output.
- **Hardware strip** — pulls vLLM `/metrics` (running/waiting/KV-cache) for the current endpoint during a run, and — optionally — **real per-GPU util / memory / temperature via SSH** (see `nodes.json`).
- **🖥️ Live Fleet** — an always-on, command-center-dense view of your **whole rig** at once, polled in parallel (starts on page load, not just during a run). Each device gets a compact header (name + reachable dot + jump badge) followed by **one mini-module per GPU** (GPU 0, GPU 1, … each its own card with an animated util bar, a colored temp pill, and VRAM used/total), plus system RAM. Devices can be **direct** (`ssh -i key user@host`) or reached through a **nested bastion** (`"jump": "jumpuser@jumphost"`): the app ssh's into the jump host, which then reaches the inner node with its own LAN access, so you only need to reach the bastion (ideal for Tailscale-gated or LAN-only cluster nodes). An optional **fabric-switch module** (RouterOS/MikroTik over SSH) shows switch/CPU temps and a dense row of active fabric ports with live Tx/Rx and link speed. Unreachable devices/GPUs/switch dim gracefully as *offline* instead of erroring. Config-driven via `fleet.json` (see below). Backed by `GET /api/fleet` (parallel SSH with short timeouts + a server-side cache so rapid polls never spam SSH; the switch is refreshed on a slow background cadence so a slow jump-host round-trip never blocks the view).
- **🧠 Live Models** — a live card per model that's actually up: it probes each unique preset endpoint's `/v1/models`, and for the ones responding shows the endpoint host plus live **requests running / waiting** and **KV-cache %** from `/metrics`. Shows *No models running* when nothing answers. Same `/api/fleet` poll.
- **Shareable summary** — on finish/stop it computes peak aggregate, sustained avg, per-stream high/low/avg tok/s, total tokens, avg TTFT/E2E, wall time, and gives you a **📋 Copy for sharing** block.
- **Zero dependencies** — pure Python stdlib (`http.server`) + one static HTML file. Runs anywhere Python 3 does.
- **🟩 Matrix Mode** — header toggle that reskins the whole monitor Matrix-style: black everything, phosphor-green glow, digital-rain canvas falling behind your streaming agents. Persists via localStorage.
- **🎨 Art Mode** — the fun experiment: N agents each render one vertical slice of the SAME ASCII canvas (conductor pattern — every agent gets the global spec + its exact column range) and the UI stitches them live into one picture. Doubles as a surprisingly brutal *spatial-sync benchmark*: small models produce chaos, big models produce recognizable shapes. Try `a sine wave` (ink positions get precomputed) or freestyle (`a dog`, `a rocket`).

## Quick start

```bash
# 1. point it at your endpoint(s)
cp presets.example.json presets.json
$EDITOR presets.json          # label -> "base_url|model"

# 2. (optional) real GPU%/RAM/temp via SSH
cp nodes.example.json nodes.json   # simple: host-substring -> [user@host, ssh_key]
# ...or the richer fleet (per-device + jump-host + switch):
cp fleet.example.json fleet.json   # devices[] with optional "jump", plus a "switch" block

# 3. run
python3 server.py             # serves on :7900
# open http://localhost:7900/
```

Set **Parallel**, **Max tokens**, **Temp**, **Thinking**, pick an endpoint, hit **▶ Run**.

## Config

**`presets.json`** — endpoint dropdown. Each value is `"<base_url>|<model_id>"`:
```json
{ "Local vLLM (:8000)": "http://localhost:8000/v1|my-model" }
```

**`nodes.json`** (optional, simple) — powers the per-run **hardware strip** and, when `fleet.json` is absent, the **Live Fleet** view as direct-only nodes. Key is a substring of the endpoint host; value is `[user@host, ssh_key_path]` **or** `[user@host, ssh_key_path, "Display Name"]` (the 3rd element is optional and just labels the card — it defaults to the host). Needs passwordless SSH + `nvidia-smi` on each host. Omit the file and the hardware strip just uses vLLM `/metrics`.
```json
{
  "192.0.2.10": ["user@192.0.2.10", "~/.ssh/id_ed25519", "GPU Rig A"],
  "192.0.2.20": ["user@192.0.2.20", "~/.ssh/id_ed25519"]
}
```

**`fleet.json`** (optional, rich) — the config-driven **Live Fleet**: per-device GPU nodes (direct **or** via a nested bastion; optional `jump_key`/`inner_key`) plus an optional fabric switch. When present it drives the fleet view (and `nodes.json` still powers the per-run hardware strip independently). See `fleet.example.json`. Shape:
```json
{
  "ssh": { "default_key": "~/.ssh/id_ed25519" },
  "devices": [
    { "name": "GPU Rig A", "user": "youruser", "host": "192.0.2.10", "ssh_key": "~/.ssh/id_ed25519", "temp_warn": 70, "temp_hot": 84 },
    { "name": "Spark 1", "user": "youruser", "host": "192.0.2.21", "jump": "bastionuser@bastion.example.local" }
  ],
  "switch": {
    "name": "Fabric Switch", "user": "admin", "host": "192.0.2.254",
    "ssh_key": "~/.ssh/id_ed25519", "jump": "bastionuser@bastion.example.local",
    "ports": ["ether1", "ether2"], "temp_warn": 55, "temp_hot": 70
  }
}
```
- **`devices[]`** — `name` (card label), `user` + `host` (→ `user@host`), optional `ssh_key` (falls back to `ssh.default_key`), optional `jump` = `"jumpuser@jumphost"` (adds `ssh -J …` ProxyJump for bastion-only boxes), optional `port`, optional `temp_warn`/`temp_hot` (°C for the GPU temp pills). Each device renders **one module per GPU**.
- **`switch`** (omit to hide) — a RouterOS/MikroTik device polled READ-ONLY over SSH (`/system health print`, `/system resource print`, `/interface print stats`, `/interface ethernet monitor`). Supports the same `ssh_key`/`jump`/`port`. `ports` lists the fabric interfaces to show (empty = auto-detect running). Returns switch/CPU temps + per-port live Tx/Rx and link speed.

**`PORT`** — the server listens on `7900` by default; override with an env var: `PORT=7905 python3 server.py`.

`presets.json`, `nodes.json`, and `fleet.json` are all **git-ignored** — keep your real hosts/keys out of the repo.

## Run history & Past Runs

Every finished run auto-saves to `runs.jsonl` — model, quant/note, cluster, parallel count, peak & sustained aggregate tok/s, per-stream high/low/avg, TTFT, E2E, total tokens, and timestamp. The **📊 Past Runs** tab renders them in a searchable, model-filterable table with a per-run throughput chart, so nothing is ever lost.

By default runs are stored next to the server. To persist them on an external drive, add a `config.json`:
```json
{ "runs_dir": "/Volumes/YourDrive/coding-agent-monitor-runs" }
```
(`config.json` and `runs.jsonl` are git-ignored.)

## Notes

- The `thinking` toggle sends `chat_template_kwargs.enable_thinking` (works with models that honor it; harmless otherwise).
- Token counts use the stream's `usage.completion_tokens` when the server sends it, else a `chars/4` estimate for the live readout.
- It only ever *calls* your endpoint — no data leaves your machine except to the endpoint you configure.

## License

MIT — see [LICENSE](LICENSE).
