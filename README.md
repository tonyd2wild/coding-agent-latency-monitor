# ⚡ Coding Agent Latency Monitor

A tiny, dependency-free live dashboard that fires **N parallel streaming coding-agent runs** at any **OpenAI-compatible** endpoint (vLLM, SGLang, llama.cpp server, TGI, …) and shows real-time **TTFT · tok/s · E2E · tokens** per run, an aggregate throughput readout, a hardware strip, and a **shareable run-summary** you can copy straight into a post.

Great for stress-testing a local rig (DGX Spark, RTX box, etc.) under concurrent agent traffic and getting honest numbers.

> Inspired by [@sonoda_mj](https://x.com/sonoda_mj)'s parallel-agent DGX Spark bench. Built to make that kind of result one click to reproduce and share.

## Features

- **True parallelism** — all N runs are launched **server-side** and multiplexed back over one connection, so you're not capped by the browser's ~6-connections-per-host limit. Goes up to 32 parallel.
- **Real kill switch** — `Stop` / `🛑 KILL ALL` (and closing the tab) tell the server to slam every upstream model connection shut, so the backend **actually aborts** the requests and your GPUs stop. No runaway jobs after you hit stop.
- **Live per-run metrics** — TTFT, live tok/s, end-to-end time, token count, streaming output.
- **Hardware strip** — pulls vLLM `/metrics` (running/waiting/KV-cache) for any endpoint, and — optionally — **real per-GPU util / memory / temperature via SSH** (see `nodes.json`).
- **Shareable summary** — on finish/stop it computes peak aggregate, sustained avg, per-stream high/low/avg tok/s, total tokens, avg TTFT/E2E, wall time, and gives you a **📋 Copy for sharing** block.
- **Zero dependencies** — pure Python stdlib (`http.server`) + one static HTML file. Runs anywhere Python 3 does.

## Quick start

```bash
# 1. point it at your endpoint(s)
cp presets.example.json presets.json
$EDITOR presets.json          # label -> "base_url|model"

# 2. (optional) real GPU%/RAM/temp via SSH
cp nodes.example.json nodes.json   # host-substring -> [user@host, ssh_key]

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

**`nodes.json`** (optional) — for real per-GPU metrics. Key is a substring of the endpoint host; value is `[user@host, ssh_key_path]`. Needs passwordless SSH + `nvidia-smi` on that host. Omit the file and the hardware strip just uses vLLM `/metrics`.

Both files are **git-ignored** — keep your real hosts/keys out of the repo.

## Notes

- The `thinking` toggle sends `chat_template_kwargs.enable_thinking` (works with models that honor it; harmless otherwise).
- Token counts use the stream's `usage.completion_tokens` when the server sends it, else a `chars/4` estimate for the live readout.
- It only ever *calls* your endpoint — no data leaves your machine except to the endpoint you configure.

## License

MIT — see [LICENSE](LICENSE).
