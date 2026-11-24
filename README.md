# Mujica

Find the moments you describe in a long video.

Upload a video and describe the scene you want. Mujica splits the video into chunks and gives each chunk to its own LangChain agent. Each agent reads per-window transcripts, looks at representative frames when it needs to, and reports the time ranges that match. The results are merged and cut into candidate clips you can watch in the browser.

## How it works

| Step | File | What it does |
|---|---|---|
| 1 | `step1_probe_video.py` | Reads duration and checks for an audio track (ffprobe) |
| 2 | `step2_split_timeline.py` | Splits the video into N chunks, each chunk into fixed-length windows |
| 3 | `step3_extract_frames.py` | Extracts one representative frame per window |
| 4 | `step4_transcribe_audio.py` | Transcribes each window's audio with a Whisper server |
| 5 | `step5_chunk_agent.py` | One agent per chunk finds matching ranges (Ollama) |
| 6 | `step6_merge_results.py` | Merges ranges, including ones split across chunk boundaries |
| 7 | `step7_cut_clips.py` | Cuts each range into an mp4 clip |

`main.py` runs the pipeline and `ui.py` is the Gradio web UI.

## Requirements

- [uv](https://docs.astral.sh/uv/) and ffmpeg
- An [Ollama](https://ollama.com) server with:
  - an agent model that supports tool calling (default `qwen3:8b`)
  - a vision model that supports image input (default `qwen2.5vl:7b`)
- An OpenAI-compatible Whisper server (`/v1/audio/transcriptions`), for example [speaches](https://github.com/speaches-ai/speaches)

## Usage

```bash
./setup.sh   # creates .venv with Python 3.12 and installs dependencies
./run.sh     # opens the UI at http://127.0.0.1:7860
```

Server addresses, models, number of agents, window length and Whisper settings can be changed under **Options** in the UI. Candidate clips are saved in `outputs/`.
