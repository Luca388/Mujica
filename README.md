# Mujica

Automated cut-editing agent for voice-intensive content.

Upload a video and describe a scene. Agents read the transcript and frames of each chunk, then return the matching parts as candidate clips.

## Requirements

- [uv](https://docs.astral.sh/uv/) and ffmpeg
- An [Ollama](https://ollama.com) server with:
  - an agent model that supports tool calling (default `qwen3:8b`)
  - a vision model that supports image input (default `qwen2.5vl:7b`)
- An OpenAI-compatible Whisper server. [speaches](https://github.com/speaches-ai/speaches)

## Usage

```bash
./setup.sh   # creates .venv and installs dependencies
./run.sh     # opens the UI at http://127.0.0.1:7860
```
