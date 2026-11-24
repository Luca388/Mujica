from pathlib import Path

import httpx

from common import run_command


SERVER_TIMEOUT_SECONDS = 120


class WhisperServer:
    """OpenAI 호환 받아쓰기 서버(/v1/audio/transcriptions)에 맡긴다. speaches, LocalAI, vLLM 등."""

    def __init__(self, server_url, model_name, language):
        self.endpoint = server_url.rstrip("/") + "/v1/audio/transcriptions"
        self.model_name = model_name
        self.language = language

    def transcribe(self, audio_path):
        with open(audio_path, "rb") as audio_file:
            response = httpx.post(
                self.endpoint,
                files={"file": (Path(audio_path).name, audio_file, "audio/wav")},
                data={"model": self.model_name, "language": self.language, "response_format": "json"},
                timeout=SERVER_TIMEOUT_SECONDS,
            )
        response.raise_for_status()
        return response.json()["text"].strip()


def transcribe_windows(video_path, chunk, work_dir, whisper_server):
    """윈도우마다 음성을 잘라 받아쓰고 transcript 에 적어 둔다."""
    for window in chunk.windows:
        audio_path = str(Path(work_dir) / f"chunk{chunk.number:02d}_window{window.index:04d}.wav")

        run_command([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{window.start:.3f}",
            "-t", f"{window.end - window.start:.3f}",
            "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            audio_path,
        ])
        window.transcript = whisper_server.transcribe(audio_path)
