import json

from common import run_command


def probe_video(video_path):
    """영상 길이(초)와 음성 트랙이 있는지를 돌려준다."""
    output = run_command([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=codec_type",
        "-of", "json",
        video_path,
    ])
    info = json.loads(output)

    duration = float(info["format"]["duration"])
    stream_types = [stream["codec_type"] for stream in info["streams"]]
    has_audio = "audio" in stream_types
    return duration, has_audio
