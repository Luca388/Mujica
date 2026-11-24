from pathlib import Path

from common import run_command


def extract_frames(video_path, chunk, work_dir):
    """윈도우마다 한가운데 시점의 프레임을 한 장씩 뽑아 frame_path 에 적어 둔다."""
    for window in chunk.windows:
        frame_path = str(Path(work_dir) / f"chunk{chunk.number:02d}_window{window.index:04d}.jpg")
        middle_time = (window.start + window.end) / 2

        run_command([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{middle_time:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-vf", "scale=640:-2",
            "-q:v", "4",
            frame_path,
        ])
        window.frame_path = frame_path
