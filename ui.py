"""
Mujica 웹 화면. 영상 업로드 / 장면 설명(+옵션) / 편집된 후보, 세 공간으로 나뉜다.
실행: ./run.sh
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr

from common import format_timestamp
from main import find_moments


OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

PROGRESS_REFRESH_SECONDS = 0.5

INITIAL_OPTIONS = {
    "ollama_url": "http://localhost:11434",
    "agent_model": "qwen3:8b",
    "vision_model": "qwen2.5vl:7b",
    "chunk_count": 4,
    "window_seconds": 10.0,
    "frames_per_look": 8,
    "whisper_server_url": "http://localhost:8000",
    "whisper_model": "Systran/faster-whisper-small",
    "language": "ko",
}

LANGUAGE_CHOICES = ["ko", "ja", "en", "zh"]


def find_problem(video_path, request_text, settings):
    if video_path is None:
        return "영상을 먼저 올려 주세요."
    if request_text == "":
        return "찾고 싶은 장면을 설명해 주세요."
    if settings["ollama_url"] == "":
        return "Ollama 서버 주소를 적어 주세요."
    if settings["agent_model"] == "":
        return "에이전트 모델 이름을 적어 주세요."
    if settings["vision_model"] == "":
        return "비전 모델 이름을 적어 주세요."
    if settings["window_seconds"] <= 0:
        return "프레임 간격은 0초보다 커야 합니다."
    if settings["whisper_server_url"] == "":
        return "Whisper 서버 주소를 적어 주세요."
    if settings["whisper_model"] == "":
        return "Whisper 모델 이름을 적어 주세요."
    return None


def show_candidates(request_text, segments):
    """후보들을 갤러리 항목 (클립, 캡션) 목록과 설명 글로 바꾼다."""
    if len(segments) == 0:
        return [], f"\"{request_text}\"에 해당하는 구간을 찾지 못했습니다."

    gallery_items = []
    report_lines = []
    for order, segment in enumerate(segments, start=1):
        time_range = f"{format_timestamp(segment.start)} ~ {format_timestamp(segment.end)}"
        chunk_numbers = ", ".join(str(number) for number in segment.chunk_numbers)

        gallery_items.append((segment.clip_path, f"후보 {order} | {time_range}"))
        report_lines.append(f"**후보 {order}** {time_range} (조각 {chunk_numbers})")
        for reason in segment.reasons:
            report_lines.append(f"- {reason}")
        report_lines.append("")

    return gallery_items, "\n".join(report_lines)


async def find_candidates(
    video_path, request_text,
    ollama_url, agent_model, vision_model,
    agent_count, window_seconds, frames_per_look,
    whisper_server_url, whisper_model, language,
):
    """'후보 찾기' 버튼. (진행 상황, 후보 갤러리, 후보 설명) 을 계속 내보낸다."""
    request_text = request_text.strip()
    settings = {
        "ollama_url": ollama_url.strip(),
        "agent_model": agent_model.strip(),
        "vision_model": vision_model.strip(),
        "chunk_count": int(agent_count),
        "window_seconds": float(window_seconds),
        "frames_per_look": int(frames_per_look),
        "whisper_server_url": whisper_server_url.strip(),
        "whisper_model": whisper_model.strip(),
        "language": language.strip(),
    }

    problem = find_problem(video_path, request_text, settings)
    if problem is not None:
        yield problem, [], ""
        return

    # 같은 초에 두 번 눌러도 폴더가 겹치지 않게 mkdtemp 로 만든다.
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S_")
    output_dir = tempfile.mkdtemp(prefix=run_name, dir=OUTPUTS_DIR)

    latest_progress = {"text": "시작하는 중"}

    async def report_progress(description, done):
        latest_progress["text"] = description

    search = asyncio.create_task(
        find_moments(request_text, video_path, settings, output_dir, report_progress)
    )
    while not search.done():
        yield latest_progress["text"], [], ""
        await asyncio.sleep(PROGRESS_REFRESH_SECONDS)

    segments = search.result()
    gallery_items, report = show_candidates(request_text, segments)
    yield f"완료: 후보 {len(segments)}개", gallery_items, report


def build_options_panel():
    """옵션 칸들을 find_candidates 인자 순서대로 돌려준다."""
    with gr.Accordion("옵션", open=False):
        gr.Markdown("**에이전트**")
        ollama_url = gr.Textbox(label="Ollama 서버 주소", value=INITIAL_OPTIONS["ollama_url"])
        with gr.Row():
            agent_model = gr.Textbox(label="에이전트 모델 (도구 호출 지원)", value=INITIAL_OPTIONS["agent_model"])
            vision_model = gr.Textbox(label="비전 모델 (이미지 입력 지원)", value=INITIAL_OPTIONS["vision_model"])
        agent_count = gr.Slider(
            label="에이전트 수 (= 영상을 나눌 조각 수)",
            minimum=1, maximum=16, step=1,
            value=INITIAL_OPTIONS["chunk_count"],
        )

        gr.Markdown("**프레임**")
        with gr.Row():
            window_seconds = gr.Number(
                label="프레임 간격(초): 대표 프레임 한 장이 맡는 시간",
                minimum=1,
                value=INITIAL_OPTIONS["window_seconds"],
            )
            frames_per_look = gr.Slider(
                label="에이전트가 한 번에 볼 수 있는 프레임 수",
                minimum=1, maximum=16, step=1,
                value=INITIAL_OPTIONS["frames_per_look"],
            )

        gr.Markdown("**받아쓰기**")
        whisper_server_url = gr.Textbox(
            label="Whisper 서버 주소 (OpenAI 호환)",
            value=INITIAL_OPTIONS["whisper_server_url"],
        )
        with gr.Row():
            whisper_model = gr.Textbox(label="Whisper 모델", value=INITIAL_OPTIONS["whisper_model"])
            # Whisper 언어 코드는 ISO 639-1 이라 일본어는 jp 가 아니라 ja 다.
            language = gr.Dropdown(
                label="언어 코드",
                choices=LANGUAGE_CHOICES,
                allow_custom_value=True,
                value=INITIAL_OPTIONS["language"],
            )

    return [
        ollama_url, agent_model, vision_model,
        agent_count, window_seconds, frames_per_look,
        whisper_server_url, whisper_model, language,
    ]


def build_app():
    with gr.Blocks(title="Mujica") as app:
        gr.Markdown("# Mujica")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## 1. 영상 업로드")
                video_input = gr.Video(label="원본 영상", sources=["upload"])

            with gr.Column():
                gr.Markdown("## 2. 장면 설명")
                request_input = gr.Textbox(
                    label="찾고 싶은 장면",
                    placeholder="예: 발표자가 매출 그래프를 설명하는 부분",
                    lines=6,
                )
                option_inputs = build_options_panel()
                find_button = gr.Button("후보 찾기", variant="primary")
                progress_text = gr.Markdown()

        gr.Markdown("## 3. 편집된 후보")
        candidate_gallery = gr.Gallery(label="후보 클립", columns=3, height="auto")
        candidate_report = gr.Markdown()

        find_button.click(
            fn=find_candidates,
            inputs=[video_input, request_input, *option_inputs],
            outputs=[progress_text, candidate_gallery, candidate_report],
        )

    return app


if __name__ == "__main__":
    # Gradio 는 허락한 폴더 밖의 파일은 화면에 내보내지 않는다.
    build_app().launch(allowed_paths=[str(OUTPUTS_DIR)])
