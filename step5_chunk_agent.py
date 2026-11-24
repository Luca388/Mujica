import base64
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from common import FoundSegment, format_timestamp


AGENT_STEP_LIMIT = 60

# Ollama 기본 컨텍스트는 작아서, 받아쓰기가 길면 앞부분이 소리 없이 잘려 나간다.
OLLAMA_CONTEXT_LENGTH = 32768


SYSTEM_PROMPT = """\
당신은 긴 영상의 한 조각을 맡은 영상 편집 보조입니다.
사용자가 원하는 장면이 당신이 맡은 조각 안에서 몇 초부터 몇 초까지인지 찾아야 합니다.

받는 자료
- 조각 전체를 일정 길이의 '윈도우'로 나눈 목록과, 윈도우별 음성 받아쓰기 결과.
- 윈도우마다 대표 프레임이 한 장씩 있습니다. look_at_frames 도구로 프레임 범위와 질문을 보내면
  프레임을 본 결과를 글로 돌려받습니다.

일하는 방법
1. 받아쓰기를 먼저 읽고 후보가 될 만한 윈도우를 고릅니다.
2. 화면이 중요한 요청이거나 받아쓰기만으로 확실하지 않으면 look_at_frames 로 확인합니다.
   질문은 요청과 관련해 무엇을 확인하고 싶은지 구체적으로 적습니다.
3. 요청에 맞는 구간을 찾을 때마다 submit_segment 로 하나씩 제출합니다.
   시각은 영상 전체 기준 초 단위이며, 반드시 맡은 조각의 범위 안이어야 합니다.
4. 맞는 구간이 없으면 아무것도 제출하지 않고 "해당 없음"이라고 답합니다.
5. 다 끝나면 무엇을 찾았는지 한두 문장으로 정리하고 마칩니다.

구간이 조각의 맨 앞이나 맨 끝에 걸쳐 있으면 조각 경계까지만 제출하세요.
이웃 조각을 맡은 에이전트의 결과와 나중에 자동으로 합쳐집니다.
"""


def create_ollama_model(model_name, ollama_url):
    return ChatOllama(model=model_name, base_url=ollama_url, num_ctx=OLLAMA_CONTEXT_LENGTH)


def build_briefing(chunk, chunk_count, user_request, frames_per_look):
    lines = [
        f"사용자 요청: {user_request}",
        "",
        f"당신이 맡은 조각: {chunk.number}번 / 전체 {chunk_count}개",
        f"조각 범위: {chunk.start:.1f}s ~ {chunk.end:.1f}s "
        f"({format_timestamp(chunk.start)} ~ {format_timestamp(chunk.end)})",
        f"윈도우 수: {len(chunk.windows)}개 (윈도우 번호 1 ~ {len(chunk.windows)})",
        f"look_at_frames 로 한 번에 볼 수 있는 프레임 수: 최대 {frames_per_look}장",
        "",
        "윈도우별 받아쓰기:",
    ]

    for window in chunk.windows:
        transcript = window.transcript
        if transcript == "":
            transcript = "(음성 없음)"
        time_range = f"{format_timestamp(window.start)} ~ {format_timestamp(window.end)}"
        lines.append(f"[윈도우 {window.index}] {time_range} ({window.start:.1f}s ~ {window.end:.1f}s) | {transcript}")

    return "\n".join(lines)


async def ask_vision_model(vision_model, windows, question):
    # Ollama 는 도구 결과 속 이미지를 모델에게 제대로 넘겨준다는 보장이 없어서,
    # 에이전트가 프레임을 직접 보는 대신 비전 모델에게 물어보고 글로 된 답을 받는다.
    instruction = ["아래 이미지들은 영상의 여러 시점에서 뽑은 프레임입니다."]
    for order, window in enumerate(windows, start=1):
        instruction.append(f"{order}번째 이미지 = 윈도우 {window.index} ({window.start:.1f}s ~ {window.end:.1f}s)")
    instruction.append("")
    instruction.append(f"질문: {question}")
    instruction.append("이미지마다 몇 번째 이미지인지 밝히고, 보이는 것을 근거로 짧게 답하세요.")

    content = [{"type": "text", "text": "\n".join(instruction)}]
    for window in windows:
        encoded = base64.b64encode(Path(window.frame_path).read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})

    response = await vision_model.ainvoke([HumanMessage(content=content)])
    return response.content


def make_tools(chunk, found_segments, vision_model, frames_per_look):
    # 도구 인자의 타입 힌트는 LangChain 이 도구 스키마를 만드는 데 쓴다.

    @tool
    async def look_at_frames(first_window: int, last_window: int, question: str):
        """first_window 번부터 last_window 번 윈도우까지(양 끝 포함)의 대표 프레임을 보고 question 에 답한다.
        한 번에 볼 수 있는 프레임 수는 처음 메시지에 적혀 있다."""
        if first_window < 1:
            return "윈도우 번호는 1부터 시작합니다."
        if last_window > len(chunk.windows):
            return f"이 조각의 마지막 윈도우는 {len(chunk.windows)}번입니다."
        if first_window > last_window:
            return "first_window 가 last_window 보다 클 수 없습니다."
        if last_window - first_window + 1 > frames_per_look:
            return f"한 번에 {frames_per_look}장까지만 볼 수 있습니다. 범위를 줄여 주세요."

        windows = chunk.windows[first_window - 1:last_window]
        return await ask_vision_model(vision_model, windows, question)

    @tool
    def submit_segment(start_seconds: float, end_seconds: float, reason: str):
        """요청에 해당하는 구간 하나를 제출한다. 시각은 영상 전체 기준 초 단위.
        reason 에는 이 구간이 왜 요청에 맞는지 짧게 적는다."""
        if start_seconds < chunk.start:
            return f"시작 시각은 조각 시작({chunk.start:.1f}s)보다 앞설 수 없습니다."
        if end_seconds > chunk.end:
            return f"끝 시각은 조각 끝({chunk.end:.1f}s)을 넘을 수 없습니다."
        if start_seconds >= end_seconds:
            return "끝 시각은 시작 시각보다 뒤여야 합니다."

        found_segments.append(FoundSegment(chunk.number, start_seconds, end_seconds, reason))
        return "제출했습니다."

    return [look_at_frames, submit_segment]


async def run_chunk_agent(agent_model, vision_model, chunk, chunk_count, user_request, frames_per_look):
    found_segments = []
    tools = make_tools(chunk, found_segments, vision_model, frames_per_look)
    agent = create_agent(agent_model, tools=tools, system_prompt=SYSTEM_PROMPT)

    briefing = build_briefing(chunk, chunk_count, user_request, frames_per_look)
    await agent.ainvoke(
        {"messages": [{"role": "user", "content": briefing}]},
        config={"recursion_limit": AGENT_STEP_LIMIT},
    )
    return found_segments
