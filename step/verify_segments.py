"""조각 에이전트들이 제출한 후보를, 후보마다 따로 다시 확인한다.

조각 에이전트는 자기 조각만 보고 찾는 일과 판정을 같이 하기 때문에, 요청의 주제가 스치기만 하는 구간이나
스스로 "해당 없음"이라고 적은 구간도 제출하곤 한다. 여기서는 후보 하나의 받아쓰기와 프레임만 주고
"이 구간이 요청한 장면인가" 하나만 묻는다. 아니라고 답한 후보는 뺀다.
"""

import base64
import re
import sys
from pathlib import Path

from common import format_timestamp
from step.step5_chunk_agent import ask_model


# 후보에 반쯤 걸친 윈도우까지 넣으면 이웃 장면이 섞이므로, 이만큼은 겹쳐야 후보의 윈도우로 본다
MIN_OVERLAP_SECONDS = 1.0
# 받아쓰기를 이 개수보다 많은 윈도우에 대해 보내지 않는다 (10초 윈도우면 20분). 넘으면 고르게 골라 보낸다.
MAX_TRANSCRIPT_WINDOWS = 120

def strip_thinking(text):
    """일부 Ollama 모델은 답 앞에 <think>...</think> 로 생각을 붙여 보낸다."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


PROMPT = """\
영상에서 "{user_request}"에 해당하는 장면을 찾고 있습니다.
아래 구간이 그 장면이라고 제출됐습니다. 정말 요청한 장면인지 판정하세요.

구간: {time_range}

구간의 받아쓰기:
{transcript}

"(노래 구간)"과 "(음악·BGM 구간)"은 말이 아니라 노래나 음악이 나오는 부분이라는 표시이고, 가사는 받아쓰지 않았습니다.
"(음성 없음)"은 그때 말소리가 없었다는 뜻입니다.

첨부한 이미지 {image_count}장은 이 구간에서 고르게 뽑은 프레임입니다.
{image_list}

판정 방법
- 요청한 일이 이 구간 안에서 실제로 일어나야 "맞음"입니다. 구간이 조금 넓은 것은 괜찮습니다.
- 요청의 주제가 말로 언급되기만 하거나, 비슷하지만 다른 일이 일어나는 구간은 "아님"입니다.
- 받아쓰기와 이미지에서 직접 확인한 것만 근거로 삼고, 확인되지 않는 것을 짐작으로 채우지 마세요.
- 노래 구간의 화면 자막(가사)을 사람이 한 말로 보지 마세요.

근거를 두세 문장으로 먼저 쓰고, 마지막 줄에는 아래 둘 중 하나만 쓰세요.
판정: 맞음
판정: 아님
"""


def pick_evenly(items, count):
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (count - 1)
    return [items[round(order * step)] for order in range(count)]


def windows_of(segment, chunks):
    """후보와 겹치는 윈도우들을 시간순으로 돌려준다."""
    overlapping = []
    for chunk in chunks:
        for window in chunk.windows:
            overlap = min(window.end, segment.end) - max(window.start, segment.start)
            if overlap > 0:
                overlapping.append((overlap, window))
    enough = [window for overlap, window in overlapping if overlap >= MIN_OVERLAP_SECONDS]
    if enough:
        return enough
    return [window for overlap, window in overlapping]


def build_prompt(segment, windows, frame_windows, user_request):
    transcript_lines = []
    for window in pick_evenly(windows, MAX_TRANSCRIPT_WINDOWS):
        heard = window.transcript if window.transcript != "" else "(음성 없음)"
        transcript_lines.append(f"[{format_timestamp(window.start)} ~ {format_timestamp(window.end)}] {heard}")

    image_lines = [
        f"{order}번째 이미지 = {format_timestamp((window.start + window.end) / 2)}"
        for order, window in enumerate(frame_windows, start=1)
    ]

    return PROMPT.format(
        user_request=user_request,
        time_range=f"{format_timestamp(segment.start)} ~ {format_timestamp(segment.end)}",
        transcript="\n".join(transcript_lines),
        image_count=len(frame_windows),
        image_list="\n".join(image_lines),
    )


def read_verdict(answer):
    """(맞는지, 근거) 를 돌려준다. 판정 줄을 못 찾으면 맞는지는 None."""
    answer = strip_thinking(answer)
    found = re.findall(r"판정\s*[:：]\s*[*\"'`]*\s*(맞음|아님)", answer)
    if not found:
        return None, answer
    reason = re.split(r"판정\s*[:：]", answer)[0].strip()
    return found[-1] == "맞음", reason


async def verify_segment(vision_model, segment, chunks, user_request, frames_per_look):
    windows = windows_of(segment, chunks)
    if not windows:
        return None, "구간과 겹치는 윈도우가 없습니다."

    frame_windows = pick_evenly(windows, frames_per_look)
    prompt = build_prompt(segment, windows, frame_windows, user_request)
    images = [base64.b64encode(Path(window.frame_path).read_bytes()).decode("ascii") for window in frame_windows]
    return read_verdict(await ask_model(vision_model, prompt, images))


async def verify_segments(vision_model, segments, chunks, user_request, frames_per_look):
    """후보를 하나씩 확인해 (남긴 후보들, 뺀 후보들) 로 나눈다. 뺀 후보에는 rejected_reason 을 적어 둔다.

    판정을 읽지 못한 후보는 남긴다. 검증이 고장 나서 맞는 후보를 잃는 것보다 오답이 남는 편이 낫다.
    """
    kept, rejected = [], []
    for segment in segments:
        # 로컬 모델 서버는 요청을 어차피 하나씩 처리하므로 차례로 묻는다
        matches, reason = await verify_segment(vision_model, segment, chunks, user_request, frames_per_look)
        time_range = f"{format_timestamp(segment.start)} ~ {format_timestamp(segment.end)}"
        if matches is False:
            segment.rejected_reason = reason
            rejected.append(segment)
            print(f"[verify] 뺌 {time_range} | {reason}", file=sys.stderr)
            continue
        if matches is None:
            print(f"[verify] 경고: {time_range} 판정을 읽지 못해 남겨 둠 | {reason[:200]}", file=sys.stderr)
        kept.append(segment)
    return kept, rejected
