from common import Chunk, Window


def split_timeline(duration, chunk_count, window_seconds):
    """
    영상을 같은 길이의 조각 chunk_count 개로 나누고, 조각마다 window_seconds 길이의 윈도우를 채운다.
    파일은 자르지 않고 시간 범위만 정한다.
    """
    chunk_length = duration / chunk_count
    chunks = []

    for chunk_index in range(chunk_count):
        start = chunk_index * chunk_length
        end = start + chunk_length
        if chunk_index == chunk_count - 1:
            # 나눗셈 오차로 마지막 조각이 영상 끝에 조금 못 미치는 걸 막는다.
            end = duration

        chunk = Chunk(chunk_index + 1, start, end)
        chunk.windows = split_into_windows(start, end, window_seconds)
        chunks.append(chunk)

    return chunks


def split_into_windows(start, end, window_seconds):
    windows = []
    window_start = start

    while window_start < end:
        window_end = min(window_start + window_seconds, end)
        windows.append(Window(len(windows) + 1, window_start, window_end))
        window_start = window_end

    return windows
