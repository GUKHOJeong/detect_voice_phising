import json
import httpx  # requests 대신 httpx 사용
import aiofiles  # 비동기 파일 처리를 위해 사용
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()


# async def 키워드를 사용하여 비동기 함수로 정의
async def stt_async(file_path: str):
    """Clova Speech API를 비동기적으로 호출하여 STT 결과를 반환합니다."""

    # userdata.get 대신 os.getenv 사용을 권장합니다.
    INVOKE_URL = os.getenv("INVOKE_URL")
    SECRET_KEY = os.getenv("SECRET_KEY")

    request_body = {
        "language": "ko-KR",
        "completion": "sync",
        "wordAlignment": True,
        "fullText": True,
        "diarization": {"enable": True},
    }

    # 비동기 HTTP 클라이언트 생성
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            # 비동기 파일 열기
            async with aiofiles.open(file_path, "rb") as f:
                files = {
                    "media": await f.read(),
                    "params": (
                        None,
                        json.dumps(request_body, ensure_ascii=False),
                        "application/json",
                    ),
                }

                headers = {
                    "Accept": "application/json; UTF-8",
                    "X-CLOVASPEECH-API-KEY": SECRET_KEY,
                }

                print("🎙️ STT API 호출 시작...")
                # client.post 대신 await client.post 사용
                response = await client.post(
                    f"{INVOKE_URL}/recognizer/upload", headers=headers, files=files
                )
                response.raise_for_status()  # 200 OK가 아니면 예외 발생

                print("   > STT API 응답 수신 완료.")
                result = response.json()

        except httpx.HTTPStatusError as e:
            print(f"STT API 오류: {e.response.status_code}, {e.response.text}")
            return None
        except Exception as e:
            print(f"STT 처리 중 예외 발생: {e}")
            return None

    # 결과 파싱 부분은 기존 로직과 동일
    full_transcription = []
    for segment in result.get("segments"):
        info = {}

        def convert_ms_to_mm_ss(ms):
            total_seconds = int(ms / 1000)

            minutes = total_seconds // 60

            seconds = total_seconds % 60
            return f"{minutes:02d}:{seconds:02d}"

        info["time"] = (
            f"[{convert_ms_to_mm_ss(segment['start'])} ~ {convert_ms_to_mm_ss(segment['end'])}]"
        )
        info["speaker"] = segment["speaker"]["name"]
        info["text"] = segment["text"]
        info["speak_time"] = int(segment["end"] / 1000) - int(segment["start"] / 1000)

        # 화자별 대화를 한 줄의 텍스트로 만듭니다.
        full_transcription.append(info)

    # 전체 대화 내용을 하나의 문자열로 합쳐서 반환
    return full_transcription
