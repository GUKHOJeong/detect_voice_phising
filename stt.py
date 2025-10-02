import json
import httpx
import aiofiles
import os
from dotenv import load_dotenv


load_dotenv()


async def stt_async(file_path: str):
    """Clova Speech API를 비동기적으로 호출하여 STT 결과를 반환합니다."""

    INVOKE_URL = os.getenv("INVOKE_URL")
    SECRET_KEY = os.getenv("SECRET_KEY")

    request_body = {
        "language": "ko-KR",
        "completion": "sync",
        "wordAlignment": True,
        "fullText": True,
        "diarization": {"enable": True},
    }

    async with httpx.AsyncClient(timeout=None) as client:
        try:

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

                response = await client.post(
                    f"{INVOKE_URL}/recognizer/upload", headers=headers, files=files
                )
                response.raise_for_status()

                print("   > STT API 응답 수신 완료.")
                result = response.json()

        except httpx.HTTPStatusError as e:
            print(f"STT API 오류: {e.response.status_code}, {e.response.text}")
            return None
        except Exception as e:
            print(f"STT 처리 중 예외 발생: {e}")
            return None

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

        full_transcription.append(info)

    return full_transcription
