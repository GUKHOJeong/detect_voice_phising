import uvicorn
import os
import shutil
from fastapi import FastAPI, WebSocket, Request, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse
from typing import List
import asyncio

# agent_v2.py와 stt.py에서 필요한 함수들을 가져옵니다.
from project.agent_v2 import run_langgraph_async, FastAPIAlertCallback
from project.stt import stt_async


# --- 1. 웹소켓 연결 관리자 (기존과 동일) ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"프론트엔드 연결: 현재 {len(self.active_connections)}명 접속 중")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"프론트엔드 연결 해제: 현재 {len(self.active_connections)}명 접속 중")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)


manager = ConnectionManager()

# --- 2. FastAPI 앱 설정 (기존과 동일) ---
app = FastAPI()
os.makedirs("public", exist_ok=True)


# --- 3. API 엔드포인트 (기존과 동일) ---


@app.get("/", response_class=HTMLResponse)
async def get_home():
    with open("public/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        manager.disconnect(websocket)


@app.post("/analyze")
async def analyze_audio(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    temp_file_path = f"./temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(run_analysis_in_background, temp_file_path)
    return {"message": "파일 분석을 시작합니다."}


# --- 4. 실제 분석 실행 로직 (병목 현상 해결 + 실시간 재생 적용) ---


async def run_analysis_in_background(file_path: str):
    """
    STT 실행 후, '실시간 대화 전송'과 'LangGraph 분석'을 병렬로 실행하며,
    무거운 분석 작업은 별도 스레드로 분리하여 병목 현상을 해결합니다.
    """
    try:
        await manager.broadcast(
            {
                "type": "STATUS_UPDATE",
                "payload": {"message": "음성 파일을 텍스트로 변환 중입니다..."},
            }
        )
        # stt_async는 이제 'speak_time'을 포함한 세그먼트 리스트를 반환한다고 가정합니다.
        conversation_segments = await stt_async(file_path)
        if not conversation_segments:
            await manager.broadcast(
                {
                    "type": "ERROR",
                    "payload": {"message": "음성을 텍스트로 변환하지 못했습니다."},
                }
            )
            return

        await manager.broadcast(
            {
                "type": "STATUS_UPDATE",
                "payload": {
                    "message": "텍스트 변환 완료. 대화 내용 표시 및 AI 분석을 시작합니다."
                },
            }
        )

        # 작업 1: 프론트엔드에 대화 내용 전송 (사용자님의 아이디어 적용)
        async def stream_transcript_to_frontend(segments: List[dict]):
            print("  -> [메인 스레드] 대화 내용 전송 시작")
            for segment in segments:
                # speak_time이 밀리초(ms) 단위라고 가정하고 초(s) 단위로 변환
                speak_duration_seconds = segment.get("speak_time") / 2
                times = 2 if speak_duration_seconds >= 2 else speak_duration_seconds
                await manager.broadcast(
                    {"type": "TRANSCRIPT_MESSAGE", "payload": segment}
                )
                # 실제 발화 시간의 절반만큼 기다려 더 자연스러운 재생 효과
                await asyncio.sleep(times)
            print("  -> [메인 스레드] 대화 내용 전송 완료")

        # 작업 2: LangGraph 에이전트 분석 (별도 스레드에서 실행)
        async def run_full_agent_analysis_in_thread(segments: List[dict]):
            print("  -> [별도 스레드] LangGraph AI 분석 시작")
            loop = asyncio.get_running_loop()

            transcription = "\n".join(
                [f"{seg['speaker']}: {seg['text']}" for seg in segments]
            )
            # 콜백이 메인 스레드의 루프를 알 수 있도록 전달해야 합니다.
            alert_callback = FastAPIAlertCallback(manager, loop)
            config = {"callbacks": [alert_callback]}

            def sync_agent_runner():
                # 1. 이 스레드만을 위한 새 이벤트 루프를 생성하고 설정합니다.
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                result = None
                try:
                    # 2. 생성한 루프 위에서 비동기 함수가 완료될 때까지 실행합니다.
                    result = loop.run_until_complete(
                        run_langgraph_async(transcription, config)
                    )
                finally:
                    # 3. 작업이 끝나면 반드시 루프를 닫아 리소스를 정리합니다.
                    loop.close()

                return result

            # asyncio.to_thread를 사용하여 동기 래퍼를 별도 스레드에서 실행
            final_state = await asyncio.to_thread(sync_agent_runner)

            # 최종 보고서는 메인 스레드의 루프를 통해 안전하게 전송
            await manager.broadcast(
                {
                    "type": "FINAL_REPORT",
                    "payload": final_state.get(
                        "crime_info", final_state.get("summarize")
                    ),
                    "crime_type": final_state.get("crime_type"),
                }
            )
            print("  -> [별도 스레드] LangGraph AI 분석 및 보고서 전송 완료")

        # asyncio.gather를 사용하여 두 작업을 동시에 시작
        await asyncio.gather(
            stream_transcript_to_frontend(conversation_segments),
            run_full_agent_analysis_in_thread(conversation_segments),
        )
        print("--- 모든 병렬 작업 완료 ---")

    except Exception as e:
        await manager.broadcast(
            {
                "type": "ERROR",
                "payload": {"message": f"분석 중 오류가 발생했습니다: {e}"},
            }
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# --- 서버 실행 ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
