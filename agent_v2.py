import os
import io
import json
from openai import OpenAI
from typing import TypedDict, List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
import pickle
from langchain.retrievers import EnsembleRetriever
from langchain_teddynote.retrievers import KiwiBM25Retriever
import faiss
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import cohere
import asyncio  # 변경점: asyncio 임포트
import asyncio
from typing import TypedDict, List, Dict, Any, TYPE_CHECKING
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI


if TYPE_CHECKING:
    from main import ConnectionManager
# --- 1. 환경 설정 및 전역 리소스 초기화 ---
load_dotenv()
open_api = os.getenv("openai")
llm = ChatOpenAI(model="gpt-5", temperature=0, api_key=open_api)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=open_api)

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

DB_PATH = os.path.join(SCRIPT_DIR, "voice_phising_v3")
BM25_OLD_PATH = os.path.join(SCRIPT_DIR, "bm25_old_v2.pkl")
BM25_NEW_PATH = os.path.join(SCRIPT_DIR, "bm25_new_v2.pkl")

# FAISS 벡터 DB 및 BM25 리트리버 로드
try:
    loaded_vector_db = FAISS.load_local(
        DB_PATH, embeddings, allow_dangerous_deserialization=True
    )
    with open(BM25_OLD_PATH, "rb") as f:
        bm25_old_retriever = pickle.load(f)
    with open(BM25_NEW_PATH, "rb") as f:
        bm25_new_retriever = pickle.load(f)
    RAG_RESOURCES_LOADED = True
except FileNotFoundError as e:
    print(
        f"오류: RAG에 필요한 파일을 찾을 수 없습니다. ({e}). 'test' 폴더 내에 파일들이 있는지 확인하세요."
    )
    RAG_RESOURCES_LOADED = False


class FastAPIAlertCallback(BaseCallbackHandler):
    def __init__(self, manager: "ConnectionManager", loop: asyncio.AbstractEventLoop):
        self.manager = manager
        self.loop = loop

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        if "dangerous" in outputs:
            level = outputs["dangerous"]
            alert_signal = 1 if "위험군" in level else 0
            message = "위험 감지" if alert_signal == 1 else "안전"
            payload = {
                "type": "ALERT",
                "payload": {"level": alert_signal, "message": message},
            }

            # 별도 스레드에서 메인 스레드로 작업을 안전하게 보냅니다.
            asyncio.run_coroutine_threadsafe(self.manager.broadcast(payload), self.loop)
            print(f"✅ [Callback/별도 스레드] 신호 전송 요청: {payload}")


# class FastAPIAlertCallback(BaseCallbackHandler):
#     """'danger' 노드 결과에 따라 FastAPI를 통해 프론트엔드에 신호를 보냅니다."""

#     def __init__(self, manager: "ConnectionManager"):
#         # main.py의 웹소켓 매니저 객체를 직접 주입받습니다.
#         self.manager = manager

#     def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
#         # danger 노드가 끝났을 때만 반응합니다.
#         if "dangerous" in outputs:
#             level = outputs["dangerous"]

#             # 프론트엔드에 보낼 신호 (0: 정상, 1: 위험)
#             alert_signal = 1 if "위험군" in level else 0
#             message = "위험 감지" if alert_signal == 1 else "안전"

#             payload = {
#                 "type": "ALERT",
#                 "payload": {"level": alert_signal, "message": message},
#             }

#             try:
#                 # 현재 실행 중인 비동기 이벤트 루프를 찾아 작업을 예약합니다.
#                 loop = asyncio.get_running_loop()
#                 loop.create_task(self.manager.broadcast(payload))
#                 print(f"✅ [Callback] 프론트엔드로 신호 전송 예약: {payload}")
#             except RuntimeError:
#                 asyncio.run(self.manager.broadcast(payload))


# --- 2. LangGraph 상태 및 Pydantic 모델 정의 ---
class State(TypedDict):
    context: str
    dangerous: str
    summarize: str
    crime_type: str
    retrive_result: List[Document]
    retrive_source: List[Dict]
    crime_info: str
    emotion_control: Dict


class TacticAnalysis(BaseModel):
    is_detected: bool = Field(description="해당 기법이 탐지되었는지 여부 (true/false)")
    reasoning: str = Field(description="탐지 근거가 되는 대화 내용 및 분석 이유")


class EmotionAnalysisResult(BaseModel):
    urgency_pressure: TacticAnalysis = Field(
        description="긴급성 및 시간 압박 조성 분석"
    )
    fear_intimidation: TacticAnalysis = Field(description="공포감 및 위협 조성 분석")
    authority_abuse: TacticAnalysis = Field(
        description="기관 사칭 등 권위를 이용한 압박 분석"
    )
    isolation_secrecy: TacticAnalysis = Field(
        description="피해자를 고립시키고 비밀을 강요하는 수법 분석"
    )
    trust_building: TacticAnalysis = Field(
        description="피해자와의 신뢰를 급격히 형성하려는 시도 분석"
    )
    final_verdict: str = Field(
        description="종합적인 위험도 판단 ('정상 대화', '의심 정황', '고위험 피싱')"
    )
    summary: str = Field(description="분석 결과에 대한 최종 요약")


# 변경점: Cohere 클라이언트를 비동기(AsyncClient)로 초기화
co = cohere.AsyncClient(os.getenv("api_key"))


# 변경점: reranker 함수를 async def로 전환
async def reranker(input_query: str, context_docs: List[Document]) -> List[Document]:
    """Cohere Reranker를 사용하여 Document 리스트를 재정렬하고 상위 결과를 Document 리스트로 반환합니다."""
    print("  - Executing Reranker...")
    if not context_docs:
        return [], []

    doc_contents = [doc.page_content for doc in context_docs]

    # 변경점: co.rerank를 await으로 비동기 호출
    reranked_results = await co.rerank(
        model="rerank-multilingual-v3.0",
        query=input_query,
        documents=doc_contents,
        top_n=3,
    )

    rank_list = []
    rank_resource = []
    for result in reranked_results.results:
        original_index = result.index
        rank_list.append(context_docs[original_index].page_content)
        rank_resource.append(context_docs[original_index].metadata)
        print(
            f"    - Reranked doc found: score={result.relevance_score:.4f}, index={original_index}"
        )

    return rank_list, rank_resource


# --- 3. LangGraph 노드 정의 ---
# 변경점: danger 함수를 async def로 전환, chain.invoke -> await chain.ainvoke
async def danger(state: State):
    print("Executing Node: danger")

    prompt = ChatPromptTemplate.from_template(
        "다음 {input}를 보고 해당 상황이 보이스피싱에 유사하면 '위험군', 그렇지 않으면 '정상'으로만 출력해주세요."
    )
    template = "다음 보이스피싱 관련 대화 내용을 RAG 시스템으로 검색하기 위한 핵심 요약문을 한 문장으로 만들어 주세요.\n\n대화 내용:\n{conversation_script}\n\n요약 쿼리:"
    prompt2 = ChatPromptTemplate.from_template(template)

    chain = prompt | llm | StrOutputParser()
    chain2 = prompt2 | llm | StrOutputParser()

    # 변경점: .invoke를 .ainvoke로 비동기 병렬 실행
    dangerous_result, summarize_result = await asyncio.gather(
        chain.ainvoke({"input": state["context"]}),
        chain2.ainvoke({"conversation_script": state["context"]}),
    )
    return {"dangerous": dangerous_result, "summarize": summarize_result}


# 변경점: retrive 함수를 async def로 전환 및 비동기 호출 사용
async def retrive(state: State):
    print("Executing Node: retrive")
    # ... (리트리버 설정은 동일) ...
    bm25_old_retriever.k = 5
    old_retriever = loaded_vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.34},
    )
    old_ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_old_retriever, old_retriever], weights=[0.5, 0.5]
    )
    bm25_new_retriever.k = 5
    new_retriever = loaded_vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "filter": {"source": "new"}, "score_threshold": 0.15},
    )
    new_ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_new_retriever, new_retriever], weights=[0.5, 0.5]
    )

    # 변경점: retriever.invoke -> await retriever.ainvoke
    old_docs_results, new_docs_results = await asyncio.gather(
        old_retriever.ainvoke(state["summarize"]),
        new_retriever.ainvoke(state["summarize"]),
    )
    old_docs = [doc.page_content for doc in old_docs_results]
    new_docs = [doc.page_content for doc in new_docs_results]

    if not old_docs and not new_docs:
        print("-> 검색된 문서가 없습니다.")
        return {"dangerous": "안전군"}  # 상태를 '안전군'으로 변경하여 라우팅에 사용

    if not new_docs:
        print("-> 이전 방식의 문서만 검색되었습니다.")
        full_context = await old_ensemble_retriever.ainvoke(state["summarize"])
        reranked_result, reranked_source = await reranker(
            state["summarize"], full_context
        )
        return {
            "crime_type": "이전 방식",
            "retrive_result": reranked_result,
            "retrive_source": reranked_source,
        }

    if not old_docs:
        print("-> 최신 방식의 문서만 검색되었습니다.")
        full_context = await new_ensemble_retriever.ainvoke(state["summarize"])
        reranked_result, reranked_source = await reranker(
            state["summarize"], full_context
        )
        return {
            "crime_type": "최신 방식",
            "retrive_result": reranked_result,
            "retrive_source": reranked_source,
        }

    print("-> 이전 방식과 최신 방식의 문서가 모두 검색되었습니다. 점수를 비교합니다.")
    # 변경점: co.rerank 비동기 호출
    old_rerank_scores, new_rerank_scores = await asyncio.gather(
        co.rerank(
            model="rerank-multilingual-v3.0",
            query=state["summarize"],
            documents=old_docs,
            top_n=1,
        ),
        co.rerank(
            model="rerank-multilingual-v3.0",
            query=state["summarize"],
            documents=new_docs,
            top_n=1,
        ),
    )
    top_old_score = (
        old_rerank_scores.results[0].relevance_score if old_rerank_scores.results else 0
    )
    top_new_score = (
        new_rerank_scores.results[0].relevance_score if new_rerank_scores.results else 0
    )
    print(f"  - Top scores: Old={top_old_score:.4f}, New={top_new_score:.4f}")

    if top_old_score >= top_new_score:
        print("-> 이전 방식의 문서가 더 관련성이 높습니다.")
        full_context = await old_ensemble_retriever.ainvoke(state["summarize"])
        reranked_result, reranked_source = await reranker(
            state["summarize"], full_context
        )
        return {
            "crime_type": "이전 방식",
            "retrive_result": reranked_result,
            "retrive_source": reranked_source,
        }
    else:
        print("-> 최신 방식의 문서가 더 관련성이 높습니다.")
        full_context = await new_ensemble_retriever.ainvoke(state["summarize"])
        reranked_result, reranked_source = await reranker(
            state["summarize"], full_context
        )
        return {
            "crime_type": "최신 방식",
            "retrive_result": reranked_result,
            "retrive_source": reranked_source,
        }


# 변경점: emotion_analysis_node를 async def로 전환, chain.invoke -> await chain.ainvoke
async def emotion_analysis_node(state: State):
    print("Executing Node: emotion_analysis")
    parser = JsonOutputParser(pydantic_object=EmotionAnalysisResult)
    prompt = ChatPromptTemplate.from_template(
        """당신은 보이스피싱 사기의 심리적 기법을 분석하는 범죄 심리 분석 전문가입니다.
        주어진 대화 내용을 분석하여 명시된 심리 조작 기법이 사용되었는지 판단하고, 최종 의견을 JSON 형식으로 제공해주세요.
        **분석할 대화 내용:** --- {conversation} ---
        **분석 지침:** 각 항목에 대해 탐지 여부(is_detected), 그리고 맥락과 상황을 충분히 고려한 판단 근거(reasoning)를 반드시 포함해야 합니다. 판단 근거는 대화의 특정 문장을 인용하여 구체적으로 작성해주세요.
        {format_instructions}""",
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    analysis_result = await chain.ainvoke({"conversation": state["context"]})
    print("-> 심리 분석 완료. 최종 판단:", analysis_result.get("final_verdict"))
    return {"emotion_control": analysis_result}


# 변경점: info 함수를 async def로 전환, chain.invoke -> await chain.ainvoke
async def info(state: State):
    print("Executing Node: info")

    retrieved_docs = state.get("retrive_result")
    retrieved_meta = state.get("retrive_source")

    # 검색 결과가 없을 경우를 대비한 방어 코드
    if not retrieved_docs or not retrieved_meta:
        return {"crime_info": "검색된 관련 사례가 없어 상세 분석을 진행할 수 없습니다."}

    top_doc = retrieved_docs[0]
    case_contents = [f"**핵심 사례 내용:**\n{top_doc}"]
    other_docs = retrieved_docs[1:]
    if other_docs:
        for i, doc in enumerate(other_docs, 2):
            case_contents.append(f"**참고 사례 {i} 내용:**\n{doc}")
    crime_info_text = "\n\n---\n\n".join(case_contents)

    url = retrieved_meta[0].get("url", "")
    source_link_markdown = f"사례 원문 바로가기({url})" if url else "정보 없음"

    emotion_control_data = state.get("emotion_control", {})
    detected_tactics = []
    tactic_names = {
        "urgency_pressure": "긴급성 및 시간 압박",
        "fear_intimidation": "공포감 및 위협 조성",
        "authority_abuse": "기관 사칭 및 권위 남용",
        "isolation_secrecy": "고립 및 비밀 강요",
        "trust_building": "신뢰 형성 시도",
    }
    for key, name in tactic_names.items():
        tactic = emotion_control_data.get(key)
        if isinstance(tactic, dict) and tactic.get("is_detected"):
            reason = tactic.get("reasoning", "근거 없음")
            detected_tactics.append(f"- **{name}**: {reason}")

    emotion_analysis_str = (
        "\n".join(detected_tactics)
        if detected_tactics
        else "탐지된 주요 심리 조작 기법이 없습니다."
    )

    # ✅ 사용자님의 새 아이디어를 반영하여 수정한 프롬프트
    final_prompt_template = """당신은 보이스피싱 패턴과 범죄 심리를 모두 분석하는 통합 분석 전문가입니다.
    주어진 정보들을 종합적으로 검토하여, 아래 형식에 맞춰 최종 분석 보고서를 작성합니다.

    1. **가장 유사한 범죄 유형**
        [검색된 실제 피해사례]의 유형을 명시합니다.
    2. **유사점 분석**
        [검색된 실제 피해사례]와 [사용자의 현재 상황]을 비교 분석합니다.
    3. **주요 심리 조작 기법**
        [심리 분석 결과]를 바탕으로 설명합니다.
    4. **종합 결론 및 대응 방안**
        모든 내용을 종합하여 최종 판단과 대응 방안을 제시합니다.
    5. **관련 자료 출처**
        [관련정보]내용 그대로 사용합니다. 임의로 추가 하지 마세요.

    ---
    [검색된 실제 피해사례]:
    {context}
    ---
    [심리 분석 결과]:
    {emotion_analysis}
    ---
    [사용자의 현재 상황]:
    {summarize}
    ---
    [관련정보]:
    {source_link}
    """
    prompt = ChatPromptTemplate.from_template(final_prompt_template)
    chain = prompt | llm | StrOutputParser()

    final_report = await chain.ainvoke(
        {
            "context": crime_info_text,
            "summarize": state["summarize"],
            "emotion_analysis": emotion_analysis_str,
            "source_link": source_link_markdown,
        }
    )
    return {"crime_info": final_report}


# --- 4. LangGraph 그래프 구성 및 라우팅 함수 ---
# 변경점: 라우팅 함수를 명확한 이름으로 변경
def should_proceed_to_analysis(state: State):
    if state["dangerous"] == "위험군":
        print("-> '위험군'으로 판단. RAG 및 심리 분석을 병렬 실행합니다.")
        return ["retrive", "emotion"]
    else:
        print("-> '정상'으로 판단. 분석을 종료합니다.")
        return END


def route_after_retrieval(state: State):
    if state.get("dangerous") == "안전군":
        print("-> 검색된 문서가 없어 '안전군'으로 재분류. 분석을 종료합니다.")
        return "stop"
    else:
        print("-> 문서 검색 성공. 최종 보고서 생성을 위해 'info' 노드로 이동합니다.")
        return "proceed"


# 변경점: 논의된 최종 그래프 구조로 재구성
graph = StateGraph(State)
graph.add_node("danger", danger)
graph.add_node("retrive", retrive)
graph.add_node("emotion", emotion_analysis_node)
graph.add_node("info", info)

graph.set_entry_point("danger")
graph.add_conditional_edges("danger", should_proceed_to_analysis)
graph.add_conditional_edges(
    "retrive", route_after_retrieval, {"proceed": "info", "stop": END}
)
graph.add_edge("emotion", "info")
graph.add_edge("info", END)

app = graph.compile()


async def run_langgraph_async(text, config: dict = None):
    initial_state = {"context": text}
    final_state = await app.ainvoke(initial_state, config=config)
    return final_state
