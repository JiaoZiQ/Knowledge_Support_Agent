import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import Settings, get_settings
from app.eval_runner import EvalRunner
from app.harness import AgentHarness
from app.knowledge import KnowledgeBase
from app.llm import AnswerGenerator
from app.schemas import ChatRequest, ChatResponse, TicketRequest, TicketResponse, UploadResponse
from app.storage import AppStorage
from app.tools import ToolRegistry
from app.vector_store import LocalVectorStore


def create_services(settings: Settings) -> dict[str, object]:
    knowledge_base = KnowledgeBase(settings.knowledge_base_path)
    items = knowledge_base.load()
    vector_store = LocalVectorStore()
    vector_store.build(items)
    storage = AppStorage(settings.database_path)
    tools = ToolRegistry(storage=storage, vector_store=vector_store)
    answer_generator = AnswerGenerator(
        use_openai=settings.use_openai_llm,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.chat_model,
    )
    harness = AgentHarness(
        settings=settings,
        storage=storage,
        tools=tools,
        answer_generator=answer_generator,
    )
    eval_runner = EvalRunner(harness=harness, dataset_path=settings.eval_dataset_path)
    return {
        "settings": settings,
        "knowledge_base": knowledge_base,
        "vector_store": vector_store,
        "storage": storage,
        "tools": tools,
        "harness": harness,
        "eval_runner": eval_runner,
    }


settings = get_settings()
services = create_services(settings)
app = FastAPI(
    title="Knowledge Support Agent",
    description="A customer support RAG Agent with a lightweight Agent Harness.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, object]:
    knowledge_base: KnowledgeBase = services["knowledge_base"]  # type: ignore[assignment]
    return {
        "status": "ok",
        "knowledge_items": len(knowledge_base.items),
        "categories": knowledge_base.summary(),
    }


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON knowledge files are supported in v1.")

    content = await file.read()
    try:
        payload = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        temp_path = Path(handle.name)

    knowledge_base = KnowledgeBase(temp_path)
    items = knowledge_base.load()
    vector_store: LocalVectorStore = services["vector_store"]  # type: ignore[assignment]
    vector_store.build(items)
    services["knowledge_base"] = knowledge_base
    return UploadResponse(loaded_items=len(items), categories=knowledge_base.summary())


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    harness: AgentHarness = services["harness"]  # type: ignore[assignment]
    return harness.chat(request)


@app.post("/tickets", response_model=TicketResponse)
def create_ticket(request: TicketRequest) -> TicketResponse:
    storage: AppStorage = services["storage"]  # type: ignore[assignment]
    ticket_id = storage.create_ticket(
        user_id=request.user_id,
        session_id=request.session_id,
        ticket_type=request.ticket_type,
        description=request.description,
        priority=request.priority,
        metadata=request.metadata,
    )
    return TicketResponse(ticket_id=ticket_id, estimated_response="1-3 个工作日")


@app.get("/tickets")
def list_tickets(limit: int = 50) -> list[dict[str, object]]:
    storage: AppStorage = services["storage"]  # type: ignore[assignment]
    return storage.list_tickets(limit=limit)


@app.get("/sessions/{session_id}")
def session_trace(session_id: str) -> dict[str, object]:
    storage: AppStorage = services["storage"]  # type: ignore[assignment]
    trace = storage.get_session_trace(session_id)
    if not trace["session"]:
        raise HTTPException(status_code=404, detail="Session not found")
    return trace


@app.post("/eval/run")
def run_eval(limit: int | None = None) -> dict[str, object]:
    eval_runner: EvalRunner = services["eval_runner"]  # type: ignore[assignment]
    return eval_runner.run(limit=limit).model_dump()
