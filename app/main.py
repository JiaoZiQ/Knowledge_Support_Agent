import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import Settings, get_settings
from app.eval_runner import EvalRunner
from app.harness import AgentHarness
from app.knowledge import KnowledgeBase
from app.llm import AnswerGenerator
from app.embeddings import HashEmbeddingModel, OpenAIEmbeddingModel
from app.schemas import ChatRequest, ChatResponse, TicketRequest, TicketResponse, UploadResponse
from app.storage import AppStorage
from app.tools import ToolRegistry
from app.vector_store import ChromaVectorStore


def create_services(settings: Settings) -> dict[str, object]:
    knowledge_base = KnowledgeBase(settings.knowledge_base_path)
    items = knowledge_base.load()
    if settings.embedding_provider == "openai" and settings.openai_api_key:
        embedding_model = OpenAIEmbeddingModel(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.embedding_model,
        )
    else:
        embedding_model = HashEmbeddingModel()
    vector_store = ChromaVectorStore(
        persist_path=str(settings.chroma_path),
        collection_name=settings.chroma_collection,
        embedding_model=embedding_model,
    )
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


app = FastAPI(
    title="Knowledge Support Agent",
    description="A customer support RAG Agent with a lightweight Agent Harness.",
    version="0.1.0",
)
services: dict[str, object] | None = None


def get_services() -> dict[str, object]:
    global services
    if services is None:
        services = create_services(get_settings())
    return services


@app.get("/health")
def health() -> dict[str, object]:
    app_services = get_services()
    knowledge_base: KnowledgeBase = app_services["knowledge_base"]  # type: ignore[assignment]
    return {
        "status": "ok",
        "knowledge_items": len(knowledge_base.items),
        "categories": knowledge_base.summary(),
        "orchestrator": "langgraph",
        "vector_store": "chroma",
        "embedding_provider": get_settings().embedding_provider,
        "llm_enabled": get_settings().use_openai_llm,
        "chat_model": get_settings().chat_model if get_settings().use_openai_llm else "template-fallback",
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
    app_services = get_services()
    vector_store: ChromaVectorStore = app_services["vector_store"]  # type: ignore[assignment]
    vector_store.build(items)
    app_services["knowledge_base"] = knowledge_base
    return UploadResponse(loaded_items=len(items), categories=knowledge_base.summary())


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    app_services = get_services()
    harness: AgentHarness = app_services["harness"]  # type: ignore[assignment]
    return harness.chat(request)


@app.post("/tickets", response_model=TicketResponse)
def create_ticket(request: TicketRequest) -> TicketResponse:
    app_services = get_services()
    storage: AppStorage = app_services["storage"]  # type: ignore[assignment]
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
    app_services = get_services()
    storage: AppStorage = app_services["storage"]  # type: ignore[assignment]
    return storage.list_tickets(limit=limit)


@app.get("/sessions/{session_id}")
def session_trace(session_id: str) -> dict[str, object]:
    app_services = get_services()
    storage: AppStorage = app_services["storage"]  # type: ignore[assignment]
    trace = storage.get_session_trace(session_id)
    if not trace["session"]:
        raise HTTPException(status_code=404, detail="Session not found")
    return trace


@app.post("/eval/run")
def run_eval(limit: int | None = None) -> dict[str, object]:
    app_services = get_services()
    eval_runner: EvalRunner = app_services["eval_runner"]  # type: ignore[assignment]
    return eval_runner.run(limit=limit).model_dump()
