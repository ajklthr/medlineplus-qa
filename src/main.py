from fastapi import FastAPI
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import torch
import os
from rag import Question
from cache import CachedRag
from nemoguardrails import LLMRails
from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.kb.kb import KnowledgeBase
from langchain.llms.base import BaseLLM
from promptflow.tracing import start_trace
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from promptflow.tracing import start_trace

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu")

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs)

vector_store = FAISS.load_local(
    "./data/faiss_index", embedding_model, allow_dangerous_deserialization=True)
llm = ChatOpenAI(model="gpt-4o-mini")

rag = CachedRag(embedding_model, vector_store, llm)


config = RailsConfig.from_path("./config")
rails = LLMRails(config)

# instrument OpenAI
start_trace()
tracer_provider = trace.get_tracer_provider()
otlp_span_exporter = OTLPSpanExporter()
tracer_provider.add_span_processor(BatchSpanProcessor(otlp_span_exporter))

instrumentor = LangchainInstrumentor()
if not instrumentor.is_instrumented_by_opentelemetry:
    instrumentor.instrument()


async def rag_action(context: dict, llm: BaseLLM, kb: KnowledgeBase) -> ActionResult:
    user_message = context.get("last_user_message")
    context_updates = {}

    response = rag.ask(Question(question=user_message))
    # 💡 Store the chunks for fact-checking
    context_updates["relevant_chunks"] = response['context']
    answer = response['answer']

    return ActionResult(return_value=answer, context_updates=context_updates)

rails.register_action(rag_action, "rag_action")


@app.post("/question/")
async def root(question: Question):
    response = await rails.generate_async(messages=[{
        "role": "user",
        "content": question.question
    }])
    return response['content']
