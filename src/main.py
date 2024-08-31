from fastapi import FastAPI
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import torch
import os
from rag import Rag
from pydantic import BaseModel

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs)

vector_store = FAISS.load_local("./data/faiss_index", embedding_model, allow_dangerous_deserialization=True)
llm= ChatOpenAI(model="gpt-4o-mini")

rag= Rag(embedding_model,vector_store,llm)

class Question(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/question/")
async def root(question:Question):
     return rag.ask(question.question)
