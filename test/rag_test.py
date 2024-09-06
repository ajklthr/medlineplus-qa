
import unittest
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import torch
import os
from ragas.metrics import faithfulness, answer_correctness
from ragas import evaluate
from datasets import Dataset
import sys
from rag import Question, Rag
import pickle


class RagTest(unittest.TestCase):

    def test_abortion_correctness(self):
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

        with open("./data/medline_plus_english_chunked.pkl", "rb") as file:
            articles = pickle.load(file)

        llm = ChatOpenAI(model="gpt-4o-mini")
        rag = Rag(articles, embedding_model, vector_store, llm)
        # result = rag.ask('What is (are) Abortion ?')
        result = rag.retrieve_context('What is (are) Abortion ?')
        print(result)
        result = rag.ask('What is (are) Abortion ?')
        print(result)


if __name__ == '__main__':
    unittest.main()
