import unittest
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import torch
from rag import Rag
import pickle
import pandas as pd
from datasets import Dataset
from ragas.metrics import answer_correctness
from ragas import evaluate


class RagasCorrectnessTest(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        device = torch.device("cuda" if torch.cuda.is_available(
        ) else "mps" if torch.backends.mps.is_available() else "cpu")
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': False}
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)

        self.vector_store = FAISS.load_local(
            "./data/faiss_index", self.embedding_model, allow_dangerous_deserialization=True)

        with open("./data/medline_plus_english_chunked.pkl", "rb") as file:
            self.articles = pickle.load(file)

        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def test_rags_answer_correctness(self):
        rag = Rag(self.articles, self.embedding_model,
                  self.vector_store, self.llm)
        df = pd.read_parquet('./test/resources/data/medline_qa_eval.parquet')
        df['answer'] = df['question'].apply(
            lambda question: rag.ask(question)['answer'])
        # https://huggingface.co/docs/datasets/loading
        dataset = Dataset.from_pandas(df)
        scored = evaluate(dataset, metrics=[answer_correctness])

        # Create a score report for further review
        scored.to_pandas().to_json('./out/ragas_correctness_scored.json', index=False)
        avg_score = scored['score'].mean()
        min_score = scored['score'].min()

        expected_avg = 80.0
        expected_min = 70

        self.assertAlmostEqual(avg_score, expected_avg, places=1)
        self.assertEqual(min_score, expected_min)


if __name__ == '__main__':
    unittest.main()
