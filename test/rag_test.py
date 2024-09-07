
import unittest
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import torch
from rag import Rag
import pickle


class RagTest(unittest.TestCase):

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


    def test_abortion_retireval_relevance(self):
        rag = Rag(self.articles, self.embedding_model, self.vector_store, self.llm)
        result = rag.retrieve_context('What is (are) Abortion ?')
        top_doc=result[0]
        self.assertEqual(top_doc.metadata["id"],0)
        self.assertEqual(top_doc.metadata["ID"],"122")
        self.assertEqual(top_doc.metadata["Title"],'Abortion')
        self.assertEqual(top_doc.metadata["URL"],'https://medlineplus.gov/abortion.html')


if __name__ == '__main__':
    unittest.main()
