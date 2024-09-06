from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


class Question(BaseModel):
    question: str


class Rag():
    def __init__(self, articles, embedding_model, vector_store, llm):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.bm25_retriever = BM25Retriever.from_documents(articles)
        self.faiss_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 10})
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.faiss_retriever], weights=[0.15, 0.85])
        self.compressor = FlashrankRerank(
            top_n=6, model="ms-marco-MiniLM-L-12-v2")
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=self.ensemble_retriever)
        self.prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
                            <context>
                            {context}
                            </context>

                            Question: {input}""")
        self.llm = llm

        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])
        self.rag_chain_with_source = RunnableParallel(
            {"context": self.compression_retriever | format_docs, "input": RunnablePassthrough()}).assign(answer=self.prompt | llm | StrOutputParser())

    def retrieve_context(self, question):
        return self.compression_retriever.invoke(question)

    def ask(self, question):
        return self.rag_chain_with_source.invoke(question)
