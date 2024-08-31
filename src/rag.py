from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel


class Question(BaseModel):
    question: str


class Rag():
    def __init__(self, embedding_model, vector_store, llm):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 10})
        self.compressor = FlashrankRerank()
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=self.retriever)
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

    def ask(self, question):
        return self.rag_chain_with_source.invoke(question)
