from rag import Rag, Question
import json
from redisvl.extensions.llmcache import SemanticCache


class CachedRag(Rag):
    def __init__(self, articles,embedding_model, vector_store, llm):
        super().__init__(articles,embedding_model, vector_store, llm)
        self.llmcache = SemanticCache(name="llmcache",                     # underlying search index name
                                      redis_url="redis://localhost:6379",  # redis connection url string TODO
                                      distance_threshold=0.1               # semantic cache distance threshold
                                      )

    def ask(self, question):
        if response := self.llmcache.check(prompt=question.question):
            # TODO Multiple responses may be returned
            return json.loads(response[0]["response"])
        else:
            response = json.dumps(super().ask(question.question))
            self.llmcache.store(prompt=question.question, response=response)
            return json.loads(response)
