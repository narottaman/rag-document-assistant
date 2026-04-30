from src.retriever import Retriever
from src.generator import Generator

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def query(self, question: str) -> dict:
        contexts = self.retriever.retrieve(question)
        answer = self.generator.generate(question, contexts)
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts
        }