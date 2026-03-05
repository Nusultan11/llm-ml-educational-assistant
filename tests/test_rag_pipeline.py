import unittest

from llm_ml_assistant.core.rag_pipeline import RAGPipeline


class FakeRetriever:
    def __init__(self):
        self.indexed_docs = None
        self.last_query = None

    def index_documents(self, docs):
        self.indexed_docs = docs

    def retrieve(self, query):
        self.last_query = query
        return ["ctx-a", "ctx-b"]


class FakePromptBuilder:
    def __init__(self):
        self.last_query = None
        self.last_contexts = None

    def build(self, query, contexts):
        self.last_query = query
        self.last_contexts = contexts
        return f"PROMPT::{query}::{len(contexts)}"


class FakeGenerator:
    def __init__(self):
        self.last_prompt = None

    def generate(self, prompt):
        self.last_prompt = prompt
        return "final-answer"


class RAGPipelineTests(unittest.TestCase):
    def test_rag_pipeline_index_delegates_to_retriever(self):
        retriever = FakeRetriever()
        pipeline = RAGPipeline(
            retriever=retriever,
            prompt_builder=FakePromptBuilder(),
            generator=FakeGenerator(),
        )

        docs = ["d1", "d2"]
        pipeline.index(docs)

        self.assertEqual(retriever.indexed_docs, docs)

    def test_rag_pipeline_ask_runs_retrieve_build_generate_flow(self):
        retriever = FakeRetriever()
        prompt_builder = FakePromptBuilder()
        generator = FakeGenerator()

        pipeline = RAGPipeline(
            retriever=retriever,
            prompt_builder=prompt_builder,
            generator=generator,
        )

        answer = pipeline.ask("hello")

        self.assertEqual(retriever.last_query, "hello")
        self.assertEqual(prompt_builder.last_query, "hello")
        self.assertEqual(prompt_builder.last_contexts, ["ctx-a", "ctx-b"])
        self.assertEqual(generator.last_prompt, "PROMPT::hello::2")
        self.assertEqual(answer, "final-answer")


if __name__ == "__main__":
    unittest.main()
