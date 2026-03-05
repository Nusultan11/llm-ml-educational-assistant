import unittest

from llm_ml_assistant.core.prompt_builder import PromptBuilder


class PromptBuilderTests(unittest.TestCase):
    def test_prompt_builder_includes_system_user_and_contexts(self):
        builder = PromptBuilder()
        prompt = builder.build(
            query="What is RAG?",
            contexts=["Context one", "Context two"],
        )

        self.assertIn("[SYSTEM]", prompt)
        self.assertIn("[USER]\nWhat is RAG?", prompt)
        self.assertIn("[CONTEXT 1]\nContext one", prompt)
        self.assertIn("[CONTEXT 2]\nContext two", prompt)
        self.assertTrue(prompt.endswith("[ASSISTANT]\n"))

    def test_prompt_builder_handles_empty_contexts(self):
        builder = PromptBuilder()
        prompt = builder.build(query="Q", contexts=[])
        self.assertIn("[CONTEXT]\n\n\n[USER]\nQ", prompt)


if __name__ == "__main__":
    unittest.main()
