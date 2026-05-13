"""
src/generator_gemini.py

Gemini-powered generator for live demo.
Uses gemini-2.5-flash-lite — cheapest stable model as of May 2026.
Cost: $0.10/1M input tokens, $0.40/1M output tokens
~$0.00032 per RAG query → $25 covers ~78,000 queries.

Does NOT replace src/generator.py (Qwen local model stays for Sol training).
Use this for the live chat demo running in Docker on your laptop.
"""

import os
import yaml
from google import genai


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class GeminiGenerator:
    """
    Drop-in replacement for Generator() using Gemini API.
    Same interface: generate(question, contexts) -> str
    """

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Run: export GEMINI_API_KEY=your-key"
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model
        print(f"[GeminiGenerator] Using {self.model}")

    def generate(self, question: str, contexts: list[str]) -> str:
        # Top 3 contexts max to save tokens
        context_str = "\n\n---\n\n".join(contexts[:3])

        prompt = f"""You are a research assistant answering questions about AI papers.
Answer using ONLY the context below. Be concise (2-4 sentences).
If the answer is not in the context, say "I don't know based on the provided papers."

Context:
{context_str}

Question: {question}
Answer:"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            print(f"[GeminiGenerator] Error: {e}")
            return f"Generation failed: {e}"


if __name__ == "__main__":
    gen = GeminiGenerator()
    answer = gen.generate(
        question="What is multi-head attention?",
        contexts=["Multi-head attention allows the model to jointly attend to "
                  "information from different representation subspaces."]
    )
    print(f"Answer: {answer}")