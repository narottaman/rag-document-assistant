from openai import OpenAI
import yaml

def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)

class Generator:
    def __init__(self):
        config = load_config()
        self.client = OpenAI()
        self.model = config["llm"]["model"]
        self.temperature = config["llm"]["temperature"]

    def generate(self, question: str, contexts: list[str]) -> str:
        context_str = "\n\n".join(contexts)
        prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't know."

Context:
{context_str}

Question: {question}
Answer:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content