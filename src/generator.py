
import os

import yaml

import time



def load_config():

    with open("configs/config.yaml") as f:

        return yaml.safe_load(f)



class Generator:

    def __init__(self):

        config = load_config()

        self.model_name = config["llm"]["model"]

        self.temperature = config["llm"]["temperature"]

        self.provider = "dummy"



        google_api_key = os.environ.get("GOOGLE_API_KEY")

        if google_api_key:

            try:

                from google import genai

                self.client = genai.Client(api_key=google_api_key)

                self.provider = "gemini"

                print("[generator] Using Gemini")

            except Exception as e:

                print(f"[generator] Gemini unavailable: {e}. Using dummy.")

        else:

            print("[generator] No GOOGLE_API_KEY — using dummy generator")



    def generate(self, question: str, contexts: list) -> str:

        if self.provider == "dummy":

            return contexts[0][:200] if contexts else "No context retrieved."



        context_str = "\n\n".join(contexts)

        prompt = f"""Answer using only the context below. If not in context, say "I don't know."



Context:

{context_str}



Question: {question}

Answer:"""



        try:

            response = self.client.models.generate_content(

                model=self.model_name,

                contents=prompt,

            )

            time.sleep(4)  # stay under 15 RPM free tier

            return response.text

        except Exception as e:

            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):

                print("[generator] Quota hit — returning dummy answer")

                return contexts[0][:200] if contexts else "Quota exhausted."

            raise

