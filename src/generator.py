"""
src/generator.py

Local HuggingFace generator — runs on GPU (A100 on Sol), zero API cost.
Default model: Qwen/Qwen2.5-3B-Instruct

Falls back to dummy (returns first context) if model fails to load.
This keeps the sweep running even if the model isn't cached yet.

FIRST TIME SETUP (run on Sol login node before submitting sweep job):
    python -c "
    from transformers import AutoModelForCausalLM, AutoTokenizer
    AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct', torch_dtype='float16')
    print('Model cached.')
    "
"""

import os
import yaml


def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


class Generator:
    def __init__(self):
        config = load_config()
        self.model_name = config["llm"].get("model", "Qwen/Qwen2.5-3B-Instruct")
        self.max_new_tokens = config["llm"].get("max_tokens", 256)
        self.temperature = config["llm"].get("temperature", 0.0)
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            from transformers import pipeline as hf_pipeline

            device = 0 if torch.cuda.is_available() else -1
            device_name = "A100 GPU" if device == 0 else "CPU"
            print(f"[generator] Loading {self.model_name} on {device_name}...")

            self.pipeline = hf_pipeline(
                "text-generation",
                model=self.model_name,
                device=device,
                torch_dtype=torch.float16,   # half precision — 6GB VRAM for 3B
                trust_remote_code=True,
            )
            print(f"[generator] ✓ Model ready on {device_name}")

        except Exception as e:
            print(f"[generator] ✗ Failed to load {self.model_name}: {e}")
            print(f"[generator] Falling back to dummy (returns first context)")
            print(f"[generator] To fix: pre-download model on login node first")
            self.pipeline = None

    def generate(self, question: str, contexts: list) -> str:
        """Generate an answer from retrieved contexts."""

        # Dummy fallback — returns first context as answer
        # context_hit_rate metric still works correctly with this
        if self.pipeline is None:
            return contexts[0][:300] if contexts else "No context retrieved."

        # Build prompt in Qwen chat format
        context_str = "\n\n".join(contexts[:3])  # top 3 contexts max
        prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't know."

Context:
{context_str}

Question: {question}
Answer:"""

        try:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 0.01),
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
                return_full_text=False,   # only return new tokens, not the prompt
            )
            answer = outputs[0]["generated_text"].strip()
            # Clean up any trailing artifacts
            for stop in ["\nQuestion:", "\nContext:", "\n\n\n"]:
                if stop in answer:
                    answer = answer[:answer.index(stop)].strip()
            return answer

        except Exception as e:
            print(f"[generator] Inference error: {e}")
            return contexts[0][:300] if contexts else "Generation failed."