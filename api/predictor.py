import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ClaimPredictor:
    def __init__(self, model_path: str = "assets/models/llama-1b-claim-ft/step-500/"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.yes_token_ids = []
        self.no_token_ids = []

    def load(self):
        """Load model and tokenizer from disk. Call once at startup."""
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Fine-tuned model not found at {path}")

        model_name = str(path)

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.yes_token_ids = self._label_token_ids(("Yes", " yes", "YES"))
        self.no_token_ids = self._label_token_ids(("No", " no", "NO"))
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded from {model_name} on {self.device}")

    def predict(self, sentence: str) -> dict:
        """Return is_claim bool and confidence float for a sentence."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI agent used to determine whether or not a "
                    "sentence is a factual claim. Only respond with Yes or No"
                ),
            },
            {
                "role": "user",
                "content": f"Is the following sentence a factual claim? {sentence}",
            },
            {"role": "assistant", "content": ""},
        ]

        chat_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            continue_final_message=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        input_ids = chat_inputs.input_ids if hasattr(chat_inputs, "input_ids") else chat_inputs
        input_ids = input_ids[:, :-1].to(self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            next_token_logits = outputs.logits[0, -1]
            no_logit = torch.logsumexp(next_token_logits[self.no_token_ids], dim=0)
            yes_logit = torch.logsumexp(next_token_logits[self.yes_token_ids], dim=0)
            probs = torch.softmax(torch.stack((no_logit, yes_logit)), dim=-1)

        claim_prob = probs[1].item()
        is_claim = claim_prob >= 0.5

        return {
            "is_claim": is_claim,
            "confidence": round(claim_prob if is_claim else 1 - claim_prob, 4),
            "claim_probability": round(claim_prob, 4),
        }

    def _label_token_ids(self, labels: tuple[str, ...]) -> list[int]:
        token_ids = []
        for label in labels:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.append(ids[0])
        if not token_ids:
            raise ValueError(f"Could not tokenize labels: {labels}")
        return sorted(set(token_ids))
