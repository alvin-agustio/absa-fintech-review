"""
Inference Module
=================
Loads the best trained model and provides prediction functions
for the Streamlit dashboard.

Usage (as module):
    from inference import ABSAPredictor
    predictor = ABSAPredictor("models/baseline/model")
    results = predictor.predict(["review text 1", "review text 2"])
"""

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import ASPECTS, ID2LABEL, MAX_LENGTH

try:
    from peft import PeftModel, PeftConfig
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


class ABSAPredictor:
    """ABSA predictor for Risk, Trust, Service Quality aspects."""

    def __init__(self, model_dir: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Auto-detect LoRA vs standard model
        if HAS_PEFT and (model_path / "adapter_config.json").exists():
            config = PeftConfig.from_pretrained(model_dir)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                config.base_model_name_or_path, num_labels=3,
            )
            self.model = PeftModel.from_pretrained(base_model, model_dir)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        self.model.to(self.device)
        self.model.eval()

    def predict_single_aspect(self, texts: list[str], aspect: str) -> list[dict]:
        """Predict sentiment for a single aspect on multiple texts."""
        task_texts = [f"[ASPECT={aspect}] {text}" for text in texts]

        encodings = self.tokenizer(
            task_texts, truncation=True, padding=True,
            max_length=self.max_length, return_tensors="pt",
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            logits = self.model(**encodings).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for i in range(len(texts)):
            pred_id = int(probs[i].argmax())
            results.append({
                "aspect": aspect,
                "sentiment": ID2LABEL[pred_id],
                "confidence": float(probs[i].max()),
                "prob_negative": float(probs[i][0]),
                "prob_neutral": float(probs[i][1]),
                "prob_positive": float(probs[i][2]),
            })
        return results

    def predict(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """
        Predict all 3 aspects for each text.
        Returns list of dicts with keys: review_text, risk, trust, service.
        """
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_results = [{
                "review_text": text,
                "risk": None,
                "trust": None,
                "service": None,
            } for text in batch_texts]

            for aspect in ASPECTS:
                aspect_preds = self.predict_single_aspect(batch_texts, aspect)
                for j, pred in enumerate(aspect_preds):
                    batch_results[j][aspect] = pred

            all_results.extend(batch_results)

        return all_results
