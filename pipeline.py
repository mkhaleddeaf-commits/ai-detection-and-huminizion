"""
Pipeline: PDF Text Extraction -> AI Detector -> (optional) Humanizer -> re-check Detector

All 3 components run locally, loaded from:
    project_package/models/ai detection/                 (HF sequence classifier)
    project_package/models/humanizer_t5_small_8epochs/    (HF T5 seq2seq)
    (PDF extraction uses pdfplumber, no trained model needed)

Detector is a DeBERTa-v2 sequence classifier (id2label: 0=human, 1=ai) —
its tokenizer requires the `sentencepiece` package.

Folder layout expected (adjust PROJECT_ROOT below if you move things):
    project_package/
        models/
            ai detection/
                config.json, model.safetensors, tokenizer.json, tokenizer_config.json
            humanizer_t5_small_8epochs/
                config.json, generation_config.json, model.safetensors,
                tokenizer.json, tokenizer_config.json
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import requests


# ---------------------------------------------------------------------------
# Adjust this if you move the project folder
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"E:\staudy\New folder\nti_nlp\project_package")
AI_DETECTOR_PATH = PROJECT_ROOT / "models" / "ai detection"
HUMANIZER_PATH = PROJECT_ROOT / "models" / "humanizer_t5_small_8epochs"


# ---------------------------------------------------------------------------
# 1) TEXT EXTRACTION COMPONENT (plain PDF -> text, no retrieval/model)
# ---------------------------------------------------------------------------

class TextExtractor(ABC):
    @abstractmethod
    def extract(self, source: str) -> str:
        """source is a path to a PDF (or .txt) file. Returns clean text."""
        ...


class LocalPDFExtractor(TextExtractor):
    """Extracts raw text from a PDF using pdfplumber. No model involved."""

    def extract(self, source: str) -> str:
        path = Path(source)

        if path.suffix.lower() == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")

        import pdfplumber

        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        return "\n".join(text_parts).strip()


class APIRAGExtractor(TextExtractor):
    """Kept for later if you ever move extraction behind an API."""

    def __init__(self, endpoint: str, api_key: Optional[str] = None):
        self.endpoint = endpoint
        self.api_key = api_key

    def extract(self, source: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = requests.post(self.endpoint, json={"source": source}, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()["text"]


# ---------------------------------------------------------------------------
# 2) AI DETECTOR COMPONENT
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    label: str            # "human" or "ai"
    score: float          # probability of "ai"
    raw: dict = field(default_factory=dict)


class AIDetector(ABC):
    @abstractmethod
    def detect(self, text: str) -> DetectionResult:
        ...


class LocalAIDetector(AIDetector):
    """
    Loads the fine-tuned sequence classifier from models/ai detection/.
    Assumes a standard HF AutoModelForSequenceClassification checkpoint
    (2 labels: human / ai). If your label order is reversed, flip
    `ai_label_index` below once you confirm it from config.json's id2label.
    """

    def __init__(
        self,
        model_path: str = str(AI_DETECTOR_PATH),
        device: Optional[str] = None,
        threshold: float = 0.5,
        max_length: int = 512,
        ai_label_index: int = 1,  # TODO: confirm against config.json id2label
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.max_length = max_length
        self.ai_label_index = ai_label_index

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, text: str) -> DetectionResult:
        import torch

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        score = probs[self.ai_label_index].item()
        label = "ai" if score >= self.threshold else "human"
        return DetectionResult(label=label, score=score, raw={"probs": probs.tolist()})


class APIAIDetector(AIDetector):
    """Kept for later if you ever move the detector behind an API."""

    def __init__(self, endpoint: str, api_key: Optional[str] = None, threshold: float = 0.5):
        self.endpoint = endpoint
        self.api_key = api_key
        self.threshold = threshold

    def detect(self, text: str) -> DetectionResult:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = requests.post(self.endpoint, json={"text": text}, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        score = data["ai_probability"]
        label = "ai" if score >= self.threshold else "human"
        return DetectionResult(label=label, score=score, raw=data)


# ---------------------------------------------------------------------------
# 3) HUMANIZER COMPONENT
# ---------------------------------------------------------------------------

class Humanizer(ABC):
    @abstractmethod
    def humanize(self, text: str) -> str:
        ...


class LocalHumanizer(Humanizer):
    """Loads the fine-tuned T5-small humanizer from models/humanizer_t5_small_8epochs/."""

    def __init__(
        self,
        model_path: str = str(HUMANIZER_PATH),
        device: Optional[str] = None,
        max_input_tokens: int = 512,
        max_new_tokens: int = 512,
        num_beams: int = 4,
    ):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def humanize(self, text: str) -> str:
        import torch

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=True,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


class APIHumanizer(Humanizer):
    """Kept for later if you ever move the humanizer behind an API."""

    def __init__(self, endpoint: str, api_key: Optional[str] = None):
        self.endpoint = endpoint
        self.api_key = api_key

    def humanize(self, text: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = requests.post(self.endpoint, json={"text": text}, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()["text"]


# ---------------------------------------------------------------------------
# PIPELINE ORCHESTRATION
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    extracted_text: str
    initial_detection: DetectionResult
    humanized_text: Optional[str] = None
    recheck_detection: Optional[DetectionResult] = None


class Pipeline:
    def __init__(
        self,
        extractor: TextExtractor,
        detector: AIDetector,
        humanizer: Optional[Humanizer] = None,
        auto_humanize_if_ai: bool = False,
        recheck_after_humanize: bool = True,
    ):
        self.extractor = extractor
        self.detector = detector
        self.humanizer = humanizer
        self.auto_humanize_if_ai = auto_humanize_if_ai
        self.recheck_after_humanize = recheck_after_humanize

    def run(self, source: str) -> PipelineResult:
        text = self.extractor.extract(source)
        detection = self.detector.detect(text)

        result = PipelineResult(extracted_text=text, initial_detection=detection)

        if detection.label == "ai" and self.auto_humanize_if_ai and self.humanizer:
            humanized = self.humanizer.humanize(text)
            result.humanized_text = humanized
            if self.recheck_after_humanize:
                result.recheck_detection = self.detector.detect(humanized)

        return result


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    extractor = LocalPDFExtractor()
    detector = LocalAIDetector()
    humanizer = LocalHumanizer()

    pipeline = Pipeline(
        extractor=extractor,
        detector=detector,
        humanizer=humanizer,
        auto_humanize_if_ai=True,
        recheck_after_humanize=True,
    )

    pdf_path = r"E:\path\to\your\document.pdf"  # TODO: put a real PDF path here
    result = pipeline.run(pdf_path)

    print("Extracted text (first 300 chars):", result.extracted_text[:300])
    print("Initial detection:", result.initial_detection)
    if result.humanized_text:
        print("Humanized text (first 300 chars):", result.humanized_text[:300])
        print("Re-check detection:", result.recheck_detection)
