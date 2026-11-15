"""Deterministic first-line scope and prompt-injection guard for the Copilot."""
import re

REFUSAL = ("I'm NeuroScan Research Copilot. I can only assist with questions related to your current brain MRI analysis, classification results, segmentation outputs and explainable AI visualizations. Please ask a question related to your NeuroScan analysis.")

_ALLOWED = re.compile(
    r"\b(mri|brain|neuro ?lens|analysis|scan|classification|class|predict|confidence|probabilit|"
    r"glioma|meningioma|pituitary|tumou?r|no[ -]?tumou?r|grad[ -]?cam|heatmap|activation|xai|"
    r"explainab|segmentation|mask|u[ -]?net|efficientnet|densenet|resnet|swin|transformer|"
    r"model|preprocess|inference|feature|layer|saliency|attribution|limitation|uncertaint|"
    r"other model|compare)\b", re.I,
)
_INJECTION = re.compile(
    r"(ignore|override|reveal|show|print|repeat).{0,48}(instructions|prompt|system|api[ _-]?key|"
    r"environment|secret)|act as|unrestricted|jailbreak", re.I,
)
_FORBIDDEN = re.compile(
    r"\b(joke|weather|president|politic|religion|celebrity|recipe|python|javascript|code|"
    r"sort an array|quantum|movie|song|treatment|medication|prescription|dosage)\b", re.I,
)
_FOLLOW_UP = re.compile(r"^(why|how|what about|explain( that| more)?|and the other model)\??$", re.I)


def validate_topic(message: str, has_history: bool = False) -> str | None:
    """Return a rejection/clarification response, or ``None`` when in scope."""
    text = message.strip()
    if _INJECTION.search(text):
        return REFUSAL
    if _FORBIDDEN.search(text):
        return REFUSAL
    if _ALLOWED.search(text) or (has_history and _FOLLOW_UP.match(text)):
        return None
    if len(text.split()) <= 5:
        return "Please clarify how your question relates to the current NeuroScan MRI analysis or its model outputs."
    return REFUSAL
