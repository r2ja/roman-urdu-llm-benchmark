from .normalize import normalize, normalize_pair
from .chrf import chrf, char_bleu, corpus_chrf
from .markers import MarkerLexicon, detect, variety_label, marker_penalty
from .language_confusion import check_language, LanguageCheck
from .classification import macro_f1, squad_f1

__all__ = [
    "normalize", "normalize_pair",
    "chrf", "char_bleu", "corpus_chrf",
    "MarkerLexicon", "detect", "variety_label", "marker_penalty",
    "check_language", "LanguageCheck",
    "macro_f1", "squad_f1",
]
