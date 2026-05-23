import re
import unicodedata
from g2p_en import G2p

_g2p = None

PHONEME_VOCAB = [
    "PAD", "AA", "AA0", "AA1", "AA2", "AE", "AE0", "AE1", "AE2",
    "AH", "AH0", "AH1", "AH2", "AO", "AO0", "AO1", "AO2",
    "AW", "AW0", "AW1", "AW2", "AY", "AY0", "AY1", "AY2",
    "B", "CH", "D", "DH", "EH", "EH0", "EH1", "EH2",
    "ER", "ER0", "ER1", "ER2", "EY", "EY0", "EY1", "EY2",
    "F", "G", "HH", "IH", "IH0", "IH1", "IH2",
    "IY", "IY0", "IY1", "IY2", "JH", "K", "L", "M",
    "N", "NG", "OW", "OW0", "OW1", "OW2",
    "OY", "OY0", "OY1", "OY2", "P", "R", "S", "SH",
    "T", "TH", "UH", "UH0", "UH1", "UH2",
    "UW", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
    "spn",
]

_p2i = {p: i for i, p in enumerate(PHONEME_VOCAB)}


def _get_g2p():
    global _g2p
    if _g2p is None:
        _g2p = G2p()
    return _g2p


def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s'.,!?-]", "", text)
    return text


def text_to_phonemes(text):
    g2p = _get_g2p()
    phonemes = g2p(normalize_text(text))
    return [p for p in phonemes if p != " "]


def phonemes_to_ids(phonemes):
    return [_p2i.get(p, _p2i["spn"]) for p in phonemes]


def text_to_ids(text):
    return phonemes_to_ids(text_to_phonemes(text))