from spacy.matcher import Matcher
from text_reconstruction import reconstruct_text

def a4d1_3(doc):
    """
    A4.1-3: superlativo relativo
    Level: A2
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"LOWER": {"IN": ["mais", "menos"]}},
        {"POS": "ADV"},
        {"LOWER": "de"},
        {"POS": {"NOT_IN": ["NOUN", "PRON", "PROPN"]}, "SENT_START": False, "OP": "{,3}", "LOWER": {"NOT_IN": ["que"]}},
        {"POS": {"IN": ["NOUN", "PRON", "PROPN"]}}
    ]

    matcher.add("a4d1_3_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a4d1_8(doc):
    """
    A4.1-8: o mais / o menos + adv. + possível
    e.g., "Tenta chegar o mais cedo possível para ficarmos com um bom lugar"
    Level: B2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"LOWER": {"IN": ["o", "a"]}},
        {"LOWER": {"IN": ["mais", "menos"]}},
        {"POS": "ADV"},
        {"NORM": "possivel"}
    ]

    matcher.add("a4d1_8_B2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]