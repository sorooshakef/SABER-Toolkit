from spacy.matcher import Matcher
from text_reconstruction import reconstruct_text

def a5d5_8(doc):
    """
    A5.5-8: "Interrogativos - uso / valor - com preposições (a, de, com, por, para...)
    - quem"
    e.g., A quem é que emprestaste o livro?
    Level: A2

    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LOWER": {"IN": ["a", "de", "com", "por", "para"]}},
                {"LOWER": "quem"}
            ]

    matcher.add("a5d5_8_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
a5d5_8.REQUIRES_SPACY = True

def a5d5_9(doc):
    """
    A5.5-9: "Interrogativos - uso / valor - com preposições (a, de, com, por, para...)
    - que, o que"
    e.g., De que estás a falar?
    Level: A2

    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LOWER": {"IN": ["a", "de", "com", "por", "para", "sobre"]}, "IS_SENT_START": True},
                {"LOWER": "o", "OP": "?"},
                {"LOWER": "que"}
            ]

    matcher.add("a5d5_9_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
a5d5_9.REQUIRES_SPACY = True


def a5d5_10(doc):
    """
    A5.5-10: "Interrogativos - uso / valor - com preposições (a, de, com, por, para...)
    - qual, quais"
    e.g., Com qual vais ficar?
    Level: A2

    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LOWER": {"IN": ["a", "de", "com", "por", "para", "sobre"]}},
                {"LEMMA": "qual"}
            ]

    matcher.add("a5d5_10_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a5d5_11(doc):
    """
    A5.5-11: "Interrogativos - uso / valor - com preposições (a, de, com, por, para...) - quanto, quanta, quantos, quantas"
    e.g., Por quanto compraste a tua mochila?
    Level: A2

    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LOWER": {"IN": ["a", "de", "com", "por", "para"]}},
                {"LEMMA": "quanto"}
            ]

    matcher.add("a5d5_11_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a5d5_14(doc):
    """
    A5.5-14: "Interrogativos - posição - posição final"
    e.g., Fazes o quê? Quem disse o quê?
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"NORM": {"IN": ["que", "quem", "porque", "quando", "como", "onde", "quanto", "qual", "aonde", "quais"]}},
                {"TEXT": "?"}
            ]

    matcher.add("a5d5_14_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]