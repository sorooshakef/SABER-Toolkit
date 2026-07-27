from spacy.matcher import Matcher, PhraseMatcher
from process_and_display import nlp_stanza
from text_reconstruction import reconstruct_text

def a7d3_3(doc):
    """
    A7.3-3: Numerais - forma - multiplicativos - em expressões com cardinal + vez/vezes
    e.g., duas/dez/cem vezes
    Level: A2

    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"POS": "NUM"},
                {"LEMMA": "vez"}
            ]

    matcher.add("a7d3_3_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a7d3_4(doc):
    """
    A7.3-4: Numerais - forma - fracionários
    e.g., meio, metade
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"MORPH": {"IS_SUPERSET": ["NumType=Ord"]}, "LOWER": {"NOT_IN": ["quarto"]}}
        # "LOWER" So that we don't match "No quarto da Ana havia um desenho de um cavalo selvagem, feito a carvão, ao lado de uma figura da Virgem Maria, em cerâmica."
    ]

    # We're basically adding a further constraint for "quarto".
    # We may have to do the same for "quinta".
    pattern_quarto = [
        {"MORPH": {"IS_SUPERSET": ["NumType=Ord"]}, "LOWER": "quarto"},
        {"POS": "NOUN"}
    ]

    pattern_string = [
                {"LEMMA": {"IN": ["meio", "metade", "oitavo"]}}
            ]

    matcher.add("a7d3_4_A2", [pattern, pattern_string, pattern_quarto])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a7d3_5(doc):
    """
    A7.3-5: Numerais - forma - multiplicativos
    e.g., dobro, triplo
    Level: B1
    """
    matcher = PhraseMatcher(doc.vocab, attr="NORM")

    patterns = list(nlp_stanza.pipe(["dobro", "triplo", "quadruplo", "quintuplo", "sextuplo", "setuplo", "octuplo", "noveplo", "decuplo"]))

    matcher.add("a7d3_5_B1", patterns)

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a7d3_6(doc):
    """
    A7.3-6: Numerais - forma - multiplicativos - em expressões com de
    e.g., o dobro do tempo
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"NORM": {"IN": ["dobro", "triplo", "quadruplo", "quintuplo", "sextuplo", "setuplo", "octuplo", "noveplo", "decuplo"]}},
        {"LOWER": "de"},
        {"POS": "DET", "OP": "?"},
        {"POS": "ADJ", "OP": "?"},
        {"POS": "NOUN"}
    ]

    matcher.add("a7d3_6_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a7d3_7(doc):
    """
    A7.3-7: Numerais - forma - multiplicativos - em expressões com cardinal + vez/vezes + mais/menos
    e.g., cem vezes mais/menos
    Level: B1

    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"POS": "NUM"},
                {"LEMMA": "vez"},
                {"LEMMA": {"IN": ["mais", "menos"]}}
            ]

    matcher.add("a7d3_7_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a7d3_8(doc):
    """
    A7.3-8: Numerais - forma - fracionários - em expressões com de
    e.g., um terço do dinheiro
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"LEMMA": {"IN": ["meio", "metade", "oitavo"]}},
        {"LOWER": "de"},
        {"POS": "DET", "OP": "?"},
        {"POS": "ADJ", "OP": "?"},
        {"POS": "NOUN"}
    ]

    pattern_alt = [
        {"MORPH": {"IS_SUPERSET": ["NumType=Ord"]}},
        {"LOWER": "de"},
        {"POS": "DET", "OP": "?"},
        {"POS": "ADJ", "OP": "?"},
        {"POS": "NOUN"}
    ]

    matcher.add("a7d3_8_B1", [pattern, pattern_alt])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a7d3_10(doc):
    """
    A7.3-10: Numerais - forma - outras expressões que exprimem número
    e.g., inúmeros, diversos, diferentes
    Level: B2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"LEMMA": {"IN": ["inúmero", "diversos", "muitíssimo"]}, "POS": {"IN": ["ADJ", "DET", "ADV"]}}
    ]

    matcher.add("a7d3_10_B2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]



