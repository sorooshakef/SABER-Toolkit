from spacy.matcher import Matcher, PhraseMatcher
from process_and_display import nlp_small
from text_reconstruction import reconstruct_text

def a7d2_1(doc):
    """
    A7.2-1: Universais - forma - variação em género e número - A2
    e.g., todo, toda, todos, todas; nenhum, nenhuma, nenhuns, nenhumas
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"POS": "DET", "LEMMA": {"IN": ["todo", "nenhum"]}},
            ]

    matcher.add("a7d2_1_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a7d2_6(doc):
    """
    A7.2-6: Universais - posição/distribuição - possibilidade de ocorrência com pronomes e determinantes
    e.g., todos eles / todos estes livros / todos os meus amigos; nenhum dos amigos/ nenhum deles/ nenhuma destas
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LEMMA": {"IN": ["todo", "nenhum"]}},
                {"LOWER": "de", "OP": "?"},
                {"POS": {"IN": ["PRON", "DET"]}},
            ]

    matcher.add("a7d2_6_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a7d2_7(doc):
    """
    A7.2-7: Universais - forma - variação em género e número - B1
    e.g., ambos, ambas; qualquer, quaisquer
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"POS": "DET", "LEMMA": {"IN": ["ambos", "qualquer"]}}
            ]

    matcher.add("a7d2_7_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a7d2_8(doc):
    """
    A7.2-8: Universais - forma - invariáveis
    e.g., cada
    Level: B1
    """
    matcher = PhraseMatcher(doc.vocab, attr="LOWER")

    patterns = list(nlp_small.pipe(["cada"]))

    matcher.add("a7d2_8_B1", patterns)

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
a7d2_8.REQUIRES_SPACY = True


def a7d2_13(doc):
    """
    A7.2-13: "Universais - uso / valor - cada ocorrência com um"
    e.g., Cada um seguiu o seu caminho.
    Level: B1
    """
    matcher = PhraseMatcher(doc.vocab, attr="LOWER")

    patterns = list(nlp_small.pipe(["cada um"]))

    matcher.add("a7d2_13_B1", patterns)

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
a7d2_13.REQUIRES_SPACY = True


def a7d2_15(doc):
    """
    A7.2-15: "Universais - distribuição - qualquer, quaisquer antes ou depois do nome, com mudança de valor"
    e.g., Ele gosta de qualquer filme / de um filme qualquer.
    Level: B2
    """
    matcher = Matcher(doc.vocab)

    # The indefinite article can be preceded by a preposition that forms a
    # contraction with it (e.g., de+um -> "dum", em+uma -> "numa"). The
    # spaCy-Stanza pipeline splits these contractions into two tokens
    # (preposition + article), so we capture the optional leading preposition
    # to include it in the highlight.
    pattern_sing = [
                {"LEMMA": {"IN": ["de", "em"]}, "POS": "ADP", "OP": "?"},
                {"LOWER": {"IN": ["um", "uma"]}},
                {"POS": "NOUN", "MORPH": {"IS_SUPERSET": ["Number=Sing"]}},
                {"LOWER": "qualquer"}
            ]

    pattern_plur = [
                {"LEMMA": {"IN": ["de", "em"]}, "POS": "ADP", "OP": "?"},
                {"LOWER": {"IN": ["uns", "umas"]}},
                {"POS": "NOUN", "MORPH": {"IS_SUPERSET": ["Number=Plur"]}},
                {"LOWER": "quaisquer"}
            ]

    # greedy="LONGEST" keeps the contraction-inclusive match instead of also
    # returning the shorter match without the leading preposition.
    matcher.add("a7d2_15_B2", [pattern_sing, pattern_plur], greedy="LONGEST")

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a7d2_16(doc):
    """
    A7.2-16: "Universais - distribuição - qualquer, quaisquer - ocorrência com outro"
    e.g., qualquer outro / outro qualquer
    Level: B2
    """
    matcher = PhraseMatcher(doc.vocab, attr="LOWER")

    patterns = list(nlp_small.pipe(["qualquer outro", "outro qualquer", "quaisquer outros", "outros quaisquer"]))

    matcher.add("a7d2_16_B2", patterns)

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
a7d2_16.REQUIRES_SPACY = True


