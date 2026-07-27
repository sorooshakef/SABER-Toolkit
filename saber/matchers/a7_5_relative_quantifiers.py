from spacy.matcher import Matcher
from text_reconstruction import reconstruct_text


def a7d5_1(doc):
    """
    A7.5-1: Relativos - forma - variação em género e número
    e.g., quanto, quanta, quantos, quantas
    Level: B2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"LEMMA": "quanto", "MORPH": {"IS_SUPERSET": ["PronType=Rel"]}}
            ]

    matcher.add("a7d5_1_B2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]    