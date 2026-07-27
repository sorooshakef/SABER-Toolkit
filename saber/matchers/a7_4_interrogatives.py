from spacy.matcher import Matcher
from text_reconstruction import reconstruct_text


def a7d4_5(doc):
    """
    A7.4-5: Interrogativos - distribuição - seguido de preposição de
    e.g., Quantos dos teus amigos são portugueses?
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"MORPH": {"INTERSECTS": ["PronType=Rel", "PronType=Int"]}},
                {"LOWER": "de"},
                {"POS": "ADP", "OP": "?"},  # To match "Quem de entre os avaliadores, se é que algum, mostrou verdadeira imparcialidade?"
                {"POS": "DET", "OP": "*"},
                {"POS": "ADJ", "OP": "?"},
                {"POS": {"IN": ["NOUN", "PRON"]}}
            ]

    matcher.add("a7d4_5_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]    