from spacy.matcher import Matcher
from process_and_display import nlp_stanza
from text_reconstruction import reconstruct_text

def b4d1_4(doc):
    """
    B4-4: negativa - marcada por outras palavras de valor negativo
    e.g. Ninguém foi à aula. Nunca visitei Portugal.
    Level: A2
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    negative_words = ["ninguem", "nada", "nenhum", "nenhuma", "nenhuns", "nenhumas", "nunca", "jamais", "nem", "nenhures"]

    pattern = [
        {"NORM": {"IN": negative_words}, "DEP": {"IN": ["advmod", "nsubj", "cc"]}},
        # "cc" to match "Nem a expectativa valeu à nossa sementinha – pôs-se a dormir também."
        {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "IS_SENT_START": False},
        {"POS": {"IN": ["VERB", "AUX"]}, "IS_SENT_START": False}
        ]

    matcher.add("b4d1_4_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]