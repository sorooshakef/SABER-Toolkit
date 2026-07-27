from spacy.matcher import Matcher, PhraseMatcher
from process_and_display import nlp_stanza
from text_reconstruction import reconstruct_text

def b3d2_8(doc):
    """
    B3.2-8: interrogativa de confirmação - expressa um pedido de confirmação da informação contida na declarativa que a precede - forma-se com não é ou não + V da frase declarativa
    e.g. Vives em Lisboa, não é? Vives em Lisboa, não vives?
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"NORM": "nao"},
        {
            "POS": {"IN": ["AUX", "VERB"]},
            "DEP": {"IN": ["parataxis", "conj"]},
            "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]},
        },
        {"LOWER": "?"}
        ]

    matcher.add("b3d2_8_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b3d2_11(doc):
    """
    B3.2-11: interrogativa de confirmação - com outras formas interrogativas
    e.g., Vives em Lisboa, ...não é verdade? não é assim?
    Level: B1
    """
    matcher = PhraseMatcher(doc.vocab, attr="NORM")

    patterns = list(nlp_stanza.pipe(["nao e verdade?", "nao e assim?", "nao e mesmo?",
                                    "nao e certo?", "estou certo?", "estou certa?",
                                    "nao lhe parece?", "nao te parece?",
                                    "ou estou enganado?", "ou estou enganada?",
                                    "estou enganado?", "estou enganada?",
                                    "nao achas?", "nao acha?",
                                    "nao concordas?", "nao concorda?",
                                    "nao foi?", "pois nao?"]))

    matcher.add("b3d2_11_B1", patterns)

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b3d2_14(doc):
    """
    B3.2-14: interrogativa parcial múltipla - contém dois ou mais elementos interrogativos
    e.g., Quem disse o quê?
    Level: B2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"NORM": {"IN": ["quem", "que", "quando", "onde", "porque", "aonde", "qual"]}},
        {"POS": {"NOT_IN": ["VERB", "PUNCT", "AUX"]}, "IS_SENT_START": False, "OP": "{,2}"},
        {"MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}, "IS_SENT_START": False},
        {"NORM": {"NOT_IN": ["quem", "que", "quando", "onde", "porque", "aonde", "qual"]}, "IS_SENT_START": False, "OP": "{,2}", "POS": {"NOT_IN": ["PUNCT"]}},
        {"NORM": {"IN": ["quem", "que", "quando", "onde", "porque", "aonde", "qual"]}, "IS_SENT_START": False}
        ]

    matcher.add("b3d2_14_B2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


# def b3d2_14(doc):
#     """
#     B3.2-14: interrogativa parcial múltipla - contém dois ou mais elementos interrogativos
#     e.g., Quem disse o quê?
#     Level: B2
#     """
#     matcher = DependencyMatcher(doc.vocab)

#     pattern = [
#         {
#             "RIGHT_ID": "first_q",
#             "RIGHT_ATTRS": {
#                 "MORPH": {"INTERSECTS": ["PronType=Int", "PronType=Rel"]}
#                 }
#         },
#         {
#             "LEFT_ID": "first_q",
#             "RIGHT_ID": "verb",
#             "RIGHT_ATTRS": {
#                 "POS": "VERB",
#                 "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}
#                 },
#             "REL_OP": "<++"           
#         },
#         {
#             "LEFT_ID": "verb",
#             "RIGHT_ID": "second_q",
#             "RIGHT_ATTRS": {
#                 "NORM": {"IN": ["quem", "que", "quando", "onde", "porque", "aonde", "qual"]},
#                 # So that we don't match "Com quem viste o Rui e onde? Para onde vais viajar e quando?"
#                 "DEP": "obj"
#             },
#             "REL_OP": ">++"
#         }
#     ]

#     pattern_alt = [
#         {
#             "RIGHT_ID": "first_q",
#             "RIGHT_ATTRS": {
#                 "MORPH": {"INTERSECTS": ["PronType=Int", "PronType=Rel"]}
#                 }
#         },
#         {
#             "LEFT_ID": "first_q",
#             "RIGHT_ID": "verb",
#             "RIGHT_ATTRS": {
#                 "POS": "VERB",
#                 "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}
#                 },
#             "REL_OP": "<++"           
#         },
#         {
#             "LEFT_ID": "verb",
#             "RIGHT_ID": "second_q",
#             "RIGHT_ATTRS": {
#                 "NORM": {"IN": ["quem", "que", "quando", "onde",
#                                 "porque", "aonde", "qual", "quais"]}
#             },
#             "REL_OP": "."
#         }
#     ]

#     pattern_alt_2 = [
#         {
#             "RIGHT_ID": "first_q",
#             "RIGHT_ATTRS": {
#                 "MORPH": {"INTERSECTS": ["PronType=Int", "PronType=Rel"]}
#                 }
#         },
#         {
#             "LEFT_ID": "first_q",
#             "RIGHT_ID": "verb",
#             "RIGHT_ATTRS": {
#                 "POS": "AUX",
#                 "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}
#                 },
#             "REL_OP": ">++"           
#         },
#         {
#             "LEFT_ID": "verb",
#             "RIGHT_ID": "second_q",
#             "RIGHT_ATTRS": {
#                 "NORM": {"IN": ["quem", "que", "quando", "onde",
#                                 "porque", "aonde", "qual", "quais"]}
#             },
#             "REL_OP": "."
#         }
#     ]

#     pattern_quando = [
#         {
#             "RIGHT_ID": "first_q",
#             "RIGHT_ATTRS": {
#                 "LOWER": "quando"
#                 }
#         },
#         {
#             "LEFT_ID": "first_q",
#             "RIGHT_ID": "verb",
#             "RIGHT_ATTRS": {
#                 "POS": "VERB",
#                 "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}
#                 },
#             "REL_OP": "<++"           
#         },
#         {
#             "LEFT_ID": "verb",
#             "RIGHT_ID": "second_q",
#             "RIGHT_ATTRS": {
#                 "NORM": {"IN": ["quem", "que", "quando", "onde",
#                                 "porque", "aonde", "qual", "quais"]}
#             },
#             "REL_OP": "."
#         }
#     ]

#     matcher.add("b3d2_14_B2", [pattern, pattern_alt, pattern_alt_2, pattern_quando])

#     matches = matcher(doc)
    
#     results = []
#     for match_id, token_ids in matches:
#         # Calculate start and end indices
#         start = min(token_ids)
#         end = max(token_ids) + 1
#         tokens_in_span = doc[start:end]
        
#         text = reconstruct_text(tokens_in_span)
#         results.append((doc.vocab.strings[match_id], text, start, end))

#     return results


# def b3d2_15(doc):
#     """
#     B3.2-15: interrogativas coordenadas - elementos interrogativos coordenados na mesma posição
#     e.g. Com quem e onde viste o Rui? Vais viajar para onde e quando?
#     Level: B2
#     """
#     matcher = Matcher(doc.vocab)

#     pattern_1 = [
#         {
#             "NORM": {"IN": ["quem", "que", "quando", "onde",
#                              "porque", "aonde", "qual", "quais", "como"]},
#             "DEP": {"IN": ["obl", "advmod", "root", "conj"]}
#         },
#         {"POS": {"NOT_IN": ["CCONJ"]}, "OP": "*", "IS_SENT_START": False},
#         {"NORM": "e", "POS": "CCONJ", "IS_SENT_START": False},
#         {
#             "NORM": {"NOT_IN": ["quem", "que", "quando", "onde",
#                              "porque", "aonde", "qual", "quais", "como"]},
#             "OP": "*",
#             "IS_SENT_START": False
#         },
#         {
#             "NORM": {"IN": ["quem", "que", "quando", "onde",
#                              "porque", "aonde", "qual", "quais", "como"]},
#             "DEP": {"IN": ["obl", "advmod", "root", "conj"]},
#             "IS_SENT_START": False
#         },
#         {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "IS_SENT_START": False},
#         {"POS": {"IN": ["VERB", "AUX"]}, "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}, "IS_SENT_START": False}

#         ]
    
#     pattern_2 = [
#         {"POS": {"IN": ["VERB", "AUX"]}, "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}},
#         {
#             "NORM": {"NOT_IN": ["quem", "que", "quando", "onde",
#                              "porque", "aonde", "qual", "quais", "como"]},
#             "OP": "*",
#             "IS_SENT_START": False
#         },
#         {
#             "NORM": {"IN": ["quem", "que", "quando", "onde",
#                              "porque", "aonde", "qual", "quais", "como"]},
#             "DEP": {"IN": ["obl", "advmod", "root", "conj"]},
#             "IS_SENT_START": False
#         },
#         {"POS": {"NOT_IN": ["CCONJ"]}, "OP": "*", "IS_SENT_START": False},
#         {"NORM": "e", "POS": "CCONJ", "IS_SENT_START": False},
#         {
#             "NORM": {"NOT_IN": ["quem", "que", "quando", "onde",
#                              "porque", "aonde", "qual", "quais", "como"]},
#             "OP": "*",
#             "IS_SENT_START": False
#         },
#         {
#             "NORM": {"IN": ["quem", "que", "quando", "onde",
#                              "porque", "aonde", "qual", "quais", "como"]},
#             "DEP": {"IN": ["obl", "advmod", "root", "conj"]},
#             "IS_SENT_START": False
#         }
#         ]

#     matcher.add("b3d2_15_B2", [pattern_1, pattern_2])

#     matches = matcher(doc)

#      # If there are fewer than two matches, no overlaps are possible.
#     if len(matches) < 2:
#         return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

#     # Sort matches by start index, then by descending end index as a tie-breaker.
#     # This places longer matches first if they share the same start token.
#     sorted_matches = sorted(matches, key=lambda m: (m[1], -m[2]))

#     filtered_matches = []
#     # Initialize with the first match from the sorted list.
#     current_best_match = sorted_matches[0]

#     for next_match in sorted_matches[1:]:
#         # Check if the next match overlaps with the current best match.
#         # An overlap occurs if the next match starts before the current one ends.
#         if next_match[1] < current_best_match[2]:
#             # Overlap detected. We keep the one with the greater end index.
#             if next_match[2] > current_best_match[2]:
#                 # The next match ends later, so it becomes the new best match.
#                 current_best_match = next_match
#             # Otherwise, we discard the next_match and keep the current_best_match.
#         else:
#             # No overlap. The current_best_match is final for its group.
#             filtered_matches.append(current_best_match)
#             # The next_match becomes the new best match for the next potential group.
#             current_best_match = next_match
    
#     # After the loop, append the last standing best match.
#     filtered_matches.append(current_best_match)

#     return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in filtered_matches]


# def b3d2_16(doc):
#     """
#     B3.2-16: interrogativas coordenadas - elementos interrogativos coordenados em posições diferentes
#     e.g. Com quem viste o Rui e onde (o viste)? Para onde vais viajar e quando (vais)?
#     Level: B2
#     """
#     matcher = Matcher(doc.vocab)

#     pattern = [
#         {
#             "NORM": {"IN": ["quem", "que", "quando", "onde",
#                              "porque", "aonde", "qual", "quais", "como"]},
#             "DEP": {"IN": ["obl", "nsubj"]}
#         },
#         {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "IS_SENT_START": False},
#         {"POS": {"IN": ["VERB", "AUX"]}, "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}, "IS_SENT_START": False},
#         {"POS": {"NOT_IN": ["CCONJ"]}, "OP": "*", "IS_SENT_START": False},
#         {"NORM": "e", "POS": "CCONJ", "IS_SENT_START": False},
#         {
#             "NORM": {"NOT_IN": ["quem", "que", "quando", "onde",
#                              "porque", "aonde", "qual", "quais", "como"]},
#             "OP": "*",
#             "IS_SENT_START": False
#         },
#         {
#             "NORM": {"IN": ["quem", "que", "quando", "onde",
#                              "porque", "aonde", "qual", "quais", "como"]},
#             "DEP": {"IN": ["obl", "advmod"]},
#             "IS_SENT_START": False
#         }
        

#         ]

#     matcher.add("b3d2_16_B2", [pattern])

#     matches = matcher(doc)

#     return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]