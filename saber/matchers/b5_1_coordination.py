from spacy.matcher import Matcher, PhraseMatcher, DependencyMatcher
from process_and_display import nlp_stanza
from text_reconstruction import reconstruct_text

# def b5d1_1(doc):
#     """
#     B5.1-1: coordenada copulativa - e
#     e.g. De manhã, leio o jornal e tomo o pequeno-almoço.
#     Level: A1
    
#     This structure will be removed for now because distinguishing between the most common
#     use of "e" as a coordenativa aditiva and its use as a cooredenativa adversativa is heavily
#     context-dependent.
#     """
#     matcher = Matcher(doc.vocab)

#     pattern = [{"LOWER": "e", "POS": "CCONJ"}]

#     matcher.add("b5d1_1_A1", [pattern])

#     matches = matcher(doc)

#     return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d1_2(doc):
    """
    B5.1-2: coordenada disjuntiva - ou
    e.g. Vamos ao cinema ou ficamos em casa?
    Level: A1
    Requires nlp_stanza

    Note: This can also be broken into 4 different features.
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "word1",
            "RIGHT_ATTRS": {}
        },
        {
            "LEFT_ID": "word1",
            "RIGHT_ID": "word2",
            "RIGHT_ATTRS": {
                "DEP": "conj"
                },
            "REL_OP": ">"             
        },
        {
            "LEFT_ID": "word2",
            "RIGHT_ID": "ou",
            "RIGHT_ATTRS": {
                "LOWER": "ou",
                "DEP": "cc"
                },
            "REL_OP": ">"             
        }
    ]
    
    matcher.add("b5d1_2_A1", [pattern])

    matches = matcher(doc)
    
    results = []
    for match_id, token_ids in matches:
        # Calculate start and end indices
        start = min(token_ids)

        span_start = max(0, start - 3)
        # Check if any token in that span is "ou"
        if any(token.text.lower() == "ou" for token in doc[span_start:start]):
            continue  # We don't want to match "ou ... ou" constructions.

        end = max(token_ids) + 1
        tokens_in_span = doc[start:end]
        
        text = reconstruct_text(tokens_in_span)
        results.append((doc.vocab.strings[match_id], text, start, end))

    return results


def b5d1_3(doc):
    """
    B5.1-3: coordenada adversativa - mas
    e.g. Vivo em Portugal, mas sou italiano.    
    Level: A1
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern = [{"NORM": "mas", "POS": "CCONJ"}]

    matcher.add("b5d1_3_A1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d1_4(doc):
    """
    B5.1-4: coordenada conclusiva - então
    e.g. Estou doente, então não vou à aula.  
    Level: A1
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern = [
    {"POS": {"IN": ["VERB", "AUX"]}},
    {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "SENT_START": False},
    {"NORM": "entao", "SENT_START": False},
    {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "SENT_START": False},
    {"POS": {"IN": ["VERB", "AUX"]}, "DEP": {"NOT_IN": ["ROOT"]}, "SENT_START": False}
    ]

    matcher.add("b5d1_4_A1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d1_5(doc):
    """
    B5.1-5: coordenada copulativa - nem
    e.g. Não comprei pão nem leite.
    Level: A2
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern = [
    {"NORM": "nao"},
    {"NORM": {"NOT_IN": ["nem"]}, "OP": "*", "SENT_START": False},
    {"NORM": "nem", "SENT_START": False}
    ]

    matcher.add("b5d1_5_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d1_6(doc):
    """
    B5.1-6:coordenada conclusiva - logo, por isso
    e.g. João está doente, logo não vem à aula. A Rita está doente, por isso não vem à aula.
    Level: A2
    Requires nlp_stanza

    Note: Perhaps it would be a better idea to constrain "logo" more to make
    sure it acts as a coordinator.
    """
    matcher = Matcher(doc.vocab)

    # pattern_logo = [
    # {"POS": {"IN": ["VERB", "AUX"]}},
    # {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "SENT_START": False},
    # {"NORM": "logo"},
    # {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "SENT_START": False, "TEXT": {"NOT_IN": ["que"]}},
    # # The last condition of the previous token is to prevent "Logo que chegaram," from matching.
    # {"POS": {"IN": ["VERB", "AUX"]}, "DEP": {"NOT_IN": ["ROOT"]}, "SENT_START": False}
    # ]

    # pattern_logo = [
    # {},
    # {"IS_PUNCT": True},
    # {"LOWER": "logo"},
    # {"TEXT": ","},
    # {}  # We need the wildcard tokens because otherwise for some weird reason (because of the commas), the pattern is not highlighted in the app.
    # ]

    pattern_logo_less_strict = [
    {},
    {"LOWER": ","},
    {"LOWER": "logo"},
    ]

    pattern_logo_sent_start = [
    {"LOWER": "logo", "IS_SENT_START": True}
    ]

    pattern_por_isso_comma = [
    {},
    {"LOWER": ","},
    {"NORM": "por"},
    {"NORM": "isso"}
    ]

    pattern_por_isso_start = [
    {"NORM": "por", "IS_SENT_START": True},
    {"NORM": "isso"}
    ]

    matcher.add("b5d1_6_A2", [pattern_logo_less_strict, pattern_logo_sent_start, pattern_por_isso_comma, pattern_por_isso_start])

    matches = matcher(doc)

    filtered_matches = []
    for match_id, start, end in matches:
        # Start the span at the conclusive connector, dropping any leading
        # wildcard/comma context the pattern used only for disambiguation, so the
        # highlight is just the connector ("logo" / "por isso") -- consistent with
        # the sentence-start variants above.
        span_start = start
        while span_start < end and doc[span_start].lower_ not in ("logo", "por"):
            span_start += 1
        if span_start >= end:
            continue
        filtered_matches.append(
            (doc.vocab.strings[match_id], reconstruct_text(doc[span_start:end]), span_start, end)
        )
    return filtered_matches

def b5d1_7(doc):
    """
    B5.1-7:coordenada explicativa - pois
    e.g. Estou cansado, pois deitei-me tarde.
    Level: A2
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern = [
    {"NORM": "pois", "POS": {"IN": ["SCONJ", "ADV"]}, "SENT_START": False}
    ]

    matcher.add("b5d1_7_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d1_8(doc):
    """
    B5.1-8: coordenada copulativa - não só…mas também; não só…como; tanto...como; nem...nem
    e.g. O João não só chegou atrasado como também não fez o trabalho. Gosto tanto de ler como de ver televisão. O João nem ligou nem deixou recado.
    Level: B1
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern_1 = [
    {"NORM": "nao"},
    {"NORM": "so"},
    {"NORM": {"NOT_IN": ["mas"]}, "OP": "{,6}", "SENT_START": False},
    # The token limit set to 6 to prevent
    # "acharam a festa um tanto barulhenta, mas ninguém ...
    # ... reclamou, pois João é conhecido como um rapaz realmente"
    # from being matched.
    {"NORM": "mas", "SENT_START": False},
    {"NORM": "tambem"}
    ]

    pattern_2 = [
    {"NORM": "nao"},
    {"NORM": "so"},
    {"NORM": {"NOT_IN": ["como"]}, "OP": "{,6}", "SENT_START": False},
    {"NORM": "como", "POS": {"IN": ["ADV", "ADP", "SCONJ"]}, "SENT_START": False}
    ]

    pattern_3 = [
    {"NORM": "tanto"},
    {"NORM": {"NOT_IN": ["como"]}, "OP": "{,6}", "SENT_START": False},
    {"NORM": "como", "POS": {"IN": ["ADV", "ADP", "SCONJ"]}, "SENT_START": False}
    ]

    pattern_4 = [
    {"NORM": "nem"},
    {"NORM": {"NOT_IN": ["nem"]}, "OP": "{,6}", "SENT_START": False},
    {"NORM": "nem", "SENT_START": False}
    ]

    matcher.add("b5d1_8_B1", [pattern_1, pattern_2, pattern_3, pattern_4])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d1_9(doc):
    """
    B5.1-9: coordenada disjuntiva - ou...ou; quer...quer; ora...ora; seja...seja
    e.g. Ou andas mais depressa ou vamos chegar atrasados.
    Level: B1
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern_1 = [
    {"NORM": "ou"},
    {"NORM": {"NOT_IN": ["ou"]}, "OP": "+", "SENT_START": False},
    {"NORM": "ou", "SENT_START": False}
    ]

    pattern_2 = [
    {"NORM": "quer"},
    {"NORM": {"NOT_IN": ["quer"]}, "OP": "+", "SENT_START": False},
    {"NORM": "quer", "SENT_START": False}
    ]

    pattern_3 = [
    {"NORM": "seja"},
    {"NORM": {"NOT_IN": ["seja"]}, "OP": "+", "SENT_START": False},
    {"NORM": "seja", "SENT_START": False}
    ]

    pattern_4 = [
    {"NORM": "ora"},
    {"NORM": {"NOT_IN": ["ora"]}, "OP": "+", "SENT_START": False},
    {"NORM": "ora", "SENT_START": False}
    ]

    matcher.add("b5d1_9_B1", [pattern_1, pattern_2, pattern_3, pattern_4])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d1_10(doc):
    """
    B5.1-10: coordenada adversativa - porém, contudo, no entanto, ainda assim
    e.g. O João está doente, porém veio trabalhar.
    Level: B1
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern_1 = [
    {"LOWER": "porém"},
    ]

    pattern_2 = [
    {"NORM": "contudo"},
    ]

    pattern_3 = [
    {"NORM": "em"},
    {"NORM": "o"},
    {"NORM": "entanto"}
    ]

    pattern_4 = [
    # {"POS": {"IN": ["VERB", "AUX"]}},
    # {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "SENT_START": False},
    {"NORM": "ainda"},
    {"NORM": "assim"},
    # {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "SENT_START": False},
    # {"POS": {"IN": ["VERB", "AUX"]}, "SENT_START": False}
    ]

    matcher.add("b5d1_10_B1", [pattern_1, pattern_2, pattern_3, pattern_4])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d1_11(doc):
    """
    B5.1-11: coordenada conclusiva - portanto, assim
    e.g. A Rita tem medo de alturas, portanto não anda de avião.
    Level: B1
    Requires nlp_stanza

    Issue: We may need more constraints for "assim".
    """
    matcher = Matcher(doc.vocab)

    pattern_1 = [
    {"POS": {"IN": ["VERB", "AUX"]}},
    {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "SENT_START": False},
    {"NORM": "portanto"},
    {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "SENT_START": False},
    {"POS": {"IN": ["VERB", "AUX"]}, "SENT_START": False}
    ]

    pattern_2 = [
    {"NORM": "assim", "SENT_START": True},
    {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "SENT_START": False, "TEXT": {"NOT_IN": ["que"]}},
    # The last condition of the previous token is to prevent "Assim que atravessaram o portão, encontraram o Tico."
    {"POS": {"IN": ["VERB", "AUX"]}, "SENT_START": False}
    ]

    pattern_3 = [
    {"NORM": {"IN": [",", ";"]}},
    {"NORM": "assim"},
    {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "SENT_START": False, "TEXT": {"NOT_IN": ["que"]}},
    {"POS": {"IN": ["VERB", "AUX"]}, "SENT_START": False}
    ]

    matcher.add("b5d1_11_B1", [pattern_1, pattern_2, pattern_3])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d1_12(doc):
    """
    B5.1-12: coordenada adversativa - todavia, não obstante
    e.g. Gosto de muitos géneros de filmes. Não obstante, não aprecio filmes de terror.
    Level: B2
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern_1 = [
    {"NORM": "todavia"},
    {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "SENT_START": False},
    {"POS": {"IN": ["VERB", "AUX"]}, "SENT_START": False}
    ]

    pattern_2 = [
    {"NORM": "nao"},
    {"NORM": "obstante"}
    ]

    matcher.add("b5d1_12_B2", [pattern_1, pattern_2])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d1_13(doc):
    """
    B5.1-13: coordenada conclusiva - por conseguinte, por consequência
    e.g. Estudei pouco para o exame; por conseguinte, não consegui passar.
    Level: B2
    Requires nlp_stanza
    """
    matcher = PhraseMatcher(doc.vocab)

    patterns = list(nlp_stanza.pipe(["por conseguinte", "por consequência",
                                     "Por conseguinte", "Por consequência"]))

    matcher.add("b5d1_12_B2", patterns)

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

# def b5d1_14(doc):
#     """
#     B5.1-14: coordenada explicativa - que
#     e.g. Ela deve ter frio, que está a tremer.
#     Level: B2
#     It's extremely tricky to distinguish between this structure and the relative clause
#     I'll remove it from the structures in the system for now.
#     """
#     matcher = Matcher(doc.vocab)

#     pattern = [
#     {"POS": "PUNCT"},  # To prevent matching "acho que ..."
#     {"NORM": "que", "POS": {"NOT_IN": ["PRON"]}},
#     {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "SENT_START": False},
#     {"POS": {"IN": ["VERB", "AUX"]}, "DEP": {"NOT_IN": ["acl:relcl"]}, "SENT_START": False}
#     ]
#     # The idea is to prevent relative clauses from being captured.

#     matcher.add("b5d1_14_B2", [pattern])

#     matches = matcher(doc)

#     return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

# def b5d1_15(doc):
#     """
#     B5.1-15: coordenada assindética - com valor expressivo
#     e.g. Olhei para trás, acenei, subi para o comboio.
#     Level: C1
#     """
#     matcher = Matcher(doc.vocab, validate=True)

#     pattern_pres = [
#     {"POS": "VERB", "DEP": {"IN": ["root", "conj", "parataxis"]}, "MORPH": {"IS_SUPERSET": ["Tense=Pres"]}},
#     {"POS": {"NOT_IN": ["VERB", "CCONJ"]}, "OP": "*", "IS_SENT_START": False},
#     {"POS": "VERB", "DEP": "acl:relcl", "OP": "?", "IS_SENT_START": False}, # So that we match "Pensei no que deixava para trás, senti um aperto no peito, hesitei."
#     {"POS": {"NOT_IN": ["VERB", "CCONJ"]}, "OP": "*", "IS_SENT_START": False},
#     {"POS": "VERB", "DEP": {"IN": ["conj", "parataxis"]}, "MORPH": {"IS_SUPERSET": ["Tense=Pres"]}, "IS_SENT_START": False},
#     {"POS": {"NOT_IN": ["VERB", "CCONJ"]}, "OP": "*", "IS_SENT_START": False},
#     {"POS": "VERB", "DEP": "acl:relcl", "OP": "?", "IS_SENT_START": False},
#     {"POS": {"NOT_IN": ["VERB", "CCONJ"]}, "OP": "*", "IS_SENT_START": False},
#     {"POS": "VERB", "DEP": {"IN": ["conj", "parataxis"]}, "MORPH": {"IS_SUPERSET": ["Tense=Pres"]}, "IS_SENT_START": False}
#     ]

#     pattern_past = [
#     {"POS": "VERB", "DEP": {"IN": ["root", "conj", "parataxis"]}, "MORPH": {"IS_SUPERSET": ["Tense=Past"]}},
#     {"POS": {"NOT_IN": ["VERB", "CCONJ"]}, "OP": "*", "IS_SENT_START": False},
#     {"POS": "VERB", "DEP": "acl:relcl", "OP": "?", "IS_SENT_START": False},
#     {"POS": {"NOT_IN": ["VERB", "CCONJ"]}, "OP": "*", "IS_SENT_START": False},
#     {"POS": "VERB", "DEP": {"IN": ["conj", "parataxis"]}, "MORPH": {"IS_SUPERSET": ["Tense=Past"]}, "IS_SENT_START": False},
#     {"POS": {"NOT_IN": ["VERB", "CCONJ"]}, "OP": "*", "IS_SENT_START": False},
#     {"POS": "VERB", "DEP": "acl:relcl", "OP": "?", "IS_SENT_START": False},
#     {"POS": {"NOT_IN": ["VERB", "CCONJ"]}, "OP": "*", "IS_SENT_START": False},
#     {"POS": "VERB", "DEP": {"IN": ["conj", "parataxis"]}, "MORPH": {"IS_SUPERSET": ["Tense=Past"]}, "IS_SENT_START": False}
#     ]

#     pattern_fut = [
#     {"POS": "VERB", "DEP": {"IN": ["root", "conj", "parataxis"]}, "MORPH": {"IS_SUPERSET": ["Tense=Fut"]}},
#     {"POS": {"NOT_IN": ["VERB", "CCONJ"]}, "OP": "*", "IS_SENT_START": False},
#     {"POS": "VERB", "DEP": "acl:relcl", "OP": "?", "IS_SENT_START": False},
#     {"POS": {"NOT_IN": ["VERB", "CCONJ"]}, "OP": "*", "IS_SENT_START": False},
#     {"POS": "VERB", "DEP": {"IN": ["conj", "parataxis"]}, "MORPH": {"IS_SUPERSET": ["Tense=Fut"]}, "IS_SENT_START": False},
#     {"POS": {"NOT_IN": ["VERB", "CCONJ"]}, "OP": "*", "IS_SENT_START": False},
#     {"POS": "VERB", "DEP": "acl:relcl", "OP": "?", "IS_SENT_START": False},
#     {"POS": {"NOT_IN": ["VERB", "CCONJ"]}, "OP": "*", "IS_SENT_START": False},
#     {"POS": "VERB", "DEP": {"IN": ["conj", "parataxis"]}, "MORPH": {"IS_SUPERSET": ["Tense=Fut"]}, "IS_SENT_START": False}
#     ]

#     pattern_imp = [
#     {"POS": "VERB", "DEP": {"IN": ["root", "conj", "parataxis"]}, "MORPH": {"IS_SUPERSET": ["Tense=Imp"]}},
#     {"POS": {"NOT_IN": ["VERB", "CCONJ"]}, "OP": "*", "IS_SENT_START": False},
#     {"POS": "VERB", "DEP": "acl:relcl", "OP": "?", "IS_SENT_START": False},
#     {"POS": {"NOT_IN": ["VERB", "CCONJ"]}, "OP": "*", "IS_SENT_START": False},
#     {"POS": "VERB", "DEP": {"IN": ["conj", "parataxis"]}, "MORPH": {"IS_SUPERSET": ["Tense=Imp"]}, "IS_SENT_START": False},
#     {"POS": {"NOT_IN": ["VERB", "CCONJ"]}, "OP": "*", "IS_SENT_START": False},
#     {"POS": "VERB", "DEP": "acl:relcl", "OP": "?", "IS_SENT_START": False},
#     {"POS": {"NOT_IN": ["VERB", "CCONJ"]}, "OP": "*", "IS_SENT_START": False},
#     {"POS": "VERB", "DEP": {"IN": ["conj", "parataxis"]}, "MORPH": {"IS_SUPERSET": ["Tense=Imp"]}, "IS_SENT_START": False}
#     ]
    

#     matcher.add("b5d1_15_C1", [pattern_pres, pattern_past, pattern_fut, pattern_imp])

#     matches = matcher(doc)

#     return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]