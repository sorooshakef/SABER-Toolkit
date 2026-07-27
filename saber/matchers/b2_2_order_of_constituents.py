from spacy.matcher import Matcher, DependencyMatcher
from process_and_display import nlp_stanza
from text_reconstruction import reconstruct_text

def b2d2_5(doc):
    """
    B2.2-5: ordem inversa (VS) - com frases exclamativas
    e.g., Que bonita é a cidade!
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"LOWER": "que", "IS_SENT_START": True},
        {"POS": {"IN": ["ADJ", "ADV"]}},
        {"LOWER": "se", "OP": "?"},
        {"POS": {"IN": ["VERB", "AUX"]}, "DEP": "root"}
    ]

    matcher.add("b2d2_5_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b2d2_11(doc):
    """
    B2.2-11: ordem inversa (VS) - nas frases com discurso indireto, com verbos de relato do discurso (dizer, sugerir, perguntar, responder...)
    e.g.,  Isso não é possível! - respondeu ela prontamente.
    Level: B1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "LEMMA": {"IN": ["dizer", "sugerir", "perguntar", "responder", "afirmar", "comentar", "explicar", "contar", "informar",
                                 "declarar", "exclamar", "gritar", "reclamar", "protestar", "lamentar", "acrescentar"]},
                "IS_SENT_START": False
                }
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "subject",
            "RIGHT_ATTRS": {
                "DEP": "nsubj"
                },
            "REL_OP": ">++"             
        }
    ]

    
    matcher.add("b2d2_11_B1", [pattern])

    matches = matcher(doc)
    
    results = []
    for match_id, token_ids in matches:
        # Calculate start and end indices
        start = min(token_ids)
        end = max(token_ids) + 1
        tokens_in_span = doc[start:end]
        
        text = reconstruct_text(tokens_in_span)
        results.append((doc.vocab.strings[match_id], text, start, end))

    return results


def b2d2_14(doc):
    """
    B2.2-14: ordem inversa (VS) - nas orações subordinadas adverbiais condicionais sem conjunção
    e.g., Tivessem eles trazido os cães e agora não poderiam entrar. Viesse a chuva mais cedo e já não era necessário continuarmos a rega.
    Level: C1

    Mesoclitics remain a problem of course:
        Conseguissem os alunos terminar o projeto até amanhã, poupar-se-ia muito stress no final do período.
        Disserem eles a verdade logo no início, evitar-se-iam muitos mal-entendidos.
    """
    matcher = Matcher(doc.vocab)

    pattern_cnd_1 = [
            {
            "IS_SENT_START": True,
            "MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Imp", "VerbForm=Fin"]},
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:sse|sses|ssemos|ssem)\\b"}
            },
            {"LOWER": {"NOT_IN": ["e", ","]}, "OP": "+"},
            {"LOWER": {"IN": ["e", ","]}},
            {"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
            # Allowing for participle adjectives
            {"MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}, "OP": "?"},
            {"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
            {"MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Cnd"]}}
            ]
    
    pattern_cnd_2 = [
            {
            "IS_SENT_START": True,
            "MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Imp", "VerbForm=Fin"]},
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:sse|sses|ssemos|ssem)\\b"}
            },
            {"LOWER": {"NOT_IN": ["e", ","]}, "OP": "+"},
            {"LOWER": {"IN": ["e", ","]}},
            {"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
            {"MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}, "OP": "?"},
            {"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
            {
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:eriamos|ariamos|iriamos|ias)\\b"},
            "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}
            }
            ]
    
    pattern_imp_1 = [
            {
            "IS_SENT_START": True,
            "MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Imp", "VerbForm=Fin"]},
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:sse|sses|ssemos|ssem)\\b"}
            },
            {"LOWER": {"NOT_IN": ["e", ","]}, "OP": "+"},
            {"LOWER": {"IN": ["e", ","]}},
            {"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
            {"MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}, "OP": "?"},
            {"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
            {
            "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Imp", "VerbForm=Fin"]},
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:ava|avas|avamos|avam|ia|ias|iamos|iam|inha|inhas|inhamos|inham)\\b"}
            }
            ]
    
    pattern_imp_2 = [
            {
            "IS_SENT_START": True,
            "MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Imp", "VerbForm=Fin"]},
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:sse|sses|ssemos|ssem)\\b"}
            },
            {"LOWER": {"NOT_IN": ["e", ","]}, "OP": "+"},
            {"LOWER": {"IN": ["e", ","]}},
            {"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
            {"MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}, "OP": "?"},
            {"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
            {"NORM": "pagavamos"}
            ]
    
    pattern_imp_3 = [
            {
            "IS_SENT_START": True,
            "MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Imp", "VerbForm=Fin"]},
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:sse|sses|ssemos|ssem)\\b"}
            },
            {"LOWER": {"NOT_IN": ["e", ","]}, "OP": "+"},
            {"LOWER": {"IN": ["e", ","]}},
            {"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
            {"MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}, "OP": "?"},
            {"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
            {
            "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Imp", "VerbForm=Fin"]},
            "LEMMA": {"IN": ["ser", "ter", "vir", "pôr", "por"]}
            }
            ]
    
    pattern_imp_4 = [
            {
            "IS_SENT_START": True,
            "MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Imp", "VerbForm=Fin"]},
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:sse|sses|ssemos|ssem)\\b"}
            },
            {"LOWER": {"NOT_IN": ["e", ","]}, "OP": "+"},
            {"LOWER": {"IN": ["e", ","]}},
            {"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
            {"MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}, "OP": "?"},
            {"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
            {"LOWER": {"IN": ["ia", "ias", "íamos", "íam"]}}
            ]
    
    # To match "Ocorrendo o erro novamente, e seremos obrigados a suspender o serviço."
    pattern_fut_1 = [
            {
            "IS_SENT_START": True,
            "MORPH": "VerbForm=Ger"
            },
            {"LOWER": {"NOT_IN": ["e", ","]}, "OP": "+"},
            {"LOWER": {"IN": ["e", ","]}},
            {"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
            {"MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}, "OP": "?"},
            {"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
            {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Ind", "Tense=Fut"]},
            "TEXT": {"REGEX": r"(ei|ás|á|emos|ão)$"}
            }
            ]
    
    pattern_fut_2 = [
            {
            "IS_SENT_START": True,
            "MORPH": "VerbForm=Ger"
            },
            {"LOWER": {"NOT_IN": ["e", ","]}, "OP": "+"},
            {"LOWER": {"IN": ["e", ","]}},
            {"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
            {"MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}, "OP": "?"},
            {"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
            {"TEXT": {"REGEX": "\\b[A-Za-z]+(?:eremos|aremos|iremos|irás|arás|erás)\\b"},
            "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]},
            "LEMMA": {"NOT_IN": ["querer"]}
            }
            ]
    
    pattern_fut_3 = [
            {
            "IS_SENT_START": True,
            "MORPH": "VerbForm=Ger"
            },
            {"LOWER": {"NOT_IN": ["e", ","]}, "OP": "+"},
            {"LOWER": {"IN": ["e", ","]}},
            {"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
            {"MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}, "OP": "?"},
            {"POS": {"NOT_IN": ["VERB"]}, "OP": "*"},
            {"LOWER": {"IN": ["direi"]}}
            ]
    

    matcher.add("b2d2_14_C1", [pattern_cnd_1, pattern_cnd_2,
                               pattern_imp_1,pattern_imp_2, pattern_imp_3, pattern_imp_4,
                               pattern_fut_1, pattern_fut_2, pattern_fut_3])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]