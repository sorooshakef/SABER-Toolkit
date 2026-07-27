from spacy.matcher import Matcher, DependencyMatcher, PhraseMatcher
from process_and_display import nlp_stanza, nlp_small
from text_reconstruction import reconstruct_text

def b5d2_1(doc):
    """
    B5.2-1: "subordinada substantiva completiva - flexionada - iniciada por que
    - com função de complemento
    - com verbos epistémicos (achar)"
    e.g. Acho que este livro é teu.
    Level: A1
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern = [
    {"LEMMA": "achar"},
    {"NORM": "que", "SENT_START": False},
    {"DEP": {"NOT_IN": ["ccomp"]}, "SENT_START": False, "OP": "*"},
    {"DEP": "ccomp", "SENT_START": False}
    ]

    matcher.add("b5d2_1_A1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d2_2(doc):
    """
    B5.2-2: "subordinada substantiva completiva - flexionada - de infinitivo (impessoal)
    com função de sujeito
    ser"
    e.g. Falar português é fácil.
    Level: A1
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern = [
    {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}, "DEP": {"IN": ["csubj", "cop"]}},
    {"LEMMA": {"NOT_IN": ["ser", "que"]}, "SENT_START": False, "OP": "{,6}", "POS": {"NOT_IN": ["PUNCT"]}},
    {"LEMMA": "ser", "SENT_START": False}
    ]

    matcher.add("b5d2_2_A1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d2_3(doc):
    """
    B5.2-3: "subordinada substantiva completiva - flexionada - de infinitivo (impessoal)
    - com função de complemento
    - com verbos optativos e volitivos (querer)"
    e.g. Quero visitar Lisboa.
    Level: A1
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern = [
    {"LEMMA": "querer", "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}},
    {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}}
    ]

    matcher.add("b5d2_3_A1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d2_4(doc):
    """
    B5.2-4: "subordinada substantiva completiva - flexionada - de infinitivo (impessoal)
    - com função de complemento
    - com verbos epistémicos (saber)"
    e.g. Sei falar português.
    Level: A1
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern = [
    {"LEMMA": "saber", "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}},
    {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}}
    ]

    matcher.add("b5d2_4_A1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d2_5(doc):
    """
    B5.2-5: "subordinada substantiva completiva - flexionada - de infinitivo (impessoal)
    - com função de complemento
    - com verbos modais (pode, dever)"
    e.g. Podes abrir a porta.
    Level: A1
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern = [
    {"LEMMA": {"IN": ["poder", "dever"]}, "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}},
    {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}}
    ]

    matcher.add("b5d2_5_A1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d2_6(doc):
    """
    B5.2-6: "subordinada adverbial - causal 
    porque (+ indicativo)"
    e.g. Gosto de Portugal, porque é um país bonito.
    Level: A1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"LOWER": "porque"},
        # Let's say there might be 6 words between "porque" and the verb 
        {"SENT_START": False, "POS": {"NOT_IN": ["AUX", "VERB"]}, "OP": "{,6}"},
        {"SENT_START": False, "POS": {"IN": ["AUX", "VERB"]}, "MORPH": {"IS_SUPERSET": ["Mood=Ind"]}}
    ]

    # A hack to accommodate indicative verbs that are misidentified as subjuncitve
    # Specifically, "Muitos deles decidem ficar após terminarem os estudos, porque se sentem bem acolhidos pela população local."
    pattern_hack = [
        {"LOWER": "porque"},
        {"SENT_START": False, "POS": {"NOT_IN": ["AUX", "VERB"]}, "OP": "{,6}"},
        {"NORM": {"REGEX": "\\b[A-Za-z]+(?:em|am)\\b"}, "SENT_START": False, "POS": {"IN": ["AUX", "VERB"]}, "MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Pres"]}}
    ]

    matcher.add("b5d2_6_A1", [pattern, pattern_hack])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

# def b5d2_7(doc):
#     """
#     B5.2-7: "subordinada adverbial - final - para (+ infinitivo)"
#     e.g. Estudo português para trabalhar em Portugal.
#     Level: A1
#     """
#     matcher = Matcher(doc.vocab)

#     pattern = [
#     {"LOWER": "para", "POS": "SCONJ"},
#     {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}, "DEP": {"IN": ["advcl", "acl"]}}
#     ]

#     matcher.add("b5d2_7_A1", [pattern])

#     matches = matcher(doc)

#     return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d2_9(doc):
    """
    B5.2-9: "subordinada substantiva completiva - flexionada - iniciada por que
    - com função de complemento
    - com verbos epistémicos (pensar, saber)"
    e.g. Penso que a Ana é uma boa amiga.
    Level: A2
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern = [
    {"LEMMA": {"IN": ["saber", "pensar"]}},
    {"NORM": "que", "SENT_START": False},
    {"DEP": {"NOT_IN": ["ccomp"]}, "SENT_START": False, "OP": "*"},
    {"DEP": "ccomp", "SENT_START": False}
    ]

    matcher.add("b5d2_9_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b5d2_11(doc):
    """
    B5.2-11: "subordinada substantiva completiva - flexionada - de infinitivo (impessoal)
    - com a função de complemento
    - com verbos optativos e volitivos (esperar, desejar, ...)"
    e.g. O Paulo deseja ter um carro novo.
    Level: A2
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    verbos_optativos_volitivos = [
    "esperar",
    "desejar",
    "pretender",
    "planear",
    "tencionar",
    "ambicionar",
    "almejar",
    "anhelar",
    "projetar",
    "pretender",
    "decidir",
    "determinar",
    "visar",
    "intentar"
    ]

    pattern = [
    {"LEMMA": {"IN": verbos_optativos_volitivos}, "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}},
    {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}}
    ]

    matcher.add("b5d2_11_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d2_12(doc):
    """
    B5.2-12: "subordinada substantiva completiva - flexionada - de infinitivo (impessoal)
    - com a função de complemento
    - com verbos epistémicos (pensar...)
    "
    e.g. A Maria pensa ir ao cinema.
    Level: A2
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    other_epistemic_verbs = [
    "achar",
    "crer",
    "imaginar",
    "julgar",
    "pensar",
    "supor"
    ]

    pattern = [
    {"LEMMA": {"IN": other_epistemic_verbs}, "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}},
    {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}}
    ]

    matcher.add("b5d2_12_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d2_13(doc):
    """
    B5.2-13: "subordinada substantiva completiva - flexionada - de infinitivo (impessoal)
    - com a função de complemento
    - com precisar, necessitar + de"
    e.g. Preciso de dormir.
    Level: A2
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern = [
    {"LEMMA": {"IN": ["precisar", "necessitar"]}, "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}},
    {"NORM": "de"},
    {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}}
    ]

    matcher.add("b5d2_13_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d2_14(doc):
    """
    B5.2-14: subordinada adverbial - temporal - enquanto (+ indicativo)
    e.g. Enquanto dormíamos, ouvimos um barulho.
    Level: A2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "enquanto",
            # Removed "POS": "SCONJ" to match "O departamento de segurança monitoriza sempre as instalações enquanto a equipa trabalha."
            "RIGHT_ATTRS": {"LOWER": "enquanto"}
        },
        {
            "LEFT_ID": "enquanto",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB", "MORPH": {"IS_SUPERSET": ["Mood=Ind"]}},
            "REL_OP": "<++"
        }
    ]
    
    matcher.add("b5d2_14_A2", [pattern])

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


def b5d2_15(doc):
    """
    B5.2-15: subordinada adverbial - temporal - desde que (+ indicativo)
    e.g. Desde que cheguei, ainda não vi o Pedro.
    Level: A2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "desde",
            "RIGHT_ATTRS": {"LOWER": "desde", "POS": "ADP"}
        },
        {
            "LEFT_ID": "desde",
            "RIGHT_ID": "que",
            "RIGHT_ATTRS": {"LOWER": "que"},
            "REL_OP": "<+"
        },
        {
            "LEFT_ID": "que",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB", "MORPH": {"IS_SUPERSET": ["Mood=Ind"]}},
            "REL_OP": "<++"
        }
    ]
    
    matcher.add("b5d2_15_A2", [pattern])

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


def b5d2_16(doc):
    """
    B5.2-16: subordinada adverbial - causal - como (+ indicativo)
    e.g. Como nos atrasámos, perdemos o comboio.
    Level: A2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "como",
            "RIGHT_ATTRS": {"LOWER": "como", "POS": "SCONJ"}
        },
        {
            "LEFT_ID": "como",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB", "MORPH": {"IS_SUPERSET": ["Mood=Ind"]}},
            "REL_OP": "<++"
            # "<<" to account for "Como estava a chover, decidimos ficar em casa."
        }
    ]

    # To match "Como estava a chover, decidimos ficar em casa.
    pattern_inf = [
        {
            "RIGHT_ID": "como",
            "RIGHT_ATTRS": {"LOWER": "como", "POS": "SCONJ"}
        },
        {
            "LEFT_ID": "como",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB", "MORPH": "VerbForm=Inf"},
            "REL_OP": "<++"
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "aux",
            "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}, "MORPH": {"IS_SUPERSET": ["Mood=Ind"]}, "DEP": "advcl"},
            "REL_OP": "<--"
        }
    ]
    
    matcher.add("b5d2_16_A2", [pattern, pattern_inf])

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

def b5d2_22(doc):
    """
    B5.2-22: "subordinada substantiva completiva - flexionada - iniciada por que - com função de sujeito - com verbos inacusativos de existência e acontecimento (acontecer, suceder, ocorrer...)"
    e.g. "Sucede que me atrasei e perdi o comboio."
    Level: B2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "root",
            "RIGHT_ATTRS": {
                "LEMMA": {"IN": ["acontecer", "suceder", "ocorrer",
                                 "verificar", "dar",
                                 "constar", "passar", "resultar",
                                 "sobrevir", "Acontece", "sucedir"]},
                # "Acontece" and "sucedir" are added due to the lemmatizer's error.
                "DEP": {"IN": ["root", "conj"]}}
                # "conj" was added to match "Ocorre que, à saída da cidade, já havia uma fila enorme, e dá-se que eu tinha esquecido de atestar o depósito na véspera."
        },
        {
            "LEFT_ID": "root",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["ccomp", "acl:relcl", "csubj"]}
                # "acl:relcl" to match "Acontece, por vezes, que os prazos são mal interpretados pelos responsáveis."
                # "csubj" to match "Ocorre com frequência que os alunos não leem atentamente os enunciados dos testes."
                },
                # Removing "POS": "VERB" to match "Consta que aquela estrada estava em obras havia semanas, mas ninguém me avisara."
            "REL_OP": ">++"
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "que",
            "RIGHT_ATTRS": {
                "LOWER": "que"},
            "REL_OP": ">--"
        }
    ]
    
    matcher.add("b5d2_22_B2", [pattern])

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


def b5d2_26(doc):
    """
    B5.2-26: "subordinada substantiva completiva - flexionada - iniciada por se - com função de complemento - com verbos de inquirição e verbos declarativos com referência ao passado
    e.g. A rapariga perguntava se já tinham chegado a casa.
    Level: B2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "root",
            "RIGHT_ATTRS": {
                "MORPH": {"INTERSECTS": ["Tense=Imp", "Tense=Past"]},
                "DEP": "root"}
        },
        {
            "LEFT_ID": "root",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["ccomp", "acl:relcl", "csubj"]},
                "POS": "VERB",
                "MORPH": {"INTERSECTS": ["VerbForm=Part", "Tense=Imp", "Mood=Sub"]}},
                # "Mood=Sub" to match "O técnico comentou, no final da reunião, se valeria a pena continuar com o projeto."
            "REL_OP": ">++"
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "se",
            "RIGHT_ATTRS": {
                "LOWER": "se",
                "POS": "SCONJ"},
            "REL_OP": ">--"
        }
    ]

    pattern_alt = [
        {
            "RIGHT_ID": "root",
            "RIGHT_ATTRS": {
                "MORPH": {"INTERSECTS": ["Tense=Imp", "Tense=Past"]},
                "DEP": "root"}
        },
        {
            "LEFT_ID": "root",
            "RIGHT_ID": "inf",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
                "DEP": "xcomp"
            },
            "REL_OP": ">+"
        },
        {
            "LEFT_ID": "inf",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["ccomp", "acl:relcl", "csubj"]},
                "POS": "VERB",
                "MORPH": {"INTERSECTS": ["VerbForm=Part", "Tense=Imp", "Mood=Sub"]}},
            "REL_OP": ">++"
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "se",
            "RIGHT_ATTRS": {
                "LOWER": "se",
                "POS": "SCONJ"},
            "REL_OP": ">--"
        }
    ]

    pattern_cop = [
        {
            "RIGHT_ID": "root",
            "RIGHT_ATTRS": {
                "MORPH": {"INTERSECTS": ["Tense=Imp", "Tense=Past"]},
                "DEP": "root"}
        },
        {
            "LEFT_ID": "root",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["ccomp", "acl:relcl", "csubj"]}},
            "REL_OP": ">++"
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "aux",
            "RIGHT_ATTRS": {
                "DEP": "cop",
                "MORPH": {"INTERSECTS": ["VerbForm=Part", "Tense=Imp", "Mood=Sub"]}
            },
            "REL_OP": ">--"
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "se",
            "RIGHT_ATTRS": {
                "LOWER": "se",
                "POS": "SCONJ"},
            "REL_OP": ">--"
        }
    ]
    
    matcher.add("b5d2_26_B2", [pattern, pattern_alt, pattern_cop])

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

def b5d2_27(doc):
    """
    B5.2-27: "subordinada substantiva completiva - flexionada - iniciada por se - com função de complemento - com verbos dubitativos (ignorar, desconhecer, ...)"
    e.g. Ignoro se a Paula vem trabalhar hoje.
    Level: B2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "root",
            "RIGHT_ATTRS": {
                "LEMMA": {"IN": ["ignorar", "desconhecer", "duvidar",
                                 "hesitar", "saber"]},
                "DEP": {"IN": ["root", "ccomp"]}}
                # "ccomp" to match "O técnico confessou que ignorava se o problema estava na configuração do sistema ou num erro humano."
        },
        {
            "LEFT_ID": "root",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["ccomp", "acl:relcl", "csubj"]}
            },
            "REL_OP": ">++"
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "se",
            "RIGHT_ATTRS": {
                "LOWER": "se",
                "POS": "SCONJ"},
            "REL_OP": ";*"
        }
    ]
    
    matcher.add("b5d2_27_B2", [pattern])

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

def b5d2_29(doc):
    """
    B5.2-29: "subordinada substantiva completiva - flexionada - de infinitivo (pessoal / impessoal) - com a função de complemento - com verbos seguidos de preposição"
    e.g. A Maria insistiu em passarmos o fim de semana em casa dela.
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    # We can't use a dependency matcher due to cases such as "O diretor pensa estarem os funcionários a fazer um bom trabalho," where
    # "pensa" leads to "fazer" and then "a," and the only way to prevent that is to implement the intuition that there's going
    # to be smaller distance between the main verb and the infitive than is errnoneously captured here. The commented out dependency matcher
    # is kept for possible future demonstration purposes.

    # matcher = DependencyMatcher(nlp_stanza.vocab)

    # pattern = [
    #     {
    #         "RIGHT_ID": "root",
    #         "RIGHT_ATTRS": {
    #             "DEP": {"IN": ["root", "ccomp"]},
    #             "LEMMA": {"NOT_IN": ["estar", "ir", "ír"]}} # We don't want to match "Eu ia a sair de casa, mas lembrei-me de que estava a chover."
    #             # "ír" due to preprocessing error of "íamos"
    #     },
    #     {
    #         "LEFT_ID": "root",
    #         "RIGHT_ID": "verb",
    #         "RIGHT_ATTRS": {
    #             "DEP": "xcomp",
    #             "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}
    #         },
    #         "REL_OP": ">++"
    #     },
    #     {
    #         "LEFT_ID": "verb",
    #         "RIGHT_ID": "pp",
    #         "RIGHT_ATTRS": {
    #             "POS": "SCONJ"},
    #         "REL_OP": ">--"
    #     }
    # ]

    # # To match "Ninguém contou com sermos forçados a mudar de planos à última hora."
    # pattern_aux = [
    #     {
    #         "RIGHT_ID": "root",
    #         "RIGHT_ATTRS": {
    #             "DEP": {"IN": ["root", "ccomp"]},
    #             "LEMMA": {"NOT_IN": ["estar", "ir", "ír"]}
    #         }
    #     },
    #     {
    #         "LEFT_ID": "root",
    #         "RIGHT_ID": "aux",
    #         "RIGHT_ATTRS": {
    #             "DEP": {"IN": ["cop", "aux:pass"]},
    #             "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
    #             "LEMMA": {"IN": ["ser", "estar"]}
    #         },
    #         "REL_OP": ">>"
    #     },
    #     {
    #         "LEFT_ID": "aux",
    #         "RIGHT_ID": "pp",
    #         "RIGHT_ATTRS": {
    #             "POS": "SCONJ"},
    #         "REL_OP": "$-"
    #     }
    # ]
    
    # matcher.add("b5d2_29_B1", [pattern, pattern_aux])

    # matches = matcher(doc)
    
    # results = []
    # for match_id, token_ids in matches:
    #     # Calculate start and end indices
    #     start = min(token_ids)
    #     end = max(token_ids) + 1
    #     tokens_in_span = doc[start:end]
        
    #     text = reconstruct_text(tokens_in_span)
        
    #     results.append((doc.vocab.strings[match_id], text, start, end))
    # return results



    # Linear signature: main verb -> (short gap) -> SCONJ preposition (not que/se)
    # -> (short gap for an optional subject/negation) -> infinitive verb.
    # The gaps are kept to at most two tokens to avoid the long-distance false
    # positives the DependencyMatcher produced.
    pattern = [
        # Main verb of the matrix clause.
        # "estar"/"ir"/"dever" excluded so we don't match e.g.
        # "Eu ia a sair de casa, mas lembrei-me de que estava a chover."
        # ("ír" is a preprocessing artefact of "íamos".)
        {"POS": {"IN": ["VERB", "AUX"]},
         "DEP": {"IN": ["root", "ccomp"]},
         "LEMMA": {"NOT_IN": ["estar", "ir", "ír", "dever"]}},
        {"POS": {"NOT_IN": ["PUNCT"]}, "SENT_START": False, "OP": "{,2}"},
        # Preposition introducing the infinitival clause (tagged SCONJ by Stanza).
        {"POS": "SCONJ", "LEMMA": {"NOT_IN": ["que", "se"]}, "SENT_START": False},
        # Optional subject/negation before the (personal or impersonal) infinitive.
        {"POS": {"NOT_IN": ["PUNCT", "VERB", "AUX"]}, "SENT_START": False, "OP": "{,2}"},
        # Infinitive: the subordinate verb, or ser/estar in a passive/copular
        # construction (e.g. "contou com sermos forçados a mudar de planos").
        {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
         "POS": {"IN": ["VERB", "AUX"]}, "SENT_START": False},
    ]

    matcher.add("b5d2_29_B1", [pattern], greedy="LONGEST")

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d2_34(doc):
    """
    B5.2-34: "subordinada substantiva relativa - quem, onde, quanto, o que - modificador do grupo verbal"
    e.g. Estacionei onde encontrei lugar.
    Level: B2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "mod",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["onde", "quanto", "quem", "que"]},
                # "nsubj" to match "Simultaneamente, o gerador auxiliar funcionou o que estava estipulado nos parâmetros de contingência."
                "DEP": {"IN": ["advmod", "obl", "nsubj"]}
            }
        },
        {
            "LEFT_ID": "mod",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["VERB", "AUX", "ADJ"]},
                # "ADJ" to match "A equipa de manutenção avançou por onde o corredor de segurança estava desimpedido."
                # "advcl" to match "Os técnicos trabalharam no isolamento da avaria quanto a iluminação de emergência permitiu."
                "DEP": {"IN": ["ccomp", "acl:relcl", "csubj", "advcl"]}
            },
            "REL_OP": "<++"
        }
    ]
    
    matcher.add("b5d2_34_B2", [pattern])

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


def b5d2_35(doc):
    """
    B5.2-35: "subordinada substantiva relativa - quem, onde, quanto, o que - predicativo do sujeito"
    e.g. Ele não é quem eu pensava.
    Level: B2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "mod",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["onde", "quanto", "quem", "que"]},
                "DEP": {"IN": ["obj", "nsubj", "obl"]}
                # "obl" added to match "O novo coordenador da equipa não é quem a administração tinha inicialmente indigitado, mas o seu posto de trabalho passará a ser onde funcionava o antigo departamento de inovação."
            }
        },
        {
            "LEFT_ID": "mod",
            "RIGHT_ID": "verb",
            # "xcomp" to match "O líder responsável por esta iniciativa deve ser quem demonstrar a maior aptidão analítica perante a volatilidade do mercado."
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["ccomp", "acl:relcl", "csubj", "xcomp"]}
            },
            "REL_OP": "<++"
        }
    ]

    pattern_alt = [
        {
            "RIGHT_ID": "root",
            "RIGHT_ATTRS": {
                "DEP": "root"
            }
        },
        {
            "LEFT_ID": "root",
            "RIGHT_ID": "ser",
            "RIGHT_ATTRS": {
                "LEMMA": "ser",
                "POS": "AUX"
            },
            "REL_OP": ">--"
        },
        {
            "LEFT_ID": "ser",
            "RIGHT_ID": "mod",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["onde", "quanto", "quem", "que"]},
                "DEP": "obl"
            },
            "REL_OP": "$++"
        }
    ]


    matcher.add("b5d2_35_B2", [pattern, pattern_alt])

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

def b5d2_42(doc):
    """
    B5.2-42: "subordinada adverbial - temporal - quando (+ conjuntivo)"
    e.g., Quando for a Lisboa, quero visitar a Torre de Belém. 
    Level: B1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "modifier",
            "RIGHT_ATTRS": {
                "LOWER": "quando"}
        
        },
        {
            "LEFT_ID": "modifier",
            "RIGHT_ID": "adverbial",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["acl", "advcl", "cop"]},
                "MORPH": {"IS_SUPERSET": ["Mood=Sub"]}
                },
            "REL_OP": "<++"             
        }
    ]
    
    matcher.add("b5d2_42_B1", [pattern])

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

def b5d2_43(doc):
    """
    B5.2-43: "subordinada adverbial - temporal - enquanto (+ conjuntivo)"
    e.g., Quero viajar pelo mundo enquanto for jovem.
    Level: B1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "modifier",
            "RIGHT_ATTRS": {
                "LOWER": "enquanto"}
        },
        {
            "LEFT_ID": "modifier",
            "RIGHT_ID": "adverbial",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["acl", "advcl", "cop"]},
                "MORPH": {"IS_SUPERSET": ["Mood=Sub"]}
                },
            "REL_OP": "<++"             
        }
    ]

    pattern_alt = [
        {
            "RIGHT_ID": "modifier",
            "RIGHT_ATTRS": {
                "LOWER": "enquanto"}
        },
        {
            "LEFT_ID": "modifier",
            "RIGHT_ID": "adverbial",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["acl", "advcl", "cop", "aux:pass"]},
                # "aux:pass" to match "Pretendo continuar a estudar enquanto me seja permitido conciliar os horários."
                "MORPH": {"IS_SUPERSET": ["Mood=Sub"]}
                },
            "REL_OP": "$++"             
        }
    ]
    
    matcher.add("b5d2_43_B1", [pattern, pattern_alt])

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

def b5d2_44(doc):
    """
    B5.2-44: "subordinada adverbial - temporal - assim que, logo que, mal (+ conjuntivo)"
    e.g., Assim que chegar ao Porto, telefono-te. Mal chegue ao trabalho, termino o relatório.
    Level: B1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "adv",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["assim", "logo"]}}
        },
        {
            "LEFT_ID": "adv",
            "RIGHT_ID": "que",
            "RIGHT_ATTRS": {
                "LOWER": "que"
            },
            "REL_OP": ">+"
        },
        {
            "LEFT_ID": "adv",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["acl", "advcl", "cop"]},
                "MORPH": {"IS_SUPERSET": ["Mood=Sub"]}
                },
            "REL_OP": "<++"             
        }
    ]

    pattern_alt = [
        {
            "RIGHT_ID": "adv",
            "RIGHT_ATTRS": {
                "LOWER": "mal"}
        },
        {
            "LEFT_ID": "adv",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                #"DEP": {"IN": ["acl", "advcl", "cop", "root"]},
                # Too many DEPs. Why do we bother?
                "MORPH": {"IS_SUPERSET": ["Mood=Sub"]}
                },
            "REL_OP": "<++"             
        }
    ]

    
    matcher.add("b5d2_44_B1", [pattern, pattern_alt])

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


def b5d2_45(doc):
    """
    B5.2-45: "subordinada adverbial - temporal - sempre que (+ conjuntivo)"
    e.g., Podes contar com a minha ajuda sempre que precisares.
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"LOWER": "sempre"},
        {"LOWER": "que"},
        {"POS": {"NOT_IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "OP": "*"},
        {"MORPH": {"IS_SUPERSET": ["Mood=Sub"]}, "IS_SENT_START": False}
    ]

    matcher.add("b5d2_45_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d2_46(doc):
    """
    B5.2-46: "subordinada adverbial - temporal - (de) cada vez que, (de) todas as vezes que (+ conjuntivo)"
    e.g., Cada vez que visitar a nossa loja, recebe um desconto.
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"LOWER": "cada"},
        {"LOWER": "vez"},
        {"LOWER": "que"},
        {"POS": {"NOT_IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "OP": "*"},
        {"MORPH": {"INTERSECTS": ["Mood=Sub", "VerbForm=Inf"]}, "IS_SENT_START": False}
    ]

    pattern_alt = [
        {"LOWER": "todas"},
        {"LOWER": "as"},
        {"LOWER": "vezes"},
        {"LOWER": "que"},
        {"POS": {"NOT_IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "OP": "*"},
        {"MORPH": {"INTERSECTS": ["Mood=Sub", "VerbForm=Inf"]}, "IS_SENT_START": False}
    ]

    matcher.add("b5d2_46_B1", [pattern, pattern_alt])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d2_51(doc):
    """
    B5.2-51: "subordinada adverbial - condicional - a menos que, a não ser que (+ conjuntivo)"
    e.g., "Iremos ao jantar de despedida, a não ser que haja algum imprevisto."
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"LOWER": "a"},
        {"LOWER": "menos"},
        {"LOWER": "que"},
        {"POS": {"NOT_IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "OP": "*"},
        {"MORPH": {"INTERSECTS": ["Mood=Sub", "VerbForm=Inf"]}, "IS_SENT_START": False}
    ]

    pattern_alt = [
        {"LOWER": "a"},
        {"NORM": "nao"},
        {"LOWER": "ser"},
        {"LOWER": "que"},
        {"POS": {"NOT_IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "OP": "*"},
        {"MORPH": {"INTERSECTS": ["Mood=Sub", "VerbForm=Inf"]}, "IS_SENT_START": False}
    ]

    matcher.add("b5d2_51_B1", [pattern, pattern_alt])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d2_53(doc):
    """
    B5.2-53: "subordinada adverbial - comparativa - como se
    e.g. Falámos como se nos conhecêssemos há muito tempo.
    Level: B1
    """
    matcher = PhraseMatcher(doc.vocab, attr="NORM")

    patterns = list(nlp_stanza.pipe(["como se"]))

    matcher.add("b5d2_53_B1", patterns)

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b5d2_54(doc):
    """
    B5.2-54: ""subordinada adverbial conformativa - consoante, conforme"
    e.g., Faz conforme achares melhor.
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"LOWER": {"IN": ["conforme", "consoante"]}},
        {"POS": {"NOT_IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "OP": "{,3}"},
        {"MORPH": {"INTERSECTS": ["Mood=Sub", "VerbForm=Inf", "Tense=Imp"]}, "IS_SENT_START": False}
    ]


    matcher.add("b5d2_54_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b5d2_56(doc):
    """
    B5.2-56: "subordinada substantiva completiva - flexionada
    iniciada por que
    - com função de complemento
    - com verbos que ocorrem em colocação com nomes (dar/fazer pena, fazer falta...)"
    e.g. Faz falta que os jovens se interessem mais pela política.
    Level: C1
    Requires nlp_stanza
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"LEMMA": {"IN": ["dar", "fazer"]}},
        {"LOWER": {"IN": ["bem", "falta", "pena", "medo", "sentido", "esperança", "tristeza", "diferença", "alívio",
                         "desgosto", "tranquilidade", "alegria", "remorso", "orgulho", "conforto", "satisfação",
                         "raiva", "desespero", "ânimo", "mal", "confusão", "questão", "importância"]}},
        {"NORM": {"NOT_IN": ["que"]}, "SENT_START": False, "OP": "{,3}"},
        # A maximum of three tokens in-between to capture possible parenthetical expressions
        {"NORM": "que"}
    ]

    matcher.add("b5d2_56_C1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

# def b5d2_57(doc):
#     """
#     B5.2-57: "subordinada substantiva completiva - flexionada
#     iniciada por que
#     - com função de complemento
#     - com verbos introduzidos por que ou se, com alterações de significado"
#     e.g. Os cientistas vão descobrir que há vida em Marte. / Os cientistas vão descobrir se há vida em Marte.
#     Level: C1
    
#     Because the tricky part about this is using "se" instead of "que", that's what we're going to identify.

#     There are some misses because sometimes the preprocessor annotates "se" as a PRON when it's actually a
#     SCONJ, but it's too dangerous to be too flexible on this issue, as we  may end up matching some clitics.
#     """
#     matcher = DependencyMatcher(nlp_stanza.vocab)

#     verbs_que_se_change = [
#     "descobrir",
#     "saber",
#     "verificar",
#     "confirmar",
#     "averiguar",
#     "determinar",
#     "compreender",
#     "aprender",
#     "ver",
#     "perceber",
#     "esclarecer",
#     "informar",
#     "acreditar",
#     "duvidar",
#     "questionar",
#     "perguntar",
#     "interrogar",
#     "ponder",
#     "averiguar",
#     "decidir"
#     # No "hesitar" because we don't want to match "Quando lho perguntei, hesitou, como se não o tivesse reconhecido."
#     ]

#     pattern = [
#         {
#             "RIGHT_ID": "verb",
#             "RIGHT_ATTRS":{
#                 "POS": "VERB",
#                 "DEP": {"IN": ["root", "conj"]},
#                 # To correctly match the following sentence: "Ontem, quando viu a Joana, imediatamente lhe perguntou se tinha planos para o dia."
#                 "LEMMA": {"IN": verbs_que_se_change}
#             }
#         },
#         {
#             "LEFT_ID": "verb",
#             "RIGHT_ID": "se",
#             "RIGHT_ATTRS":{
#                 "LOWER": "se",
#                 "POS": "SCONJ"
#             },
#             "REL_OP": ".*"
#         }
#     ]

#     matcher.add("b5d2_57_C1", [pattern])

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


def b5d2_58(doc):
    """
        B5.2-58: subordinada substantiva completiva - flexionada
        de infinitivo (pessoal)
    - com diferentes verbos, com infinitivo pessoal composto   
    e.g. O Ivo reconheceu ter sido muito teimoso. Os alunos lamentaram terem chegado atrasados à aula.

    DON'T MATCH: fosse por terem seguido (We're not.)
    Level: C1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    # The preprocessor may identify past participle as either a verb or an adjective.
    # In case it's identified as an adjective:
    pattern_adj_dep1 = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}
                         }
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"}},
            "REL_OP": ">"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
            "TEXT": {"REGEX": ".*(?:mos|es|em)$"},
            "LEMMA": {"IN": ["ter", "terem"]}},
            "REL_OP": "."            
        }
    ]
    
    # In case it's identified as a verb

    pattern_verb_dep1 = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}
                         }
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "pp",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": ">"             
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
            "TEXT": {"REGEX": ".*(?:mos|es|em)$"},
            # "termos" to match "A equipa de investigação lamentou termos perdido o prazo de submissão do artigo, mas os gestores negaram terem ignorado os alertas do sistema."
            "LEMMA": {"IN": ["ter", "terem", "termos"]}},
            "REL_OP": "."
        }
    ]

    # Adressing the issue of "termos" being identified as a noun

    pattern_adj_termos = [
        {   
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}
                         }
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "termos",
            "RIGHT_ATTRS":{
                "LOWER": "termos",
                "POS": "NOUN"
            },
            "REL_OP": ">+"
        },
        {
            "LEFT_ID": "termos",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da)$"}},
            "REL_OP": ">+"             
        }
    ] 

    
    pattern_verb_termos = [
        {   
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}
                         }
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "termos",
            "RIGHT_ATTRS":{
                "LOWER": "termos",
                "POS": "NOUN"
            },
            "REL_OP": ">+"
        },
        {
            "LEFT_ID": "termos",
            "RIGHT_ID": "pp",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": ">+"
        }
    ]

    # An ADV (typically a negador such as "não"/"nunca"/"jamais") can wedge itself
    # into the slot the infinitive occupies in pattern_verb_dep1/pattern_adj_dep1,
    # breaking the "verb . ter" adjacency, as in "...e fingiram não terem percebido...".
    # Here the personal infinitive ("terem") and the ADV that immediately precedes it
    # are both children of the past participle (which is itself the child of the finite
    # verb), and it is the ADV — not the infinitive — that fills the slot right after the
    # finite verb. (The DependencyMatcher allows only one relation per node, so we pin the
    # ADV to the infinitive via "immediately precedes"; the finite-verb adjacency then
    # follows from the participle being the finite verb's dependent.)
    pattern_verb_adv = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}
                         }
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "pp",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "pp",
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
            "TEXT": {"REGEX": ".*(?:mos|es|em)$"},
            "LEMMA": {"IN": ["ter", "terem", "termos"]}},
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adv",
            "RIGHT_ATTRS": {"POS": "ADV"},
            "REL_OP": ";"  # the ADV immediately precedes the infinitive
        }
    ]

    # Same consideration for when the past participle is identified as an adjective.
    pattern_adj_adv = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}
                         }
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"}},
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "adj",
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
            "TEXT": {"REGEX": ".*(?:mos|es|em)$"},
            "LEMMA": {"IN": ["ter", "terem", "termos"]}},
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adv",
            "RIGHT_ATTRS": {"POS": "ADV"},
            "REL_OP": ";"  # the ADV immediately precedes the infinitive
        }
    ]

    matcher.add("b5d2_58_C1", [
                                pattern_adj_dep1, pattern_verb_dep1,
                                pattern_adj_termos, pattern_verb_termos,
                                pattern_verb_adv, pattern_adj_adv
                            ]
                )

    matches = matcher(doc)

    results = []
    for match_id, token_ids in matches:
        # Calculate start and end indices
        start = min(token_ids)
        end = max(token_ids) + 1
        tokens_in_span = doc[start:end]

        text = reconstruct_text(tokens_in_span)

        results.append((doc.vocab.strings[match_id], text, start, end))

    # -------------------------------------------------------------------------
    # Disambiguating the AMBIGUOUS singular "ter".
    #
    # The compound personal infinitive is only morphologically distinct in the
    # 2nd person sing. (teres), 1st/2nd person plur. (termos/terdes) and 3rd
    # person plur. (terem); the patterns above rely on those endings. In the 1st
    # and 3rd person singular, however, "ter + particípio" is spelled exactly
    # like the impersonal form, so it slips through.
    #
    # We recover it with the following intuition: when a finite verb (a)
    # immediately precedes the infinitive ("REL_OP": ".") and (b) carries its own
    # overt subject (an "nsubj" child), the infinitive is personal
    # (e.g. "A coordenadora declarou ter concluído a auditoria.").
    #
    # The matrix verb is restricted to exclude the semi-auxiliaries / modals /
    # aspectuais listed below, which instead govern an *impersonal* infinitive
    # (e.g. "Ela deve ter feito o trabalho." is not a personal infinitive).
    aux_lemmas = ["dever", "poder", "querer", "ir", "vir", "andar", "estar",
                  "ficar", "começar", "continuar", "acabar", "deixar", "costumar",
                  "haver", "ter", "passar", "voltar", "chegar", "conseguir",
                  "saber", "tornar"]

    matcher_nsubj = DependencyMatcher(nlp_stanza.vocab)

    # Participle attached directly to the matrix verb (e.g. "declarou ter concluído").
    pattern_nsubj_pp = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]},
                "LEMMA": {"NOT_IN": aux_lemmas}
            }
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "nsubj",
            "RIGHT_ATTRS": {"DEP": "nsubj"},
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "pp",
            "RIGHT_ATTRS": {"MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "LEMMA": "ter",
                "LOWER": "ter",
                "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}
            },
            "REL_OP": "."
        }
    ]

    # Copular variant, where the participle ("sido"/"estado") hangs off the
    # predicate rather than the matrix verb (e.g. "reconheceu ter sido teimoso").
    pattern_nsubj_cop = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]},
                "LEMMA": {"NOT_IN": aux_lemmas}
            }
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "nsubj",
            "RIGHT_ATTRS": {"DEP": "nsubj"},
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "pred",
            "RIGHT_ATTRS": {},
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "pred",
            "RIGHT_ID": "pp",
            "RIGHT_ATTRS": {"MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "LEMMA": "ter",
                "LOWER": "ter",
                "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}
            },
            "REL_OP": "."
        }
    ]

    matcher_nsubj.add("b5d2_58_C1", [pattern_nsubj_pp, pattern_nsubj_cop])

    seen_spans = {(s, e) for _, _, s, e in results}
    for match_id, token_ids in matcher_nsubj(doc):
        # "nsubj" is always the 2nd node in the patterns above; drop it so the
        # highlighted span covers the matrix verb + infinitive, not the subject.
        span_ids = [tid for i, tid in enumerate(token_ids) if i != 1]
        start = min(span_ids)
        end = max(span_ids) + 1
        if (start, end) in seen_spans:
            continue
        seen_spans.add((start, end))
        text = reconstruct_text(doc[start:end])
        results.append((doc.vocab.strings[match_id], text, start, end))

    return results


def b5d2_62(doc):
    """
    B5.2-62: "subordinada adverbial - causal - participial"
    e.g. Lesionado, o atleta não participou na maratona.
    Level: C1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "pp",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
                "NORM": {"REGEX": "\\b[A-Za-z]+(?:do|da|dos|das)\\b"},
                "IS_SENT_START": True,
                "DEP": "advcl"
            }
        },
        {
            "LEFT_ID": "pp",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB", "DEP": "root"},
            "REL_OP": "<++"
        }
    ]

    pattern_adj = [
        {
            "RIGHT_ID": "pp",
            "RIGHT_ATTRS": {
                "POS": "ADJ",
                "NORM": {"REGEX": "\\b[A-Za-z]+(?:do|da|dos|das)\\b"},
                "IS_SENT_START": True,
                "DEP": "amod"
            }
        },
        {
            "LEFT_ID": "pp",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB", "DEP": "root"},
            "REL_OP": "<++"
        }
    ]
    
    matcher.add("b5d2_62_C1", [pattern, pattern_adj])

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


def b5d2_63(doc):
    """
    B5.2-63: "subordinada adverbial - causal - devido a, devido ao facto de (+ infinitivo)"
    e.g. Este livro foi um sucesso de vendas devido ao facto de ter recebido um prémio.
    Level: C1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
    {"LOWER": "devido"},
    {"SENT_START": False, "OP": "*"},
    {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}, "SENT_START": False}
    ]

    matcher.add("b5d2_63_C1", [pattern])

    matches = matcher(doc)
    
    # Filter overlapping matches, keeping the one with smallest end index
    filtered_matches = []
    for match_id, start, end in matches:
        # Check if this match overlaps with any existing filtered match
        overlap_found = False
        for i, (f_match_id, f_start, f_end) in enumerate(filtered_matches):
            # Check for overlap
            if (start < f_end and end > f_start):
                overlap_found = True
                # If current match has smaller end index, replace the existing one
                if end < f_end:
                    filtered_matches[i] = (match_id, start, end)
                break
        
        # If no overlap found, add this match
        if not overlap_found:
            filtered_matches.append((match_id, start, end))

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in filtered_matches]


def b5d2_64(doc):
    """
    B5.2-64: "subordinada adverbial - causal - graças a, por culpa de (+ infinitivo)"
    e.g. Ele tem dificuldade em respirar por culpa de fumar tanto.
    Level: C1
    """
    matcher = Matcher(doc.vocab)

    pattern_culpa = [
    {"LOWER": "por"},
    {"LOWER": "culpa"},
    {"LOWER": "de"},
    {"SENT_START": False, "OP": "*"},
    {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}, "SENT_START": False}
    ]

    pattern_gracas = [
    {"NORM": "gracas"},
    {"LOWER": "a"},
    {"SENT_START": False, "OP": "*"},
    {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}, "SENT_START": False}
    ]

    matcher.add("b5d2_64_C1", [pattern_culpa, pattern_gracas])

    matches = matcher(doc)
    
    # Filter overlapping matches, keeping the one with smallest end index
    filtered_matches = []
    for match_id, start, end in matches:
        # Check if this match overlaps with any existing filtered match
        overlap_found = False
        for i, (f_match_id, f_start, f_end) in enumerate(filtered_matches):
            # Check for overlap
            if (start < f_end and end > f_start):
                overlap_found = True
                # If current match has smaller end index, replace the existing one
                if end < f_end:
                    filtered_matches[i] = (match_id, start, end)
                break
        
        # If no overlap found, add this match
        if not overlap_found:
            filtered_matches.append((match_id, start, end))

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in filtered_matches]


def b5d2_65(doc):
    """
    B5.2-65: "subordinada adverbial - temporal - mal...(logo) (+ indicativo / conjuntivo)"
    e.g. Mal entrei em casa, apercebi-me logo de que alguém lá tinha estado.
    Level: C1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
    {"LOWER": "mal", "DEP": "advmod"},
    {"LOWER": {"NOT_IN": ["logo"]}, "OP": "*", "IS_SENT_START": False},
    {"LOWER": "logo", "IS_SENT_START": False, "DEP": "advmod"}
    ]

    matcher.add("b5d2_65_C1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b5d2_66(doc):
    """
    B5.2-66: "subordinada adverbial - temporal - à medida que, ao passo que (+ indicativo / conjuntivo)"
    e.g. À medida que a economia melhora, o desemprego diminui.
    Level: C1
    """
    matcher = PhraseMatcher(doc.vocab, attr="NORM")

    patterns = list(nlp_stanza.pipe(["ao passo que", "a medida que"]))

    matcher.add("b5d2_66_C1", patterns)

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b5d2_67(doc):
    """
    B5.2-67: ""subordinada adverbial - temporal - no momento em que, na altura em que (+ indicativo)""
    e.g. Fez-se silêncio no momento em que ele subiu ao palco.
    Level: C1
    """
    matcher = PhraseMatcher(doc.vocab, attr="LOWER")

    patterns = list(nlp_small.pipe(["no momento em que", "na altura em que"]))

    matcher.add("b5d2_67_C1", patterns)

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
b5d2_67.REQUIRES_SPACY = True


def b5d2_69(doc):
    """
    B5.2-69: "subordinada adverbial - concessiva - por muito...que (+ conjuntivo)"
    e.g. Por muito empenhada que seja a Maria, não creio que consiga este emprego.
    Level: C1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
    {"LOWER": "por"},
    {"LEMMA": "muito"},
    {"LOWER": {"NOT_IN": ["que"]}, "IS_SENT_START": False, "OP": "*"},
    {"LOWER": "que", "IS_SENT_START": False},
    {"POS": {"NOT_IN": ["VERB"]}, "IS_SENT_START": False, "OP": "*"},
    {"MORPH": {"IS_SUPERSET": ["Mood=Sub", "VerbForm=Fin"]}, "IS_SENT_START": False}
    ]

    matcher.add("b5d2_69_C1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b5d2_70(doc):
    """
    B5.2-70: "subordinada adverbial - concessiva - gerundiva"
    e.g., Mesmo não gostando de calor, fui à praia.
    Level: C1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern_gerund = [
        {
            "RIGHT_ID": "mesmo",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["mesmo", "ainda"]},
                "POS": "ADV",
                "DEP": "advmod"}
        },
        {
            "LEFT_ID": "mesmo",
            "RIGHT_ID": "adverbial",
            "RIGHT_ATTRS": {
                "DEP": "advcl",
                "MORPH": {"INTERSECTS": ["VerbForm=Ger"]}
                },
            "REL_OP": "<"
        },
        {
            "LEFT_ID": "adverbial",
            "RIGHT_ID": "root",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["root", "xcomp"]}
                # "xcomp" for "O Pedro continuou a treinar, mesmo estando lesionado."
                },
            "REL_OP": "<"           
        }
    ]

    pattern_gerund_pp = [
        {
            "RIGHT_ID": "mesmo",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["mesmo", "ainda"]},
                "POS": "ADV",
                "DEP": "advmod"}
        },
        {
            "LEFT_ID": "mesmo",
            "RIGHT_ID": "pp",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
                },
            "REL_OP": "<"           
        },
                {
            "LEFT_ID": "pp",
            "RIGHT_ID": "aux",
            "RIGHT_ATTRS": {
                "POS": "AUX",
                "MORPH": {"IS_SUPERSET": ["VerbForm=Ger"]}
                },
            "REL_OP": ">--"
        },
        {
            "LEFT_ID": "pp",
            "RIGHT_ID": "root",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["root", "xcomp"]}
            },
            "REL_OP": "<"
        }
    ]

    pattern_gerund_pp_adj = [
        {
            "RIGHT_ID": "mesmo",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["mesmo", "ainda"]},
                "POS": "ADV",
                "DEP": "advmod"}
        },
        {
            "LEFT_ID": "mesmo",
            "RIGHT_ID": "pp",
            "RIGHT_ATTRS": {
                "POS": "ADJ",
                "TEXT": {"REGEX": ".*(?:do|da)$"}
                },
            "REL_OP": "<"           
        },
                {
            "LEFT_ID": "pp",
            "RIGHT_ID": "aux",
            "RIGHT_ATTRS": {
                "POS": "AUX",
                "MORPH": {"IS_SUPERSET": ["VerbForm=Ger"]}
                },
            "REL_OP": ">--"
        },
        {
            "LEFT_ID": "pp",
            "RIGHT_ID": "root",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["root", "xcomp"]}
            },
            "REL_OP": "<"
        }
    ]


    pattern_adj = [
        {
            "RIGHT_ID": "mesmo",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["mesmo", "ainda"]},
                "POS": "ADV",
                "DEP": "advmod"}
        },
        {
            "LEFT_ID": "mesmo",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
                "POS": "ADJ",
                "DEP": "advcl"
                },
            "REL_OP": "<++"           
        },
                {
            "LEFT_ID": "adj",
            "RIGHT_ID": "aux",
            "RIGHT_ATTRS": {
                "POS": "AUX",
                "MORPH": {"IS_SUPERSET": ["VerbForm=Ger"]},
                "DEP": "cop"
                },
            "REL_OP": ">--"
        },
        {
            "LEFT_ID": "adj",
            "RIGHT_ID": "root",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["root", "xcomp"]}
            },
            "REL_OP": "<"
        }
    ]

    matcher.add("b5d2_70_C1", [pattern_gerund, pattern_gerund_pp, pattern_gerund_pp_adj,
                               pattern_adj])

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


def b5d2_71(doc):
    """
    B5.2-71: "subordinada adverbial - concessiva - participial"
    e.g. Embora cansado, trabalhei até de madrugada.
    Level: C1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "embora",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["embora", "ainda", "mesmo"]}
                }
        },
        {
            "LEFT_ID": "embora",
            "RIGHT_ID": "pp",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
                # "NORM": {"REGEX": "\\b[A-Za-z]+(?:do|da|dos|das)\\b"},
                "DEP": "advcl"
            },
            "REL_OP": "<++"
        },
        {
            "LEFT_ID": "pp",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB", "DEP": "root"},
            "REL_OP": "<++"
        }
    ]

    pattern_adj = [
        {
            "RIGHT_ID": "embora",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["embora", "ainda", "mesmo"]}
                }
        },
        {
            "LEFT_ID": "embora",
            "RIGHT_ID": "pp",
            "RIGHT_ATTRS": {
                "POS": "ADJ",
                "NORM": {"REGEX": "\\b[A-Za-z]+(?:do|da|dos|das)\\b"},
                "DEP": "amod"
            },
            "REL_OP": "<++"
        },
        {
            "LEFT_ID": "pp",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB", "DEP": "root"},
            "REL_OP": "<++"
        }
    ]

    pattern_reversed = [
        {
            "RIGHT_ID": "embora",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["embora", "ainda", "mesmo"]}
                # Removed "POS": "SCONJ" and "DEP": "mark" to match "Mesmo alertados para as anomalias geológicas do terreno, os responsáveis mantiveram as especificações das fundações originais."
            }
        },
        {
            "LEFT_ID": "embora",
            "RIGHT_ID": "pp",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
                # "NORM": {"REGEX": "\\b[A-Za-z]+(?:do|da|dos|das)\\b"},
                "DEP": "advcl"
            },
            "REL_OP": ">++"
        },
        {
            "LEFT_ID": "pp",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB", "DEP": "root"},
            "REL_OP": "<++"
        }
    ]

    pattern_adj_reversed = [
        {
            "RIGHT_ID": "embora",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["embora", "ainda", "mesmo"]}
                }
        },
        {
            "LEFT_ID": "embora",
            "RIGHT_ID": "pp",
            "RIGHT_ATTRS": {
                "POS": "ADJ",
                "NORM": {"REGEX": "\\b[A-Za-z]+(?:do|da|dos|das)\\b"},
                "DEP": "amod"
            },
            "REL_OP": "<++"
        },
        {
            "LEFT_ID": "pp",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB", "DEP": "root"},
            "REL_OP": ">++"
        }
    ]

    # "se bem que" concessive: anchor "se" with fixed children "bem" and "que"
    pattern_se = [
        {
            "RIGHT_ID": "se",
            "RIGHT_ATTRS": {"LOWER": "se"}
        },
        {
            "LEFT_ID": "se",
            "RIGHT_ID": "bem",
            "RIGHT_ATTRS": {"LOWER": "bem", "DEP": "fixed"},
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "se",
            "RIGHT_ID": "que",
            "RIGHT_ATTRS": {"LOWER": "que", "DEP": "fixed"},
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "se",
            "RIGHT_ID": "pp",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
                # "NORM": {"REGEX": "\\b[A-Za-z]+(?:do|da|dos|das)\\b"},
                "DEP": "advcl"
            },
            "REL_OP": "<++"
        },
        {
            "LEFT_ID": "pp",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB", "DEP": "root"},
            "REL_OP": "<++"
        }
    ]

    pattern_se_adj = [
        {
            "RIGHT_ID": "se",
            "RIGHT_ATTRS": {"LOWER": "se"}
        },
        {
            "LEFT_ID": "se",
            "RIGHT_ID": "bem",
            "RIGHT_ATTRS": {"LOWER": "bem", "DEP": "fixed"},
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "se",
            "RIGHT_ID": "que",
            "RIGHT_ATTRS": {"LOWER": "que", "DEP": "fixed"},
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "se",
            "RIGHT_ID": "pp",
            "RIGHT_ATTRS": {
                "POS": "ADJ",
                "NORM": {"REGEX": "\\b[A-Za-z]+(?:do|da|dos|das)\\b"},
                "DEP": "amod"
            },
            "REL_OP": "<++"
        },
        {
            "LEFT_ID": "pp",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB", "DEP": "root"},
            "REL_OP": "<++"
        }
    ]

    pattern_se_reversed = [
        {
            "RIGHT_ID": "se",
            "RIGHT_ATTRS": {"LOWER": "se"}
        },
        {
            "LEFT_ID": "se",
            "RIGHT_ID": "bem",
            "RIGHT_ATTRS": {"LOWER": "bem", "DEP": "fixed"},
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "se",
            "RIGHT_ID": "que",
            "RIGHT_ATTRS": {"LOWER": "que", "DEP": "fixed"},
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "se",
            "RIGHT_ID": "pp",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
                # "NORM": {"REGEX": "\\b[A-Za-z]+(?:do|da|dos|das)\\b"},
                "DEP": "advcl"
            },
            "REL_OP": ">++"
        },
        {
            "LEFT_ID": "pp",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB", "DEP": "root"},
            "REL_OP": "<++"
        }
    ]

    pattern_se_adj_reversed = [
        {
            "RIGHT_ID": "se",
            "RIGHT_ATTRS": {"LOWER": "se"}
        },
        {
            "LEFT_ID": "se",
            "RIGHT_ID": "bem",
            "RIGHT_ATTRS": {"LOWER": "bem", "DEP": "fixed"},
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "se",
            "RIGHT_ID": "que",
            "RIGHT_ATTRS": {"LOWER": "que", "DEP": "fixed"},
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "se",
            "RIGHT_ID": "pp",
            "RIGHT_ATTRS": {
                "POS": "ADJ",
                "NORM": {"REGEX": "\\b[A-Za-z]+(?:do|da|dos|das)\\b"},
                "DEP": "amod"
            },
            "REL_OP": "<++"
        },
        {
            "LEFT_ID": "pp",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB", "DEP": "root"},
            "REL_OP": ">++"
        }
    ]

    matcher.add("b5d2_62_C1", [pattern, pattern_adj, pattern_reversed, pattern_adj_reversed,
                               pattern_se, pattern_se_adj, pattern_se_reversed, pattern_se_adj_reversed])

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


def b5d2_73(doc):
    """
    B5.2-73: "subordinada adverbial - condicional - se porventura"
    e.g. Teria ajudado o João, se porventura tivesse sabido da sua situação.
    Level: C1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
    {"LOWER": "se", "POS": "SCONJ", "DEP": "mark"},
    {"LOWER": "porventura"},
    {"POS": {"NOT_IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "OP": "*", "IS_SENT_START": False},
    {"MORPH": {"IS_SUPERSET": ["Mood=Sub", "VerbForm=Fin"]}, "IS_SENT_START": False}
    ]

    matcher.add("b5d2_73_C1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b5d2_74(doc):
    """
    B5.2-74: ""subordinada adverbial - condicional - salvo se (+ conjuntivo)""
    e.g. O contrato é automaticamente renovado, salvo se uma das partes decidir de outro modo.
    Level: C1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
    {"LOWER": "salvo"},
    {"LOWER": "se"},
    {"POS": {"NOT_IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "OP": "*", "IS_SENT_START": False},
    {"MORPH": {"INTERSECTS": ["Mood=Sub", "VerbForm=Inf"]}, "IS_SENT_START": False}
    ]

    matcher.add("b5d2_74_C1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b5d2_75(doc):
    """
    B5.2-75: "subordinada substantiva completiva - flexionada - de infinitivo (pessoal) - com verbos declarativos e epistémicos com alteração de ordem de palavras"
    e.g. A Paula afirmou ter o João sido o responsável pela situação. O diretor pensa terem os funcionários feito um bom trabalho.
    Level: C2

    This pattern used to be more complicated and encompassed participle and progressive verbs as well.
    I've now simplified it. If the precision or recall are unacceptable, we may now be able to explore using
    dependency matchers as well now that the pattern is more simple.
    """
    matcher = Matcher(doc.vocab)

    # This construction (personal-infinitive completive clause with subject
    # inversion) is licensed only by declarative (verba dicendi) and epistemic
    # (verba cogitandi) matrix verbs. Constraining the first verb to this closed
    # semantic class prevents false positives with volitional/control verbs such
    # as "querer", e.g. "Ela quer ter boas notas, logo não pode faltar às aulas."
    declarative_epistemic = [
        # Declarative (verba dicendi)
        "afirmar", "dizer", "declarar", "anunciar", "comunicar", "referir",
        "mencionar", "alegar", "garantir", "assegurar", "confirmar", "negar",
        "admitir", "reconhecer", "revelar", "sustentar", "defender", "jurar",
        "prometer", "informar", "relatar", "contar", "proclamar", "salientar",
        "sublinhar", "apontar", "indicar", "esclarecer", "notar", "observar",
        "comentar", "argumentar", "responder", "acrescentar", "frisar",
        # Epistemic (verba cogitandi / putandi)
        "pensar", "crer", "achar", "considerar", "julgar", "supor", "imaginar",
        "acreditar", "presumir", "entender", "calcular", "estimar", "concluir",
        "deduzir", "suspeitar", "saber", "verificar", "constatar", "assumir",
        "depreender", "inferir",
    ]

    pattern = [
    {"POS": "VERB", "DEP": "root", "LEMMA": {"IN": declarative_epistemic}},
    {"POS": {"NOT_IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "OP": "{,3}"},
    # "DEP": "cop" to match "Durante a audiência preliminar, o perito financeiro afirmou ser o esquema contabilístico altamente complexo e ardiloso."
    {"LEMMA": {"IN": ["ter", "ser", "estar", "haver"]}, "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}, "DEP": {"IN": ["xcomp", "cop"]}, "IS_SENT_START": False},
    {"DEP": {"NOT_IN": ["nsubj", "obj", "nsubj:pass", "xcomp"]}, "IS_SENT_START": False, "OP": "{,3}"},
    # "DEP": "xcomp" to match "Durante a audiência preliminar, o perito financeiro afirmou ser o esquema contabilístico altamente complexo e ardiloso."
    {"DEP": {"IN": ["nsubj", "obj", "nsubj:pass", "xcomp"]}, "IS_SENT_START": False}
    ]


    # Gerund variants: the first verb is a gerund (VerbForm=Ger) instead of the root.
    pattern_ger = [
    {"POS": "VERB", "MORPH": {"IS_SUPERSET": ["VerbForm=Ger"]}, "LEMMA": {"IN": declarative_epistemic}},
    {"POS": {"NOT_IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "OP": "{,3}"},
    # "DEP": "cop" to match "Durante a audiência preliminar, o perito financeiro afirmou ser o esquema contabilístico altamente complexo e ardiloso."
    {"LEMMA": {"IN": ["ter", "ser", "estar", "haver"]}, "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}, "DEP": {"IN": ["xcomp", "cop"]}, "IS_SENT_START": False},
    {"DEP": {"NOT_IN": ["nsubj", "obj", "nsubj:pass", "xcomp"]}, "IS_SENT_START": False, "OP": "{,3}"},
    # "DEP": "xcomp" to match "Durante a audiência preliminar, o perito financeiro afirmou ser o esquema contabilístico altamente complexo e ardiloso."
    {"DEP": {"IN": ["nsubj", "obj", "nsubj:pass", "xcomp"]}, "IS_SENT_START": False},
    ]


    matcher.add("b5d2_75_C2", [pattern, pattern_ger])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b5d2_76(doc):
    """
    B5.2-76: "subordinada substantiva completiva - flexionada - de infinitivo (pessoal) - nominalizada, precedida de artigo"
    e.g. O que me surpreendeu foi o ele ter ido sem nos dizer nada.
    Level: C2
    """
    matcher = Matcher(doc.vocab)

    pattern_singular = [
        {"DEP": "root"}, # So we don't match " O vinho tinto, eu abro-o antes de os convidados chegarem."
        # {"POS": {"NOT_IN": ["DET"]}, "IS_SENT_START": False, "OP": "{,3}"},
        {"MORPH": {"IS_SUPERSET": ["PronType=Art", "Number=Sing"]}, "IS_SENT_START": False},
        {"DEP": {"NOT_IN": ["nsubj", "nsubj:pass"]}, "IS_SENT_START": False, "OP": "?"},
        {"DEP": {"IN": ["nsubj:pass", "nsubj"]}, "MORPH": {"IS_SUPERSET": ["Number=Sing"]}, "IS_SENT_START": False},
        #"nsubj:pass" to match "Surpreendeu todos a Maria ser despedida."
        {"POS": {"NOT_IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "OP": "{,3}", "LOWER":{"NOT_IN": ["a"]}},
        # "O diretor pensa estarem os funcionários a fazer um bom trabalho." should not be matched, so we exclude "a" from the pattern.
        {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf", "Number=Sing"]}, "IS_SENT_START": False},
    ]

    pattern_plural = [
        {"DEP": "root"}, # So we don't match " O vinho tinto, eu abro-o antes de os convidados chegarem."
        # {"POS": {"NOT_IN": ["DET"]}, "IS_SENT_START": False, "OP": "{,3}"},
        {"MORPH": {"IS_SUPERSET": ["PronType=Art", "Number=Sing"]}, "IS_SENT_START": False},
        {"DEP": {"NOT_IN": ["nsubj", "nsubj:pass"]}, "IS_SENT_START": False, "OP": "?"},
        {"DEP": {"IN": ["nsubj:pass", "nsubj"]}, "MORPH": {"IS_SUPERSET": ["Number=Plur"]}, "IS_SENT_START": False},
        #"nsubj:pass" to match "Surpreendeu todos a Maria ser despedida."
        {"POS": {"NOT_IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "OP": "{,3}", "LOWER":{"NOT_IN": ["a"]}},
        # "O diretor pensa estarem os funcionários a fazer um bom trabalho." should not be matched, so we exclude "a" from the pattern.
        {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf", "Number=Plur"]}, "IS_SENT_START": False},
    ]

    pattern_singular_start = [
        {"MORPH": {"IS_SUPERSET": ["PronType=Art", "Number=Sing"]}, "IS_SENT_START": True},
        {"DEP": {"NOT_IN": ["nsubj", "nsubj:pass"]}, "IS_SENT_START": False, "OP": "?"},
        {"DEP": {"IN": ["nsubj:pass", "nsubj"]}, "MORPH": {"IS_SUPERSET": ["Number=Sing"]}, "IS_SENT_START": False},
        #"nsubj:pass" to match "Surpreendeu todos a Maria ser despedida."
        {"POS": {"NOT_IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "OP": "{,3}", "LOWER":{"NOT_IN": ["a"]}},
        # "O diretor pensa estarem os funcionários a fazer um bom trabalho." should not be matched, so we exclude "a" from the pattern.
        {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf", "Number=Sing"]}, "IS_SENT_START": False},
    ]

    pattern_plural_start = [
        {"MORPH": {"IS_SUPERSET": ["PronType=Art", "Number=Sing"]}, "IS_SENT_START": True},
        {"DEP": {"NOT_IN": ["nsubj", "nsubj:pass"]}, "IS_SENT_START": False, "OP": "?"},
        {"DEP": {"IN": ["nsubj:pass", "nsubj"]}, "MORPH": {"IS_SUPERSET": ["Number=Plur"]}, "IS_SENT_START": False},
        #"nsubj:pass" to match "Surpreendeu todos a Maria ser despedida."
        {"POS": {"NOT_IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "OP": "{,3}", "LOWER":{"NOT_IN": ["a"]}},
        # "O diretor pensa estarem os funcionários a fazer um bom trabalho." should not be matched, so we exclude "a" from the pattern.
        {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf", "Number=Plur"]}, "IS_SENT_START": False},
    ]

    pattern_singular_start_alt = [
        {"IS_SENT_START": True},
        {"OP": "{,3}"}, 
        {"MORPH": {"IS_SUPERSET": ["PronType=Art", "Number=Sing"]}, "IS_SENT_START": False},
        {"DEP": {"NOT_IN": ["nsubj", "nsubj:pass"]}, "IS_SENT_START": False, "OP": "?"},
        {"DEP": {"IN": ["nsubj:pass", "nsubj"]}, "MORPH": {"IS_SUPERSET": ["Number=Sing"]}, "IS_SENT_START": False},
        #"nsubj:pass" to match "Surpreendeu todos a Maria ser despedida."
        {"POS": {"NOT_IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "OP": "{,3}", "LOWER":{"NOT_IN": ["a"]}},
        # "O diretor pensa estarem os funcionários a fazer um bom trabalho." should not be matched, so we exclude "a" from the pattern.
        {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf", "Number=Sing"]}, "IS_SENT_START": False},
    ]

    pattern_plural_start_alt = [
        {"IS_SENT_START": True},
        {"OP": "{,3}"}, # To match "Na verdade, o eles cederem em pontos que antes defendiam com unhas e dentes revelou que a pressão exterior fora maior do que admitiam."
        {"MORPH": {"IS_SUPERSET": ["PronType=Art", "Number=Sing"]}, "IS_SENT_START": False},
        {"DEP": {"NOT_IN": ["nsubj", "nsubj:pass"]}, "IS_SENT_START": False, "OP": "?"},
        {"DEP": {"IN": ["nsubj:pass", "nsubj"]}, "MORPH": {"IS_SUPERSET": ["Number=Plur"]}, "IS_SENT_START": False},
        #"nsubj:pass" to match "Surpreendeu todos a Maria ser despedida."
        {"POS": {"NOT_IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "OP": "{,3}", "LOWER":{"NOT_IN": ["a"]}},
        # "O diretor pensa estarem os funcionários a fazer um bom trabalho." should not be matched, so we exclude "a" from the pattern.
        {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf", "Number=Plur"]}, "IS_SENT_START": False},
    ]

    pattern_singular_start_alt_subj = [
        {"IS_SENT_START": True},
        {"OP": "{,3}"}, 
        {"MORPH": {"IS_SUPERSET": ["PronType=Art", "Number=Sing"]}, "IS_SENT_START": False},
        {"DEP": {"IN": ["nsubj:pass", "nsubj"]}, "MORPH": {"IS_SUPERSET": ["Number=Sing"]}, "IS_SENT_START": False},
        {"MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Fut", "Number=Sing"]}, "IS_SENT_START": False},
    ]

# A more strict pattern that allows the infinitive to be identified as future subjunctive, and matches
# "Na verdade, o eles cederem em pontos que antes defendiam com unhas e dentes revelou que a pressão exterior fora maior do que admitiam."
    pattern_plural_start_alt_subj = [
        {"IS_SENT_START": True},
        {"OP": "{,3}"}, 
        {"MORPH": {"IS_SUPERSET": ["PronType=Art", "Number=Sing"]}, "IS_SENT_START": False},
        {"DEP": {"IN": ["nsubj:pass", "nsubj"]}, "MORPH": {"IS_SUPERSET": ["Number=Plur"]}, "IS_SENT_START": False},
        {"MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Fut", "Number=Plur"]}, "IS_SENT_START": False},
    ]

    pattern_singular_start_subj = [
        {"MORPH": {"IS_SUPERSET": ["PronType=Art", "Number=Sing"]}, "IS_SENT_START": True},
        {"DEP": {"IN": ["nsubj:pass", "nsubj"]}, "MORPH": {"IS_SUPERSET": ["Number=Sing"]}, "IS_SENT_START": False},
        {"MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Fut", "Number=Sing"]}, "IS_SENT_START": False},
    ]

    pattern_plural_start_subj = [
        {"MORPH": {"IS_SUPERSET": ["PronType=Art", "Number=Sing"]}, "IS_SENT_START": True},
        {"DEP": {"IN": ["nsubj:pass", "nsubj"]}, "MORPH": {"IS_SUPERSET": ["Number=Plur"]}, "IS_SENT_START": False},
        {"MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Fut", "Number=Plur"]}, "IS_SENT_START": False},
    ]

    matcher.add("b5d2_76_C2", [pattern_singular, pattern_plural, pattern_singular_start,
                               pattern_plural_start, pattern_singular_start_alt, pattern_plural_start_alt,
                               pattern_singular_start_alt_subj, pattern_plural_start_alt_subj,
                               pattern_singular_start_subj, pattern_plural_start_subj], greedy="LONGEST")

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]