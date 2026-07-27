from spacy.matcher import Matcher, DependencyMatcher, PhraseMatcher
from process_and_display import nlp_stanza
from text_reconstruction import reconstruct_text

def a3d1_11(doc):
    """
    A3.1-11: pretérito perfeito simples do indicativo - forma - verbos regulares [-ar, -er -ir]
    e.g., falar (falei, falaste, falou, falámos, falaram)
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern_1 = [
        {
            "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Past", "VerbForm=Fin"]},
            "LEMMA": {"NOT_IN": ["dar", "estar", "caber", "haver", "poder", "querer", "saber", "ter", "ver", 
            "dizer", "fazer", "ir", "ser", "trazer", "vir", "pôr", "rever", "deter", 
            "compôr", "intervir", "contrair", "sair", "rir", "valer", "cair", "vistir"]},  # "vistir" to deal with preprocessing issue of "viste"
            "NORM": {"REGEX": "^(?!.*(erei|irei|arei|ramos)$).*"},  # To avoid matching A3.1-105 and a3.1-38 due to preprocessing error
            "TEXT": {"REGEX": "^(?:.*(?:ámos|aste|ei|ou|aram|i|iu|iste|imos|iram|este|eu|emos|eram))$"}
        }
    ]

    # Ad-hoc fixes for patterns that are not captured as a result of the restrictiveness of the first pattern
    pattern_2 = [
        {
            "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Past", "VerbForm=Fin"]},
            "LEMMA": "preparar"
        }
    ]

    pattern_3 = [
        {
            "MORPH": {"IS_SUPERSET": ["Mood=Ind", "VerbForm=Fin"]},
            "LEMMA": {"NOT_IN": ["dar", "estar", "caber", "haver", "poder", "querer", "saber", "ter", "ver", 
            "dizer", "fazer", "ir", "ser", "trazer", "vir", "pôr", "rever", "deter", 
            "compôr", "intervir", "contrair", "sair", "rir", "valer", "cair", "vistir"]},  # "vistir" to deal with preprocessing issue of "viste"
            "TEXT": {"REGEX": "^(?:.*(?:ámos|aste|aram))$"}  # To address the cases which the preprocessor doesn't attribute a tense to the token.
            # "aram" for "cancelaram"
        }
    ]

    pattern_ad_hoc = [
        {"LOWER": {"IN": ["enviaste", "pediste", "descreveram"]}}
    ]

    matcher.add("a3d1_11_A2", [pattern_1, pattern_2, pattern_3, pattern_ad_hoc])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a3d1_12(doc):
    """
    A3.1-12: pretérito perfeito simples do indicativo - forma - verbos irregulares [-ar, -er -ir]
    e.g., ser, estar, ter, haver, ir, vir, ver, querer, sair, fazer, trazer, dizer, dar, saber, pôr...
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {
            "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Past", "VerbForm=Fin"]},
            "LEMMA": {"IN": ["dar", "estar", "caber", "haver", "poder", "querer", "saber", "ter", "ver", 
            "dizer", "fazer", "ser", "trazer", "vir", "pôr", "rever", "deter", 
            "compôr", "intervir", "contrair", "sair", "rir", "valer", "cair", "por"]}
        }
    ]

    pattern_ad_hoc = [
        {
            "LOWER": {"IN": ["fui", "foste", "foi", "fomos", "foram", "viste"]}
        }
    ]

    matcher.add("a3d1_12_A2", [pattern, pattern_ad_hoc])

    matches = matcher(doc)

    # Adding an ad-hoc fix for "hei".
    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches if doc[start].text.lower() != "hei"]


def a3d1_14(doc):
    """
    A3.1-14: pretérito imperfeito do indicativo - forma - verbos regulares [-ar, -er -ir]
    e.g., falar (falava, falavas, falava, falávamos , falavam)
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {
            "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Imp", "VerbForm=Fin"]},
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:ava|avas|avamos|avam|ia|ias|iamos|iam|inha|inhas|inhamos|inham)\\b"},
            "LEMMA": {"NOT_IN": ["ser", "ter", "vir", "pôr", "ir", "por"]}
        }
    ]

    pattern_ad_hoc = [
        {
            "NORM": {"IN": ["pagavamos", "franziam", "comias"]} 
        }
    ]

    matcher.add("a3d1_14_A2", [pattern, pattern_ad_hoc])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a3d1_15(doc):
    """
    A3.1-15: pretérito imperfeito do indicativo - forma - verbos irregulares
    e.g., ser, ter, vir, pôr
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {
            "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Imp", "VerbForm=Fin"]},
            "LEMMA": {"IN": ["ser", "ter", "vir", "pôr", "por"]}
        }
    ]

    pattern_ir = [
        {
            "LOWER": {"IN": ["ia", "ias", "íamos", "íam"]}
        }
    ]

    matcher.add("a3d1_15_A2", [pattern, pattern_ir])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a3d1_28(doc):
    """
    A3.1-28: particípio passado - forma - particípios regulares
    e.g., cantar: cantado
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    # Verbs whose participle should NOT surface as a "regular -do" model.
    part_exclusions = [
        # --- Irregular participles that themselves end in -do (slip past the
        #     regex, so exclusion here actually matters). LEMMA is exact, so
        #     every compound needs its own entry. ---
        "vir", "convir", "provir", "intervir", "advir", "sobrevir",

        # --- Abundant participles whose regular -do form is dispreferred in EP
        #     (EP prefers the short/irregular form: pago, ganho, gasto, morto,
        #     entregue, aceite ...). Excluding avoids modelling a dead form. ---
        "pagar", "ganhar", "gastar", "matar", "salvar", "soltar",
        "limpar", "entregar", "aceitar",

        # --- Redundant given the regex (forms end in -to/-sto, not -do): kept
        #     only to document intent. Remove if you want a lean list. ---
        "ver", "pôr", "abrir", "dizer", "escrever",
        "fazer", "desfazer", "satisfazer",
        "cobrir", "descobrir", "compor", "dispor", "descrever", "rever",
    ]

    # True (non-deverbal) adjectives ending in -do/-da that are NOT participles.
    # Filters the ADJ branch, which otherwise has no exclusion at all.
    adj_exclusions = [
        "rápido", "sólido", "húmido", "úmido", "lindo", "gordo", "surdo",
        "mudo", "agudo", "ácido", "lúcido", "vívido", "válido", "pálido",
        "tímido", "rígido", "frígido", "cómodo", "sórdido", "mórbido",
        "estúpido", "límpido", "árido", "fundo", "redondo", "imundo",
        "segundo", "oriundo",
    ]

    # ^\w+(?:...)$  — \w is Unicode-aware in Python 3, so accented -ído stems
    # (saído, caído, concluído, distribuído ...) are matched instead of dropped.
    suffix_regex = r"^\w+(?:do|da|dos|das)$"

    pattern = [
        {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
            "LEMMA": {"NOT_IN": part_exclusions},
            "NORM": {"REGEX": suffix_regex}
        }
    ]

    pattern_adj = [
        {
            "POS": "ADJ",
            "LEMMA": {"NOT_IN": adj_exclusions},
            "NORM": {"REGEX": suffix_regex}
        }
    ]

    matcher.add("a3d1_28_A2", [pattern, pattern_adj])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a3d1_29(doc):
    """
    A3.1-29: particípio passado - forma - particípios irregulares (frequentes)
    e.g., ver (visto), pôr (posto), abrir (aberto), pagar (pago), vir (vindo), dizer (dito), escrever (escrito)
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
            "LEMMA": {"IN": ["ver", "pôr", "abrir", "pagar", "vir", "dizer", "escrever"]}
        }
    ]

    matcher.add("a3d1_29_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a3d1_31(doc):
    """
    A3.1-31: particípio passado - forma - auxiliar ter no imperfeito do indicativo + particípio passado do verbo principal
    e.g. tinha saído
    Level: A2
    Requires nlp_stanza
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)
    
    # The preprocessor may identify past participle as either a verb or an adjective.
    # In case it's identified as an adjective:
    pattern_adj = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {"LEMMA": "ter",
                            "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Imp", "VerbForm=Fin"]}}
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"}},
            "REL_OP": "<"             
        }
    ]
    
    # In case it's identified as a verb
    pattern_verb = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {"LEMMA": "ter",
                            "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Imp", "VerbForm=Fin"]}}
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": "<"             
        }
    ]

    # To match "Havia algumas crianças a brincar entre as folhas, enquanto os adultos, sentados nos bancos, comentavam que nenhum outono anterior tinha sido tão bonito."
    pattern_ser = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {"LEMMA": "ter",
                            "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Imp", "VerbForm=Fin"]}}
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "ser",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
            "LEMMA": "ser"
            },
            "REL_OP": "$+"             
        }
    ]
    
    matcher.add("a3d1_31_A2", [pattern_adj, pattern_verb, pattern_ser])

    matches = matcher(doc)
    
    # Dictionary to keep matches keyed by start index.
    # For any start index, we keep the match with the larger end index.
    filtered_matches = {}
    for match_id, token_ids in matches:
        start = min(token_ids)
        end = max(token_ids) + 1  # end index is non-inclusive
        if start not in filtered_matches or filtered_matches[start][3] < end:
            tokens_in_span = doc[start:end]
            text = reconstruct_text(tokens_in_span)
            filtered_matches[start] = (doc.vocab.strings[match_id], text, start, end)
    
    return list(filtered_matches.values())


def a3d1_34(doc):
    """
    A3.1-34: pretérito perfeito composto do indicativo - forma - auxiliar ter no presente do indicativo + particípio passado do verbo principal
    e.g., tenho entrado
    Level: B1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)
    
    # The preprocessor may identify past participle as either a verb or an adjective.
    # In case it's identified as an adjective:
    pattern_adj = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                            "LEMMA": "ter", "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Pres", "VerbForm=Fin"]},
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da)$"}},
            "REL_OP": "<"             
        }
    ]
    
    # In case it's identified as a verb
    pattern_verb = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                            "LEMMA": "ter", "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Pres", "VerbForm=Fin"]},
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": "<"             
        }
    ]
    
    matcher.add("a3d1_34_B1", [pattern_adj, pattern_verb])

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


def a3d1_36(doc):
    """
    A3.1-36: infinitivo pessoal simples - forma - [-ar, -er -ir]
    e.g., falar (falar, falares, falar, falarmos, falarem)
    Level: B1

    Tips: If we have a pronoun right before the infinitive
    Sometimes the suject can come after it.
    """
    matcher = Matcher(doc.vocab)

    pattern_1 = [
        {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf", "Person=1"]}}
    ]

    # If we have a perosnal pronoun right before the infinitive, it's definitely
    # a personal infinitive.
    pattern_pronoun = [
        {"POS": "PRON", "MORPH": {"IS_SUPERSET": ["PronType=Prs"]}},
        {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}}
    ]

    pattern_2 = [
        {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf", "Person=2"]}}

    ]

    pattern_3 = [
        {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf", "Person=3"]}}  
    ]

    pattern_form = [
        {"LOWER": "de"},
        {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}, "LOWER": {"REGEX": ".*es$"}}
    ]

    matcher.add("a3d1_36_B1", [pattern_1, pattern_2, pattern_3, pattern_pronoun, pattern_form])

    matches = matcher(doc)

    results = []

    for match_id, start, end in matches:
        # If the matched span contains more than one token,
        # retain only the last token.
        if end - start > 1:
            last_token = doc[end - 1]
            results.append((doc.vocab.strings[match_id], last_token.text, end - 1, end))
        else:
            results.append((doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end))

    return results

def a3d1_38(doc): 
    """
    A3.1-38: futuro simples do indicativo - forma - verbos regulares [-ar, -er -ir]
    e.g., falar (falarei, falarás, falará, falar emos, falarão)
    Level: B1
    Issue: We still cannot match forms such as "responder-te-ei" due to how Stanza preprocesses the text. When a solution is found, we must also implement it in A3.1-60.
    """
    matcher = Matcher(doc.vocab)

    pattern_1 = [
        {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Ind", "Tense=Fut"]},
            "TEXT": {"REGEX": r"(ei|ás|á|emos|ão)$"}
            }
        ]

    # Some were not captured using pattern_1
    # due to preprocessing error, so we use the following hack for them:
    pattern_2 = [
        {"TEXT": {"REGEX": "\\b[A-Za-z]+(?:eremos|aremos|iremos|irás|arás|erás)\\b"},
         "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}}
        ]

    matcher.add("a3d1_38_B1", [pattern_1, pattern_2])

    matches = matcher(doc)

    # Filter matches to ensure they begin with the lemma of each token and are therefore regular.
    final_matches = []
    for match_id, start, end in matches:
        token = doc[start]
        if token.lower_.startswith(token.lemma_):
            final_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end))

    return final_matches

def a3d1_39(doc):
    """
    A3.1-39: futuro simples do indicativo - forma - verbos irregulares
    e.g., dizer (direi, dirás...), fazer (farei, farás...), trazer (trarei, trarás...)
    Level: B1
    Issue: We still cannot match forms such as "responder-te-ei" due to how Stanza preprocesses the text. When a solution is found, we must also implement it in A3.1-60.
    """
    matcher = Matcher(doc.vocab)

    pattern_1 = [{"MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Ind", "Tense=Fut"]}}]

    # Some first person plural future tense verbs were not captured using pattern_1
    # due to preprocessing error, so we use the following hack for them:
    pattern_2 = [
        {
         "NORM": {"REGEX": "\\b[A-Za-z]+(?:eremos|aremos|iremos)\\b"},
         "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]},
         "LEMMA": {"NOT_IN": ["querer"]}
        }
        ]
    
    # Ad-hoc fixes for the preprocessing error
    pattern_3 = [{"LOWER": {"IN": ["direi"]}}]

    matcher.add("a3d1_39_B1", [pattern_1, pattern_2, pattern_3])

    matches = matcher(doc)

    # Filter matches to ensure they do not begin with the lemma of each token and are therefore irregular.
    final_matches = []
    for match_id, start, end in matches:
        token = doc[start]
        if (not token.lower_.startswith(token.lemma_)) or (token.lower_ == "direi"):
            final_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end))

    return final_matches

def a3d1_42(doc):
    """
    A3.1-42: condicional simples - forma - verbos regulares [-ar, -er -ir]
    e.g., falar (falaria, falarias, falaria, falar íamos, falariam)
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern_1 = [{"MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Cnd"]}}]

    # Due to preprocessing errors of identifying the mood
    pattern_2 = [
        {"NORM": {"REGEX": "\\b[A-Za-z]*(?:eriamos|ariamos|iriamos|rias|ria|riam)\\b"},
         # We don't want to match "ia", but we want to match "íriamos."
         "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}}
        ]

    matcher.add("a3d1_42_B1", [pattern_1, pattern_2])

    matches = matcher(doc)

    # Filter matches to ensure they begin with the lemma of each token and are therefore regular.
    final_matches = []
    for match_id, start, end in matches:
        token = doc[start]
        # The condition after "or" is to match preprocessing errors such as
        # "comeríar" being the lemma of "comeríamos".
        if token.lower_.startswith(token.lemma_) or token.lower_.startswith(token.lemma_[:-1]):
            final_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end))

    return final_matches

def a3d1_43(doc):
    """
    A3.1-43: condicional simples - forma - verbos irregulares
    e.g., dizer (diria, dirias...), fazer (faria, farias...), trazer (traria, trarias...)
    Level: B1
    """
    matcher = PhraseMatcher(doc.vocab, attr="NORM")

    irregular_forms = [
    # dizer
    "diria",
    "dirias",
    "diria",
    "diríamos",
    "diríeis",
    "diriam",

    # fazer
    "faria",
    "farias",
    "faria",
    "faríamos",
    "faríeis",
    "fariam",

    # trazer
    "traria",
    "trarias",
    "traria",
    "traríamos",
    "traríeis",
    "trariam",
]

    patterns = list(nlp_stanza.pipe(irregular_forms))

    matcher.add("a3d1_43_B1", patterns)

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a3d1_46(doc):
    """
    A3.1-46: presente do conjuntivo - forma - verbos regulares [-ar, -er -ir]
    e.g., falar (fale, fales, fale, falemos, falem)
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
    {
        "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Sub", "Tense=Pres"]},
        "TEXT": {"REGEX": "^(?!.*\\b\\w*(sse|sses|ssemos|ssem|iste)\\b).*"},
        "LEMMA": {"NOT_IN":["ser", "estar", "ir", "querer", "saber",
                        "haver", "dar", "ver", "vir", "pôr", "ter",
                        "fazer", "dizer", "trazer", "sair", "ouvir",
                        "pedir", "perder", "valer", "caber"]
        },
        "LOWER": {"NOT_IN": ["apoiaremos"]} #Ad-hoc fix
    }
]


    matcher.add("a3d1_46_B1", [pattern])

    matches = matcher(doc)

    # Filter matches to ensure they are formed from the 1.ª pessoa sing.
    # do presente do indicativo are therefore regular.
    final_matches = []
    for match_id, start, end in matches:
        token = doc[start]
        # The condition after and is to avoid matching short irregular forms like seja.
        if token.lower_.startswith(token.lemma_[:-2]) and len(token.lemma_[:-2]) > 2:
            final_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end))

    return final_matches

def a3d1_47(doc):
    """
    A3.1-47: presente do conjuntivo - forma - alternância vocálica na 1.ª pessoa de alguns verbos
    e.g., vestir (vista), preferir (prefira), conseguir (consiga); sentir (sinta), dormir (durma, dormimos), subir (suba, subamos)
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"POS": "VERB", "LOWER": {"IN": ["vista", "prefira", "consiga", "sinta",
                          "durma", "suba", "minta", "siga", "sirva",
                          "ouça", "peça", "dispa", "surja", "fuja",
                          "aja", "cubra", "redija", "meça"]}}
    ]

    matcher.add("a3d1_47_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a3d1_48(doc):
    """
    A3.1-48: presente do conjuntivo - forma - verbos irregulares [-ar, -er -ir] não formados a partir da 1.ª pessoa sing. do presente do indicativo
    e.g., ser (seja), estar (esteja), ir (vá), querer (queira), saber (saiba), haver (haja)
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {"LEMMA": {"IN":["ser", "estar", "ir", "saber",
                        "haver", "dar"]
        },
        "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Sub", "Tense=Pres"]}
        }
    ]

    matcher.add("a3d1_48_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a3d1_57(doc):
    """
    A3.1-57: particípio passado - forma - forma irregular de particípio (particípios duplos), com variação em género e número
    e.g., regular: entregado
    irregular: entregue
    Ex. «Ela tinha entregado o relatório.» / «O relatório foi entregue ontem.»
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern_verb = [
        {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
            "LEMMA": {"IN": ["aceitar", "acender", "entregar", "enxugar", "expressar",
                             "ganhar", "gastar", "imprimir", "limpar", "pagar",
                             "prender", "salvar", "secar", "soltar", "suspender"]},
            "NORM": {"REGEX": "^(?!.*\\b\\w*(do|da|dos|das)\\b).*"}

        }
    ]

    # There are cases where the token is identified as an adjective but the lemma is a verb,
    # such as in "A conta já está paga." This doesn't work, however, with
    # "A casa está limpa agora."
    pattern_adj = [
        {
            "POS": "ADJ",
            "LEMMA": {"IN": ["aceitar", "acender", "entregar", "enxugar", "expressar",
                             "ganhar", "gastar", "imprimir", "limpar", "pagar",
                             "prender", "salvar", "secar", "soltar", "suspender"]},
            "NORM": {"REGEX": "^(?!.*\\b\\w*(do|da|dos|das)\\b).*"}

        }
    ]
    

    matcher.add("a3d1_57_B1", [pattern_verb, pattern_adj])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def a3d1_60(doc):
    """
    A3.1-60: futuro simples do indicativo - uso/valor - possibilidade (construção condicional)
    e.g., Se vieres cedo, assistirás ao debate.
    Level: B2
    Issue: We still cannot match forms such as "responder-te-ei" due to how Stanza preprocesses the text. When a solution is found, we must also implement it in A3.1-60.
    """
    matcher = Matcher(doc.vocab)

    pattern_1 = [
        {"LOWER": "se", "POS": "SCONJ"},
        {"POS": {"NOT_IN": ["VERB, AUX"]}, "IS_SENT_START": False, "OP": "{,4}"},
        {"POS": {"IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "DEP": {"NOT_IN": ["root"]}},
        {"DEP": {"NOT_IN": ["root"]}, "OP": "*", "IS_SENT_START": False},
        {"MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Ind", "Tense=Fut"]}, "IS_SENT_START": False, "DEP": "root"}
        ]

    # Some were not captured using pattern_1
    # due to preprocessing error, so we use the following hack for them:
    pattern_2 = [
        {"LOWER": "se", "POS": "SCONJ"},
        {"POS": {"NOT_IN": ["VERB, AUX"]}, "IS_SENT_START": False, "OP": "{,4}"},
        {"POS": {"IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "DEP": {"NOT_IN": ["root"]}},
        {"DEP": {"NOT_IN": ["root"]}, "OP": "*", "IS_SENT_START": False},
        {"NORM": {"REGEX": "\\b[A-Za-z]+(?:eremos|aremos|iremos|iras|aras|eras)\\b"}, "IS_SENT_START": False}
        ]

    matcher.add("a3d1_60_B2", [pattern_1, pattern_2])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a3d1_61(doc):
    """
    A3.1-61: condicional simples - uso/valor - ação dependente de uma condição que, no momento, não se concretiza ou é pouco provável (estrutura condicional)
    e.g., Se tivesse tempo, iria ajudar na preparação da festa.
    Level: B2
    """
    matcher = Matcher(doc.vocab)

    pattern_1 = [
        {"LOWER": "se", "POS": "SCONJ"},
        {"POS": {"NOT_IN": ["VERB, AUX"]}, "IS_SENT_START": False, "OP": "{,4}"},
        {"POS": {"IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "DEP": {"NOT_IN": ["root"]}, "MORPH": {"IS_SUPERSET": ["Mood=Sub"]}},
        {"DEP": {"NOT_IN": ["root"]}, "OP": "*", "IS_SENT_START": False},
        {"MORPH": {"IS_SUPERSET": ["Mood=Cnd"]}, "IS_SENT_START": False,}
        ]
    
    # Due to preprocessing errors of identifying the mood
    pattern_2 = [
        {"LOWER": "se", "POS": "SCONJ"},
        {"POS": {"NOT_IN": ["VERB, AUX"]}, "IS_SENT_START": False, "OP": "{,4}"},
        {"POS": {"IN": ["VERB", "AUX"]}, "IS_SENT_START": False, "DEP": {"NOT_IN": ["root"]}, "MORPH": {"IS_SUPERSET": ["Mood=Sub"]}},
        {"DEP": {"NOT_IN": ["root"]}, "OP": "*", "IS_SENT_START": False},
        {"NORM": {"REGEX": "\\b[A-Za-z]*(?:eriamos|ariamos|iriamos|ias|ia|iam)\\b"},
         "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}, "IS_SENT_START": False}
        ]

    matcher.add("a3d1_61_B2", [pattern_1, pattern_2])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a3d1_62(doc):
    """
    A3.1-62: condicional simples - uso/valor - ação posterior ao momento da fala/escrita
    e.g., (Ela disse que) Iríamos ao concerto na semana seguinte.
    Level: B2
    """
    matcher = Matcher(doc.vocab)

    verbs = ["dizer", "perguntar", "responder", "afirmar", "declarar", "contar",
    "explicar", "sugerir", "mencionar", "admitir", "negar", "confirmar",
    "pedir", "falar", "comunicar", "informar", "relatar", "anunciar", "escrever",
    "mencionar", "acordar", "avisar", "recomendar", "propor", "suplicar", "implorar",
    "assumir", "admitir", "negar", "confirmar", "pedir", "falar", "prever", "garantir",
    "estimar", "calcular", "acordar",
    # Verbs of thinking/feeling
    "achar", "pensar", "crer", "acreditar", "saber", "considerar",
    "imaginar", "supor", "duvidar", "perceber", "notar", "sentir",
    "ver", "ouvir", "concluir", "descobrir", "lembrar", "esquecer",
    # Other relevant verbs
    "sonhar", "prometer", "jurar"]

    pattern_1 = [
        {"LEMMA": {"IN": verbs}, "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Past", "VerbForm=Fin"]}},
        {"POS": {"NOT_IN": ["AUX", "VERB"]}, "OP": "*", "IS_SENT_START": False},
        {"MORPH": {"IS_SUPERSET": ["Mood=Cnd"]}, "IS_SENT_START": False}
        ]
    
    # Due to preprocessing errors of identifying the mood
    pattern_2 = [
        {"LEMMA": {"IN": verbs}, "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Tense=Past", "VerbForm=Fin"]}},
        {"POS": {"NOT_IN": ["AUX", "VERB"]}, "OP": "*", "IS_SENT_START": False},
        {"NORM": {"REGEX": "\\b[A-Za-z]*(?:eriamos|ariamos|iriamos|ias|ia|iam)\\b"},
         "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}, "IS_SENT_START": False}
        ]

    matcher.add("a3d1_62_B2", [pattern_1, pattern_2])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a3d1_64(doc):
    """
    A3.1-64: pretérito imperfeito do conjuntivo - forma - verbos regulares [-ar, -er -ir]
    e.g., falar (falasse, falasses, falasse, falássemos, falassem)
    Level: B2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {
            "MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Imp", "VerbForm=Fin"]},
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:sse|sses|ssemos|ssem)\\b"},
            "LEMMA": {"NOT_IN": ["dizer", "ter", "fazer", "trazer", "vir", "dar", "estar",
                                 "poder", "querer", "saber", "ver", "ir", "ser"]}
        }
    ]

    pattern_ad_hoc = [
        {"NORM": {"IN": ["perdessemos", "ouvisses", "interrompesses"
                         "ladrasse"]}}
    ]


    matcher.add("a3d1_64_B2", [pattern, pattern_ad_hoc])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a3d1_65(doc):
    """
    A3.1-65: pretérito imperfeito do conjuntivo - forma - verbos irregulares [-ar, -er -ir] (sempre formados a partir da 3.ª pess. pl. do pret. perf. do indicativo, substituindo-se -ram por - sse)
    e.g., dizer (dissesse, dissesses...), ter (tivesse, tivesses...), fazer, trazer, vir
    Level: B2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {
            "MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Imp", "VerbForm=Fin"]},
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:sse|sses|ssemos|ssem)\\b"},
            "LEMMA": {"IN": ["dizer", "ter", "fazer", "trazer", "vir", "dar", "estar",
                                 "poder", "querer", "saber", "ver", "ir", "ser"]}
        }
    ]


    matcher.add("a3d1_65_B2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a3d1_72(doc):
    """
    A3.1-72: futuro simples do conjuntivo - forma - verbos regulares [-ar, -er -ir]
    e.g., falar (falar, falares, falar, falarmos, falarem)
    Level: B2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {
            "MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Fut", "VerbForm=Fin"]},
            "LEMMA": {"NOT_IN": ["dizer", "ter", "fazer", "trazer", "vir", "dar", "estar",
                                 "poder", "querer", "saber", "ver", "ir", "ser", "haver"]},
            # Ad-hoc fixes
            "LOWER": {"NOT_IN": ["enviaste"]}
        }
    ]


    matcher.add("a3d1_72_B2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a3d1_73(doc):
    """
    A3.1-73: futuro simples do conjuntivo - forma - verbos irregulares [-ar, -er -ir] (sempre formados a partir da 3.ª pess. pl. do pret. perf. do indicativo, substituindo-se -ram pelas terminações de futuro)
    e.g., dizer (disser, disseres...), ter (tiver, tiveres...), fazer, trazer, vir...
    Level: B2
    """
    
    matcher = Matcher(doc.vocab)

    irregular_future_subjunctive_lemmas = [
    # Primitive Verbs
    "dizer", "ter", "fazer", "trazer", "vir", "dar", "estar", 
    "poder", "querer", "saber", "ver", "ir", "ser", "haver", "pôr",
    
    # Derivatives of 'dizer'
    "contradizer", "desdizer", "bendizer", "maldizer", "predizer",
    
    # Derivatives of 'ter'
    "manter", "obter", "reter", "conter", "deter", "abster", "ater", "entreter",
    
    # Derivatives of 'fazer'
    "satisfazer", "refazer", "desfazer", "perfazer", "liquefazer", "rarefazer",
    
    # Derivatives of 'vir'
    "intervir", "convir", "advir", "provir", "sobrevir", "desavir",
    
    # Derivatives of 'ver'
    "rever", "prever", "antever", "entrever",
    
    # Derivatives of 'haver'
    "reaver",
    
    # Derivatives of 'pôr'
    "supor", "compor", "impor", "dispor", "propor", "expor", 
    "repor", "opor", "depor", "justapor", "pospor", "pressupor", 
    "predispor", "sobrepor", "transpor", "antepor"
    ]

    pattern = [
        {
            "MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Fut", "VerbForm=Fin"]},
            "LEMMA": {"IN": irregular_future_subjunctive_lemmas}
        }
    ]


    matcher.add("a3d1_73_B2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a3d1_78(doc):
    """
    A3.1-78: mais-que-perfeito composto do conjuntivo - forma - auxiliar ter no imperfeito do conjuntivo + particípio passado do verbo principal
    e.g., tivesse participado
    Level: B2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)
    
    # The preprocessor may identify past participle as either a verb or an adjective.
    # In case it's identified as an adjective:
    pattern_adj = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Imp", "VerbForm=Fin"]},
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:sse|sses|ssemos|ssem)\\b"},
            "DEP": {"NOT_IN": ["advcl"]},
            "LEMMA": "ter"
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"}},
            "REL_OP": "<"             
        }
    ]
    
    # In case it's identified as a verb
    pattern_verb = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["Mood=Sub", "Tense=Imp", "VerbForm=Fin"]},
            "NORM": {"REGEX": "\\b[A-Za-z]+(?:sse|sses|ssemos|ssem)\\b"},
            "DEP": {"NOT_IN": ["advcl"]},
            "LEMMA": "ter"
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": "<"             
        }
    ]
    
    matcher.add("a3d1_78_B2", [pattern_adj, pattern_verb])

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


def a3d1_81(doc):
    """
    A3.1-81: condicional composto - forma - auxiliar ter no condicional simples + particípio passado do verbo principal
    e.g., teríamos conseguido
    Level: B2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)
    
    # The preprocessor may identify past participle as either a verb or an adjective.
    # In case it's identified as an adjective:
    pattern_adj = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["Mood=Cnd", "VerbForm=Fin"]},
            "LEMMA": "ter"
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"}},
            "REL_OP": "<"             
        }
    ]
    
    # In case it's identified as a verb
    pattern_verb = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["Mood=Cnd", "VerbForm=Fin"]},
            "LEMMA": "ter"
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": "<"             
        }
    ]
    
    matcher.add("a3d1_81_B2", [pattern_adj, pattern_verb])

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


def a3d1_83(doc):
    """
    A3.1-83: infinitivo impessoal composto - forma - auxiliar ter no infinitivo impessoal simples + particípio passado do verbo principal
    e.g., ter falado
    Level: B2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)
    
    # The preprocessor may identify past participle as either a verb or an adjective.
    # In case it's identified as an adjective:
    pattern_adj = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": "VerbForm=Inf",
            "LEMMA": "ter"
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"}},
            "REL_OP": "<"             
        }
    ]
    
    # In case it's identified as a verb
    pattern_verb = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": "VerbForm=Inf",
            "LEMMA": "ter"
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": "<"             
        }
    ]
    
    matcher.add("a3d1_83_B2", [pattern_adj, pattern_verb])

    matches = matcher(doc)

    # Semi-auxiliaries / modais / aspectuais: when one of these governs the
    # infinitive it stays impersonal (e.g. "Ela deve ter feito..."), so they must
    # NOT trigger the personal-infinitive exclusion below.
    aux_lemmas = ["dever", "poder", "querer", "ir", "vir", "andar", "estar",
                  "ficar", "começar", "continuar", "acabar", "deixar", "costumar",
                  "haver", "ter", "passar", "voltar", "chegar", "conseguir",
                  "saber", "tornar"]

    results = []
    for match_id, token_ids in matches:
        # Calculate start and end indices
        start = min(token_ids)
        end = max(token_ids) + 1

        # This is to avoid matching "ao ter feito tantas tarefas ao mesmo tempo," which looks like it's
        # impersonal but is actually personal.
        # Check if the token immediately before 'start' is "o"
        if start > 0:  # Make sure we're not at the beginning of the doc
            prev_2_tokens = doc[start - 2]
            prev_token = doc[start - 1]
            # "PRON" to prevent " A maior surpresa foi o ele ter admitido o erro em público" from matching
            if prev_token.text.lower() == "o" or prev_token.pos_ == "PRON":
                # Skip if the previous token is "o"
                continue
            elif prev_token.text.lower() == "de":
                if prev_2_tokens.text.lower() in ["apesar", "depois", "antes"]:
                    continue
            # A finite verb that immediately precedes "ter" and carries its own
            # overt subject (nsubj) signals a *personal* infinitive
            # (e.g. "A coordenadora declarou ter concluído..."), which belongs to
            # a3d1_101 and not here. Semi-auxiliaries/modais are exempt.
            elif ("VerbForm=Fin" in prev_token.morph
                  and prev_token.lemma_ not in aux_lemmas
                  and any(child.dep_ == "nsubj" for child in prev_token.children)):
                continue
            elif doc[start].text.lower() != doc[start].lemma_:  # If it's different from the lemma (teres), it's personal.
                continue

        tokens_in_span = doc[start:end]
        
        text = reconstruct_text(tokens_in_span)
        
        results.append((doc.vocab.strings[match_id], text, start, end))
    return results


def a3d1_85(doc):
    """
    A3.1-85: gerúndio simples - forma - verbos regulares [-ar, -er -ir]
    e.g., falar (falando)
    Level: B2
    The structure says "verbos regulares," but because there is no other structure listed in the Referencial
    that indicates the irregular verbs of this form are taught at a later stage, and because distinguishing
    them is a lot of trouble, we're treating them both as simply "gerúndio simples" in one structure.

    The trick is not to match "Gerúndio composto." But I don't know how without matching irrelevant tokens.
    Maybe we can let it slide since gerúndio composto is made up of gerúndio simples and pp.
    """
    matcher = Matcher(doc.vocab)

    pattern = [
        {
            "MORPH": "VerbForm=Ger"
        }
    ]


    matcher.add("a3d1_85_B2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a3d1_93(doc):
    """
    A3.1-93: futuro composto do indicativo - forma - auxiliar ter no futuro simples do indicativo + particípio passado do verbo principal
    e.g., terás visto
    Level: C1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)
    
    # The preprocessor may identify past participle as either a verb or an adjective.
    # In case it's identified as an adjective:
    pattern_adj = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Ind", "Tense=Fut"]},
                "LEMMA": "ter"
                }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"}},
            "REL_OP": "<++"             
        }
    ]
    
    # In case it's identified as a verb
    pattern_verb = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Ind", "Tense=Fut"]},
                "LEMMA": "ter"
                }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": "<++"             
        }
    ]

    pattern_cop_adj = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Ind", "Tense=Fut"]},
                "LEMMA": "ter"
                }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"},
            "LEMMA": {"IN": ["estar", "ser"]}},
            "REL_OP": "."             
        }
    ]

    pattern_cop_verb = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Ind", "Tense=Fut"]},
                "LEMMA": "ter"
                }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
            "LEMMA": {"IN": ["ser", "estar"]}},
            "REL_OP": "."             
        }
    ]
    
    matcher.add("a3d1_93_C1", [pattern_adj, pattern_verb, pattern_cop_adj, pattern_cop_verb])

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

def a3d1_97(doc):
    """
    A3.1-97: pretérito perfeito composto do conjuntivo - forma - auxiliar ter no presente do conjuntivo + particípio passado do verbo principal
    tenha referido
    Level: C1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)
    
    # The preprocessor may identify past participle as either a verb or an adjective.
    # In case it's identified as an adjective:
    pattern_adj = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Sub", "Tense=Pres"]},
                "LEMMA": "ter"
                }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"},
            "DEP": {"NOT_IN": ["ccomp", "csubj"]}},  # The DEP information was newly added.
            "REL_OP": "<"             
        }
    ]
    
    # In case it's identified as a verb
    pattern_verb = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Sub", "Tense=Pres"]},
                "LEMMA": "ter"
                }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": "<"             
        }
    ]

    pattern_cop_adj = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Sub", "Tense=Pres"]},
                "LEMMA": "ter"
                }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"},
            "LEMMA": {"IN": ["estar", "ser"]}},
            "REL_OP": "."             
        }
    ]

    pattern_cop_verb = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Sub", "Tense=Pres"]},
                "LEMMA": "ter"
                }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
            "LEMMA": {"IN": ["ser", "estar"]}},
            "REL_OP": "."             
        }
    ]
    
    matcher.add("a3d1_97_C1", [pattern_adj, pattern_verb, pattern_cop_adj, pattern_cop_verb])

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

def a3d1_99(doc):
    """
    A3.1-99: futuro composto do conjuntivo - forma - auxiliar ter no futuro simples do conjuntivo + particípio passado do verbo principal
    e.g., tiver acabado
    Level: C1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)
    
    # The preprocessor may identify past participle as either a verb or an adjective.
    # In case it's identified as an adjective:
    pattern_adj = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Sub", "Tense=Fut"]},
                "LEMMA": "ter"
                }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"}},
            "REL_OP": "<"             
        }
    ]
    
    # In case it's identified as a verb
    pattern_verb = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Sub", "Tense=Fut"]},
                "LEMMA": "ter"
                }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": "<"             
        }
    ]
    
    matcher.add("a3d1_99_C1", [pattern_adj, pattern_verb])

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

def a3d1_101(doc):
    """
    A3.1-101: infinitivo pessoal composto - forma - auxiliar ter no infinitivo pessoal simples + particípio passado do verbo principal
    e.g., ter falado
    Level: C1
    Issue: "Termos" is being identified as a noun by the preprocessor, which results in constructions such as
    termos terminado not being identified. We could implement an ad-hoc fix to this, but then it could possibly
    result in unanticipated false positives, which I don't want to risk, especially because we're already
    implementing such an ad-hoc fix for identifying past participles when they are recognized as an adjective
    instead of a verb, and the combination of these  two ad-hoc fixes could get messy.

    Instead of specifying "Person=*", let's use REGEX to match the ending.
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    # The preprocessor may identify past participle as either a verb or an adjective.
    # In case it's identified as an adjective:
    pattern_adj = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
            "TEXT": {"REGEX": ".*(?:mos|es|em)$"},
            "LEMMA": {"IN": ["ter", "termos"]}
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"}},
            "REL_OP": "<++"             
        }
    ]
    
    # In case it's identified as a verb
    pattern_verb = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
            "TEXT": {"REGEX": ".*(?:mos|es|em)$"},
            "LEMMA": {"IN": ["ter", "termos"]}
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
            "LOWER": {"NOT_IN": ["cuidado"]}},
            "REL_OP": "<++"             
        }
    ]

    # To match "ao ter feito"
    pattern_ao_verb = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
            "LEMMA": "ter"
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "ao",
            "RIGHT_ATTRS": {
            "LOWER": {"IN": ["a", "ao"]},
            "POS": "SCONJ"},
            "REL_OP": ";*"             
        }
    ]

    pattern_ao_adj = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
            "LEMMA": "ter"
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"}},
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "ao",
            "RIGHT_ATTRS": {
            "LOWER": {"IN": ["a", "ao"]},
            "POS": "SCONJ"},
            "REL_OP": ";*"             
        }
    ]

    # To match "depois de ter estudado"
    pattern_de_verb = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
            "LEMMA": "ter"
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "de",
            "RIGHT_ATTRS": {
            "LOWER": {"IN": ["de", "após"]}},
            "REL_OP": ";"             
        }
    ]

    pattern_de_adj = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
            "LEMMA": "ter"
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"}},
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "de",
            "RIGHT_ATTRS": {
            "LOWER": {"IN": ["de", "após"]}},
            "REL_OP": ";"             
        }
    ]

    # To match "O facto de eu ter estudado no estrangeiro ajudou-me imenso na carreira."
    pattern_eu_verb = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
            "LEMMA": "ter"
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "eu",
            "RIGHT_ATTRS": {
            "LOWER": {"IN": ["eu"]}},
            "REL_OP": "$-"             
        }
    ]

    pattern_eu_adj = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
            "LEMMA": "ter"
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"}},
            "REL_OP": "<++"             
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "eu",
            "RIGHT_ATTRS": {
            "LOWER": {"IN": ["eu"]}},
            "REL_OP": "$-"             
        }
    ]

    # Now let's include the cases where the dependency relationship is identified as
    # the opposite direction by the dependency parser:

    pattern_adj_reverse = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
            "TEXT": {"REGEX": ".*(?:mos|es|em)$"},
            "LEMMA": {"IN": ["ter", "termos"]}
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"}},
            "REL_OP": ">++"             
        }
    ]
    
    # In case it's identified as a verb
    pattern_verb_reverse = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
            "TEXT": {"REGEX": ".*(?:mos|es|em)$"},
            "LEMMA": {"IN": ["ter", "termos"]}
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
            "LOWER": {"NOT_IN": ["cuidado"]}
            },
            "REL_OP": ">++"             
        }
    ]

    # To match "ao ter feito"
    pattern_ao_verb_reverse = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
            "LEMMA": "ter"
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "ao",
            "RIGHT_ATTRS": {
            "LOWER": {"IN": ["a", "ao"]},
            "POS": "SCONJ"},
            "REL_OP": ";*"             
        }
    ]

    pattern_ao_adj_reverse = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
            "LEMMA": "ter"
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"}},
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "ao",
            "RIGHT_ATTRS": {
            "LOWER": {"IN": ["a", "ao"]},
            "POS": "SCONJ"},
            "REL_OP": ";*"             
        }
    ]
    
    # If we have "ao" before "termos," we know it's definitely not a noun here:
    pattern_termos_adj = [
        {
            "RIGHT_ID": "termos",
            "RIGHT_ATTRS": {"LOWER": "termos", "POS": "NOUN"}
        },
        {
            "LEFT_ID": "termos",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"}},
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "termos",
            "RIGHT_ID": "o",
            "RIGHT_ATTRS": {"LOWER": "o"},
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "termos",
            "RIGHT_ID": "a",
            "RIGHT_ATTRS": {"LOWER": "a"},
            "REL_OP": ">"
        }
    ]

    pattern_termos_verb = [
        {
            "RIGHT_ID": "termos",
            "RIGHT_ATTRS": {"LOWER": "termos", "POS": "NOUN"}
        },
        {
            "LEFT_ID": "termos",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": ">++"             
        },
        {
            "LEFT_ID": "termos",
            "RIGHT_ID": "o",
            "RIGHT_ATTRS": {"LOWER": "o"},
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "termos",
            "RIGHT_ID": "a",
            "RIGHT_ATTRS": {"LOWER": "a"},
            "REL_OP": ">"
        }
    ]

    # "Ter cuidado de" and "ter cuidado com". The former is infinitivo pessoal composto while the latter isn't.

    pattern_cuidado = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]},
            "TEXT": {"REGEX": ".*(?:mos|es|em)$"},
            "LEMMA": {"IN": ["ter", "termos"]}
            }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "cuidado",
            "RIGHT_ATTRS": {
            "LOWER": "cuidado"
            },
            "REL_OP": ">++"          
        },
        {
            "LEFT_ID": "cuidado",
            "RIGHT_ID": "de",
            "RIGHT_ATTRS": {
                "LOWER": "de"
            },
            "REL_OP": ">>"
        },
        {
            "LEFT_ID": "de",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["NOUN", "PRON", "PROPN"]}
            },
            "REL_OP": "<"
        }
    ]

    pattern_ter = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Inf", "Person=3", "Number=Sing"]},
            "LEMMA": "ter"
                         }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
            "LOWER": {"NOT_IN": ["cuidado"]}},
            "REL_OP": "<++"             
        }
    ]

    
    
    
    matcher.add("a3d1_101_C1", [
                                pattern_adj, pattern_verb, pattern_adj_reverse, pattern_verb_reverse,
                                pattern_ao_verb, pattern_ao_adj,
                                pattern_ao_verb_reverse, pattern_ao_adj_reverse,
                                pattern_termos_adj, pattern_termos_verb, pattern_de_adj, pattern_de_verb,
                                pattern_cuidado, pattern_ter, pattern_eu_adj, pattern_eu_verb
                            ]
                )

    matches = matcher(doc)
    
    # A dictionary to group matches by their start token
    grouped_matches = {}
    for match_id, token_ids in matches:
        original_start = min(token_ids)
        start = original_start
        # Check if the original start token is 'de' and handle the previous token condition
        if doc[original_start].text.lower() == "de":
            prev_token_idx = original_start - 1
            if prev_token_idx >= 0 and doc[prev_token_idx].lemma_ in ["precisar", "haver", "necessitar", "ter", "precisáver"]:  # The last item is to address the preprocessing error in the case of "Precisávamos".
                continue  # Skip this match
            start = original_start + 1
        elif doc[original_start].text.lower() in ["após", "eu"]:
            start = original_start + 1
        end = max(token_ids) + 1
        
        # Group matches by start position
        if start not in grouped_matches:
            grouped_matches[start] = []
        grouped_matches[start].append((match_id, start, end))
    
    results = []
    # Process each group of matches that share the same start position
    for start_pos, match_group in grouped_matches.items():
        # If there are multiple matches with the same start, keep only the one with the largest span
        if match_group:
            # Sort by end position in descending order and take the first one
            match_group.sort(key=lambda x: x[2], reverse=True)
            match_id, start, end = match_group[0]
            
            tokens_in_span = doc[start:end]
            text = reconstruct_text(tokens_in_span)

            # So that we don't match "Eles terem o cuidado de avisar é muito positivo."
            exclude_match = False
            for token in tokens_in_span:
                if token.lower_ == "cuidado":
                    for child in token.children:
                        if child.lower_ == "o":
                            exclude_match = True
                            break
                    if exclude_match:
                        break
            if exclude_match:
                continue


            results.append((doc.vocab.strings[match_id], text, start, end))

    # -------------------------------------------------------------------------
    # Disambiguating the AMBIGUOUS singular "ter".
    #
    # The patterns above catch the compound personal infinitive through the
    # unambiguous endings (teres, termos, terem) or when the preprocessor tags
    # "ter" as Person=3|Number=Sing. In the 1st/3rd person singular, though,
    # "ter + particípio" is spelled exactly like the impersonal form and is often
    # left untagged for person, so it is missed (e.g. "declarou ter concluído").
    #
    # We recover it with the following intuition: when a finite verb (a)
    # immediately precedes the infinitive ("REL_OP": ".") and (b) carries its own
    # overt subject (an "nsubj" child), the infinitive is personal. The matrix
    # verb is restricted to exclude semi-auxiliaries / modals / aspectuais, which
    # instead govern an *impersonal* infinitive ("Ela deve ter feito o trabalho.").
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

    matcher_nsubj.add("a3d1_101_C1", [pattern_nsubj_pp, pattern_nsubj_cop])

    seen_spans = {(s, e) for _, _, s, e in results}
    for match_id, token_ids in matcher_nsubj(doc):
        # In both patterns above "ter" is the last node and the participle the
        # second-to-last; the personal-infinitive *form* is just "ter + particípio".
        ter_idx, pp_idx = token_ids[-1], token_ids[-2]
        start = min(ter_idx, pp_idx)
        end = max(ter_idx, pp_idx) + 1
        if (start, end) in seen_spans:
            continue
        seen_spans.add((start, end))
        text = reconstruct_text(doc[start:end])
        results.append((doc.vocab.strings[match_id], text, start, end))

    return results

def a3d1_103(doc):
    """
    A3.1-103: gerúndio composto - forma - auxiliar ter no futuro do gerúndio + particípio passado do verbo principal
    e.g., tendo conseguido
    Level: C1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)
    
    # The preprocessor may identify past participle as either a verb or an adjective.
    # In case it's identified as an adjective:
    pattern_adj = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "MORPH": "VerbForm=Ger",
                "LEMMA": "ter"
                }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"}},
            "REL_OP": ">"             
        }
    ]
    
    # In case it's identified as a verb
    pattern_verb = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "MORPH": "VerbForm=Ger",
                "LEMMA": "ter"
                }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": ">"             
        }
    ]

    # Cases where the dependency relation is identified as reversed: (Tendo eles terminado o trabalho, decidiram sair para jantar.)
    pattern_adj_reverse = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "MORPH": "VerbForm=Ger",
                "LEMMA": "ter"
                }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {
            "POS": "ADJ",
            "TEXT": {"REGEX": ".*(?:do|da|dos|das)$"}},
            "REL_OP": "<"             
        }
    ]
    
    pattern_verb_reverse = [
        {
            "RIGHT_ID": "ter",
            "RIGHT_ATTRS": {
                "MORPH": "VerbForm=Ger",
                "LEMMA": "ter"
                }
        },
        {
            "LEFT_ID": "ter",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}},
            "REL_OP": "<"             
        }
    ]
    
    matcher.add("a3d1_103_C1", [pattern_adj, pattern_verb, pattern_adj_reverse, pattern_verb_reverse])

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

def a3d1_105(doc):
    """
    A3.1-105: pretérito mais-que-perfeito simples do indicativo - forma - verbos regulares [-ar, -er -ir]
    e.g., falar (falara, falaras, falara, faláramos , falaram)
    Level: C1

    The approach we are taking to identify this structure is really a hack based on the output of Stanza's
    preprocessor and the ending we expect. We won't be trying to identify the third person plural form of
    this structure (falaram), as it's impossible to distinguish it from Pretérito perfeito simples do indicativo
    without context, and perhaps we can later identify it when we're targeting the uses of this structure using
    contextual information.
    """
    matcher = Matcher(doc.vocab)

    pattern_1 = [
        {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Tense=Imp", "Mood=Ind"]},
            "TEXT": {"REGEX": "^.+(?:ara|aras|áramos|era|eras|êramos|ira|iras|íramos)$"},
            "LENGTH": {">=": 5}
        }
    ]

    pattern_2 = [
        {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Tense=Pqp", "Mood=Ind"]},
        }
    ]

    pattern_3 = [
        {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Ind"]},
            "TEXT": {"REGEX": "^.+(?:áramos|êramos|íramos|era)$"},
            "LENGTH": {">=": 5} # To avoid matching "era."
        }
    ]

    matcher.add("a3d1_105_C1", [pattern_1, pattern_2, pattern_3])

    matches = matcher(doc)

    # To check if it's regular
    filtered_matches = []
    for match_id, start, end in matches:
        token = doc[start]  # Get the matched token
        # Convert both text and lemma to lowercase for case-insensitive comparison
        # Including an ad-hoc fix for "trouxer"
        if token.text.lower().startswith(token.lemma_) and token.lemma_ != "trouxer":
            filtered_matches.append((match_id, start, end))

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in filtered_matches]


def a3d1_106(doc):
    """
    A3.1-106: pretérito mais-que-perfeito simples do indicativo - forma - verbos irregulares [-ar, -er -ir] (sempre formados a partir da 3.ª pess. pl. do p.p.s. do indicativo, substituindo-se -ram pelas terminações de pretérito mais-que-perfeito)
    e.g., dizer (dissera, disseras...), ter (tivera, tiveras...), fazer, trazer, vir...
    Level: C1

    The approach we are taking to identify this structure is really a hack based on the output of Stanza's
    preprocessor and the ending we expect.

    We are now able to disambiguate between the pretérito and mais-que-perfeito thanks to adding the extension
    attribute "past_in_sentence". Now, if we have other tokens that have the "past" or "pqp" tense, this 
    attribute is set to True, and then we'll consider the ambiguous form as mais-que-perfeito.

    We decided that we can't disambiguate using rule-based methods. It will be interesting to see if LLMs do better.
    """
    matcher = Matcher(doc.vocab)

    pattern_1 = [
        {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Tense=Imp", "Mood=Ind"]},
            "TEXT": {"REGEX": "^.+(?:ara|aras|áramos|era|eras|êramos|ira|iras|íramos)$"}
        }
    ]

    pattern_2 = [
        {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Tense=Pqp", "Mood=Ind"]},
        }
    ]

    pattern_3 = [
        {
            "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Ind"]},
            "TEXT": {"REGEX": "^.+(?:áramos|êramos|íramos)$"}
        }
    ]

    # pattern_disambiguate = [
    #     {
    #         "MORPH": {"IS_SUPERSET": ["Mood=Ind", "Number=Plur", "Person=3", "Tense=Past"]},
    #         "_": {"past_in_sentence": True},
    #         "DEP": {"NOT_IN": ["conj", "parataxis"]}
    #     }
    # ] # It doesn't work with "O público adorou a atuação e, no final, aplaudiu tanto que os alunos tiveram de repeti-la."


    matcher.add("a3d1_106_C1", [pattern_1, pattern_2, pattern_3])

    matches = matcher(doc)

    # To check that it's irregular
    filtered_matches = []
    for match_id, start, end in matches:

        # Skip matches that immediately precede a gerund.
        if end < len(doc):
            next_token = doc[end]
            if next_token.pos_ == "VERB" and "VerbForm=Ger" in next_token.morph:
                continue 

        token = doc[start]
        # Convert both text and lemma to lowercase for case-insensitive comparison
        # Including an ad-hoc fix for "trouxer"
        if not token.text.lower().startswith(token.lemma_.lower()) or token.lemma_ == "trouxer":
            filtered_matches.append((match_id, start, end))

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in filtered_matches]