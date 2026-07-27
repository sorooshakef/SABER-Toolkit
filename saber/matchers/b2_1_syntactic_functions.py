from spacy.matcher import Matcher, DependencyMatcher
from process_and_display import nlp_stanza
from text_reconstruction import reconstruct_text

def b2d1_111(doc):
    """
    B2.1-111: sujeito - sujeito nulo - sem interpretação/referente (expletivo) - A2:
    e.g., [-] Choveu muito.
    Level: A2
    """
    matcher = Matcher(doc.vocab)
    pattern = [
        {"MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}}
    ]
    matcher.add("b2d1_111_A2", [pattern])
    matches = matcher(doc)

    valid_matches = []
    for match_id, start, end in matches:
        token = doc[start]  # our pattern matches a single token
        # Exclude if this token has any child with the 'nsubj' dependency
        if any(child.dep_ == "nsubj" for child in token.children):
            continue
        valid_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end))
    
    return valid_matches


def b2d1_117(doc):
    """
    B2.1-117: sujeito - sujeito nulo - sem interpretação/referente (expletivo) - B1:
    e.g., [-] Parece que eles também vêm.
    Level: B1
    """
    matcher = Matcher(doc.vocab)
    pattern = [
        {"LEMMA": "parecer", "MORPH": {"IS_SUPERSET": ["VerbForm=Fin"]}},
        {"LOWER": "que"}
    ]
    matcher.add("b2d1_117_B1", [pattern])
    matches = matcher(doc)

    valid_matches = []
    for match_id, start, end in matches:
        token = doc[start]  # our pattern matches a single token
        # Exclude if this token has any child with the 'nsubj' dependency
        if any(child.dep_ == "nsubj" for child in token.children):
            continue
        valid_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end))
    
    return valid_matches


def b2d1_118(doc):
    """
    B2.1-118: vocativo - em posição não inicial (vd. ordem dos constituintes)
    e.g., Sabes, [Paula], se a secretaria já abriu? Não creio, [José], que possa ajudar-te.
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"DEP": "vocative", "IS_SENT_START": False}
            ]
    
    pattern_alt = [
        {"LOWER": ","},
        {"POS": "PROPN"},
        {"LOWER": "."}
    ]

    matcher.add("b2d1_118_B1", [pattern, pattern_alt])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b2d1_119(doc):
    """
    B2.1-119: sujeito - oracional - com oração relativa
    e.g., Quem chegar atrasado não vai ao passeio.
    Level: B2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "pronoun",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["PronType=Rel"]}
                }
        },
        {
            "LEFT_ID": "pronoun",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["acl:relcl", "csubj"]}
                #"acl:relcl" to match "O que fizeste foi lamentável."
                },
            "REL_OP": "<"             
        }
    ]
    
    matcher.add("b2d1_119_B2", [pattern])

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


def b2d1_120(doc):
    """
    B2.1-120: sujeito - oracional - com oração completiva
    e.g., É verdade que eles se conhecem?
    Level: B2

    The parser labels the completive verb (the head of "que") very inconsistently:
    across attested positives it appears as "csubj", "ccomp" and even "nsubj"
    (when the verb is mistagged as a NOUN), so a strict DEP="csubj" constraint
    yields poor recall. Instead of relying on the dependency label, we drop it and
    disambiguate the subject completive from its confusables using the heuristic
    (cf. b2d1_111) that a subject completive clause has no overt subject preceding
    "que" in the main clause:
      - object completives ("Ele disse que...") have a subject before "que", so the
        main predicate carries an nsubj that precedes it;
      - exclamatives ("Que pena que chegaste tão tarde") introduce the predicate
        with an exclamative determiner ("Que"/"Quão").
    Both are excluded below.
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "que",
            "RIGHT_ATTRS": {
                "LOWER": "que",
                "POS": "SCONJ"
                }
        },
        {
            "LEFT_ID": "que",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                # DEP intentionally left unconstrained (see docstring); the
                # completive head must still be a possible clause predicate.
                "POS": {"IN": ["VERB", "AUX", "ADJ", "NOUN"]}
                },
            "REL_OP": "<"
        }
    ]

    matcher.add("b2d1_120_B2", [pattern])

    matches = matcher(doc)

    results = []
    for match_id, token_ids in matches:
        que_tok = doc[token_ids[0]]
        verb_tok = doc[token_ids[1]]
        # The predicate the completive clause attaches to (e.g. "verdade", "Importa").
        predicate = verb_tok.head

        # Exclude object completives: the main predicate has an overt subject
        # occurring before "que" (e.g. "Ele disse que o curso era longo.").
        if any(child.dep_ in ("nsubj", "nsubj:pass") and child.i < que_tok.i
               for child in predicate.children):
            continue

        # Exclude exclamatives introduced by an exclamative determiner
        # (e.g. "Que pena que chegaste tão tarde.").
        if any(child.pos_ == "DET" and child.lower_ in ("que", "quão")
               for child in predicate.children):
            continue

        # Calculate start and end indices
        start = min(token_ids)
        end = max(token_ids) + 1
        tokens_in_span = doc[start:end]

        text = reconstruct_text(tokens_in_span)
        results.append((doc.vocab.strings[match_id], text, start, end))

    return results


def b2d1_213(doc):
    """
    B2.1-213: complemento direto - nominal (pronome pessoal)
    e.g., A Joana comprou-[as] ontem. Ainda não [o] vi.
    Level: A2   
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {
                    "LOWER": {"IN": ["o", "a", "se", "nos", "vos",
                                     "os", "as", "te", "me"]},
                    "DEP": "obj",
                    "POS": "PRON"
                }
            ]
    
    """
    A assembleia geral elegeu o Miguel presidente da associação, apesar da oposição interna. Mais tarde, o conselho diretivo nomeou a Joana coordenadora do projeto europeu, reconhecendo o seu contributo anterior. Alguns membros consideraram o plano uma verdadeira revolução, enquanto outros julgaram as medidas um erro estratégico.

    Durante a reunião, a diretora tornou o estagiário responsável pela comunicação interna, uma decisão que surpreendeu muitos. A imprensa, por sua vez, considerou o ministro o principal culpado pela crise, embora alguns analistas o tenham julgado vítima de circunstâncias externas.

    Com o tempo, os colegas passaram a considerar a Ana uma referência na área, e a universidade a nomeou, por unanimidade, professora catedrática. Em contextos informais, até o chamavam génio da matemática, embora nem todos o julgassem merecedor de tanto reconhecimento.

    A equipa técnica tornou o campo de treinos um espaço multifuncional, adaptando-o às novas necessidades. Por fim, a comunidade escolar elegeu o professor Rui mentor do ano, reconhecendo o seu trabalho exemplar.
    """

    matcher.add("b2d1_213_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b2d1_214(doc):
    """
    B2.1-214: complemento direto - oracional - verbos como 'dizer' (completiva com verbo no indicativo)
    e.g., Ele disse [que o curso é/era muito longo].
    Level: A2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    TRIGGER_LEMMAS = {
    # Verbs of saying
    "dizer", "perguntar", "responder", "afirmar", "declarar", "contar",
    "explicar", "sugerir", "mencionar", "admitir", "negar", "confirmar",
    "pedir", "falar", "comunicar", "informar", "relatar",
    # Verbs of thinking/feeling
    "achar", "pensar", "crer", "acreditar", "saber", "considerar",
    "imaginar", "supor", "duvidar", "perceber", "notar", "sentir",
    "ver", "ouvir", "concluir", "descobrir", "lembrar", "esquecer",
    # Other relevant verbs
    "sonhar", "prometer", "jurar"
}

    pattern = [
        {
            "RIGHT_ID": "que",
            "RIGHT_ATTRS": {
                "LOWER": "que",
                "POS": "SCONJ"
                }
        },
        {
            "LEFT_ID": "que",
            "RIGHT_ID": "ccomp",
            "RIGHT_ATTRS": {
                "DEP": "ccomp",
                "POS": {"NOT_IN": ["VERB", "AUX"]}
                },
            "REL_OP": "<"             
        },
        {
            "LEFT_ID": "ccomp",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["Mood=Ind"]}
                },
            "REL_OP": ">"             
        }
    ]


    pattern_verb = [
        {
            "RIGHT_ID": "que",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["que", "se"]},
                "POS": "SCONJ"
                }
        },
        {
            "LEFT_ID": "que",
            "RIGHT_ID": "ccomp",
            "RIGHT_ATTRS": {
                "DEP": "ccomp",
                "POS": {"IN": ["VERB", "AUX"]},
                "MORPH": {"IS_SUPERSET": ["Mood=Ind"]}
                },
            "REL_OP": "<"             
        }
    ]

    # To match "Eles perguntaram se tínhamos recebido a encomenda."
    pattern_part = [
        {
            "RIGHT_ID": "que",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["que", "se"]},
                "POS": "SCONJ"
                }
        },
        {
            "LEFT_ID": "que",
            "RIGHT_ID": "ccomp",
            "RIGHT_ATTRS": {
                "DEP": "ccomp",
                "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}
                },
            "REL_OP": "<"             
        },
        {
            "LEFT_ID": "ccomp",
            "RIGHT_ID": "aux",
            "RIGHT_ATTRS": {
                "POS": "AUX",
                "MORPH": {"IS_SUPERSET": ["Mood=Ind"]}
                },
            "REL_OP": ">"             
        }
    ]

    
    matcher.add("b2d1_214_A2", [pattern, pattern_verb, pattern_part])

    matches = matcher(doc)
    
    results = []
    for match_id, token_ids in matches:
        # Calculate the start index of the matched span
        start = min(token_ids)
        
        # Check if there is a token before the match
        if start > 0:
            # Get the token immediately before the start of the span
            prev_token = doc[start - 1]
            
            # Check if the lemma of the preceding token is in our trigger list
            if prev_token.lemma_ in TRIGGER_LEMMAS:
                end = max(token_ids) + 1
                tokens_in_span = doc[start:end]
                
                text = reconstruct_text(tokens_in_span)
                results.append((doc.vocab.strings[match_id], text, start, end))

    return results

def b2d1_215(doc):
    """
    B2.1-215: complemento indireto - pronominal (sem preposição)
    e.g., A Rita deu[-lhe] um livro.
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {
                    "LOWER": {"IN": ["me", "te", "lhe",
                                     "nos", "vos", "lhes"]},
                    "DEP": "iobj",
                    "POS": "PRON"
                }
            ]

    matcher.add("b2d1_215_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

def b2d1_216(doc):
    """
    B2.1-216: modificador - oração - subordinada adverbial (temporal/causal/final)
    e.g., O Gonçalo chegou [quando tu saíste]. Ele vestiu o casaco [porque tinha frio]. Elas vieram para Lisboa [para trabalhar].   
    Level: A2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "modifier",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["SCONJ", "ADV"]},
                "LOWER": {"IN": ["quando", "porque", "para"]}
                }
        },
        {
            "LEFT_ID": "modifier",
            "RIGHT_ID": "adverbial",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["acl", "advcl"]}
                # "acl" added for "Estudo português para trabalhar em Portugal."
                },
            "REL_OP": "<"             
        }
    ]
    
    matcher.add("b2d1_216_A2", [pattern])

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

def b2d1_218(doc):
    """
    B2.1-218: complemento direto - oracional - verbos da subordinada completiva no presente do conjuntivo
    e.g., Ele não quer [que o projeto se atrase].
    Level: B1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "que",
            "RIGHT_ATTRS": {
                "LOWER": "que",
                "POS": "SCONJ"
                }
        },
        {
            "LEFT_ID": "que",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "DEP": "ccomp",
                "MORPH": {"IS_SUPERSET": ["VerbForm=Fin", "Mood=Sub", "Tense=Pres"]}
                },
            "REL_OP": "<"             
        }
    ]
    
    matcher.add("b2d1_218_B1", [pattern])

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


def b2d1_219(doc):
    """
    B2.1-219: predicativo do sujeito - oracional (oração substantiva predicativa)
    e.g., É muito importante [lavar bem as mãos antes de comer].
    Level: B1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "cop",
            "RIGHT_ATTRS": {
                "DEP": "cop"
                }
        },
        {
            "LEFT_ID": "cop",
            "RIGHT_ID": "clausal_subject",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["csubj"]}
                },
            "REL_OP": "$++"
        }
    ]

    # To match "A garantia da administração é que não haverá cortes no orçamento estrutural."
    pattern_ccomp = [
        {
            "RIGHT_ID": "cop",
            "RIGHT_ATTRS": {
                "LOWER": "é"
                }
        },
        {
            "LEFT_ID": "cop",
            "RIGHT_ID": "clausal_subject",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["ccomp"]}
                },
            "REL_OP": "$++"
        }
    ]

    # Alternative structure in which the copula attaches directly to the
    # predicative infinitive (which is the head of the clause), e.g.,
    # "A nossa principal prioridade é [concluir a implementação técnica]."
    pattern_infinitive = [
        {
            "RIGHT_ID": "infinitive",
            "RIGHT_ATTRS": {
                "POS": "VERB",
                "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}
                }
        },
        {
            "LEFT_ID": "infinitive",
            "RIGHT_ID": "cop",
            "RIGHT_ATTRS": {
                "DEP": "cop"
                },
            "REL_OP": ">--"
        }
    ]

    matcher.add("b2d1_219_B1", [pattern, pattern_infinitive, pattern_ccomp])

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


def b2d1_220(doc):
    """
    B2.1-220: "predicativo do complemento direto - com 'nomear'/'eleger', 'considerar'/'julgar', 'tornar' GN"
    e.g., A direção nomeou a Maria [chefe de departamento].
    Level: B1
    """
    # matcher = Matcher(doc.vocab)

    # pattern = [
    #             {"LEMMA": {"IN": ["nomear", "eleger", "considerar", "julgar", "tornar"]}},
    #             {"POS": {"NOT_IN": ["PROPN", "PRON", "NOUN"]}, "OP": "*"},
    #             {"POS": {"IN": ["PROPN", "PRON", "NOUN"]}, "DEP": "obj"},
    #             {"POS": {"NOT_IN": ["PROPN", "PRON", "NOUN"]}, "OP": "*", "LOWER": {"NOT_IN": ["de"]}},
    #             {"POS": "NOUN"}
    # ]

    # pattern_de = [
    #             {"LEMMA": {"IN": ["nomear", "eleger", "considerar", "julgar", "tornar"]}},
    #             {"POS": {"NOT_IN": ["PROPN", "PRON", "NOUN"]}, "OP": "*"},
    #             {"POS": {"IN": ["PROPN", "PRON", "NOUN"]}, "DEP": "obj"},
    #             {"LOWER": "de"},
    #             {"POS": "DET", "OP": "?"},
    #             {"LOWER": {"IN": ["PRON", "NOUN"]}},
    #             {"POS": {"NOT_IN": ["PROPN", "PRON", "NOUN"]}, "OP": "*"},
    #             {"POS": "NOUN"}
    # ]

    # matcher.add("b2d1_220_B1", [pattern, pattern_de])

    # matches = matcher(doc)

    # return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]



    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "LEMMA": {"IN": ["nomear", "eleger", "considerar", "julgar", "tornar"]},
                }
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "obj",
            "RIGHT_ATTRS": {
                "DEP": "obj"
                },
            "REL_OP": ">"             
        },
        {
            "LEFT_ID": "obj",
            "RIGHT_ID": "predicative",
            "RIGHT_ATTRS": {
                "POS": "NOUN"
                },
            "REL_OP": ">"
        }
    ]

    pattern_siblings = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "LEMMA": {"IN": ["nomear", "eleger", "considerar", "julgar", "tornar"]},
                }
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "obj",
            "RIGHT_ATTRS": {
                "DEP": "obj"
                },
            "REL_OP": ">"             
        },
        {
            "LEFT_ID": "obj",
            "RIGHT_ID": "predicative",
            "RIGHT_ATTRS": {
                "POS": "NOUN",
                "DEP": "obj"
                },
            "REL_OP": "$++"
        }
    ]


    # Using the alternative pattern below will result in many double matches due to the greedy
    # nature of the search.

    # pattern_alt = [
    #     {
    #         "RIGHT_ID": "verb",
    #         "RIGHT_ATTRS": {
    #             "LEMMA": {"IN": ["nomear", "eleger", "considerar", "julgar", "tornar"]},
    #             }
    #     },
    #     {
    #         "LEFT_ID": "verb",
    #         "RIGHT_ID": "obj",
    #         "RIGHT_ATTRS": {
    #             "DEP": "obj"
    #             },
    #         "REL_OP": ">"             
    #     },
    #     {
    #         "LEFT_ID": "verb",
    #         "RIGHT_ID": "predicative",
    #         "RIGHT_ATTRS": {
    #             "POS": "NOUN",
    #             "DEP": "obj"
    #             },
    #         "REL_OP": ">"
    #     }
    # ]
    
    matcher.add("b2d1_220_B1", [pattern, pattern_siblings])

    matches = matcher(doc)

    # DependencyMatcher has no `greedy` option, so keep only the shortest
    # match for each start index. The de/em exclusion below must run *before*
    # this selection: otherwise a spurious short match (e.g. the noun-internal
    # complement in "julgou o [sistema de operações] um fracasso", where the
    # predicative "operações" is an nmod child ending on "de") would win the
    # shortest-per-start comparison and suppress the correct sibling match
    # ("... um fracasso"), and then be dropped itself, yielding no result.
    shortest = {}
    for match_id, token_ids in matches:
        start = min(token_ids)
        end = max(token_ids) + 1

        last_token = doc[end - 1]

        # So that we don't match "O professor considerou a apresentação dos alunos claramente insuficiente para os padrões exigidos. "
        if any(child.lower_ in ["de", "em"] for child in last_token.children):
            continue

        if start not in shortest or end < shortest[start][1]:
            shortest[start] = (match_id, start, end)

    results = []
    for match_id, start, end in shortest.values():
        tokens_in_span = doc[start:end]
        text = reconstruct_text(tokens_in_span)
        results.append((doc.vocab.strings[match_id], text, start, end))

    return results


def b2d1_221(doc):
    """
    B2.1-221: "predicativo do complemento direto - com 'nomear'/'eleger', 'considerar'/'julgar', 'tornar' + GAdj"
    e.g., Todos consideraram a viagem [muito cansativa]. Eles tornaram o passeio [insuportável]!
    Level: B1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "LEMMA": {"IN": ["nomear", "eleger", "considerar", "julgar", "tornar"]},
            }
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "obj",
            "RIGHT_ATTRS": {
                "DEP": "obj"
            },
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "obj",
            "RIGHT_ID": "predicative",
            "RIGHT_ATTRS": {
                "DEP": "amod"
            },
            "REL_OP": ">++"
        }
    ]

    pattern_alt = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "LEMMA": {"IN": ["nomear", "eleger", "considerar", "julgar", "tornar"]},
            }
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "obj",
            "RIGHT_ATTRS": {
                "DEP": "obj",
                "POS": "NOUN"  # So that we don't match "Mais tarde, o conselho diretivo nomeou a Joana coordenadora do projeto europeu"
            },
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "obj",
            "RIGHT_ID": "modifier",
            "RIGHT_ATTRS": {
                "DEP": "nmod"
            },
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "modifier",
            "RIGHT_ID": "predicative",
            "RIGHT_ATTRS": {
                "DEP": "amod"
            },
            "REL_OP": ">++"
        }
    ]
    
    matcher.add("b2d1_221_B1", [pattern, pattern_alt])
    matches = matcher(doc)

    # Dictionary to keep the best match (largest end index) for each start index.
    best_matches = {}
    for match_id, token_ids in matches:
        start = min(token_ids)
        end = max(token_ids) + 1
        
        # Check if the span has more than one token with dependency 'obj'
        # if sum(1 for token in doc[start:end] if token.dep_ == "obj") > 1:
        #     continue  # Exclude this match

        # To exclude "a universidade a nomeou, por unanimidade, professora catedrática."
        if sum(1 for token in doc[start].children if token.dep_ == "obj") > 1:
            continue
        
        text = reconstruct_text(doc[start:end])
        # Only update if this match has a larger end index for the same start.
        if start not in best_matches or end > best_matches[start][3]:
            best_matches[start] = (doc.vocab.strings[match_id], text, start, end)

    results = list(best_matches.values())
    return results


def b2d1_224(doc):
    """
    B2.1-224: modificador - oração - subordinada adverbial (condicional/concessiva)
    e.g., Eu trago-te o livro, [no caso de precisares/caso precises]. [Apesar de estar frio/Embora esteja frio], eles vão andar de barco.
    Level: B1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "modifier",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["SCONJ", "ADV"]},
                "LOWER": {"IN": ["caso", "apesar", "embora", "se", "mesmo"]},
                # "ainda" to match "Ainda que seja difícil de acreditar, ele conseguiu terminar tudo em dois dias."
                # Removed "ainda" not to match "Desde que cheguei, ainda não vi o Pedro."
                # "mesmo" to match "Mesmo que não concordem connosco, temos de seguir com o plano."
                }
        },
        {
            "LEFT_ID": "modifier",
            "RIGHT_ID": "adverbial",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["advcl", "xcomp", "advmod", "root", "obl"]}
                # "xcomp" to match "Vamos caminhar até ao cume, apesar de estar a chover."
                # "advmod" to match "Embora já fosse tarde, continuaram a trabalhar no projeto."
                # "root" to match "Embora, segundo disseram, o relatório esteja incompleto, decidiram publicá-lo."
                # "obl" to match ""
                },
            "REL_OP": "<++"             
        }
    ]
    
    # To match "Eu trago-te o livro, no caso de precisares."
    pattern_alt = [
        {
            "RIGHT_ID": "modifier",
            "RIGHT_ATTRS": {
                "POS": "NOUN",
                "LOWER": "caso"
                }
        },
        {
            "LEFT_ID": "modifier",
            "RIGHT_ID": "adverbial",
            "RIGHT_ATTRS": {
                "DEP": "acl"
                },
            "REL_OP": ">++"             
        }
    ]

    # To match "Supondo que chegue a horas, podemos começar a reunião sem atrasos."
    pattern_supondo = [
        {
            "RIGHT_ID": "modifier",
            "RIGHT_ATTRS": {
                "LOWER": "supondo"
                }
        },
        {
            "LEFT_ID": "modifier",
            "RIGHT_ID": "adverbial",
            "RIGHT_ATTRS": {
                "DEP": "ccomp"
                },
            "REL_OP": ">++"             
        }
    ]

    matcher.add("b2d1_224_B1", [pattern, pattern_alt, pattern_supondo])

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


def b2d1_225(doc):
    """
    B2.1-225: complemento direto - oracional - verbos da subordinada completiva em diferentes tempos e modos (indicativo/conjuntivo/condicional)
    e.g., A Carolina queria [que ela também fosse ao jantar]. Ele pensou [que iríamos todos juntos]. Eles perguntaram [se tínhamos gostado do almoço].
    Level: B2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "que",
            "RIGHT_ATTRS": {
                "LOWER": "que",
                "POS": "SCONJ"
                }
        },
        {
            "LEFT_ID": "que",
            "RIGHT_ID": "ccomp",
            "RIGHT_ATTRS": {
                "DEP": "ccomp",
                "POS": {"NOT_IN": ["VERB", "AUX"]}
                },
            "REL_OP": "<"             
        },
        {
            "LEFT_ID": "ccomp",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "MORPH": {"INTERSECTS": ["Mood=Cnd", "Mood=Sub"]}
                },
            "REL_OP": ">"             
        }
    ]


    pattern_verb = [
        {
            "RIGHT_ID": "que",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["que", "se"]},
                "POS": "SCONJ"
                }
        },
        {
            "LEFT_ID": "que",
            "RIGHT_ID": "ccomp",
            "RIGHT_ATTRS": {
                "DEP": "ccomp",
                "POS": {"IN": ["VERB", "AUX"]},
                "MORPH": {"INTERSECTS": ["Mood=Cnd", "Mood=Sub"]}
                },
            "REL_OP": "<"             
        }
    ]

    pattern_part = [
        {
            "RIGHT_ID": "que",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["que", "se"]},
                "POS": "SCONJ"
                }
        },
        {
            "LEFT_ID": "que",
            "RIGHT_ID": "ccomp",
            "RIGHT_ATTRS": {
                "DEP": "ccomp",
                "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}
                },
            "REL_OP": "<"             
        },
        {
            "LEFT_ID": "ccomp",
            "RIGHT_ID": "aux",
            "RIGHT_ATTRS": {
                "POS": "AUX",
                "MORPH": {"INTERSECTS": ["Mood=Cnd", "Mood=Sub"]}
                },
            "REL_OP": ">"             
        }
    ]

    # Tense=Imp to match "Eles perguntaram se tínhamos gostado do almoço."
    pattern_part_alt = [
        {
            "RIGHT_ID": "que",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["que", "se"]},
                "POS": "SCONJ"
                }
        },
        {
            "LEFT_ID": "que",
            "RIGHT_ID": "ccomp",
            "RIGHT_ATTRS": {
                "DEP": "ccomp",
                "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}
                },
            "REL_OP": "<"             
        },
        {
            "LEFT_ID": "ccomp",
            "RIGHT_ID": "aux",
            "RIGHT_ATTRS": {
                "POS": "AUX",
                "MORPH": {"IS_SUPERSET": ["Tense=Imp"]}
                },
            "REL_OP": ">"             
        }
    ]
    
    matcher.add("b2d1_225_B2", [pattern, pattern_verb, pattern_part, pattern_part_alt])

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


def b2d1_227(doc):
    """
    B2.1-227: predicativo do complemento direto - GP
    e.g., Ela tratava os filhos [por 'criaturas']. Considerámos a exposição [sem interesse de maior]. Estariam a tomá-lo [por idiota]?
    Level: B2
    """
    matcher = DependencyMatcher(doc.vocab)

    pattern = [
        {
            "RIGHT_ID": "preposition",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["ADP", "SCONJ"]},
                # "SCONJ" added to match "Outros a classificaram de acertada, apesar das críticas."
                "LOWER": {"IN": ["por", "como", "sem"]}
            }
        },
        {
            "LEFT_ID": "preposition",
            "RIGHT_ID": "xcomp",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["iobj", "obl", "xcomp"]},
                # "xcomp" to match "Todos começaram a vê-lo como uma ameaça."
                "LOWER": {"NOT_IN": ["isso"]}
            },
            "REL_OP": "<"
        }
    ]
    
    matcher.add("b2d1_227_B2", [pattern])
    matches = matcher(doc)
    
    allowed_lemmas = ["considerar", "tratar", "tomar", "imaginar",
                      "ver", "classificar", "avaliar", "ter",
                      "vir", "aclamar", "rotular", "achar", "apresentar",
                      "julgar"]
    # "vir" to account for the preprocessing error of "vissem"

    # Dictionary to store matches keyed by the head token's index
    results_dict = {}
    
    for match_id, token_ids in matches:
        last_token = doc[max(token_ids)]
        # Continue to the next match if the last token's head lemma is not allowed
        if last_token.head.lemma_ not in allowed_lemmas or last_token.lemma_ == "imediato":
            # So that we don't match "os críticos rotularam-no, quase de imediato, por exagerado."
            continue
        
        # Calculate start and end indices of the matched span
        start = min(token_ids)
        end = max(token_ids) + 1
        tokens_in_span = doc[start:end]
        text = reconstruct_text(tokens_in_span)
        
        head_index = last_token.head.i
        
        # If there's already a match with the same head, only keep the one with the smaller end index
        # So that we don't match "sem provas" in "A imprensa tratou o político por traidor, mesmo sem provas conclusivas."
        if head_index in results_dict:
            _, _, _, stored_end = results_dict[head_index]
            if end < stored_end:
                results_dict[head_index] = (doc.vocab.strings[match_id], text, start, end)
        else:
            results_dict[head_index] = (doc.vocab.strings[match_id], text, start, end)

    # Convert dictionary values to a list of results
    results = list(results_dict.values())
    return results


def b2d1_229(doc):
    """
    B2.1-229: modificador - de constituinte verbal/predicado - oração - não finita (gerundiva/infinitiva - diferentes valores)
    e.g., Sabendo que vinhas, preparei um jantar especial. Ao entrar em casa, reparou na janela partida.
    Level: B2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern_gerund = [
        {
            "RIGHT_ID": "adverbial",
            "RIGHT_ATTRS": {
                "DEP": "advcl",
                "MORPH": {"INTERSECTS": ["VerbForm=Ger"]}
                }
        },
        {
            "LEFT_ID": "adverbial",
            "RIGHT_ID": "root",
            "RIGHT_ATTRS": {
                "DEP": "root"
                },
            "REL_OP": "<"           
        }
    ]
    
    # To match "Sendo professor há mais de vinte anos, conhece bem os desafios do sistema educativo."
    pattern_alt_gerund = [
        {
            "RIGHT_ID": "aux",
            "RIGHT_ATTRS": {
                "POS": "AUX",
                # So that we match "Por ter perdido os documentos, teve de repetir todo o processo."
                "MORPH": {"INTERSECTS": ["VerbForm=Ger"]}
                }
        },
        {
            "LEFT_ID": "aux",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "DEP": "advcl",
                "POS": {"IN": ["NOUN", "PROPN"]}
                },
            "REL_OP": "<"           
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "root",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["root", "xcomp"]}
                # "xcomp" so that we match "Por ter perdido os documentos, teve de repetir todo o processo."
            },
            "REL_OP": "<"
        }
    ]

    # So that we match "Por ter perdido os documentos, teve de repetir todo o processo."
    pattern_inf = [
        {
            "RIGHT_ID": "aux",
            "RIGHT_ATTRS": {
                "POS": "AUX",
                "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}
                }
        },
        {
            "LEFT_ID": "aux",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "DEP": "advcl",
                "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]}
                },
            "REL_OP": "<"           
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "sub",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["ao", "por"]}
                },
            "REL_OP": ">"           
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "root",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["root", "xcomp"]}
            },
            "REL_OP": "<"
        }
    ]

    pattern_inf_alt = [
        {
            "RIGHT_ID": "adverbial",
            "RIGHT_ATTRS": {
                "DEP": "advcl",
                "MORPH": {"INTERSECTS": ["VerbForm=Inf"]}
                }
        },
        {
            "LEFT_ID": "adverbial",
            "RIGHT_ID": "sub",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["ao", "por"]}
                },
            "REL_OP": ">"           
        },
        {
            "LEFT_ID": "adverbial",
            "RIGHT_ID": "root",
            "RIGHT_ATTRS": {
                "DEP": "root"
                },
            "REL_OP": "<"           
        }
    ]

    matcher.add("b2d1_229_B2", [pattern_gerund, pattern_inf, pattern_inf_alt, pattern_alt_gerund])

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


def b2d1_230(doc):
    """
    B2.1-230: complemento indireto - pleonástico - um dos complementos é obrigatoriamente pronome pessoal átono
    e.g., Quem [lhe] disse [a ela] que estávamos a organizar uma festa?
    Level: C1
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    # spaCy's DependencyMatcher has no plain "$" sibling operator; siblings are
    # expressed with "$++" (right sibling) or "$--" (left sibling). Since the
    # átono pronoun may sit on either side of the strong pronoun, each pattern
    # below is split into a "$++" and a "$--" variant.
    pattern = [
        {
            "RIGHT_ID": "lhe",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["me", "te", "lhe",
                                 "nos", "vos", "lhes"]},
                "DEP": "iobj",
                "POS": "PRON"
                }
        },
        {
            "LEFT_ID": "lhe",
            "RIGHT_ID": "pron",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["PRON", "PROPN"]},
                # "obl" to account for "A mim, o conselho de administração exigiu-me um relatório detalhado sobre o último trimestre."
                "DEP": {"IN": ["obj", "iobj", "obl"]}
                },
            "REL_OP": "$++"
        },
        {
            "LEFT_ID": "pron",
            "RIGHT_ID": "prep",
            "RIGHT_ATTRS": {
                "POS": "ADP"
            },
            "REL_OP": ">"
        }
    ]

    pattern_rev = [
        {
            "RIGHT_ID": "lhe",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["me", "te", "lhe",
                                 "nos", "vos", "lhes"]},
                "DEP": "iobj",
                "POS": "PRON"
                }
        },
        {
            "LEFT_ID": "lhe",
            "RIGHT_ID": "pron",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["PRON", "PROPN"]},
                # "obl" to account for "A mim, o conselho de administração exigiu-me um relatório detalhado sobre o último trimestre."
                "DEP": {"IN": ["obj", "iobj", "obl"]}
                },
            "REL_OP": "$--"
        },
        {
            "LEFT_ID": "pron",
            "RIGHT_ID": "prep",
            "RIGHT_ATTRS": {
                "POS": "ADP"
            },
            "REL_OP": ">"
        }
    ]

    pattern_alt = [
        {
            "RIGHT_ID": "prep",
            "RIGHT_ATTRS": {
                "POS": "ADP"
            }
        },
        {
            "LEFT_ID": "prep",
            "RIGHT_ID": "pron",
            "RIGHT_ATTRS": {
                # Excluding "NOUN" because it would match "A nós, contudo, ninguém nos reportou o erro no sistema operativo."
                "POS": {"IN": ["PRON", "PROPN"]},
                "DEP": {"IN": ["obj", "iobj", "obl"]}
                },
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "pron",
            "RIGHT_ID": "lhe",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["me", "te", "lhe",
                                 "nos", "vos", "lhes"]},
                "DEP": "iobj",
                "POS": "PRON"
                },
            "REL_OP": "$++"
        }

    ]

    pattern_alt_rev = [
        {
            "RIGHT_ID": "prep",
            "RIGHT_ATTRS": {
                "POS": "ADP"
            }
        },
        {
            "LEFT_ID": "prep",
            "RIGHT_ID": "pron",
            "RIGHT_ATTRS": {
                # Excluding "NOUN" because it would match "A nós, contudo, ninguém nos reportou o erro no sistema operativo."
                "POS": {"IN": ["PRON", "PROPN"]},
                "DEP": {"IN": ["obj", "iobj", "obl"]}
                },
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "pron",
            "RIGHT_ID": "lhe",
            "RIGHT_ATTRS": {
                "LOWER": {"IN": ["me", "te", "lhe",
                                 "nos", "vos", "lhes"]},
                "DEP": "iobj",
                "POS": "PRON"
                },
            "REL_OP": "$--"
        }

    ]


    matcher.add("b2d1_230_C1", [pattern, pattern_rev, pattern_alt, pattern_alt_rev])

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


def b2d1_231(doc):
    """
    B2.1-231: modificador - oracional - não finita (gerundiva - diferentes valores)
    e.g., Tendo sabido que vinhas, podia ter preparado um jantar especial.
    Level: C1

    The distincion between this property and b2d1_229 seems to be that in 231,
    we have a compound gerund while 229 we have a simple grund.
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    
    pattern = [
        {
            "RIGHT_ID": "aux",
            "RIGHT_ATTRS": {
                "POS": "AUX",
                "MORPH": {"IS_SUPERSET": ["VerbForm=Ger"]}
                }
        },
        {
            "LEFT_ID": "aux",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "DEP": "advcl",
                "POS": "VERB"
                },
            "REL_OP": "<"           
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "root",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["root", "xcomp"]}
            },
            "REL_OP": "<"
        }
    ]

    matcher.add("b2d1_231_C1", [pattern])

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


def b2d1_232(doc):
    """
    B2.1-232: modificador - oracional - não finita (partcipial/gerundiva - expressão de circunstâncias de tempo e condição)
    e.g., Em chegando o João, vamos./ Em tendo tempo, vou contigo. Aberta a porta, puderam entrar./ Chegada a hora, partiram.
    Level: C2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    
    pattern_ger = [
        {
            "RIGHT_ID": "em",
            "RIGHT_ATTRS": {
                "LOWER": "em"
            }
        },
        {
            "LEFT_ID": "em",
            "RIGHT_ID": "ger",
            "RIGHT_ATTRS": {
                # Removed the verb constraint to match "Em estando tudo pronto, começamos a apresentação." 
                "MORPH": {"IS_SUPERSET": ["VerbForm=Ger"]}
                },
            "REL_OP": "<"
        },
        {
            "LEFT_ID": "ger",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["NOUN", "PROPN", "PRON", "ADJ", "ADV"]}
                # Added "ADV" to match "Em voltando cedo, ainda podemos jantar juntos."
                },
            "REL_OP": ">"           
        },
        {
            "LEFT_ID": "ger",
            "RIGHT_ID": "root",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["root", "xcomp"]}
                # Added "xcomp to match "Em voltando cedo, ainda podemos jantar juntos."
            },
            "REL_OP": "<"
        }
    ]


    pattern_part = [
        {
            "RIGHT_ID": "part",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
                "IS_SENT_START": True
                # So that we don't match "Tendo vivido muitos anos no estrangeiro, compreendia bem as dificuldades dos imigrantes."
                }
        },
        {
            "LEFT_ID": "part",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["NOUN", "PROPN", "PRON", "ADJ", "ADV"]}
                },
            "REL_OP": ">"           
        },
        {
            "LEFT_ID": "part",
            "RIGHT_ID": "root",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["root", "xcomp"]}
            },
            "REL_OP": "<"
        }
    ]

    # To match "Em sendo aceite a proposta, iniciaremos os trabalhos na próxima semana."
    pattern_ger_compound = [
        {
            "RIGHT_ID": "em",
            "RIGHT_ATTRS": {
                "LOWER": "em"
            }
        },
        {
            "LEFT_ID": "em",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {
                "DEP": "advcl",
                "POS": "VERB"
                },
            "REL_OP": "<"           
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "ger",
            "RIGHT_ATTRS": {
                "POS": "AUX",
                "MORPH": {"IS_SUPERSET": ["VerbForm=Ger"]}
                },
            "REL_OP": ">"
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["NOUN", "PROPN", "PRON", "ADJ", "ADV"]}
                },
            "REL_OP": ">"           
        },
        {
            "LEFT_ID": "verb",
            "RIGHT_ID": "root",
            "RIGHT_ATTRS": {
                "DEP": {"IN": ["root", "xcomp"]}
            },
            "REL_OP": "<"
        }
    ]

    # To match "Instalada a nova versão, o programa passou a funcionar corretamente."
    pattern_part_alt = [
        {
            "RIGHT_ID": "part",
            "RIGHT_ATTRS": {
                "MORPH": {"IS_SUPERSET": ["VerbForm=Part"]},
                "DEP": "acl",
                "IS_SENT_START": True
                }
        },
        {
            "LEFT_ID": "part",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["NOUN", "PROPN", "PRON", "ADJ", "ADV"]}
                },
            "REL_OP": ">"           
        },
        {
            "LEFT_ID": "part",
            "RIGHT_ID": "noun2",
            "RIGHT_ATTRS": {
                "POS": "NOUN",
                "DEP": "nsubj"
                },
            "REL_OP": "<"           
        },
        {
            "LEFT_ID": "noun2",
            "RIGHT_ID": "root",
            "RIGHT_ATTRS": {
                "DEP": "root"
            },
            "REL_OP": "<"
        }
    ]

    matcher.add("b2d1_232_C2", [pattern_ger, pattern_part, pattern_ger_compound, pattern_part_alt])

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


def b2d1_308(doc):
    """
    B2.1-308: complemento do nome - com nomes icónicos, como 'foto'/'fotografia', 'imagem', 'desenho', 'figura' (GP)
    e.g., O desenho [de Lisboa] estava muito bonito.
    Level: A2
    """
    matcher = DependencyMatcher(nlp_stanza.vocab)

    pattern = [
        {
            "RIGHT_ID": "iconic",
            "RIGHT_ATTRS": {
                "LEMMA": {"IN": ["foto", "fotografia", "imagem", "desenho", "figura"]}
                }
        },
        {
            "LEFT_ID": "iconic",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {
                "POS": {"IN": ["NOUN", "PROPN"]}
                },
            "REL_OP": ">++"           
        },
        {
            "LEFT_ID": "noun",
            "RIGHT_ID": "de",
            "RIGHT_ATTRS": {
                "LOWER": "de"
            },
            "REL_OP": ">--"
        }
    ]

    matcher.add("b2d1_308_A2", [pattern])

    matches = matcher(doc)

    # Collect all matches with their start and end indices
    all_matches = []
    for match_id, token_ids in matches:
        # Calculate start and end indices
        start = min(token_ids)
        end = max(token_ids) + 1
        tokens_in_span = doc[start:end]
        
        text = reconstruct_text(tokens_in_span)
        all_matches.append((doc.vocab.strings[match_id], text, start, end))
    
    # Group matches by start index
    start_index_groups = {}
    for match in all_matches:
        match_id, text, start, end = match
        if start not in start_index_groups:
            start_index_groups[start] = []
        start_index_groups[start].append(match)
    
    # For each group with the same start index, keep only the match with the largest end index
    results = []
    for start, matches_with_same_start in start_index_groups.items():
        # Find the match with the largest end index
        best_match = max(matches_with_same_start, key=lambda x: x[3])
        results.append(best_match)
    
    return results


def b2d1_311(doc):
    """
    B2.1-311: complemento do nome - com nomes que regem preposição (GP oracional - oração não finita)
    e.g., Sentia saudades [de estar com os amigos]. A ideia [de irem ao parque] era ótima. Estive a pensar no facto [de não poderes vir nessa data].
    Level: B1

    The possible dependency relations are too varied to result in a reliable outcome.
    Opting for pattern matching instead.
    """
    matcher = Matcher(doc.vocab)

    pattern = [

                {"POS": "NOUN"},
                {"POS": "ADJ", "OP": "{,2}"},
                # "ADV" excluded not to match "É muito importante lavar bem as mãos antes de comer."
                # To match "desejo constante de reencontrar os velhos amigos"
                {"LOWER": "de"},
                {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "{,3}"},
                {"MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}}
            ]
    
    matcher.add("b2d1_311_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def b2d1_312(doc):
    """
    B2.1-312: modificador de nome - apositivo - GN
    e.g., D. Afonso Henriques, primeiro rei de Portugal, teve sete filhos.
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [

                {"ENT_TYPE": {"IN": ["PER", "LOC", "ORG"]}, "LOWER": {"NOT_IN": ["malta"]}, "OP": "+"},
                {"LOWER": ","},
                {"POS": {"NOT_IN": ["NOUN", "NUM"]}, "OP": "{,3}"},
                # "NUM" so that we can match "Foi aqui que viveu Fernando Pessoa, um dos maiores poetas portugueses de sempre."
                {"DEP": "appos", "POS": {"IN": ["NOUN", "NUM"]}}
            ]
    pattern_reverse = [
                # "LOWER" is ad-hoc fix so that we don't match "Bom dia, Dr. Andrade."
                {"POS": "NOUN", "LOWER": {"NOT_IN": ["dia", "tarde", "noite", "semana"]}},
                {"POS": {"NOT_IN": ["NOUN"]}, "OP": "{,2}"},
                {"LOWER": ","},
                {"ENT_TYPE": {"IN": ["PER", "LOC", "ORG"]}, "DEP": "appos", "OP": "+"},
            ]

    matcher.add("b2d1_312_B1", [pattern, pattern_reverse], greedy="LONGEST")

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
b2d1_312.REQUIRES_SPACY = True


def b2d1_315(doc):
    """
    B2.1-315: complemento do nome - com nomes que regem preposição (GP oracional - oração finita)   
    e.g., Tinha vontade de que lhe trouxessem doces regionais. A hipótese de que conseguisse o trabalho era grande.
    Level: B2
    """
    matcher = Matcher(doc.vocab)

    pattern = [

                {"POS": "NOUN"},
                {"POS": "SCONJ"},
                {"LOWER": {"IN": ["que", "se"]}},
            ]

    matcher.add("b2d1_315_B2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]

b2d1_315.REQUIRES_SPACY = True