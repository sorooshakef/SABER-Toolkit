from spacy.matcher import Matcher
from text_reconstruction import reconstruct_text

# Too unreliable despite the many accommodations:  A desinformação nas redes sociais constitui uma ameaça à democracia, ameaça essa que se tem vindo a intensificar nos últimos anos.
# def a5d1_5(doc):
#     """
#     A5.1-5: ausência do pronome (Sujeito omitido (elíptico ou subentendido))
#     e.g., Ela abriu a porta e entrou. / A minha irmã ligou quando chegou a casa.
#     Level: A2

#     The constructions that this structure matches are quite constrained, but perhaps
#     that's desirable considering the prevalence of constructions without an explicit subject.
#     """
#     matcher = Matcher(doc.vocab)

#     # Singular patterns

#     # First person singular
#     pattern_sing_1 = [
#     {"POS": {"IN": ["NOUN", "PRON", "PROPN"]}, "MORPH": {"IS_SUPERSET": ["Number=Sing"]}, "DEP": "nsubj"},
#     # Excluding "que" so that we don't match "todas as pessoas que estavam na rua pareciam tranquilas"
#     {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "IS_SENT_START": False, "NORM": {"NOT_IN": ["que"]}},
#     {"POS": {"IN": ["VERB", "AUX"]}, "MORPH": {"IS_SUPERSET": ["Number=Sing", "VerbForm=Fin", "Person=1"]}, "IS_SENT_START": False, "DEP": {"NOT_IN": ["acl:relcl"]}},
#     # So that we don't match "todas as versões que fizemos continuaram fúcsia."
#     # So that we don't match "Não é."
#     {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "IS_SENT_START": False, "DEP": {"NOT_IN": ["nsubj"]}, "NORM": {"NOT_IN": ["que", "nao"]}},
#     {"POS": {"IN": ["VERB", "AUX"]}, "MORPH": {"IS_SUPERSET": ["Number=Sing", "VerbForm=Fin", "Person=1"]}, "IS_SENT_START": False, "DEP": {"NOT_IN": ["acl:relcl", "ccomp", "root"]}, "LEMMA": {"NOT_IN": ["parecer", "haver"]}}
#     # "ccomp" so that we don't match "O médico lhe explicou à paciente como funcionava o tratamento, passo a passo."
#     # "root" so that we don't match "Isso não é possível! - respondeu ela prontamente."
#     ]

#     # Second person singular
#     pattern_sing_2 = [
#     {"POS": {"IN": ["NOUN", "PRON", "PROPN"]}, "MORPH": {"IS_SUPERSET": ["Number=Sing"]}, "DEP": "nsubj"},
#     {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "IS_SENT_START": False, "NORM": {"NOT_IN": ["que"]}},
#     {"POS": {"IN": ["VERB", "AUX"]}, "MORPH": {"IS_SUPERSET": ["Number=Sing", "VerbForm=Fin", "Person=2"]}, "IS_SENT_START": False, "DEP": {"NOT_IN": ["acl:relcl"]}},
#     # So that we don't match "todas as versões que fizemos continuaram fúcsia."
#     {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "IS_SENT_START": False, "DEP": {"NOT_IN": ["nsubj"]}, "NORM": {"NOT_IN": ["que", "nao"]}},
#     {"POS": {"IN": ["VERB", "AUX"]}, "MORPH": {"IS_SUPERSET": ["Number=Sing", "VerbForm=Fin", "Person=2"]}, "IS_SENT_START": False, "DEP": {"NOT_IN": ["acl:relcl", "ccomp", "root"]}}
#     ]

#      # Third person singular
#     pattern_sing_3 = [
#     {"POS": {"IN": ["NOUN", "PRON", "PROPN"]}, "MORPH": {"IS_SUPERSET": ["Number=Sing"]}, "DEP": "nsubj"},
#     {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "IS_SENT_START": False, "NORM": {"NOT_IN": ["que"]}},
#     {"POS": {"IN": ["VERB", "AUX"]}, "MORPH": {"IS_SUPERSET": ["Number=Sing", "VerbForm=Fin", "Person=3"]}, "IS_SENT_START": False, "DEP": {"NOT_IN": ["acl:relcl"]}, "LEMMA": {"NOT_IN": ["haver"]}},
#     # So that we don't match "todas as versões que fizemos continuaram fúcsia."
#     {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "IS_SENT_START": False, "DEP": {"NOT_IN": ["nsubj"]}, "NORM": {"NOT_IN": ["que", "nao"]}},
#     {"POS": {"IN": ["VERB", "AUX"]}, "MORPH": {"IS_SUPERSET": ["Number=Sing", "VerbForm=Fin", "Person=3"]}, "IS_SENT_START": False, "DEP": {"NOT_IN": ["acl:relcl", "ccomp", "root"]}, "LEMMA": {"NOT_IN": ["parecer", "haver"]}}
#     # The last condition is so that we don't match "Esta situação, na verdade, já era previsível há muito tempo."
#     ]

#     # Plural patterns
    
#     # First person plural
#     pattern_plur_1 = [
#     {"POS": {"IN": ["NOUN", "PRON", "PROPN"]}, "MORPH": {"IS_SUPERSET": ["Number=Plur"]}, "DEP": "nsubj"},
#     {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "IS_SENT_START": False, "NORM": {"NOT_IN": ["que"]}},
#     {"POS": {"IN": ["VERB", "AUX"]}, "MORPH": {"IS_SUPERSET": ["Number=Plur", "VerbForm=Fin", "Person=1"]}, "IS_SENT_START": False, "DEP": {"NOT_IN": ["acl:relcl"]}},
#     # So that we don't match "todas as versões que fizemos continuaram fúcsia."
#     {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "IS_SENT_START": False, "DEP": {"NOT_IN": ["nsubj"]}, "NORM": {"NOT_IN": ["que", "nao"]}},
#     {"POS": {"IN": ["VERB", "AUX"]}, "MORPH": {"IS_SUPERSET": ["Number=Plur", "VerbForm=Fin", "Person=1"]}, "IS_SENT_START": False, "DEP": {"NOT_IN": ["acl:relcl", "ccomp", "root"]}}
#     ]

#     # Second person plural
#     pattern_plur_2 = [
#     {"POS": {"IN": ["NOUN", "PRON", "PROPN"]}, "MORPH": {"IS_SUPERSET": ["Number=Plur"]}, "DEP": "nsubj"},
#     {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "IS_SENT_START": False, "NORM": {"NOT_IN": ["que"]}},
#     {"POS": {"IN": ["VERB", "AUX"]}, "MORPH": {"IS_SUPERSET": ["Number=Plur", "VerbForm=Fin", "Person=2"]}, "IS_SENT_START": False, "DEP": {"NOT_IN": ["acl:relcl"]}},
#     # So that we don't match "todas as versões que fizemos continuaram fúcsia."
#     {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "IS_SENT_START": False, "DEP": {"NOT_IN": ["nsubj"]}, "NORM": {"NOT_IN": ["que", "nao"]}},
#     {"POS": {"IN": ["VERB", "AUX"]}, "MORPH": {"IS_SUPERSET": ["Number=Plur", "VerbForm=Fin", "Person=2"]}, "IS_SENT_START": False, "DEP": {"NOT_IN": ["acl:relcl", "ccomp", "root"]}}
#     ]

#      # Third person singular
#     pattern_plur_3 = [
#     {"POS": {"IN": ["NOUN", "PRON", "PROPN"]}, "MORPH": {"IS_SUPERSET": ["Number=Plur"]}, "DEP": "nsubj"},
#     {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "IS_SENT_START": False, "NORM": {"NOT_IN": ["que"]}},
#     {"POS": {"IN": ["VERB", "AUX"]}, "MORPH": {"IS_SUPERSET": ["Number=Plur", "VerbForm=Fin", "Person=3"]}, "IS_SENT_START": False, "DEP": {"NOT_IN": ["acl:relcl"]}},
#     # So that we don't match "todas as versões que fizemos continuaram fúcsia."
#     {"POS": {"NOT_IN": ["VERB", "AUX"]}, "OP": "*", "IS_SENT_START": False, "DEP": {"NOT_IN": ["nsubj"]}, "NORM": {"NOT_IN": ["que", "nao"]}},
#     {"POS": {"IN": ["VERB", "AUX"]}, "MORPH": {"IS_SUPERSET": ["Number=Plur", "VerbForm=Fin", "Person=3"]}, "IS_SENT_START": False, "DEP": {"NOT_IN": ["acl:relcl", "ccomp", "root"]}}
#     ]
    

#     matcher.add("a5d1_5_A2", [pattern_sing_1, pattern_sing_2, pattern_sing_3,
#                               pattern_plur_1, pattern_plur_2, pattern_plur_3])

#     matches = matcher(doc)

#     filtered_matches = []
#     # So that we don't match "Sim, é nossa, mas quem trata do jardim é mais o Pedro; cuida dele com muito carinho."
#     for match_id, start, end in matches:
#         token = doc[start]
#         # Check the morphological feature for relative pronoun
#         if "Rel" in token.morph.get("PronType"):
#             continue
#         filtered_matches.append((doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end))

#     return filtered_matches

def a5d1_6(doc):
    """
    A5.1-6: pronomes complemento - forma - variação em pessoa, número e género
    e.g., me, te, o, a, se; nos, vos, os, as, se; me, te, lhe;
    nos, vos, lhes;
    a mim, a ti, a si, ele, ela; a nós, a vós, a eles, a elas;
    comigo, contigo, consigo, com ele, com ela;
    connosco, convosco, consigo, com eles, com elas
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {
                    "LOWER": {"IN": ["o", "a", "se", "nos", "vos",
                                     "os", "as"]},
                    "MORPH": {"IS_SUPERSET": ["PronType=Prs"]}
                }
            ]

    pattern_1 = [
                {
                    "LOWER": {"IN": ["lhe", "lhes", "comigo", "contigo", "me", "te",
                                     "consigo", "connosco", "conosco", "convosco"]},
                    "POS": {"IN": ["ADV", "PRON", "ADJ"]} # ADJ for "contigo"
                }
            ]
    
    pattern_2 = [
        {"LOWER": {"IN": ["a", "para", "em"]}},
        {"NORM": {"IN": ["mim", "ti", "si", "ele", "ela", "nos", "eles", "elas", "voces"]}}
    ]

    pattern_3 = [
        {"LOWER": "com"},
        {"NORM": {"IN": ["ele", "ela", "eles", "elas", "voces"]}}
    ]

    matcher.add("a5d1_6_A2", [pattern, pattern_1, pattern_2, pattern_3])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a5d1_7(doc):
    """
    A5.1-7: colocação do pronome - antes da forma verbal
    Ele não se chama João. Como te chamas?
    Level: A2
    """
    matcher = Matcher(doc.vocab)

    # The idea is that we don't want to match e.g. "Ele perguntou se estava pronto."
    verbs_que_se_change = [
    "descobrir",
    "saber",
    "verificar",
    "confirmar",
    "averiguar",
    "determinar",
    "compreender",
    "aprender",
    "ver",
    "perceber",
    "esclarecer",
    "informar",
    "acreditar",
    "duvidar",
    "questionar",
    "perguntar",
    "interrogar",
    "ponder",
    "averiguar",
    "hesitar",
    "decidir"
    ]

    pattern = [
                {
                    "LEMMA": {"NOT_IN": verbs_que_se_change}
                },
                {
                "MORPH": {"IS_SUPERSET": ["PronType=Prs"]},
                "LOWER": {"IN": ["te", "se", "me", "lhe", "vos", "nos", "lhes", "o", "a", "os", "as"]}
                },
                {"POS": "VERB"}
            ]

    matcher.add("a5d1_7_A2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start+1:end]), start+1, end) for match_id, start, end in matches]
a5d1_7.REQUIRES_SPACY = True
# Stanza performs poorly with "Talvez nos vejamos amanhã à tarde."
# But spaCy doesn't correctly parse "Quando te deitas normalmente?"


def a5d1_9(doc):
    """
    A5.1-9: pronomes complemento - forma - variantes morfológicas de o, a, os, as - com verbos terminados em <r>, <s> ou <z>, com desaparecimento da consoante
    e.g., lo, la, los, las
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {
                    "LOWER": {"IN": ["lo", "la", "los", "las"]}
                }
            ]

    matcher.add("a5d1_9_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a5d1_10(doc):
    """
    A5.1-10: pronomes complemento - forma - variantes morfológicas de o, a, os, as - com verbos terminados em nasal
    e.g., no, na, nos, nas
    "Eles cantaram-na na festa."
    Level: B1
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {"POS": "VERB", "MORPH": {"IS_SUPERSET": ["Person=3", "Number=Plur"]}},
                {"TEXT": {"REGEX": r"(no|na|nos|nas)$"}, "POS": "PRON"}
            ]

    matcher.add("a5d1_10_B1", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]


def a5d1_12(doc):
    """
    A5.1-12: pronomes complemento - forma - grupos de pronomes (formas contraídas)
    e.g., me + o, a, os, as = mo, ma, mos, mas; te + o, a, os, as = to, ta, tos, tas; lhe + o, as, os, as = lho, lha, lhos, lhas
    Level: B2
    """
    matcher = Matcher(doc.vocab)

    # Enclitic/mesoclitic contracted forms attached to a verb, e.g. "devolveu-mo".
    pattern = [
                {
                    "LOWER": {"REGEX": r"^[A-Za-zÀ-ÖØ-öø-ÿ]+-(?:mo|ma|mos|mas|to|ta|tos|tas|lho|lha|lhos|lhas)$"}
                }
            ]

    # Standalone (proclitic) contracted forms, e.g. "ele nunca mo devolveu".
    # Exclude conjunctions so we don't match "mas" ("but"), which is tagged CCONJ.
    pattern_standalone = [
                {
                    "LOWER": {"REGEX": r"^(?:mo|ma|mos|mas|to|ta|tos|tas|lho|lha|lhos|lhas)$"},
                    "POS": {"NOT_IN": ["CCONJ", "SCONJ"]}
                }
            ]


    matcher.add("a5d1_12_B2", [pattern, pattern_standalone])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
a5d1_12.REQUIRES_SPACY = True


from spacy.matcher import Matcher

def a5d1_15(doc):
    """
    A5.1-15: colocação dos pronomes - no interior da forma verbal - com formas do futuro e do condicional
    e.g., O professor informar-vos-á da data da prova. Se eu soubesse, ter-lhe-ia dito a verdade.
    Level: B2
    """
    matcher = Matcher(doc.vocab)

    pattern = [
                {
                    "LOWER": {"REGEX": r"^[A-Za-zÀ-ÖØ-öø-ÿ]+-(?:me|te|se|nos|vos|lhe|lhes|lo|la|los|las|mo|ma|mos|mas|to|ta|tos|tas|lho|lha|lhos|lhas)-[A-Za-zÀ-ÖØ-öø-ÿ]+$"}
                }
            ]

    matcher.add("a5d1_15_B2", [pattern])

    matches = matcher(doc)

    return [(doc.vocab.strings[match_id], reconstruct_text(doc[start:end]), start, end) for match_id, start, end in matches]
a5d1_15.REQUIRES_SPACY = True


