"""Labels for the multiword-expression (MWE) matchers in ``saber/matchers_mwe``.

This is the project-owned counterpart of :mod:`saber.label_translation`, which
is copied verbatim from the upstream SABER application and replaced wholesale on
every matcher update. Everything that is *not* part of upstream lives here and
in ``saber/matchers_mwe`` so that such an update cannot delete it.

The identifiers follow the same scheme as the upstream ones
(``<category>_<number>_<suffix>``), but the suffix is ``MWE`` instead of a CEFR
level: these patterns are not tied to a level of the Referencial Camões, so
:mod:`saber.registry` reports their level as ``NA``.
"""

# Structure id -> description. Declaration order is the order in which the
# structures are listed and in which their feature columns are emitted (after
# all the upstream structures).
LABEL_TRANSLATIONS_MWE = {
    "b2d4_1_MWE": "Expressões multipalavra - nome + adjetivo (relação amod)",
    "b2d4_2_MWE": "Expressões multipalavra - nome + preposição + nome",
    "b2d4_3_MWE": "Expressões multipalavra - verbo + (determinante) + nome (complemento direto)",
    "b2d4_4_MWE": "Expressões multipalavra - verbo + preposição + (determinante) + nome",
    "b2d4_5_MWE": "Expressões multipalavra - locuções fixas (relação fixed)",
}

# Group / category key -> display name, merged over the upstream
# LEVEL_DESCRIPTIONS. Only keys that upstream does not define belong here.
LEVEL_DESCRIPTIONS_MWE = {
    "b2d4": "Expressões multipalavra",
}
