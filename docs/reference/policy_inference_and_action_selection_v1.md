# Inferència de política i selecció d'accions v1

## Objectiu i contracte

El PR-15 avalua simultàniament tota la capacitat `C` de `GenomeBatch` sobre observacions locals
`float32[C,15]` i produeix scores `float32[C,7]`. Després converteix les preferències neuronals
en `proposed_actions`, no en accions executades. La
[validació local v1](action_contract_and_validation_v1.md) conserva la proposta i produeix una
acció encaminada; només substitueix rebutjos per `STAY`, no produeix accions executades.

| Element | Forma | Dtype | Semàntica |
|---|---:|---|---|
| Observacions | `[C,15]` | `float32` | Entrada local |
| Hidden | `[C,16]` | `float32` | Activació `tanh` temporal |
| Raw scores | `[C,7]` | `float32` | Preferències neuronals |
| Scores canònics | `[C,7]` | `float32` | Zero en inactius o invàlids |
| Propostes | `[C]` | `int32` | `ActionCode` proposat |
| Scores invàlids | `()` | `int32` | Agents actius afectats |
| Empats exactes | `()` | `int32` | Agents actius amb més d'un màxim |

## Fórmula batched

Les quatre fulles de `GenomeBatch` tenen formes `[C,16,15]`, `[C,16]`, `[C,7,16]` i
`[C,7]`. La primera contracció és `hidden = tanh(einsum("chi,ci->ch", W1, observations) + b1)`;
la segona és `raw_scores = einsum("coh,ch->co", W2, hidden) + b2`. Aquesta fórmula és el
forward exacte de `PolicyMLP` aplicat sobre l'eix poblacional, sense construir models Python per
agent. La sortida és lineal: no es normalitza i un score no és una probabilitat. `hidden` no es
retorna ni es persisteix.

Els scores mantenen l'ordre únic d'`ActionCode`: `STAY`, `MOVE_NORTH`, `MOVE_SOUTH`,
`MOVE_EAST`, `MOVE_WEST`, `EAT`, `REPRODUCE`.

## Selecció, canonicalització i diagnòstics

Per cada fila activa finita es calcula el màxim. Hi ha empat només si dos o més valors
`float32` són exactament iguals al màxim, sense tolerància. Una màscara de candidats i el mínim
índex seleccionen explícitament el codi menor. Aquest criteri estable introdueix un biaix
deliberat cap als codis baixos: un empat total prefereix especialment `STAY`; no és neutral des
del punt de vista ecològic.

Un slot inactiu conserva la forma però rep scores zero i proposa `STAY`, fins i tot si conté
residus. Una fila activa amb qualsevol `NaN`, `+inf` o `-inf` també rep scores zero i proposa
`STAY`; incrementa `invalid_active_score_count` i no incrementa `exact_tie_count`. No es llancen
excepcions numèriques dins JIT. La selecció no usa RNG, món, energia, identitats ni màscares de
legalitat. Una proposta com moure's fora del món continua sent la preferència neuronal fins que
el PR-16 la validi.

## Esquema versionat i compilació

- Nom: `deterministic_max_score_lowest_action_code_v1`.
- Versió: `1`.
- Digest: `fb5398375ce760eb4335f353167359555d7518aec47ce257e79c5b8d6056603f`.
- Desempat: `lowest_action_code`; fallbacks inactiu i invàlid: `stay`.

El payload canònic JSON inclou el digest de política, set scores, l'ordre complet d'`ActionCode`,
la definició exacta del màxim i l'empat, els fallbacks, l'absència de normalització i RNG, i la
semàntica de proposta. Canviar el desempat o l'ordre exigeix versió i digest nous i revisar la
compatibilitat experimental. `CompileSignature` v7 incorpora versió i digest de selecció i el
recompte d'accions, a més dels digests d'observació i política i la topologia 15–16–7. N'exclou
scores, propostes, `alive`, seeds, keys, genomes i observacions concrets.

Les operacions són pures, vectoritzades, de formes fixes i compatibles amb eager, JIT i
`lax.scan`. El cost neuronal es paga per tota la capacitat, no només per `sum(alive)`, a canvi
d'evitar formes dinàmiques i recompilacions.

| Funcionalitat | PR responsable |
|---|---|
| Esquema d'observacions | PR-12 |
| `PolicyMLP` individual | PR-13 |
| `GenomeBatch` | PR-14 |
| Inferència poblacional | PR-15 |
| Selecció determinista | PR-15 |
| Validació de propostes | PR-16 |
| Moviment | PR-17 |
| Alimentació | PR-18 |
| Reproducció | PR-22 |
| Mètriques de decisions | PR-25 |
| Integració del `step` | PR-27 |
| `lax.scan` compilat | PR-28 |

La saturació de `tanh` pot reduir diferències neuronals i genomes futurs mutats poden generar
no finits. PR-16 i els PR posteriors resoldran legalitat, moviment, alimentació, reproducció,
costos i conflictes; cap d'aquestes responsabilitats forma part d'aquest contracte.
