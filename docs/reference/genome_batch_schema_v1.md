# Esquema batched de genomes v1

## Contracte

`GenomeBatch` és el bloc numèric heretable de la població. No és la xarxa individual
(`PolicyMLP`), la identitat de l'agent ni el seu comportament. La versió 1 hereta només pesos i
biaixos; no inclou metabolisme, costos, llindars, memòria, aprenentatge individual ni topologia
variable.

- Nom: `neural_genome_policy_mlp_15_16_7_v1`
- Versió: `1`
- Digest: `b80abe3d615a50b7f7cf533918a175b9be69e19e0e235343e976ef98b1069d0b`
- Inicialització: `glorot_uniform_zero_bias_v1`, versió `1`
- Paràmetres per genoma: `375`

El descriptor incorpora el digest de `PolicyMLP`, l'esquema d'observacions locals i l'ordre
estable d'`ActionCode`. Un canvi funcional requereix revisar versió, digest i compatibilitat.

| Fulla | Forma individual | Forma batched | Dtype | Inicialització | Slot inactiu |
|---|---:|---:|---|---|---|
| `layer1.weight` | `[16,15]` | `[C,16,15]` | `float32` | Glorot uniform | zero |
| `layer1.bias` | `[16]` | `[C,16]` | `float32` | Zero | zero |
| `layer2.weight` | `[7,16]` | `[C,7,16]` | `float32` | Glorot uniform | zero |
| `layer2.bias` | `[7]` | `[C,7]` | `float32` | Zero | zero |

## Slot, identitat i inicialització

Per a cada slot `i`, `population.alive[i]`, `population.genome_id[i]` i les quatre fulles
`genomes.*[i]` descriuen el mateix agent. El slot és ubicació computacional; `genome_id` és la
identitat persistent. No hi ha cap taula dinàmica ID-fila. Quan en el futur es mogui o s'alliberi
un slot, totes les dades s'hauran de moure coherentment o tornar a zero.

Els pesos `[fan_out, fan_in]` segueixen `U(-L,L)`, on `L = sqrt(6/(fan_in+fan_out))`:
`sqrt(6/31)` per a la primera capa i `sqrt(6/23)` per a la segona. Els biaixos són exactament
zero. El stream global és `RngStreamCode.GENOME_INITIALIZATION`; cada clau d'entitat es deriva
de `genome_id` i després dels substreams `LAYER1_WEIGHT=0` i `LAYER2_WEIGHT=1`. Per tant, una
permutació de slots només permuta genomes i ampliar `C` no altera els genomes comuns.

Un slot inactiu sempre té les quatre fulles a zero, fins i tot amb un ID residual. Un slot actiu
amb `genome_id < 0` també queda a zero i incrementa `invalid_active_genome_id_count`; no es fa
wrapping ni es llança una excepció dins JIT. La reproduïbilitat pressuposa el mateix entorn de
JAX, jaxlib, XLA i maquinari.

## Estat i orquestració futura

`SimulationState` acaba amb `genomes`, després de `population`. `state` només usa una anotació
type-only per no crear una dependència runtime cap a `policy`. Un futur orquestrador crearà la
població, passarà només `alive`, `genome_id` i la root key a `initialize_genome_batch`, comprovarà
`overflow == False` i el diagnòstic genòmic igual a zero, i construirà l'estat amb arguments per
nom. Aquest PR no implementa aquesta factoria.

| Funcionalitat | PR responsable |
|---|---|
| Model individual | PR-13 |
| Representació batched | PR-14 |
| Inicialització fundadora | PR-14 |
| Inferència poblacional | PR-15 |
| Selecció d'accions | PR-15 |
| Còpia parental | PR-22 |
| Mutació | PR-23 |
| Genealogia genòmica | PR-24 |
| Diversitat genètica | PR-44 |
| Persistència de genomes | PR-39 |
| Checkpoints | PR-37 i PR-41 |

`GenomeBatch` continua sense `__call__`; la [inferència batched i selecció](policy_inference_and_action_selection_v1.md)
la consumeixen sense construir una xarxa Python per agent. Mutació, reproducció, genealogia i
persistència continuen fora del contracte.
