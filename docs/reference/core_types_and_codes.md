# Tipus, dtypes i codis del nucli

Aquest contracte centralitza el vocabulari estable dels futurs arrays del nucli. No implementa
estat, comportament ni lògica de simulació.

## Política de dtypes

| Ús | Dtype |
|---|---|
| Valors reals | `float32` |
| Índexs, identificadors, comptadors, passos i codis | `int32` |
| Màscares | `bool` |

`float32` és el format real portàtil i eficient del prototip, sense activar globalment JAX x64.
`int32` fixa una representació homogènia per als enters del dispositiu i `bool` representa les
màscares sense codificacions numèriques alternatives. Les conversions dels consumidors han de
ser explícites i usar les constants de `evolucio.core`. La clau de `RngState` és l’única excepció: usa un dtype PRNG tipat, no un dtype numèric comú.

Els aliases d'identificadors són semàntica host; l'estat vectoritzat futur els emmagatzemarà en
arrays amb `ID_DTYPE`.

## Codis d'acció

| Nom | Valor | Significat |
|---|---:|---|
| `STAY` | 0 | Inacció funcional. |
| `MOVE_NORTH` | 1 | Intenció de moviment cap al nord. |
| `MOVE_SOUTH` | 2 | Intenció de moviment cap al sud. |
| `MOVE_EAST` | 3 | Intenció de moviment cap a l'est. |
| `MOVE_WEST` | 4 | Intenció de moviment cap a l'oest. |
| `EAT` | 5 | Intenció explícita d'alimentar-se. |
| `REPRODUCE` | 6 | Intenció explícita de reproduir-se. |

`ACTION_COUNT` val 7 i és la font de veritat per a la mida de sortida de la política neuronal.
L'espai d'accions és fix durant el prototip inicial i no és configurable.

## Codis de causa de mort

| Nom | Valor | Significat |
|---|---:|---|
| `NONE` | 0 | Agent viu o cap causa assignada; no és una mort real. |
| `ENERGY_DEPLETION` | 1 | Energia igual o inferior al llindar. |
| `MAX_AGE` | 2 | Edat màxima assolida. |
| `ENVIRONMENTAL_STRESS` | 3 | Causa ambiental terminal definida pel model. |
| `COMPETITIVE_EXCLUSION` | 4 | Exclusió competitiva explícita futura. |
| `INVALID_STATE` | 5 | Estat impossible o corrupte. |

`DEATH_CAUSE_COUNT` val 6. Els valors numèrics **no defineixen cap prioritat de mortalitat**.

## Representació i estabilitat

Els `IntEnum` són representacions host del contracte. Els arrays JAX guarden els valors enters
amb `CODE_DTYPE`, mai strings ni objectes `Enum`. Els noms, l'ordre i els valors numèrics són API
estable: canviar-los requeriria versionar i migrar qualsevol dada persistida que els contingui.

Aquest PR no resol accions, moviment, alimentació, reproducció o mortalitat, ni assigna causes.


## Codis de streams RNG

Els onze valors explícits de `RngStreamCode`, de `WORLD_INITIALIZATION = 0` a `GENOME_MUTATION = 10`, identifiquen dominis estables d’aleatorietat. La taula completa i el contracte de derivació són a [RNG determinista i identificadors interns](rng_and_identifiers.md).

## Codis del món

`BoundaryModeCode.CLOSED = 0` és l'únic límit admès. `ResourceDistributionCode` fixa
`UNIFORM = 0` i `PATCHES = 1`. Els valors són contractes host estables; no existeix cap codi
toroidal ni cap distribució aleatòria blanca.

## Codi de col·locació fundadora

`InitialPlacementCode.UNIFORM_WITH_REPLACEMENT = 0` identifica l'únic algoritme inicial. En
l'esquema host 1.3 conserva l'spelling establert `placement: random`; no hi ha aliases ni modes
alternatius. El recompte `INITIAL_PLACEMENT_COUNT` val 1.

## Sortides de la política

La [PolicyMLP v1](policy_mlp_schema_v1.md) produeix set scores lineals en l'ordre numèric estable d'`ActionCode`; no selecciona ni valida cap acció.
La [selecció determinista v1](policy_inference_and_action_selection_v1.md) converteix aquests
scores en propostes i reserva la validació de legalitat per al PR-16.

El [contracte d'accions i validació local v1](action_contract_and_validation_v1.md) afegeix sis
`ActionValidationCode` estables, conserva la proposta original i encamina qualsevol rebuig a
`STAY`; aquests codis no són causes de mort ni indiquen execució final.
