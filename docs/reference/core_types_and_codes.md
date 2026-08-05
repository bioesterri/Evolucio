# Tipus, dtypes i codis del nucli

Aquest contracte centralitza el vocabulari estable dels futurs arrays del nucli. No implementa
estat, comportament ni lògica de simulació.

## Política de dtypes

| Ús | Dtype |
|---|---|
| Valors reals | `float32` |
| Índexs, identificadors, comptadors i passos | `int32` |
| Codis categòrics d’acció, mort i flux RNG | `int32` |
| Màscares | `bool` |

`float32` és el format real portàtil i eficient del prototip, sense activar globalment JAX x64.
`int32` fixa una representació homogènia per a índexs, identificadors, comptadors, passos i codis categòrics que poden participar en selecció d’accions, indexació i contractes de compilació; i `bool` representa les màscares sense codificacions numèriques alternatives. Les conversions dels consumidors han de ser explícites i usar les constants de `evolucio.core`.

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

## Codis de flux RNG

| Nom | Valor | Significat |
|---|---:|---|
| `WORLD_INITIALIZATION` | 0 | Flux per inicialitzar l’estat del món. |
| `RESOURCE_INITIALIZATION` | 1 | Flux per inicialitzar recursos. |
| `AGENT_INITIALIZATION` | 2 | Flux per inicialitzar agents o ocupació inicial. |
| `GENOME_INITIALIZATION` | 3 | Flux per inicialitzar genomes. |
| `ENVIRONMENT_UPDATE` | 4 | Flux per actualitzacions ambientals reproduïbles. |
| `ACTION_TIE_BREAK` | 5 | Flux per desempats de selecció d’accions. |
| `MOVEMENT_CONFLICT` | 6 | Flux per conflictes de moviment. |
| `RESOURCE_CONFLICT` | 7 | Flux per conflictes de recursos. |
| `REPRODUCTION_CONFLICT` | 8 | Flux per conflictes reproductius. |
| `BIRTH_PLACEMENT` | 9 | Flux per col·locació de naixements. |
| `GENOME_MUTATION` | 10 | Flux per mutació genòmica. |

`RNG_STREAM_COUNT` val 11. Aquests codis només identifiquen streams de manera estable; no creen claus, no divideixen PRNGs i no implementen cap política d’aleatorietat en aquest PR.

## Representació i estabilitat

Els `IntEnum` són representacions host del contracte. Els arrays JAX guarden els valors enters
amb `CODE_DTYPE`, mai strings ni objectes `Enum`. Els noms, l'ordre i els valors numèrics són API
estable: canviar-los requeriria versionar i migrar qualsevol dada persistida que els contingui.

Aquest PR no resol accions, moviment, alimentació, reproducció o mortalitat, ni assigna causes.
