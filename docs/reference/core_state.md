# Estat del nucli

`SimulationState` és el contracte estructural de l'estat dinàmic del nucli. És un PyTree
d'Equinox immutable per convenció, format exclusivament per arrays JAX i sense configuració,
efectes laterals ni lògica d'inicialització.

```text
SimulationState
├── step
├── rng: RngState
│   └── key
├── ids: IdCounters
│   ├── next_agent_id
│   ├── next_genome_id
│   └── next_lineage_id
├── world: WorldState
│   ├── resources
│   ├── environment
│   └── occupancy
└── population: PopulationState
    ├── alive
    ├── agent_id, parent_id, lineage_id, genome_id
    ├── generation
    ├── position
    ├── energy
    ├── birth_step
    └── age
```

## Dimensions i camps

Per a una execució, `C = max_agents`, `H = world_height` i `W = world_width`. Aquestes dimensions
són fixes i no es desen duplicades a l'estat.

| Subestat | Camp | Forma | Dtype | Semàntica |
|---|---|---:|---|---|
| `SimulationState` | `step` | `()` | `STEP_DTYPE` (`int32`) | Pas temporal actual. |
| `RngState` | `key` | `()` | dtype PRNG tipat | Única clau arrel persistent. |
| `IdCounters` | `next_agent_id` | `()` | `ID_DTYPE` (`int32`) | Següent ID d’agent. |
| `IdCounters` | `next_genome_id` | `()` | `ID_DTYPE` (`int32`) | Següent ID de genoma. |
| `IdCounters` | `next_lineage_id` | `()` | `ID_DTYPE` (`int32`) | Següent ID de llinatge. |
| `WorldState` | `resources` | `[H, W]` | `REAL_DTYPE` (`float32`) | Recurs actual per cel·la. |
| `WorldState` | `environment` | `[H, W]` | `REAL_DTYPE` (`float32`) | Valor ambiental local. |
| `WorldState` | `occupancy` | `[H, W]` | `COUNT_DTYPE` (`int32`) | Recompte derivat d'agents vius per cel·la; pot superar un. |
| `PopulationState` | `alive` | `[C]` | `MASK_DTYPE` (`bool`) | Única font de veritat sobre l'activitat del slot. |
| `PopulationState` | `agent_id` | `[C]` | `ID_DTYPE` (`int32`) | Identitat estable de l'agent. |
| `PopulationState` | `parent_id` | `[C]` | `ID_DTYPE` (`int32`) | Identitat del progenitor. |
| `PopulationState` | `lineage_id` | `[C]` | `ID_DTYPE` (`int32`) | Identitat del llinatge. |
| `PopulationState` | `genome_id` | `[C]` | `ID_DTYPE` (`int32`) | Referència lògica al genoma. |
| `PopulationState` | `generation` | `[C]` | `COUNT_DTYPE` (`int32`) | Profunditat reproductiva. |
| `PopulationState` | `position` | `[C, 2]` | `INDEX_DTYPE` (`int32`) | Coordenades discretes en ordre `[x, y]`. |
| `PopulationState` | `energy` | `[C]` | `REAL_DTYPE` (`float32`) | Reserva energètica actual. |
| `PopulationState` | `birth_step` | `[C]` | `STEP_DTYPE` (`int32`) | Pas de naixement. |
| `PopulationState` | `age` | `[C]` | `COUNT_DTYPE` (`int32`) | Edat en passos. |

En accedir en el futur a un mapa `[H, W]`, una posició `[x, y]` s'indexarà com `[y, x]`.
`occupancy` és un recompte derivat de `population.position` i `population.alive`. El PR-10 el
construeix mitjançant el component general de `core.world` del PR-11. És l'únic mapa poblacional
persistent: les densitats global i local es deriven sota demanda i no formen part de l'estat.

## Capacitat fixa i slots

`PopulationState` és una estructura d'arrays, no una col·lecció d'objectes agent. Tots els camps
poblacionals mantenen el primer eix `C`, independentment del nombre d'agents vius. Els naixements
i les morts futurs canviaran màscares i valors, però no afegiran, eliminaran ni compactaran files.

`alive[index]` determina si un slot participa en la simulació. Quan és `False`, el slot continua
existint i la resta dels seus camps conserva forma i dtype, però els valors no tenen significat
biològic: poden ser neutres o residuals, no formen un historial i només es poden interpretar sota
la màscara. `NULL_ID = -1` és el sentinel explícit per a referències absents o slots inactius; els camps només s’interpreten sota la màscara `alive`.

L'índex del slot és una ubicació reutilitzable dins els arrays; no és `agent_id`. La identitat d'un
agent és estable encara que els slots es reutilitzin. Confondre ambdós conceptes introduiria errors
genealògics i dependència de l'ordre físic dels arrays.

La capacitat fixa estabilitza formes i compilació, però una `C` massa baixa pot rebutjar naixements
per una limitació tècnica, mentre que una `C` excessiva consumeix memòria i temps de compilació.

## Ampliacions deliberadament absents

El PR-10 inicialitza slots i fundadors, però no genomes neuronals. El PR-14 incorpora genomes
batched i el PR-25, acumuladors de mètriques i buffers d'esdeveniments.

## Bloc genòmic (PR-14)

`SimulationState.genomes` és el `GenomeBatch` de capacitat fixa associat slot a slot amb
`PopulationState.genome_id`. Consulteu [l'esquema batched v1](genome_batch_schema_v1.md).
