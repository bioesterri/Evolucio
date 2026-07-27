# Estat del nucli

`SimulationState` és el contracte estructural de l'estat dinàmic del nucli. És un PyTree
d'Equinox immutable per convenció, format exclusivament per arrays JAX i sense configuració,
efectes laterals ni lògica d'inicialització.

```text
SimulationState
├── step
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
`occupancy` és un recompte derivat de `population.position` i `population.alive`; aquest contracte
només l'emmagatzema i encara no el calcula.

## Capacitat fixa i slots

`PopulationState` és una estructura d'arrays, no una col·lecció d'objectes agent. Tots els camps
poblacionals mantenen el primer eix `C`, independentment del nombre d'agents vius. Els naixements
i les morts futurs canviaran màscares i valors, però no afegiran, eliminaran ni compactaran files.

`alive[index]` determina si un slot participa en la simulació. Quan és `False`, el slot continua
existint i la resta dels seus camps conserva forma i dtype, però els valors no tenen significat
biològic: poden ser neutres o residuals, no formen un historial i només es poden interpretar sota
la màscara. No s'estableix encara cap sentinel per als slots inactius.

L'índex del slot és una ubicació reutilitzable dins els arrays; no és `agent_id`. La identitat d'un
agent és estable encara que els slots es reutilitzin. Confondre ambdós conceptes introduiria errors
genealògics i dependència de l'ordre físic dels arrays.

La capacitat fixa estabilitza formes i compilació, però una `C` massa baixa pot rebutjar naixements
per una limitació tècnica, mentre que una `C` excessiva consumeix memòria i temps de compilació.

## Ampliacions deliberadament absents

Aquesta primera versió no inclou placeholders. El PR-07 afegirà RNG i comptadors d'identificadors;
el PR-14, genomes batched; i el PR-25, acumuladors de mètriques i buffers d'esdeveniments. Els PR-08
i PR-10 seran responsables d'inicialitzar el món i la població, respectivament.
