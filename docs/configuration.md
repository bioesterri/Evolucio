# Configuració d'experiments

La configuració host descriu i valida els paràmetres científics abans de qualsevol simulació. L'esquema **1.5** conté els blocs `world`, `population`, `policy`, `observations`, `energy`, `evolution`, `runtime` i `persistence` i `genome`, a més de la llavor explícita.

## Versions i immutabilitat

`schema_version` versiona aquest contracte i és independent de la versió del paquet i de la futura `model_version`. `observations.schema_version` i `action_schema_version` evolucionen independentment. Un canvi incompatible incrementa la versió major; un camp compatible o una semàntica ampliada incrementa la menor. Tot canvi exigeix proves i actualitzar el JSON Schema. Mai no es reinterpreta retroactivament una versió publicada ni es migra silenciosament.

Els models Pydantic són estrictes, rebutgen camps desconeguts i queden immutables. No hi ha variables d'entorn, herència ni fallbacks. `freeze_config` inclou defaults i nuls en JSON UTF-8 compacte amb claus ordenades; `config_hash` és el SHA-256 hexadecimal d'aquests bytes canònics.

## Blocs

- `world`: dimensions, límits, recursos i fases ambientals ordenades i no solapades.
- `population`: capacitats, col·locació i política d'ocupació de cel·les.
- `policy`: versions d'observació/acció i topologia fixa declarada.
- L'espai d'accions del prototip és fix: `ACTION_COUNT` deriva dels set codis públics del nucli i
  no és un paràmetre de la configuració host.
- `energy`: reserves, costos i viabilitat reproductiva. `reproduction_cost` és el cost addicional i `offspring_initial_energy` es transfereix al descendent; ambdós es resten al progenitor. El PR-21 revalidarà la viabilitat efectiva.
- `evolution`: edats i paràmetres explícits de mutació.
- `runtime`: passos, chunk, mostreig i backend host.
- `persistence` i `genome`: nivell, destins i lots host-only, sense comprovar connexions.

## Formats i exemple

S'admeten YAML (`.yaml`, `.yml`) i JSON (`.json`) UTF-8, amb claus úniques. Exemple complet de validació estructural (els valors **no estan calibrats científicament**):

```yaml
schema_version: "1.5"
seed: 42
world: {width: 64, height: 64, boundary_mode: closed, resource_capacity: 10.0, initial_resource_mean: 5.0, resource_distribution: patches, resource_patch_count: 8, resource_patch_radius: 5.0, resource_patch_contrast: 0.8, environment_initial_value: 0.0, regeneration_rate: 0.05, environment_schedule: []}
population: {initial_agents: 128, max_agents: 1024, max_births_per_step: 64, placement: random, allow_multiple_agents_per_cell: true}
policy: {action_schema_version: "1.0", schema_version: 1, input_size: 15, hidden_size: 16, output_size: 7, activation: tanh, use_bias: true}
observations: {schema_version: 1, perception_radius: 2}
energy: {initial_energy: 20.0, max_energy: 100.0, death_threshold: 0.0, basal_cost: 0.1, movement_cost: 0.05, feeding_cost: 0.0, feeding_conversion: 1.0, reproduction_threshold: 40.0, reproduction_cost: 5.0, offspring_initial_energy: 10.0, failed_action_cost: 0.0}
evolution: {min_reproduction_age: 5, max_age: 1000, mutation_rate: 0.05, mutation_sigma: 0.02, mutation_clip_abs: 5.0}
runtime: {steps: 10000, chunk_size: 128, record_stride: 10, snapshot_stride: 1000, backend: cpu}
persistence: {level: none, destinations: [], output_dir: runs, batch_size: 1024, checkpoint_stride: null}
```

Errors habituals: versions desconegudes, nombres expressats com strings, claus duplicades, fases solapades, capacitat espacial insuficient o reproducció energèticament inviable. `CoreConfig` i la transformació cap al nucli corresponen al PR-04.

Des del PR-10, `population.initial_agents` admet zero i no s'exigeix que càpiga en el nombre de
cel·les quan `allow_multiple_agents_per_cell` és cert. `placement: random` significa col·locació
uniforme amb reemplaçament. El producte `world.width * world.height` no pot superar el màxim
`int32`, necessari per a l'índex lineal d'ocupació.

## Frontera host-core

`ExperimentConfig` és el contracte host complet i `config_hash` n'identifica tots els valors
canònics. `compile_config` en crea una projecció `CoreConfig` PyTree: els paràmetres numèrics
variables són arrays JAX amb dtype explícit, mentre que les formes i els selectors de control són
primitives Python. Les dades operatives exclusives del host no entren al PyTree.

`CompileSignature` és una allowlist immutable dels camps estàtics. El seu digest SHA-256 canònic
identifica de manera persistent una classe d'executable; a diferència de `config_hash`, no canvia
per costos, taxes, llavor, passos totals o persistència. Per tant, dos runs amb la mateixa
signatura poden reutilitzar en el futur un executable, encara que els seus valors dinàmics siguin
diferents.

| Camps reals | Categoria | Representació compilada | Motiu |
|---|---|---|---|
| `schema_version` | estàtic | `str` a `CompileSignature` | Versiona el contracte interpretat. |
| `world.width`, `world.height`, `boundary_mode`, `resource_distribution`, `resource_patch_count` | estàtic | primitives Python | Defineixen formes o selecció de l'algoritme del món. |
| `world.environment_schedule` (longitud) | estàtic | `int` a `CompileSignature` | Determina la forma dels vectors ambientals. |
| `world.resource_capacity`, `initial_resource_mean`, `resource_patch_radius`, `resource_patch_contrast`, `environment_initial_value`, `regeneration_rate` | dinàmic | escalars `float32` | Canvien valors, no formes. |
| valors de `environment_schedule` | dinàmic | vectors `int32`/`float32` | La longitud és fixa, però els valors poden variar. |
| `population.max_agents`, `max_births_per_step`, `placement`, `allow_multiple_agents_per_cell` | estàtic | primitives Python | Defineixen capacitat, buffers o control compilat. |
| `population.initial_agents` | dinàmic | escalar `int32` | Ocupació inicial dins una capacitat fixa. |
| versions, topologia, activació i radi de `policy` | estàtic | primitives Python | Defineixen esquema, topologia o observació. |
| tots els camps d'`energy` | dinàmic | escalars `float32` | Són costos i llindars sense efecte sobre formes. |
| edats d'`evolution` | dinàmic | escalars `int32` | Són comptadors i llindars. |
| mutació d'`evolution` | dinàmic | escalars `float32` | Són taxes, sigma i clipping numèric. |
| `runtime.chunk_size`, `record_stride`, `snapshot_stride`, `backend` | estàtic | primitives a la signatura; controls de sortida al bloc runtime quan escau | Determinen l'executable, els buffers o la política de compilació. |
| implementació PRNG `threefry2x32` | estàtic | `str` a `CompileSignature` | Afecta el dtype de clau i potencialment l’executable. |
| `seed`, `runtime.steps` i tot `persistence` i `genome` | només host | exclosos | La seed identifica el run, però no formes ni topologia; orquestració i I/O són responsabilitats host. |


La versió 8 de `CompileSignature` afegeix versió i digest del contracte de validació local; la
versió 7 afegeix la selecció determinista i el recompte d'accions, la versió 6 el genoma i la
versió 5 l'esquema complet de PolicyMLP. Aquest canvi versiona el contracte serialitzat de
compilació; la seed continua exclosa perquè canvia la trajectòria del run, no la classe
d'executable.

## Calendari ambiental del PR-09

La longitud d'`environment_schedule` ja forma part de `CompileSignature`. Les dates, els multiplicadors, `stress_level` i `regeneration_rate` són dinàmics: canvien `config_hash`, però amb la mateixa longitud no canvien la signatura. El calendari compilat usa intervals semioberts `[start_step, end_step)` i vectors `int32`/`float32`.

## Observacions locals i política a la signatura v5

El bloc `observations` fixa `schema_version: 1` i valida `perception_radius` com enter estricte entre 1 i 3. La `CompileSignature` v4 incorpora versió, mida 15, digest canònic i radi. Les escales d’energia, edat i recursos són dinàmiques i no alteren la signatura. Vegeu [Esquema d’observacions locals v1](reference/local_observation_schema_v1.md).

## Política neuronal fixa

El bloc `policy` de l’esquema host 1.5 valida exclusivament la versió 1, 15 entrades, 16 unitats ocultes, 7 sortides, `tanh` i biaixos. `PolicyCoreConfig` conserva aquestes primitives com a camps estàtics i `CompileSignature` v5 incorpora també el digest de [PolicyMLP v1](reference/policy_mlp_schema_v1.md). Ni pesos, llavor ni paràmetres d’inicialització o mutació formen part de la signatura.

## Genoma i signatura v6

`genome` accepta només `schema_version: 1` i `initialization:
glorot_uniform_zero_bias_v1`. `GenomeCoreConfig` conserva versió, digest, inicialitzador i
recompte 375 com a primitives estàtiques. `CompileSignature` v6 inclou aquests camps; no inclou
llavor, `initial_agents`, IDs ni valors dels pesos. El bloc sí forma part de `config_hash`.

## Selecció determinista i signatura v7

La selecció no és configurable. `PolicyCoreConfig` i `CompileSignature` incorporen la versió 1 i
el digest canònic de [la selecció d'accions](reference/policy_inference_and_action_selection_v1.md),
i la signatura registra set accions. Scores, propostes, `alive`, genomes, observacions, seed i
root key són dades d'execució deliberadament excloses.

## Validació local d'accions i signatura v8

La `CompileSignature` incorpora la versió 1 i el digest canònic del
[contracte d'accions](reference/action_contract_and_validation_v1.md). Propostes, accions
encaminades, causes concretes, posicions, recursos, `alive`, costos i RNG continuen exclosos.
