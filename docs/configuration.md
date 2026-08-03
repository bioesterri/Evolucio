# Configuració d'experiments

La configuració host descriu i valida els paràmetres científics abans de qualsevol simulació. L'esquema **1.0** conté els blocs `world`, `population`, `policy`, `energy`, `evolution`, `runtime` i `persistence`, a més de la llavor explícita.

## Versions i immutabilitat

`schema_version` versiona aquest contracte i és independent de la versió del paquet i de la futura `model_version`. `observation_schema_version` i `action_schema_version` evolucionen independentment. Un canvi incompatible incrementa la versió major; un camp compatible o una semàntica ampliada incrementa la menor. Tot canvi exigeix proves i actualitzar el JSON Schema. Mai no es reinterpreta retroactivament una versió publicada ni es migra silenciosament.

Els models Pydantic són estrictes, rebutgen camps desconeguts i queden immutables. No hi ha variables d'entorn, herència ni fallbacks. `freeze_config` inclou defaults i nuls en JSON UTF-8 compacte amb claus ordenades; `config_hash` és el SHA-256 hexadecimal d'aquests bytes canònics.

## Blocs

- `world`: dimensions, límits, recursos i fases ambientals ordenades i no solapades.
- `population`: capacitats, col·locació i política d'ocupació de cel·les.
- `policy`: versions d'observació/acció i topologia fixa declarada.
- `energy`: reserves, costos i viabilitat reproductiva. `reproduction_cost` és el cost addicional i `offspring_initial_energy` es transfereix al descendent; ambdós es resten al progenitor. El PR-21 revalidarà la viabilitat efectiva.
- `evolution`: edats i paràmetres explícits de mutació.
- `runtime`: passos, chunk, mostreig i backend host.
- `persistence`: nivell, destins i lots host-only, sense comprovar connexions.

## Formats i exemple

S'admeten YAML (`.yaml`, `.yml`) i JSON (`.json`) UTF-8, amb claus úniques. Exemple complet de validació estructural (els valors **no estan calibrats científicament**):

```yaml
schema_version: "1.0"
seed: 42
world: {width: 64, height: 64, boundary_mode: closed, resource_capacity: 10.0, initial_resource_fraction: 0.5, resource_distribution: patches, regeneration_rate: 0.05, environment_schedule: []}
population: {initial_agents: 128, max_agents: 1024, max_births_per_step: 64, placement: random, allow_multiple_agents_per_cell: true}
policy: {observation_schema_version: "1.0", action_schema_version: "1.0", hidden_size: 16, activation: tanh, perception_radius: 2}
energy: {initial_energy: 20.0, max_energy: 100.0, death_threshold: 0.0, basal_cost: 0.1, movement_cost: 0.05, feeding_cost: 0.0, feeding_conversion: 1.0, reproduction_threshold: 40.0, reproduction_cost: 5.0, offspring_initial_energy: 10.0, failed_action_cost: 0.0}
evolution: {min_reproduction_age: 5, max_age: 1000, mutation_rate: 0.05, mutation_sigma: 0.02, mutation_clip_abs: 5.0}
runtime: {steps: 10000, chunk_size: 128, record_stride: 10, snapshot_stride: 1000, backend: cpu}
persistence: {level: none, destinations: [], output_dir: runs, batch_size: 1024, checkpoint_stride: null}
```

Errors habituals: versions desconegudes, nombres expressats com strings, claus duplicades, fases solapades, capacitat espacial insuficient o reproducció energèticament inviable. `CoreConfig` i la transformació cap al nucli corresponen al PR-04.
