# Checkpoints i snapshots Zarr (PR-37)

La persistència local utilitza **Zarr 3**, amb un store independent per captura.
Un checkpoint conserva el carry necessari per reprendre el futur; un snapshot és
un DTO host analític. Els dos esquemes són independents i no són intercanviables.

| Propietat | Snapshot | Checkpoint |
|---|---|---|
| Analític/visual | sí | secundari |
| Reprenable | no | sí |
| Població | vius compactats, ordre agent_id | tota la capacitat i slots inactius |
| Genomes complets | no | sí |
| RNG | no | sí |
| MetricsAccumulator | no | sí |

No existeix `snapshot_to_simulation_state()`. `load_snapshot()` retorna
`SimulationSnapshotDTO`; `load_checkpoint()` retorna un `CheckpointRestore` amb
els PyTrees reals `SimulationState` i `MetricsAccumulator`.

## Layout i esquemes

```text
runs/<run_id>/artifacts/zarr/
├── index.json
├── checkpoints/
│   ├── step-000000010.zarr/
│   └── step-000000020.zarr/
└── snapshots/
    ├── step-000000005.zarr/
    ├── step-000000010.zarr/
    ├── step-000000015.zarr/
    └── step-000000020.zarr/
```

El nom identifica `SimulationState.step`, el següent pas que s'executaria:
`state.step == 10` significa que s'han completat 0..9. No s'utilitza l'últim
`metrics.step` com a identificador de captura. No hi ha store gegant append-only.

Checkpoint: `simulation_checkpoint_zarr_v1`, versió 1. Arrays explícits:

- `step`: escalar int32.
- `world/{resources,environment,occupancy}`: [H,W], float32/float32/int32.
- `population/alive`: [C], bool.
- `population/{agent_id,parent_id,lineage_id,genome_id,generation,birth_step,age}`:
  [C], int32.
- `population/position`: [C,2], int32; `population/energy`: [C], float32.
- `genomes/layer1_weight`: [C,16,15]; `layer1_bias`: [C,16];
  `layer2_weight`: [C,7,16]; `layer2_bias`: [C,7], tots float32.
- `rng/key_data`: [2], uint32, implementació `threefry2x32` a metadata.
- `ids/{next_agent_id,next_genome_id,next_lineage_id}`: escalars int32.
- `accumulator/{steps_accumulated,births_total,birth_rejected_capacity_total,
  deaths_total,movement_success_total,feeding_success_total,
  reproduction_success_total,alive_count_sum}`: escalars int32.
- `accumulator/deaths_by_cause_total`: [6], int32.
- `accumulator/{resource_consumed_total,feeding_energy_gained_total,
  basal_energy_cost_total,movement_energy_cost_total,feeding_energy_cost_total,
  reproduction_energy_cost_total,death_energy_removed_total,death_age_sum,
  energy_alive_sum}`: escalars float32.
- `accumulator/count_overflow`: escalar bool.

Es conserven els valors signats legítims, inclosa l'energia retirada per morts.
Tots els slots, també els inactius canònics i els seus genomes zero, es persisteixen.
El guard de schema rebutja un carry amb camps nous no versionats; no omet fulles.

Snapshot: `analysis_snapshot_zarr_v1`, versió 1, independent de checkpoint v1.
Reutilitza el DTO PR-31, sense una segona compactació:

- `population/{slot_index,agent_id,parent_id,lineage_id,genome_id,generation,
  position,energy,birth_step,age}`: només vius, [N] o [N,2].
- `world/{resources,environment,occupancy}`: món complet [H,W].
- `state_step`, comptadors d'IDs, dimensions, capacity i alive_count: metadata.

El snapshot preserva els dtypes NumPy del DTO; la metadata enumera el dtype de
cada array i el reader valida tipus numèric, forma i contingut. No conté `rng/`,
`genomes/`, `accumulator/`, pesos neuronals ni història d'events. En extinció N=0
és vàlid; el món es manté complet. No es crea cap agent fictici.

## Política i scheduler

`ZarrCapturePolicy` és host, immutable i separada de `CoreConfig`, `config_hash`
i `CompileSignature`. Camps:

- `checkpoint_interval`, `snapshot_interval`: None o enter positiu, mai bool.
- `checkpoint_on_completion`, `snapshot_on_completion`: bool, per defecte false.
- `keep_last_checkpoints`: None o enter positiu; None conserva tots.

La captura periòdica compleix `state.step % interval == 0`, després d'executar
pasos; no es crea una captura inicial implícita al pas 0.
El planner executa `min(chunk_steps, remaining, distància al checkpoint,
distància al snapshot)`. Amb chunk=256 i interval=1000 produeix
256+256+256+232, arribant exactament a 1000. La unió dels límits dels dos tipus
es respecta també en continuacions amb un pas inicial diferent de zero.

El runtime reutilitza els modes existents per segments i reindexa la telemetria
amb un chunk_index global del nou run. No modifica kernels ni regles científiques.
Cada boundary transfereix explícitament el carry complet per al checkpoint o
utilitza `simulation_state_to_snapshot()` per a la captura analítica.
L'escriptura és síncrona, fora de JIT, amb un únic writer per run.

Una captura periòdica que coincideix amb el final es crea una sola vegada encara
que també estigui activada la captura final. Checkpoint i snapshot al mateix pas
són dos stores diferents. Sense intervals ni flags finals no es crea cap índex ni
store Zarr. `interval=1` és vàlid, però pot generar un cost elevat de transferència,
espai, sincronització i escriptura. No hi ha prohibicions arbitràries.

Els modes smoke/debug conserven els seus diagnòstics en cada segment; benchmark
reutilitza el seu warmup descartat i agrega temps dels segments. El temps de
captura queda fora del temps mesurat de simulació.

## Restore i continuació

```python
from evolucio.persistence.zarr import load_checkpoint, load_snapshot

restored = load_checkpoint(
    checkpoint_path=path,
    expected_config_hash=compiled.config_hash,
    expected_compile_signature=signature_digest,
)
# restored.state i restored.accumulator són carry de dispositiu reprenable.
# restored.warnings informa de diferències de package/Git.
snapshot = load_snapshot(snapshot_path)  # només DTO host
```

Abans de reconstruir el carry es validen bytes, schema, digests del core, formes,
dtypes, config_hash esperat i CompileSignature esperada. No hi ha `force=True`
ni migracions automàtiques. Després es reconstrueixen les classes reals, es
restaura la clau tipada amb `jax.random.wrap_key_data` i s'executen les invariants
estructurals PR-26. No es reparen ni es reinicialitzen arrays o acumuladors.

La metadata guarda run_id, state_step, config_hash, CompileSignature, digests de
simulation/metrics/invariants/genomes, implementació RNG, dimensions/capacitat,
resource_capacity, timestamp UTC, versió de paquet i commit Git. Git diferent
produeix un advertiment al caller, no un rebuig quan els contractes coincideixen.

La continuació explícita està disponible amb:

```python
continued = run_recorded_simulation(
    config=original_config,
    source_checkpoint=checkpoint_path,
    plan=RunPlan(total_steps=60, chunk_steps=7),
    output_root=runs_root,
)
```

Això crea **un run_id nou**, amb `initial_step` del checkpoint. El temps i els
steps Parquet continuen sent absoluts. `artifacts/checkpoint_source.json` registra
source_run_id, source_checkpoint_step i source_checkpoint_hash. Els acumuladors
i els totals del resum inclouen la història original; Parquet del nou run només
conté la seva continuació. Els snapshots no són necessaris per restaurar.

No hi ha resume transparent del mateix run, rollback de Parquet ni recuperació
transaccional completa de manifest, events i captures. Aquesta coordinació queda
per a PRs futurs; la font no es modifica durant una continuació.

## Atomicitat, hash i retenció

Cada captura s'escriu a `step-<step>.zarr.tmp-<token>`. Només després d'escriure i
validar arrays, metadata i integritat es fa rename al nom definitiu. Si el nom
final existeix es rebutja l'operació. Els temporals no s'indexen i es netegen quan
és possible. Es fan servir les APIs natives Zarr 3 i chunking conservador de fins
a 128 elements per dimensió, sense tuning de codecs.

`integrity.json` és el catàleg intern de paths relatius, mides i hashes de tots
els fitxers Zarr generats. No és un array ni una peça del carry. La verificació
rebutja fitxers catalogats absents/modificats, symlinks i fitxers Zarr addicionals
(metadata `zarr.json` o chunks sota `c/`). Fitxers aliens no catalogats, com
`notes.txt`, no formen part de la identitat. No s'adopten silenciosament chunks.

El tree SHA-256 comença amb els bytes `evolucio-zarr-tree-v1` seguits de NUL.
Per cada path catalogat, inclòs el mateix `integrity.json`, en ordre lexicogràfic:
longitud del path UTF-8 en 8 bytes big-endian, path UTF-8, mida en 8 bytes
big-endian i contingut en blocs de màxim 1 MiB. No depèn de rutes absolutes,
timestamps del filesystem ni permisos. La mida registrada suma aquests fitxers.
El reader comprova també la identitat externa de l'índex quan hi ha un registre
coincident, independentment de l'estat d'altres captures del run.

`index.json` es reemplaça atòmicament i conté la política, el context del run,
complete/partial i registres amb kind, state_step, relative_path, size, SHA-256,
schema_version i created_at_utc. Latest/list només consulten aquest índex.
Un store final no indexat és orfe; verify el reporta sense adoptar-lo ni reparar-lo.

Retenció: primer s'escriu i verifica el checkpoint nou, després s'indexa, i només
llavors s'eliminen els més antics i es publica l'índex reduït. Amb keep_last=2 i
captures 10/20/30 es conserven 20/30. No s'aplica retenció als snapshots. Una
interrupció durant l'eliminació pot deixar inconsistències reportables a l'índex,
però el checkpoint nou vàlid ja hi està registrat.

En fallada es conserven les captures anteriors i es finalitza l'índex com parcial.
El manifest PR-35 només registra `artifacts/zarr/index.json`, amb kind
`zarr_artifact_index`; no enumera chunks interns. `verify_local_run()` delega a
Parquet i Zarr. Els checkpoints no inclouen Parquet ni història d'events.

## CLI i verificació

```console
evolucio run config.json --steps 20 --chunk-steps 7 \
  --checkpoint-every 10 --snapshot-every 5 --mode experiment --output-root runs
evolucio checkpoint list runs/<run_id>
evolucio checkpoint verify runs/<run_id>/artifacts/zarr/checkpoints/step-000000010.zarr
evolucio snapshot list runs/<run_id>
evolucio artifacts verify runs/<run_id>
```

Flags opcionals: `--checkpoint-on-completion`, `--snapshot-on-completion`,
`--keep-last-checkpoints N`. Dry-run mostra intervals i steps esperats sense
crear stores, inicialitzar simulació ni compilar. List mostra metadata, mida i
prefix de hash sense carregar arrays. Verify no executa passos de simulació.

Les proves comparen 0→100 amb checkpoint/restore/continuació als passos 1, 17, 32
i 40, en un escenari amb moviment conflictiu, feeding, births, mutació i morts.
S'aplica la política PR-29: igualtat exacta discreta, RNG i IDs; tolerància float32
1e-6 per al run continuat i igualtat bit a bit del carry immediat després de restore.

## Límits

Filesystem local, un writer, sense PostgreSQL, MLflow, object storage, locks
concurrents, checkpoint distribuït, deltes, trajectories, observations/actions,
genome registry analític ni tuning avançat. Els genomes complets només es guarden
al checkpoint reprenable; no s'afegeixen a snapshots ni Parquet.
