# Registre relacional PostgreSQL

## Responsabilitat

PostgreSQL és l'índex consultable d'identitats, relacions i metadades. El `run_id` és el mateix
UUID creat pel registre local del PR-35; no existeix cap traducció d'identitats. La connexió es
configura exclusivament amb `EVOLUCIO_DATABASE_URL` (format
`postgresql+psycopg://...`) i mai no s'escriu als manifests.

```text
experiment 1 ── N run
                  ├── 1 config_snapshot
                  ├── 0..1 run_summary
                  ├── N artifact
                  ├── N dataset_index
                  └── N state_capture
```

`experiment` agrupa runs comparables. `run` conserva llavor, mode, hash de configuració,
versions, horitzó i lifecycle. `config_snapshot` conserva una còpia JSONB immutable de la
configuració normalitzada i la referència/hash del `config.json` portable. Les claus foranes de
les taules filles tenen `ON DELETE CASCADE`; experiment → run usa `RESTRICT`.

## Bytes i índexs

Les sèries de mètriques, naixements, morts i genealogia continuen en **Parquet**. Els
checkpoints i snapshots complets continuen en **Zarr**. PostgreSQL només registra els índexs,
paths relatius, schemas, intervals de passos, mides i hashes; no desa blobs, pesos de genomes,
arrays del món ni files per agent/pas. El manifest local continua sent la font portable i els
artefactes locals són la font primària dels bytes.

## Transaccions i lifecycle

`create_run` insereix el run i el snapshot de configuració en una sola transacció: una fallada
del snapshot desfà també el run. `mark_run_completed` sincronitza els índexs verificats, insereix el resum immutable i
tanca el lifecycle en una transacció. `mark_run_failed` conserva error i timestamp sense exigir resum.
Els runs terminals no es poden reobrir i no s'exposa cap operació normal de supressió.

`sync_run_artifacts` exigeix primer un `verify_local_run` correcte. Sincronitza artefactes,
índexs Parquet i captures Zarr de forma idempotent i conserva els UUIDs de les files existents. Un mateix path
amb hash diferent és un conflicte, no un update. `reconcile_run` és només lectura: informa
absències, hashes i metadata/status divergents, sense decidir guanyador ni reparar contingut.
No hi ha transacció distribuïda filesystem/SQL; si falla SQL, els bytes locals es conserven i el
run centralitzat no s'ha de declarar complet.

## Migracions i operació

La revisió inicial és `0001_initial_experiment_registry`. Producció usa Alembic, no
`Base.metadata.create_all()`:

```console
evolucio db status
evolucio db upgrade
evolucio experiment create --name "Mutació baixa vs alta" --objective "Comparació"
evolucio experiment list
evolucio run config.json --steps 128 --experiment-id <uuid>
```

Sense `--experiment-id`, el run continua sent estrictament local. Amb l'opció, la CLI valida
l'experiment abans de crear el directori, registra el mateix `run_id`, sincronitza els índexs i
manté els lifecycle local i SQL. Una fallada SQL obligatòria es propaga i mai no produeix un
fals estat `completed` a PostgreSQL.

`db status` fa `SELECT 1` i compara la revisió actual amb `head`. `chunk_steps`, política de
persistència i credencials són infraestructura host i no formen part de `CoreConfig`.

## Batches i consultes

```console
evolucio batch run --config baseline=baseline.json --config variant=variant.json \
  --seeds 1,2,3 --steps 128 --chunk-steps 32 --experiment-id <uuid>
```

El batch registra l'experiment amb un únic pool de connexions i conserva `case_id`,
`batch_ordinal` i la llavor de cada run. La façana exposa les consultes següents.

```python
registry.list_runs(experiment_id=experiment_id)
registry.list_runs(config_hash=config_hash, seed=seed)
registry.list_runs_for_comparison(experiment_id)
registry.list_artifacts(run_id)
registry.list_datasets(run_id)
registry.list_state_captures(run_id)
```

`list_runs_for_comparison` retorna les metadades i el resum disponible de cada run.
Les captures inclouen `capture_kind` i `state_step` per identificar checkpoints i snapshots.

El manifest es publica com a completat després del commit PostgreSQL. Si la publicació
local falla després del commit, la reconciliació detecta la divergència d'estat. Si falla
el commit, el runtime conserva els fitxers, inclòs el resum local si ja s'havia escrit,
i intenta registrar la fallada en tots dos registres. Aquest intent inclou els artefactes parcials que superen la verificació.
