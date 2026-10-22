# Persistència Parquet de telemetria (PR-36)

Els runs registrats, inclosos els casos i seeds del batch seqüencial PR-34, escriuen
les dades host del PR-31 mitjançant un consumer síncron. El core i el runtime compilat
no coneixen Parquet. Cada run té el seu writer i índex; només s'admet un writer per
directori, sense concurrència ni resume.

## Captura i esquemes v1

`ParquetPersistencePolicy` és una política host immutable. No entra a `CoreConfig`,
`config_hash` ni `CompileSignature`. Queda registrada íntegrament a `datasets.json`.

| Dataset | Clau temporal | Identitat dins del run | Captura |
|---|---|---|---|
| metrics | step | step | `step % metrics_interval == 0` |
| births | birth_step | child_agent_id | sempre |
| deaths | death_step | agent_id | sempre |
| genealogy | birth_step | child_agent_id | sempre, una aresta per birth |

Totes les files contenen `run_id: String`. Els IDs, coordenades, passos i comptadors
són `Int64`; els codis de mort/fase són `Int32`; els valors reals són `Float32` per
preservar la precisió del DTO original. Els esquemes explícits i el seu ordre es
defineixen a `persistence/parquet/schemas.py` i es congelen amb proves. El contracte
és `run_telemetry_parquet_v1`, versió 1. Un canvi incompatible exigeix nova versió.
Els readers rebutgen versions desconegudes, sense migració automàtica.

Metrics conserva tots els camps de `StepMetricsDTO`, amb `deaths_by_cause` expandit
en aquest ordre: `deaths_none`, `deaths_energy_depletion`, `deaths_max_age`,
`deaths_environmental_stress`, `deaths_competitive_exclusion`, `deaths_invalid_state`.
La correspondència v1 és exactament la dels codis 0–5 de `DeathCauseCode`.
Deaths conserva els codis estables de causa i fase (preacció=1, postacció=2).
Births conserva identitats, posició, energies i tots els mutation summaries del DTO.
No s'escriuen slots, màscares ni files sentinella dels buffers.

`metrics_interval` és un enter positiu, no booleà. El mostreig usa el pas absolut,
independent del chunking, sense excepcions per al primer o últim pas.
**`metrics_interval` no pot afectar births, deaths ni genealogy.** Desactivar aquests
tres datasets és un error de política científica.

Genealogy és una projecció incremental de cada birth: parent → child, birth_step,
lineage, generation, genome IDs i mutation selected/effective count, sum/max delta.
No es reconstrueix des de `SimulationState` i no s'inventen arestes de fundadors.
La verificació comprova una sola aresta per fill i la igualtat de la projecció amb
births, incloent-hi els mutation summaries compartits.

## Layout i atomicitat

```text
runs/<run_id>/
├── manifest.json
├── config.json
├── summary.json
└── artifacts/parquet/
    ├── datasets.json
    ├── metrics/part-00000000.parquet
    ├── births/part-00000000.parquet
    ├── deaths/part-00000000.parquet
    └── genealogy/part-00000000.parquet
```

El número de part és el `chunk_index` del DTO, amb un mínim de vuit dígits.
No es creen parts buides. Cada part és immutable: un nom existent produeix error;
no hi ha reprocessament destructiu ni append sobre un únic fitxer.

Polars construeix columnes amb esquema explícit per chunk i escriu Parquet natiu
amb compressió `zstd` per defecte, sense pandas ni PyArrow. Primer s'escriu un fitxer
temporal al mateix directori, es tanca i sincronitza i s'aplica `os.replace`.
Es netegen temporals quan és possible. Les parts anteriors no es reescriuen.
El consumer allibera els DTOs i DataFrames de cada chunk; conserva només metadades
proporcionals al nombre de parts, no totes les files del run.

L'índex també es reemplaça atòmicament, després de totes les parts del chunk.
Conté política, complete/partial, schema version, directoris relatius, nombres de
files i parts, rangs temporals i, per part, chunk index, mida i SHA-256 calculat
amb el helper streaming de PR-35 després de tancar el fitxer definitiu.

## Finalització i runs parcials

Abans de marcar el run completed, `finalize()` verifica hashes, esquemes, counts,
mostreig, identitats úniques i correspondència births/genealogy. Les consultes lazy
comproven també que un agent nascut durant el run no mor abans de néixer. Després
es registra només `datasets.json` al manifest PR-35 com `parquet_dataset_index`;
el resum conté counts compactes i la referència a l'índex.

Una fallada atura l'execució, finalitza l'índex amb `complete=false` i marca el run
failed. Les parts dels chunks confirmats continuen llegibles. Si la fallada deixa
parts finals fora de l'índex (interrupció entre replace i commit), es reporten com
orfes; no s'adopten ni s'eliminen silenciosament. Una fallada durant la pròpia
finalització parcial es conserva com a nota de l'excepció original.

## Lectura i verificació

```python
from evolucio.persistence.parquet import (
    scan_metrics, scan_births, scan_deaths, scan_genealogy,
)

metrics = scan_metrics(run_dir)  # pl.LazyFrame; no collect automàtic
births = scan_births(run_dir)
deaths = scan_deaths(run_dir)
genealogy = scan_genealogy(run_dir)
```

Els readers validen versions i paths, rebutgen escapes i symlinks i només escanegen
parts registrades a l'índex. Un dataset sense parts retorna un LazyFrame buit amb
l'esquema correcte. Les comprovacions d'integritat es fan explícitament amb
`verify_parquet_datasets()` o `verify_local_run()`; llegir no recalcula tots els hashes.

La CLI existent `evolucio artifacts verify <run_dir>` delega a la verificació Parquet
i retorna un codi diferent de zero si hi ha danys. `artifacts inspect` mostra counts,
parts, interval i complete/partial sense imprimir taules completes.
La verificació és de només lectura i detecta parts absents, SHA-256 diferents,
esquemes incorrectes, counts incoherents, duplicats, arestes incoherents i orfes.
Un run parcial coherent pot verificar correctament; la seva incompletesa es manté
explícita a l'índex i el manifest failed.

## Límits

No hi ha PostgreSQL, SQL, MLflow, Zarr, snapshots, trajectòries, observacions/accions,
pesos de genomes, checkpoints, resume, object storage, writers paral·lels, compaction
ni API d'estadística científica. La genealogia Parquet és la còpia analítica; una
capa relacional futura podrà indexar-ne referències.
