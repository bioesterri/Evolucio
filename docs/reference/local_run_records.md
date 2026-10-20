# Registres locals de runs v1

El registre local identifica i conserva el context d’una execució sense afegir I/O al core.
La capa `persistence/local` registra; `runtime/recorded_run.py` orquestra la inicialització
nova i `run_with_mode`. No hi ha cap modificació a `simulation_step` ni a `run_chunk`.

## Directori i identitat

L’arrel per defecte és la constant host `DEFAULT_RUNS_ROOT = Path("runs")`, relativa al
directori des d’on s’invoca la CLI. `--output-root PATH` la substitueix explícitament.
No s’utilitza cap home ocult ni directori temporal implícit.

```text
runs/<UUID>/
├── manifest.json
├── config.json
├── summary.json       # només si hi ha resum final
└── artifacts/         # artefactes explícits; buit per defecte
```

Si falla la inicialització o l’execució, es conserva `config.json` i s’escriu `error.json`.
No es crea `summary.json` per una fallada de simulació sense resultat final.

El `run_id` és un UUID canònic generat amb `uuid.uuid4()`, independent de la seed, el hash,
els timestamps i l’ordinal. Dues execucions idèntiques tenen UUIDs diferents. Es pot injectar
`run_id` a `create_local_run` o `run_recorded_simulation`; un directori existent causa
`FileExistsError` i no es reutilitza, buida ni sobreescriu. La verificació exigeix que el nom
del directori coincideixi amb el UUID. Es pot moure tot el directori preservant aquest nom.

## Lifecycle i manifest

`RUN_MANIFEST_SCHEMA_NAME = "local_run_manifest_v1"`, versió 1, és un contracte host de
persistència. No entra a `config_hash` ni a `CompileSignature`.

Transicions permeses:

```text
created → running → completed
   │         └───→ failed
   └─────────────→ failed
```

Un estat terminal no es reobre ni passa de completed a failed. L’API no simula suport
per stopped o discarded. Els timestamps són ISO 8601 UTC amb sufix `Z` i microsegons.

`RunManifest` és un DTO immutable. Registra UUID, estat, timestamps, seed, hash de configuració,
referència a `config.json`, pla (total/chunk), mode, passos inicial/final i executats,
`compile_signature`, artefactes i error opcional. `code_version` agrupa versió canònica del
paquet, commit i dirty Git; `environment` agrupa Python, JAX, Equinox i plataforma. Git usa
subprocessos host amb timeout de dos segons; si no està disponible, els valors són `None`.
Les versions de dependències es llegeixen amb `importlib.metadata`, sense pip freeze.

`schemas` conté la versió real de configuració i el digest dels DTOs host. L’orquestració
real hi afegeix els digests existents de simulation step i mètriques, des de la signatura
canònica compilada. Un recorder creat sense runtime deixa aquests digests i la signatura
com a `None`, sense inventar-los.

Es pot afegir `case_id` i `batch_ordinal`; el UUID continua sent la identitat primària.
`manifest_to_payload` produeix el JSON. `load_run_manifest` valida schema, camps obligatoris,
tipus JSON estrictes, estats, UUID, timestamps, coherència terminal, hashes, paths i unicitat.
Rebutja claus duplicades, NaN/Infinity i camps desconeguts al manifest.

## Configuració congelada i hashes

El runtime enregistrat rep `ExperimentConfig`, perquè `CoreConfig` no conserva tot el
context host necessari per congelar el document executat. La seed és explícita dins d’aquest
model canònic; no s’afegeix una segona font de seeds ni es modifica el seu contracte.

`config.json` s’escriu amb `serialize_config`, sobre el model normalitzat validat: JSON
UTF-8 amb claus ordenades, indentació estable, newline final i sense NaN/Infinity. No és una
còpia cega del fitxer font. El registre no torna a escriure aquest fitxer durant el run.

- `config_hash`: hash canònic del projecte, calculat amb `freeze_config`; inclou la seed
  segons el contracte existent.
- `ArtifactRecord.sha256`: SHA-256 dels bytes exactes del fitxer, inclosa la presentació.
  Pot ser diferent de `config_hash`; no el substitueix.

Els overrides de total_steps, chunk_steps i mode queden al manifest i no es projecten dins
la configuració per alterar-ne el hash. En batches es reutilitzen les còpies normalitzades
per seed del PR-34; no es modifica cap configuració base.

## Artefactes i atomicitat

`LocalArtifactRegistry` no importa JAX ni Equinox. Cada `ArtifactRecord` immutable conté
nom, tipus, path relatiu, MIME type, mida en bytes i SHA-256 lowercase. El helper
`sha256_file` llegeix blocs binaris d’1 MiB, sense carregar el fitxer sencer.

Els paths són POSIX relatius al run, per exemple `artifacts/notes.json`. Es rebutgen paths
absoluts, drives Windows, backslashes, segments buits, `.`/`..`, traversal i symlinks en el
camí de l’artefacte. No es copien fitxers externs silenciosament. El directori es considera
propietat exclusiva del procés recorder; això no és una sandbox contra un procés adversari
que canviï simultàniament el filesystem.

Noms i paths són únics. Registrar exactament el mateix contingut i metadata és idempotent;
canviar-los provoca error. `write_json_artifact` no sobreescriu un fitxer existent no registrat
ni un artefacte immutable diferent. `manifest.json` està reservat i mai no s’indexa a si mateix.

El JSON es serialitza abans d’escriure, es desa a un temporal al mateix directori, es fa
flush i fsync, i es publica amb `os.replace`. Una fallada abans del replace conserva el
manifest anterior vàlid; el temporal es neteja. El canvi en memòria només es publica després
de l’escriptura correcta. L’atomicitat és per fitxer, no una transacció de múltiples fitxers:
una interrupció pot deixar un artefacte encara no indexat, però no mig manifest JSON.

## Execució, resum i fallades

L’ordre és: validar/convertir configuració → UUID/directori → config + manifest created →
inicialitzar estat/acumulador nous → running → runtime/mode existent → summary → completed.

`run_recorded_simulation` retorna `RecordedRunResult`, amb resultat runtime, resum escalar,
manifest i directori. El resum usa només observacions finals i totals produïts pel runtime:
població viva, naixements, morts, llinatges actius, energia i recursos. No genera conclusions
científiques. Benchmark conserva l’absència de telemetria per chunk; el registre es fa fora
del timing intern del PR-33.

Una configuració que no passa parseig/validació/conversió no crea run. Una fallada posterior
a la creació conserva el directori, marca failed i escriu un error segur (tipus, missatge
acotat a 4096 caràcters i pas conegut si és un diagnòstic). No serialitza traceback, locals,
estat ni tensors. Els passos executats i el final són `None` quan el runtime no retorna carry
parcial; no s’inventen dades. L’excepció original es propaga.

Si el mateix filesystem impedeix registrar la fallada, es conserva l’excepció original i
s’hi afegeix una nota amb l’error de persistència, visible també a la CLI. El manifest previ
pot continuar en created/running; no s’afirma falsament que s’ha pogut desar failed.

## CLI i batch

```console
uv run evolucio run config.json --steps 1000 --chunk-steps 128 \
  --mode experiment --output-root runs
uv run evolucio artifacts inspect runs/<UUID>
uv run evolucio artifacts verify runs/<UUID>
```

Les execucions reals de `evolucio run` creen registre per defecte. Mostren UUID i directori
abans de la inicialització, i el manifest en acabar. Amb `--quiet` es conserva la supressió
del report normal; les fallades sempre mostren el context del registre a stderr.

`--dry-run` no genera UUIDs, no crea directoris ni escriu fitxers, tampoc en batch.

```console
uv run evolucio batch run --config baseline=A.json --config variant=B.json \
  --seeds 11,22 --steps 4 --chunk-steps 2 --output-root runs
```

La CLI batch crea un run local independent per cada (case, seed). L’API `run_batch` admet
`output_root=Path("runs")` per activar aquest comportament, i conserva el mode sense
persistència si s’omet. Cada `BatchRunResult` pot conservar `run_id` i `run_dir`, també en
fallada. Els manifests inclouen case i ordinal; continue_on_error/fail-fast es mantenen.
Tota la prevalidació del lot continua sent anterior a la creació del primer run.

## Verificació posterior

`verify_local_run(run_dir)` és de només lectura i retorna `LocalRunVerification` amb
`ok`, artefactes absents, mismatches de mida/hash i paths invàlids. Comprova també la seed i
el hash semàntic de la configuració. Un manifest/path d’entrada que no es pot carregar és
un error d’input, no un run científic fallit. No s’arregla ni es reescriu res.

`artifacts inspect` mostra manifest i verificació; `artifacts verify` mostra només la
verificació. Codis: 0 integritat correcta, 1 artefactes danyats/absents/paths incoherents,
2 input o manifest invàlids. Un failed íntegre pot donar 0: integritat i èxit científic/runtime
són conceptes diferents.

Modificar manualment `config.json`, eliminar un artefacte o canviar-ne els bytes fa fallar
la verificació. El manifest no està signat: els hashes detecten alteracions dels artefactes,
no autentiquen un directori manipulat íntegrament per un tercer.

## Abast i evolució futura

Un registre local **no és un checkpoint**: no desa SimulationState, RNG ni genomes complets,
i no permet reprendre la simulació. No hi ha sèries temporals JSON per chunk, snapshots
automàtics, Parquet/Zarr, PostgreSQL, MLflow, upload remot ni retenció automàtica.

Una capa futura podrà ser propietària del UUID i associar el manifest i els artefactes a
PostgreSQL o MLflow. Aquest PR només prepara identitat, metadades i integritat local;
no implementa migracions, experiment_id ni cap integració remota.
