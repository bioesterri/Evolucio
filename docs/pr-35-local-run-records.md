# PR-35 — Manifest de run, configuració congelada i artefactes locals

Branca: `pr-35/local-run-manifest-artifacts`.

## Descripció del PR

Les execucions acabaven sense una identitat local ni context auditable fora de memòria.
Aquest PR afegeix un registre mínim a `runs/<UUID>/`: manifest, configuració normalitzada,
resum final o error i un índex d’artefactes amb paths relatius, mida i SHA-256.

`create_local_run` genera UUID v4 o valida un UUID injectat, reserva un directori exclusiu,
escriu `config.json` amb el serializer canònic i publica el manifest created. L’adaptador
`run_recorded_simulation` exigeix el model host `ExperimentConfig` per preservar exactament
la configuració congelada; valida la conversió a `CoreConfig` abans de crear el run.
Després inicialitza estat/RNG/IDs/acumulador nous, marca running i reutilitza `run_with_mode`.
El UUID, el directori i el manifest existeixen abans de la inicialització.

En completar escriu summary i marca completed. En fallada conserva config i directori,
escriu error segur, marca failed i propaga l’excepció original. Els runs terminals no es
reobren. Si falla també el filesystem en registrar l’error, l’excepció original conserva
una nota amb la fallada de persistència; no s’afirma un estat terminal que no s’ha pogut desar.

El schema `local_run_manifest_v1` registra seed, pla, mode, timestamps UTC, configuració,
compile signature, versions de paquet/Git/Python/JAX/Equinox/plataforma, digests reals,
artefactes i error opcional. Les versions de codi i entorn s’agrupen en DTOs immutables.
Git té timeout i permet informació desconeguda explícita. El loader valida JSON estricte,
identitats, lifecycle, hashes, paths i unicitat; rebutja schema desconegut, claus duplicades
i manifestos terminals incoherents.

`LocalArtifactRegistry` valida paths i rebutja traversal, drives, backslashes i symlinks.
No sobreescriu registres immutables; una repetició exacta és idempotent. SHA-256 es calcula
per blocs. Tots els JSON es serialitzen abans d’escriure, es publiquen amb temporal al mateix
directori + flush/fsync + os.replace, i es netegen temporals en fallada. El manifest no es
registra a si mateix. L’atomicitat és per fitxer, sense prometre una transacció multi-fitxer.

`verify_local_run` no modifica dades: comprova UUID/directori, fitxers, mida, hashes,
paths, seed i hash canònic de config. `config_hash` continua sent el hash canònic del projecte;
el SHA del fitxer protegeix els bytes exactes amb indentació/newline. El pla, mode, UUID i
timestamps no alteren `config_hash` ni `CompileSignature`.

## CLI i batch

Les execucions reals de `evolucio run` creen registre local per defecte sota `runs/`, amb
`--output-root` per substituir l’arrel. Mostren UUID/directori abans d’inicialitzar i manifest
al final; les fallades mostren context a stderr. `--quiet` conserva la supressió del report
normal. `--dry-run` no genera UUID, no crea directoris ni escriu res.

S’afegeixen `evolucio artifacts inspect <run_dir>` i `artifacts verify <run_dir>` sense
trencar el verb `run`. Codis: 0 íntegre, 1 dany d’artefactes/integritat, 2 input/manifest
invàlids. Un run failed pot ser íntegre. Parseig/config invàlids continuen sent exit 2;
una fallada d’inicialització posterior a la creació és tècnica, exit 1, amb rastre local.

La CLI batch crea un registre independent per (case, seed). L’API PR-34 admet `output_root`
opcional i conserva l’execució sense persistència si s’omet. Els resultats poden incloure
run_id i run_dir, incloses fallades. Cada manifest conserva case_id i batch_ordinal;
no es crea experiment_id ni un únic manifest de run per al lot. La prevalidació completa
continua sent anterior a qualsevol creació i es mantenen continue_on_error/fail-fast.

## Layout

```text
runs/<UUID>/
├── manifest.json
├── config.json
├── summary.json       # completed amb resum final
└── artifacts/
```

En fallada de simulació, `error.json` substitueix l’absència de resum final. La guia completa
és [Registres locals de runs v1](reference/local_run_records.md).

## Abast

Sense canvis al core, dependències noves ni canvis de schema científic. S’afegeix un contracte
core → persistence prohibit i es comprova la independència del registre d’artefactes de
llibreries de dispositiu. La importació de la capa local no importa JAX/Equinox.

Un registre local no és un checkpoint. Sense PostgreSQL, MLflow, Parquet/Zarr, sèries temporals,
registre complet de genomes, snapshots automàtics, checkpoints, resume, cloud storage,
upload remot ni retenció automàtica. No es fa merge automàtic.

## Fitxers

- `persistence/__init__.py` i `persistence/local/{__init__,artifacts,manifest,recorder,schema,verification,version_info}.py`.
- `runtime/recorded_run.py`: adaptador d’execució enregistrada i resum escalar.
- `cli/{run_command,batch_command,main,artifact_commands}.py`: registre i inspecció/verificació.
- `experiments/{batch,types}.py`: registre local opcional i identitat en resultats.
- `pyproject.toml`, `.gitignore`: contractes d’importació i exclusió dels runs generats.
- `tests/unit/persistence/local/`: artefactes, manifest, lifecycle, metadata, schema i verificació.
- `tests/unit/cli/test_local_records.py`, fixtures CLI i expectatives de context operacional.
- `tests/integration/test_recorded_run.py`, `test_recorded_batch.py`.
- Guia local, guies CLI/batch, README i índex documental; aquesta descripció del PR.

## Proves i verificacions

- Suite prèvia sense canvis: `uv run pytest`, **753 proves passades**.
- Suite final: `uv run pytest`, **836 proves passades** (83 noves), sense skips ni xfails.
- `uv run coverage erase`, `uv run coverage run -m pytest`, `uv run coverage report`:
  **836 proves passades**, **97 % global**; recorder i versions 100 %, artefactes 98 %,
  adaptador runtime enregistrat 97 %, verificació 93 %, manifest 86 %.
- `uv run pytest tests/unit/persistence/local`: **66 proves passades**.
- `uv run pytest tests/integration/test_recorded_run.py`: **10 proves passades**.
- `uv run pytest tests/integration/test_recorded_batch.py`: **3 proves passades**.
- `uv sync --group dev`, `uv sync --locked --group dev`, `uv lock --check`: correctes.
- Ruff lint i format: correctes; Pyright: 0 errors i 0 warnings.
- `uv run lint-imports`: **8 contractes conservats**, 0 trencats.
- `uv pip check`: 36 paquets compatibles; cap dependència nova ni canvi a uv.lock.
- `uv build`: wheel i distribució font construïts correctament.
- `git status --short` revisat i `git diff --check` correcte.

Les proves cobreixen UUID generat/injectat, no overwrite, configuració normalitzada immutable,
dos tipus de hash, lifecycle, terminalitat, timestamps UTC, traversal/symlinks, hashing en
streaming, duplicats/idempotència, escriptures atòmiques interrompudes i validació estricta
JSON. També comproven corrupció/eliminació de fitxers, CLI, dry-run sense UUID/directoris,
fallades d’inicialització/runtime/consumer i excepció original conservada si falla el registre.

La prova d’auditoria executa els quatre modes, descarta el resultat Python i torna a carregar
només manifest/config/summary. Es recuperen identitat, seed, configuració/hash, pla, mode,
versions, estat i artefactes. La prova de neutralitat compara població, món, genomes, RNG,
IDs i acumulador contra el runtime sense registre amb les toleràncies del PR-29. Els batches
comproven directoris/UUIDs separats, atribució de case/seed/ordinal, continuació i fail-fast.

## Smoke CLI real i integritat

Ordres executades sobre una configuració petita vàlida:

```console
uv run evolucio run /workspace/work/pr35/configs/A.json \
  --steps 4 --chunk-steps 2 --mode experiment --output-root runs
uv run evolucio artifacts verify runs/f5c85105-ecb0-43d6-89b1-17ae3039bd3f
```

Run completat amb UUID `f5c85105-ecb0-43d6-89b1-17ae3039bd3f`, seed 42, pas final 4 i
4 passos executats. El procés CLI ja havia acabat quan es va tornar a auditar només amb
els fitxers locals. Resultat real:

```text
LocalRunVerification(ok=True, missing_artifacts=(), hash_mismatches=(), invalid_paths=())
```

La CLI verify retorna codi 0. El directori conté manifest.json, config.json, summary.json
i artifacts/ buit; no hi ha telemetria massiva ni estat reprenable. Els runs generats són
locals i queden exclosos del control de versions amb .gitignore.

Cap comprovació local pendent. No s’ha fet merge.
