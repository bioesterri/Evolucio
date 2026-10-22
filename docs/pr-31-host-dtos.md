# PR-31 — DTOs host per mètriques, esdeveniments i snapshots

Branca: `pr-31/host-dtos`

## Resum

El runtime lliurava PyTrees materialitzats que obligaven els consumidors externs a conèixer
Equinox, masks i sentinels. Aquest PR afegeix `evolucio.host`, independent de JAX/Equinox/core,
amb DTOs immutables de mètriques, naixements, morts, chunks i snapshots.

```text
core PyTrees → runtime device_get → conversion → host DTO → persistència futura
```

Els adaptadors host converteixen totes les mètriques reals a escalars Python, compacten només
events reals i els ordenen per pas i ID biològic. `DTOChunkConsumer` manté compatible el
consumer del PR-30, sense transferències addicionals ni historial acumulat. Es rebutgen
incoherències temporals, valors no finits, IDs invàlids i formes incorrectes.

Els snapshots són explícits: transfereixen pas, comptadors, població i món, sense genomes ni
RNG. Contenen només agents vius, ordenats per `agent_id`, i arrays NumPy propis de només
lectura amb dtypes preservats. `slot_index` és diagnòstic; `state_step` és el següent pas.
No són checkpoints ni permeten reprendre la simulació.

El schema host v1 té digest SHA-256 sobre JSON canònic i queda fora de `CompileSignature`
i del hash científic. Els contractes d'importació impedeixen `host → core/runtime/JAX/Equinox`
i `core → host`. No es modifica cap fitxer del core ni `runtime/runner.py`.

## Validació

- `uv run pytest`: 602 proves passades.
- Suite amb cobertura: 602 proves passades; 97 % global, 100 % als DTOs host i conversions,
  i 98 % a snapshots. Sense skips ni xfails.
- Proves específiques: 6 de host, 36 de conversions, 23 de snapshots i 1 d'integració, passades.
- Ruff, format, Pyright, els quatre contractes d'importació, lockfile, sincronització,
  compatibilitat de dependències, build i `git diff --check`, superats.

Les proves cobreixen tipus Python, tots els camps reals de mètriques, ordre temporal,
compactació i ordre independent dels slots, genealogia, mutació, causes/fases i corrupció.
Els snapshots cobreixen població buida, un agent, slots dispersos, ordenació, dtypes, còpies,
read-only, món i comptadors. Es comprova que no són entrades de simulació reprenibles.
Una prova en un procés separat impedeix importar JAX, Equinox, core i runtime des del host.
La integració executa 10 passos com 4+4+2 amb el consumer DTO, comprova totes les mètriques
i events i confirma que el snapshot només es transfereix quan es demana explícitament.

## Abast

Sense persistència, JSON/CSV de dades, Parquet/Zarr, PostgreSQL, SQLAlchemy, Polars, MLflow,
registre de runs, `run_id`, checkpoints, resume, hashes d'artefactes, sampling, política de
snapshots, genomes destacats, CLI ni visualització. No s'ha fet merge automàtic.

Contracte detallat: [DTOs host v1](reference/host_metrics_events_snapshots_v1.md).
