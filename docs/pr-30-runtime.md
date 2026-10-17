# PR-30 — Runtime de runs, chunking i frontera dispositiu-host

Branca: `pr-30/run-runtime-device-host`

## Resum

El nucli podia executar chunks individuals, però faltava completar un horitzó exacte des del
host sense retenir tota la trajectòria. Aquest PR introdueix `evolucio.runtime` amb `RunPlan`
immutable, `run_simulation`, `HostChunkBatch`, `ChunkConsumer` i `RunRuntimeResult`.

El bucle Python reutilitza `core.simulation.run_chunk` compilat i planifica amb `divmod`.
Per exemple, 10 passos amb chunks de 4 executen **4 + 4 + 2**, sense padding. El remainder
pot provocar una compilació addicional amb la seva longitud exacta.

```text
core.run_chunk → runtime → HostChunkBatch → consumer futur
```

## Frontera dispositiu-host

Estat i acumulador passen directament al chunk següent i continuen al dispositiu. Després de
cada chunk, només les mètriques i els events complets passen per `jax.device_get`. Del carry
només es llegeixen els escalars de pas inicial i final.

Abans de lliurar el batch al consumer host síncron, es valida l'interval de `metrics.step`.
Els errors del kernel i del consumer es propaguen i impedeixen executar chunks posteriors.
No es mostregen ni es filtren naixements o morts.

El runtime allibera les referències a les sortides abans del chunk següent. La seva memòria
és l'estat, l'acumulador i els outputs d'un chunk; només un consumer pot decidir conservar
historial. El resultat retorna el carry final i els comptadors d'execució.

Una població extingida continua fins a l'horitzó sol·licitat. Aturar al límit d'un chunk faria
dependre l'horitzó executat d'un paràmetre tècnic.

## Arquitectura i schema

El schema `host_chunked_run_runtime_v1`, versió 1, té digest SHA-256 sobre JSON canònic.
El pla i el schema queden fora de `CoreConfig`, `CompileSignature` i el hash científic.
Un contracte d'importació prohibeix `core → runtime`. No es modifica cap fitxer del core.

El contracte detallat és a [Runtime host v1](reference/host_chunked_run_runtime_v1.md).

## Proves i verificacions

- Suite prèvia: 492 proves passades.
- `uv run pytest` final seqüencial: 536 proves passades.
- Suite completa amb cobertura: 536 proves passades; 97 % global i 100 % al runtime.
- Runtime unitari: 34 proves passades; integració específica: 10 proves passades.
- Exactitud de passos, intervals, remainder, inici no nul i acumulador preexistent.
- Neutralitat en 20 passos amb chunks d'1, 2, 4, 5, 7, 10 i 16: estat final, RNG,
  IDs, acumulador i trajectòria completa de mètriques i events equivalents a un chunk únic.
- Repetibilitat, població buida, extinció, preservació d'events i errors del consumer.
- Arrays host, ordre kernel-transferència-consumer i alliberament d'outputs anteriors.
- Lockfile, sincronització, Ruff, format, Pyright, imports, compatibilitat de paquets,
  build i comprovació de whitespace superats.

Les dades discretes exigeixen igualtat exacta; els float32 segueixen el contracte del PR-29
amb `atol=rtol=1e-6`, amb el mateix backend i versions. Les proves no requereixen GPU.

Una passada global en paral·lel amb cobertura va ser interrompuda pel límit de memòria de
l'entorn (OOM, codi 137). La passada amb cobertura i la repetició final seqüencial es van
completar correctament, sense skips ni xfails.

## Abast

Sense PostgreSQL, Parquet, Zarr, MLflow, DTOs persistents, checkpoints, resume, CLI,
benchmarks, early stopping, concurrència ni multi-device. El consumer és la frontera per a
adaptadors futurs. No s'ha fet merge automàtic.
