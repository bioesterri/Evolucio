# Runtime host de runs per chunks v1

`evolucio.runtime.run_simulation` executa un `RunPlan` immutable amb dos enters Python
estrictament positius: `total_steps` i `chunk_steps`. Rebutja booleans. El pla queda separat
de `CoreConfig`, del hash científic i de `CompileSignature`.

```text
core.run_chunk → runtime → HostChunkBatch → consumer futur
```

El runtime reutilitza el `run_chunk` compilat del PR-28. Planifica amb `divmod` i un bucle
Python; per exemple, 10 passos amb chunks de 4 executen **4 + 4 + 2**. El remainder usa un
chunk compilat amb la seva longitud exacta, sense padding. Pot requerir una segona compilació.
La mida tècnica dels chunks no canvia la trajectòria científica.

`SimulationState` i `MetricsAccumulator` passen directament al chunk següent i continuen al
dispositiu. Només es llegeixen els escalars de pas inicial i final del carry. Després de cada
chunk, `jax.device_get((chunk.metrics, chunk.events))` transfereix les sortides completes i
sincronitza aquestes dades. No hi ha callbacks ni transferències dins el JIT.

Abans de construir `HostChunkBatch`, el runtime comprova que `metrics.step` té la longitud
prevista i conté exactament els passos de `[start_step, end_step_exclusive)`. Els índexs dels
chunks comencen a zero. Les incoherències es propaguen com a errors, sense reparar mètriques.

El `ChunkConsumer` és síncron i opcional. Rep les estructures originals de mètriques i events
amb arrays host; no rep l'estat, genomes ni transitoris. No es mostregen naixements ni morts.
Qualsevol excepció del consumer o del kernel es propaga i impedeix executar chunks posteriors.

No es reté historial: després de lliurar un batch, el runtime allibera les seves referències
a les sortides abans de continuar. La memòria pròpia és l'estat actual, l'acumulador i les
sortides d'un chunk. Un consumer que conservi batches assumeix el cost d'aquell historial.
`RunRuntimeResult` retorna només el carry final i els passos i chunks executats.

La població buida o extingida continua fins a `total_steps`. Aturar al final d'un chunk faria
dependre l'horitzó executat d'un paràmetre tècnic. L'early stop exacte queda per a un altre PR.

El schema `host_chunked_run_runtime_v1`, versió 1, té digest SHA-256 sobre JSON canònic i no
entra a la identitat científica ni de compilació ecològica. No incorpora persistència,
tracking, DTOs persistents, checkpoints, resume, CLI, benchmarks ni execució asíncrona.
