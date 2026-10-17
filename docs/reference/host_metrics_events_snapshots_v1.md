# DTOs host de mètriques, esdeveniments i snapshots v1

`evolucio.host` defineix el contracte de dades entre el runtime i les futures capes externes.
Importar-lo no importa JAX, Equinox, `evolucio.core` ni `evolucio.runtime`. Utilitza dataclasses
congelades amb slots, tuples, escalars Python i arrays NumPy. No afegeix dependències.

```text
core PyTrees → runtime device_get → conversion → host DTO → persistència futura
```

| DTO | Contingut | Granularitat | Ús previst |
| --- | --- | --- | --- |
| `StepMetricsDTO` | Tots els camps reals de `StepMetrics`; causes com a tuple | Un pas | Sèries i anàlisi |
| `BirthEventDTO` | Identitats, genealogia, posició, energia i resum de mutació | Un naixement real | Genealogia i events |
| `DeathEventDTO` | Identitats, edat, energia, posició, causa i fase | Una mort real | Mortalitat i events |
| `EventBatchDTO` | Tuples compactes de naixements i morts | Un lot | Persistència futura |
| `ChunkTelemetryDTO` | Versió, interval, mètriques i events | Un chunk | Consumer temporal |
| `PopulationSnapshotDTO` | Només agents vius i índexs tècnics dels slots | Una captura | Inspecció poblacional |
| `WorldSnapshotDTO` | Recursos, ambient i ocupació `[H,W]` | Una captura | Anàlisi espacial |
| `SimulationSnapshotDTO` | Població, món, comptadors i `state_step` | Una captura | Fotografia analítica |

## Telemetria

`metrics_to_dtos` opera sobre mètriques temporals `[T,...]` ja materialitzades al host. Retorna
T registres ordenats, amb `int`/`float` Python i `tuple[int, ...]` per causes. Inclou també
`birth_rejected_capacity` i `death_energy_removed`, presents al PR-25 real.

`birth_events_to_dtos` selecciona exclusivament `born=True` i ordena per
`(birth_step, child_agent_id)`. `death_events_to_dtos` selecciona exclusivament `died=True`
i ordena per `(death_step, agent_id)`. Els slots buits i els seus valors de farciment no
generen DTOs. Es preserven genealogia, resum de mutació, causa i fase. Un `parent_id=-1`
continua representant un fundador; un ID nul d'agent real es rebutja.

`host_chunk_batch_to_dto` valida la longitud i l'inici de la sèrie consecutiva de mètriques,
i que tots els events siguin dins `[start_step, end_step_exclusive)`. Les conversions exigeixen
arrays NumPy: no provoquen transferències implícites de `jax.Array`. Les formes, dtypes,
valors no finits, IDs obligatoris nuls, posicions negatives i metadades de mort incoherents
es rebutgen amb `ValueError`. Les posicions dels events no disposen de dimensions del món;
el snapshot sí que comprova els límits superiors espacials. No es reordenen mètriques ni es
reparen dades corruptes.

`DTOChunkConsumer(consumer)` adapta `ChunkConsumer(HostChunkBatch)` a
`consumer(ChunkTelemetryDTO)` sense modificar el contracte PR-30. Les excepcions es propaguen.
La conversió és per chunk i no conserva historial.

## Snapshots explícits

`simulation_state_to_snapshot(state)` és l'única entrada de captura. Fa un `device_get` de
`(state.step, state.ids, state.population, state.world)`, sense transferir genomes ni RNG.
La població es filtra per `alive` i s'ordena per `agent_id` ascendent, permutant tots els camps.
`slot_index` conserva el slot físic original i **no és identitat biològica**. Els identificadors
externs són `agent_id`, `genome_id` i `lineage_id`.

Els arrays de població tenen forma `[N]`, excepte `position`, de forma `[N,2]`; N pot ser zero.
Els arrays de món són `[height,width]`. Cada array té emmagatzematge host propi, conserva
el dtype original i queda marcat amb `write=False`, també en construir els DTOs directament.
Els snapshots amb arrays es comparen camp a camp amb NumPy, sense igualtat implícita d'arrays.

`state_step` és el següent pas que s'executaria des de `SimulationState`, i no el pas de
l'última mètrica. Un snapshot és una fotografia analítica, **no és un checkpoint**: no inclou
RNG, genomes complets, acumulador, configuració, events, observacions, scores, accions ni
`run_id`. No es pot usar com a entrada de `simulation_step`/`run_chunk` ni es proporciona
cap conversió inversa a `SimulationState`.

Capturar població i camps del món pot ser car. Aquesta funció és host-side, fora del JIT i
del fast path. No s'invoca per defecte, cada pas ni després de cada chunk. No s'introdueix
cap política de captura vinculada a `chunk_steps`.

## Versionat i exclusions

El schema és `host_metrics_events_snapshots_v1`, versió 1, amb SHA-256 sobre JSON canònic.
Descriu camps, compactació, ordre i semàntica de snapshots. No entra a `CompileSignature`
ni modifica el hash científic, el core o les formes del device.

No hi ha serialització de DTOs a JSON, CSV, SQL, Parquet o Zarr; tampoc persistència concreta,
registre de runs, checkpoints, resume, sampling, genomes destacats, CLI ni visualització.
La persistència futura associarà el context de run als DTOs sense repetir `run_id` a cada registre.
