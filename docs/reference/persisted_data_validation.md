# Lectura i validació de dades persistides

La CLI i l’API llegeixen el directori portable del run sense executar la simulació. Les
consultes de Parquet retornen un `LazyFrame`. Els arrays de les captures i els
paràmetres neuronals seleccionats es llegeixen quan es demanen. L’API retorna les
fulles NumPy dels genomes en ordre de l’esquema neuronal.

## Formats

| Format | Versió persistida | Loader |
| --- | --- | --- |
| Manifest local | `local_run_manifest_v1` (1), `local_run_manifest_v2` (2) | `load_run_manifest` |
| Configuració congelada | `3.0` | `load_config` |
| Resum final | Llegat, sense camp propi de versió | Contracte del manifest, `open_run` |
| Índex `run_telemetry_parquet_v1` | 1 i 2 | `parquet.index.load_index` |
| `metrics`, `births`, `deaths`, `genealogy` | 1, declarat a l’índex | `scan_dataset` i funcions específiques |
| `founders_v1`, `agent_lifecycle_summary_v1`, `lineage_outcome_summary_v1` | 1, als peus Parquet | Consultes lazy de `RunData` i validació del peu |
| Índex de captures Zarr | 1 i 2 | `zarr.index.load_index` |
| `analysis_snapshot_zarr_v1` | 1 | `load_snapshot` |
| `simulation_checkpoint_zarr_v1` | 1 | `read_checkpoint_arrays` i `load_checkpoint` |
| Índex de genomes destacats i fulles NPZ | 1, amb esquema neuronal declarat per registre | `read_genome_index`, `read_genome_leaves` |
| Provenance de recuperació | 1 | `read_provenance` |
| Metadades de tracking MLflow | 1 | `load_tracking_state` |

La versió de l’índex de telemetria és independent de la versió de les seves
taules. L’índex v2 incorpora el registre de chunks confirmats. El manifest
conserva els digests de DTO, simulació i mètriques que l’execució va declarar.

Per afegir una versió, registra el parell exacte de nom i versió, afegeix el
loader i la validació del format nou, i conserva el lector i la fixture de la
versió anterior. Un canvi compatible d’índex pot reaprofitar el mateix lector
si les dues variants es discriminen inequívocament. Un camp de versió absent és
un error per als formats que el declaren. El resum local es valida dins del
contracte del manifest perquè el writer històric el va publicar sense versió pròpia.

## Nivells i cost

| Nivell | Què comprova i què llegeix |
| --- | --- |
| `metadata` | Manifest, configuració, resum i índexs registrats. El cost depèn de la mida dels índexs. |
| `integrity` | Afegeix mides, hashes i artefactes orfes dins dels magatzems indexats. Pot llegir tots els bytes. |
| `structural` | Afegeix esquemes, tipus, nul·labilitat, recomptes i intervals declarats. Pot obrir parts Parquet i totes les captures d’una en una. |
| `scientific` | Afegeix consultes i creuaments sobre tots els esdeveniments, genealogia, població, resums, captures i recuperació. El cost depèn de les files i de la quantitat de captures. |

Els estats d’una comprovació són `passed`, `failed`, `not_checked` i
`not_applicable`. Cada incidència indica el component, un codi estable, la
severitat i el context disponible. `data_complete` descriu el lifecycle conegut.
Un run complet pot fallar una comprovació d’integritat. El camp
`local_validation_passed` permet llegir el resultat local independentment
de les comprovacions remotes sol·licitades. Per validar un prefix
fallit o aturat, passa explícitament `--allow-partial`. Si el writer té el
bloqueig de PR-41, es permet la inspecció amb un avís i es bloqueja l’auditoria
exhaustiva. La lectura concurrent de metadades és transitòria.

La genealogia ha de projectar exactament les identitats de cada naixement.
Les morts del pas es resolen abans de la reproducció i un descendent pot
reproduir-se a partir del pas següent al seu naixement. Els esdeveniments
complets reconstrueixen el balanç fins a cada mètrica mostrejada. Les captures
de l’estat `S` es comparen amb la mètrica `S - 1` quan existeix. Els resums
d’agents i llinatges es contrasten amb els esdeveniments i les projeccions canòniques.

Els runs recuperats conserven els founders originals i els esdeveniments
del prefix importat. La cadena es comprova quan les fonts són accessibles
com a directoris germans. Una font absent deixa la verificació externa
pendent i la recuperació continua identificada com una trajectòria relacionada.

## Compatibilitat

`open_run` avalua la lectura dels formats. `compare_runs` avalua la
comparabilitat de la telemetria mostrejada amb els digests, les versions
disponibles i la cobertura. Una diferència de `config_hash` queda informada i
pot correspondre al disseny de l’experiment. El digest del pas incorpora l’esquema de mutació.
`None` indica una comprovació sense
resultat establert. `check_restore_compatibility` demana explícitament al lector
de checkpoint que reconstrueixi el PyTree amb la configuració i la signatura
aplicables.

`open_registered_run` resol un run per UUID sota l’arrel de fitxers indicada.
`validate_experiment` retorna informes individuals i totals operatius.
`validate_run(..., registry=..., tracking_uri=...)` sol·licita la reconciliació
amb PostgreSQL i, si s’indica un URI, MLflow. Les discrepàncies d’aquests serveis
s’identifiquen dins del seu component. La reconciliació és de lectura.

## CLI

```console
evolucio data schemas
evolucio data inspect runs/<run-id> --format json
evolucio data validate runs/<run-id> --level scientific --format json
evolucio data validate runs/<run-id> --level integrity --component parquet --allow-partial
```

L’informe JSON té `schema_version: 1`. El codi de sortida és 0 per a les
comprovacions sol·licitades superades, 1 per a errors o comprovacions requerides
bloquejades i 2 per a arguments invàlids. Per exemple, un run complet pot
retornar `parquet=passed` al nivell `integrity` i `parquet=failed` al nivell
`scientific` amb el codi `genealogy_projection` després d’un canvi coherent de
hashes i índexs.

## Límits de l’evidència

- L’índex Parquet v1 no acredita la cobertura per chunks. Els runs iniciats
  des d’un checkpoint sense història importada poden tenir un origen analític desconegut.
- Una captura sense mostra corresponent deixa pendent la comparació de mètriques.
  Sense captura final o mostra final, l’energia i els recursos del resum no es
  poden contrastar amb aquestes fonts.
- Les marques de saturació de l’acumulador estan disponibles als checkpoints.
  Els artefactes històrics poden no conservar aquesta informació a l’estat final.
- Les versions de codi i els digests disponibles aporten evidència sobre el
  contracte del model. Les conclusions científiques depenen també del disseny
  experimental i de la cobertura.
