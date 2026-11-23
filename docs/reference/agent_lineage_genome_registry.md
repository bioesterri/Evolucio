# Registre de genomes destacats i resums d’agents i llinatges

L’anàlisi del PR-39 descriu què ha passat durant un run. Es construeix al
host a partir dels fundadors inicials, els naixements i morts observats,
la projecció genealògica validada de PR-36 i la població viva a l’estat
final. La simulació només s’executa una vegada.

## Artefactes analítics locals

| Fitxer | Esquema | Contingut |
| --- | --- | --- |
| `artifacts/analysis/founders.parquet` | `founders_v1` | Identitat, llinatge, genoma, generació, pas de naixement, energia i posició inicials. |
| `artifacts/analysis/agent_summary.parquet` | `agent_lifecycle_summary_v1` | Una fila per agent, ordenada per `agent_id`. |
| `artifacts/analysis/lineage_summary.parquet` | `lineage_outcome_summary_v1` | Una fila per llinatge, ordenada per `lineage_id`. |
| `artifacts/genomes/index.json` | Versió 1 | Identitats seleccionades, motius i passos de selecció, compatibilitat de política i referències als NPZ. |

Els resums Parquet porten el nom i la versió d’esquema a les metadades.
Els identificadors, passos, edats i recomptes són `Int64`. Els codis de
mort són `Int32`, les energies, fraccions i deltes acumulats són
`Float64`, i `alive_at_end` és booleà.

Un `AgentSummary` enllaça cada origen únic (fundador o naixement) amb un
únic final (mort o supervivència al final). Les morts aporten l’edat,
l’energia i els codis de l’esdeveniment. Els supervivents aporten l’edat i
l’energia de l’últim estat, i els camps de mort queden nuls. El
progenitor d’un fundador és nul. El recompte de fills directes surt
d’agrupar els naixements pel progenitor.

Un `LineageSummary` agrega els agents que tenen el mateix fundador.
El nombre de descendents és el total d’agents menys el fundador i
coincideix amb el nombre de naixements del llinatge. La generació màxima
mesura la profunditat genealògica assolida. Si hi ha supervivents, el pas
d’extinció és nul i la durada és `final_state_step - founder_birth_step`.
Si l’última mort és a `D`, el pas d’extinció és `D` i la durada es
compta fins a l’estat següent, `(D + 1) - founder_birth_step`. Quan la
població final és zero, totes les fraccions de població final són zero.

El líder reproductiu és l’agent amb més fills directes. El líder de
supervivència és l’agent amb més edat al seu punt final observat. En tots
dos casos, un empat es resol amb l’`agent_id` més petit. Aquestes
estadístiques descriuen l’individu i no assignen cap qualitat al seu
genoma ni intervenen en la simulació. Els totals de mutació compten
seleccions, canvis efectius i la suma de deltes absoluts observats als
naixements. Aquesta suma no mesura la diversitat genètica.

Els runs fallits conserven la telemetria disponible i la captura
inicial quan existeix. Els resums finals es retiren si falla la
finalització. En mode benchmark la telemetria interna es persisteix
sense incloure el temps d’escriptura en la mesura reportada.

Les continuacions de checkpoints amb artefactes PR-39 recuperen els
fundadors del run font i incorporen els naixements i morts anteriors al
pas restaurat. Aquest prefix queda als fitxers
`artifacts/analysis/historical_births.parquet` i
`artifacts/analysis/historical_deaths.parquet`. Un checkpoint anterior
al PR-39 o distribuït sol encara es pot reprendre. Quan no hi ha dades
d’origen suficients, la continuació indica
`analysis_status=unavailable_pre_checkpoint_history` al resum del run.

## Genomes recuperables

Es desen tots els genomes fundadors després de la inicialització i abans
del primer pas. De cada llinatge amb supervivents es desa el genoma de
l’agent de generació més alta. En cas d’empat s’escull l’`agent_id` més
petit. Si el mateix genoma és fundador i representant final, té una
entrada amb els dos motius. Per als descendents el genoma parental es
recupera del naixement.

Cada NPZ conté les quatre fulles de pesos i biaixos de la política.
El SHA-256 semàntic concatena la capçalera amb l’esquema i la seva
versió, i després, per cada fulla, el nom, el dtype, la forma i els bytes
float32 little-endian en ordre C. L’ordre de les fulles és
`layer1.weight`, `layer1.bias`, `layer2.weight`, `layer2.bias`.
Els identificadors no formen part del hash. Dos genomes amb contingut
idèntic comparteixen `artifacts/genomes/<genome_hash>.npz`.

`artifact_sha256` comprova els bytes del fitxer NPZ. El manifest
registra l’índex JSON, que al seu torn verifica els NPZ i detecta fitxers
orfes. `load_highlighted_genome(run_dir, genome_id)` comprova la
compatibilitat de política i els dos hashes abans de reconstruir una
`PolicyMLP`. Els resums retenen els identificadors dels líders morts.
El genoma complet és recuperable si s’havia conservat per un dels
motius admesos.

```console
evolucio genome list runs/<run_id>
evolucio genome verify runs/<run_id>
evolucio run inspect runs/<run_id>
```

## PostgreSQL

La revisió `0002_analysis_registry` afegeix `agent_summary` i
`lineage_summary`, amb una clau composta per run i identitat.
`genome_registry` enllaça cada genoma amb la fila d’agent corresponent
i desa el camí i les empremtes de l’NPZ. `genome_highlight` desa una fila
per motiu de cada genoma. L’índex principal s’incorpora a la taula
`artifact` existent.

El flux amb `--experiment-id` verifica els artefactes locals abans
d’indexar-los. Els inserts per lots, els índexs i la transició a
`completed` es confirmen en una única transacció SQL. Els runs només
locals poden completar i produir tots els artefactes sense PostgreSQL.
El fitxer provisional `summary.json` forma part dels artefactes que
es verifiquen abans de la transacció SQL. El manifest local es publica
com a `completed` quan es confirma la transacció. Els dos sistemes
tenen commits independents.

`PostgresRegistry` pot consultar un agent o llinatge, llistar agents
d’un llinatge, ordenar llinatges pels seus recomptes o generació, i
llistar motius de conservació. `lineage_outcomes_for_experiment`
retorna, per run completat, l’identificador del run, el cas, la seed,
el nombre de llinatges, la generació màxima i la mida màxima d’un
llinatge. Són mesures descriptives per run.
