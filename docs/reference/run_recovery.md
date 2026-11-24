# Recuperació de runs d’experiment

Un run passa de `CREATED` a `RUNNING` i acaba `COMPLETED`, `FAILED` o
`STOPPED`. `COMPLETED` requereix assolir l’horitzó i publicar les dades
científiques obligatòries. L’extinció és un resultat ecològic i l’execució arriba
igualment a l’horitzó. `FAILED` registra una fallada tècnica, amb la fase i
l’interval afectat quan es coneixen. `STOPPED` registra el motiu, les marques
temporals disponibles i si la interrupció ha estat ordenada. Els estats
terminals conserven el seu resultat històric.

## Interrupció i fronteres

El primer Ctrl+C sol·licita l’aturada al host. El runtime acaba el chunk actual,
confirma la telemetria i, si l’estat és vàlid, escriu un checkpoint a la
frontera. Torna a utilitzar el checkpoint indexat si ja existeix en aquella
frontera. `--no-checkpoint-on-stop` desactiva aquesta captura. Un segon Ctrl+C
pot interrompre el procés abans que acabi l’escriptura. Una caiguda, un procés
terminat per força o una pèrdua d’alimentació poden deixar el manifest en
`RUNNING`.

El fitxer `.writer.lock` es manté al directori. El bloqueig del sistema operatiu
s’allibera quan el procés mor. Un bloqueig ocupat indica activitat. Si el fitxer
falta o el sistema de fitxers no permet comprovar-lo, la propietat és
indeterminada. `recovery inspect` és de lectura i
`recovery reconcile` pot registrar explícitament un `RUNNING` sense escriptor
com a `STOPPED` per `external_interruption`. El motiu indica que l’execució
s’ha interromput i deixa la causa concreta oberta.

El darrer estat observat, el darrer final exclusiu de telemetria confirmada i
el darrer checkpoint vàlid són tres fronteres diferents. Un checkpoint a
`S` conserva l’estat després de `[0, S)` i el següent pas és `S`. El runtime
executa `[S, target_step)`. Una fallada conserva els checkpoints publicats
prèviament. Els fitxers temporals i les parts o stores sense indexar queden
disponibles per inspecció.

## Integritat i compatibilitat

L’índex Parquet v2 confirma cada chunk amb el número, l’interval temporal
semiobert, els recomptes de tots quatre datasets i les referències a les parts
existents. Els recomptes de zero també queden confirmats. El commit és la
substitució atòmica de l’índex després d’escriure les parts.

El registre Zarr vincula el SHA-256 de la captura al final exclusiu confirmat i
al digest de telemetria. El digest usa SHA-256 del JSON UTF-8 canònic produït per
`json_text` amb claus ordenades, indentació de dos espais i un salt de línia
final. El document conté `chunks` fins a `S` en ordre i `parts` agrupades per
dataset per als mateixos chunks i `metrics_interval` amb el mostreig canònic. Cada part inclou el seu hash, mida, nombre de
files i rang temporal. Afegir chunks posteriors deixa aquest digest intacte.

Un checkpoint computacionalment vàlid supera les comprovacions de bytes,
esquemes, RNG, formes, tipus i invariants. La recuperació experimental també
comprova la configuració congelada, la signatura de compilació, una revisió Git
neta i idèntica, les versions de Python, JAX, jaxlib i Equinox i el backend.
La recuperació requereix cobertura completa i mostreig verificat del prefix,
amb fundadors i genomes originals, projecció genealògica, identitats vives i
comptadors coherents. Els índexs llegats sense evidència de chunks i la
configuració o l’entorn amb identitat desconeguda poden impedir la recuperació
estricta. La CLI avalua els checkpoints indexats en ordre descendent, mostra els
errors detectats i informa del checkpoint seleccionat. Una selecció explícita
rebutjada acaba amb error.

## Identitat i persistència

La recuperació crea un nou run amb la mateixa configuració, seed, cas i
horitzó. Conserva la identitat de la trajectòria amb `root_run_id`. Qualsevol
continuació posterior apunta al seu origen immediat i conserva l’arrel i la
profunditat. La provenance és a `artifacts/recovery/provenance.json` i al
manifest.

Les parts confirmades es llegeixen i materialitzen per chunk amb el nou
`run_id`. Això consumeix CPU, I/O i espai de disc proporcionals al prefix. La
validació conserva en memòria les identitats necessàries per comprovar
duplicats i reconstruir els agents vius. Es copien els fundadors i els seus
genomes i es crea un checkpoint inicial propi. Els snapshots anteriors queden
al run d’origen. Els representants finals se seleccionen a partir de la
trajectòria recuperada. Els resums d’agents i llinatges combinen fundadors,
prefix, telemetria nova i estat final.

`steps_executed` compta els passos nous d’una invocació. El resum recuperat
afegeix el pas inicial de la trajectòria, el pas inicial de l’execució,
l’horitzó, els passos heretats i els passos totals de la trajectòria. El mode
`experiment` admet recuperació. Els lots poden aturar el run actiu i marquen
els pendents com a `not_started`. La recuperació opera sobre un run individual.

## PostgreSQL i MLflow

Aplica la migració Alembic `0004_run_recovery` per registrar les continuacions
de runs vinculats. Configura la mateixa connexió PostgreSQL de l’origen amb
`EVOLUCIO_DATABASE_URL`. La taula `run_recovery` vincula cada execució amb
l’origen i l’arrel, valida la identitat i permet consultar els intents.
Les consultes comparatives identifiquen els intents de la mateixa trajectòria
i requereixen una tria quan hi ha diverses continuacions completes.

Si l’origen té seguiment MLflow, la continuació utilitza el mateix experiment
amb un nou identificador MLflow. Els tags identifiquen l’origen, l’arrel,
el checkpoint i la profunditat. La sincronització usa mètriques canòniques
amb passos absoluts i recupera el prefix mitjançant el Parquet materialitzat.
Si l’interval MLflow de l’origen era incompatible amb el mostreig Parquet, la
continuació usa la intersecció dels dos mostreigs. La inspecció ho indica.

Els estats MLflow corresponents són `FINISHED`, `FAILED` i `KILLED`.
La sincronització pot fallar independentment de l’execució científica. Les
escriptures al directori local, a PostgreSQL i a MLflow tenen fronteres de
confirmació pròpies. `db verify-run` i `mlflow verify` permeten detectar
diferències. `db sync-run` publica els índexs verificats i una aturada local.
`mlflow sync` reintenta el seguiment quan els registres canònics coincideixen.

## Exemple

Des d’un checkout net amb les dependències bloquejades:

```console
evolucio run tests/fixtures/config/valid_v1.json --steps 100 --chunk-steps 10 --checkpoint-every 20
# Prem Ctrl+C durant l'execució i conserva el run_dir que ha imprès la CLI.
evolucio recovery inspect <run-dir>
evolucio recovery resume <run-dir> --latest-recoverable --dry-run
evolucio recovery resume <run-dir> --latest-recoverable --chunk-steps 7
evolucio artifacts verify <new-run-dir>
```

Si la inspecció indica un `RUNNING` sense escriptor, executa
`evolucio recovery reconcile <run-dir>` abans de recuperar. Per seleccionar
un pas concret, usa `--checkpoint-step` en lloc de `--latest-recoverable`.
El dry-run llegeix i valida els artefactes sense crear cap run ni executar el
bucle de simulació. Els codis de sortida són `0` per èxit, `1` per error
d’execució, integritat o persistència, `2` per arguments o pla invàlids i
`130` per interrupció en els camins registrats de la CLI.
