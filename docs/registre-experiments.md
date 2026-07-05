# Projecte Evolució — Registre d'experiments

**Especificació funcional per garantir que cada execució guardi configuració, llavor aleatòria, versió del model, paràmetres principals, mètriques i artefactes rellevants.**

## Metadades

| Camp | Valor |
| --- | --- |
| Tipus de document | Especificació funcional del registre d'experiments |
| Punt del full de ruta | Punt 13 del pla funcional |
| Àmbit | Traçabilitat d'execucions, configuracions, llavors, versions, mètriques, artefactes i estat final |
| Objectiu | Garantir que cada simulació rellevant deixi una traça completa, ordenada i consultable. |
| Resultat esperat | Traçabilitat completa: qualsevol resultat important s'ha de poder revisar, comparar i, idealment, repetir. |
| Criteri de tancament | No hi ha experiments orfes: tota execució important queda identificada, documentada i vinculada als seus resultats. |
| Estat | Versió funcional per a revisió interna |
| Versió | 1.0 |
| Data | Juliol de 2026 |

## Índex

1. [Objectiu del registre d'experiments](#1-objectiu-del-registre-dexperiments)
2. [Principi rector](#2-principi-rector)
3. [Què es considera una execució experimental](#3-què-es-considera-una-execució-experimental)
4. [Jerarquia d'identificació](#4-jerarquia-didentificació)
5. [Informació obligatòria per execució](#5-informació-obligatòria-per-execució)
6. [Configuració congelada](#6-configuració-congelada)
7. [Gestió de llavors aleatòries](#7-gestió-de-llavors-aleatòries)
8. [Versions del model i del codi](#8-versions-del-model-i-del-codi)
9. [Mètriques registrades](#9-mètriques-registrades)
10. [Artefactes experimentals](#10-artefactes-experimentals)
11. [Persistència recomanada](#11-persistència-recomanada)
12. [Flux funcional de registre](#12-flux-funcional-de-registre)
13. [Estat de l'execució](#13-estat-de-lexecució)
14. [Validacions mínimes abans i després del run](#14-validacions-mínimes-abans-i-després-del-run)
15. [Repetibilitat i comparabilitat](#15-repetibilitat-i-comparabilitat)
16. [Exclusions temporals](#16-exclusions-temporals)
17. [Riscos principals](#17-riscos-principals)
18. [Resultat esperat](#18-resultat-esperat)
19. [Criteris funcionals de tancament](#19-criteris-funcionals-de-tancament)
20. [Decisió funcional recomanada](#20-decisió-funcional-recomanada)
21. [Definició de tancament](#21-definició-de-tancament)
22. [Annex. Checklist mínima per cada execució important](#22-annex-checklist-mínima-per-cada-execució-important)

## 1. Objectiu del registre d'experiments

L'objectiu del registre d'experiments és garantir que cada simulació rellevant del Projecte Evolució deixi una traça completa, ordenada i consultable. Una execució no s'ha de considerar vàlida només perquè produeixi gràfics o resultats aparentment interessants: ha de poder-se identificar, revisar, comparar i, quan sigui possible, repetir sota les mateixes condicions.

Aquest registre és una peça central del projecte, no una capa administrativa secundària. Sense registre rigorós, les diferències observades entre llinatges, configuracions o poblacions poden ser impossibles d'interpretar: podrien venir de la selecció ambiental, de la deriva, d'una llavor aleatòria concreta, d'un canvi de codi no documentat o d'un error de configuració.

## 2. Principi rector

El principi funcional és simple: **cap resultat important pot existir sense context experimental**. Qualsevol mètrica, genoma destacat, gràfic o conclusió ha d'estar vinculat a una execució concreta i a una configuració immutable.

Cada execució important ha d'incloure, com a mínim:

- un identificador únic d'execució;
- una configuració completa i congelada;
- una llavor aleatòria explícita;
- una versió del model i del codi;
- mètriques mínimes de població, energia, reproducció, mortalitat i diversitat;
- artefactes associats, com genomes, genealogies, sèries temporals o gràfics;
- estat final de l'execució: completada, fallida, interrompuda o descartada.

Aquesta disciplina evita experiments orfes i permet construir coneixement acumulatiu en lloc d'una col·lecció de proves inconnexes.

## 3. Què es considera una execució experimental

Una execució experimental és qualsevol simulació que es vulgui conservar per anàlisi, comparació o depuració. No totes les proves exploratòries han de tenir el mateix nivell de detall, però tota execució que pugui influir en decisions del projecte ha de quedar registrada.

| Tipus d'execució | Finalitat | Nivell de registre recomanat |
| --- | --- | --- |
| Exploratòria | Provar una idea, detectar errors o ajustar paràmetres inicials. | Registre bàsic: configuració, llavor, versió i estat final. |
| Experimental | Comparar configuracions o validar una hipòtesi funcional. | Registre complet: configuració, mètriques, artefactes i resum analític. |
| Referència | Servir com a base estable per comparar canvis futurs. | Registre complet i immutable, amb artefactes clau conservats. |
| Benchmark | Mesurar rendiment computacional o escalabilitat. | Registre complet més dades de temps, memòria i volum de dades. |
| Fallida significativa | Execució que col·lapsa, dona error o revela un problema important. | Registre de configuració, error, logs i estat parcial si existeix. |

## 4. Jerarquia d'identificació

El registre ha de distingir clarament entre projecte, experiment, execució i artefacte. Barrejar aquests nivells és una font habitual de confusió i impedeix comparar resultats de manera neta.

| Nivell | Què representa | Exemple funcional |
| --- | --- | --- |
| Projecte | El marc global de treball. | Projecte Evolució. |
| Experiment | Una pregunta o comparació concreta. | Comparar baixa i alta taxa de mutació. |
| Execució / run | Una simulació concreta amb configuració i llavor determinades. | Run 2026-07-05-001, mutació 0,01, seed 42. |
| Pas temporal | Un estat parcial dins una execució. | Step 10.000 amb població 732. |
| Agent o llinatge | Unitat individual o genealògica dins una execució. | Agent A-1932, fill de A-0871. |
| Artefacte | Fitxer o resultat vinculat a una execució. | `metrics.parquet`, `best_genomes.zarr`, `lineage_summary.csv`. |

## 5. Informació obligatòria per execució

Cada execució important ha de guardar un paquet mínim d'informació. Aquest paquet ha de ser suficient per entendre què s'ha executat, amb quin codi, amb quina configuració, amb quines dades inicials i amb quins resultats.

| Bloc d'informació | Contingut mínim | Raó funcional |
| --- | --- | --- |
| Identitat | `run_id`, `experiment_id`, nom curt, data d'inici, data de finalització i estat. | Permet trobar i citar una execució sense ambigüitat. |
| Configuració | Paràmetres complets de món, agents, recursos, energia, reproducció, mutació, mort, mètriques i durada. | Evita resultats sense context. |
| Aleatorietat | Llavor mestra i, si escau, derivació de subllavors per inicialització, mutació i entorn. | Permet repetir o auditar la font de variabilitat. |
| Versió | Versió del model, versió de l'esquema de dades, commit o identificador de codi. | Permet saber si dues execucions són realment comparables. |
| Inicialització | Distribució inicial de recursos, població inicial, genomes inicials o criteri de generació. | Permet interpretar l'arrencada de la dinàmica. |
| Mètriques | Sèries temporals i agregats finals de població, energia, naixements, morts, diversitat i llinatges. | Converteix la simulació en material analitzable. |
| Artefactes | Genomes destacats, genealogies, snapshots, gràfics, logs i resums. | Permet revisar resultats sense reexecutar-ho tot. |
| Execució tècnica | Durada, nombre de passos, errors, avisos, rendiment i consum aproximat de recursos. | Ajuda a depurar i dimensionar experiments futurs. |

## 6. Configuració congelada

La configuració d'una execució ha de quedar congelada en el moment d'arrencar. No n'hi ha prou amb guardar els paràmetres importants: cal conservar tot allò que pugui modificar el resultat.

La configuració congelada ha d'incloure:

- dimensions i límits del món 2D;
- regles de recursos i regeneració;
- nombre inicial d'agents i criteris de col·locació;
- estructura de la xarxa neuronal;
- rang i inicialització del genoma;
- observacions disponibles per als agents;
- espai d'accions;
- costos energètics i metabolisme basal;
- regles de reproducció i transferència d'energia;
- taxes, intensitat i tipus de mutació;
- causes de mort i llindars associats;
- canvis ambientals programats;
- freqüència de registre de mètriques;
- durada màxima de la simulació i criteris d'aturada anticipada.

La configuració ha de guardar-se com un document estructurat, preferiblement en YAML o JSON, i associar-se de manera immutable al `run_id`.

## 7. Gestió de llavors aleatòries

La llavor aleatòria és part de la definició de l'experiment. En un sistema evolutiu, dues execucions amb la mateixa configuració però llavors diferents poden divergir molt. Això no és un problema: és una propietat del sistema. El problema seria no saber quina llavor ha produït cada resultat.

| Element | Requisit funcional |
| --- | --- |
| Llavor mestra | Ha de quedar registrada sempre. |
| Subllavors | Han de derivar de manera sistemàtica per inicialització, entorn, mutació i desempats. |
| Repetició | Una mateixa configuració amb una mateixa llavor ha de donar el mateix resultat mentre no canviï el codi ni l'entorn d'execució. |
| Comparació | Els lots experimentals han d'utilitzar conjunts de llavors comparables quan es vulgui contrastar configuracions. |
| Depuració | Qualsevol error ha de poder vincular-se a la llavor que l'ha generat. |

En JAX, aquest punt és especialment important perquè la gestió de claus pseudoaleatòries ha de ser explícita. Funcionalment, el registre ha de permetre reconstruir l'arbre de generació aleatòria, encara que el detall tècnic es defineixi més endavant.

## 8. Versions del model i del codi

Cap execució hauria de quedar deslligada de la versió del sistema que l'ha produït. Una mètrica obtinguda abans i després d'un canvi en el metabolisme, en la xarxa o en l'ordre del bucle de simulació pot no ser comparable.

Cal registrar:

- versió funcional del model;
- versió de l'esquema de configuració;
- versió de l'esquema de mètriques;
- commit de Git o identificador equivalent;
- estat del repositori si hi ha canvis locals no incorporats;
- versions principals de dependències quan puguin afectar resultats;
- notes de migració quan un canvi trenqui compatibilitat amb experiments antics.

No cal sobrecarregar el prototip amb una infraestructura complexa des del primer dia, però sí cal evitar una situació en què no es pugui saber amb quin model es va generar un resultat.

## 9. Mètriques registrades

El registre d'experiments ha de separar les mètriques de seguiment temporal dels agregats finals. Les primeres expliquen la dinàmica; els segons faciliten la comparació entre execucions.

| Categoria | Exemples | Ús |
| --- | --- | --- |
| Població | Població viva, naixements, morts, taxa de creixement. | Detectar estabilitat, expansió o col·lapse. |
| Energia | Energia mitjana, mínima, màxima, distribució, consum total. | Avaluar pressió energètica i eficiència. |
| Recursos | Recursos totals, taxa de consum, zones exhaurides. | Mesurar competència indirecta i càrrega ambiental. |
| Reproducció | Descendència per agent, edat reproductiva, llinatges actius. | Quantificar èxit evolutiu. |
| Mort | Causes de mort, edat de mort, energia a la mort. | Entendre la pressió selectiva real. |
| Diversitat | Distància genètica aproximada, nombre de llinatges, dominància. | Distingir adaptació, deriva o col·lapse genètic. |
| Resposta ambiental | Canvi abans/després d'un xoc ambiental. | Avaluar adaptabilitat. |
| Rendiment | Temps per pas, agents per segon, memòria aproximada. | Dimensionar simulacions més grans. |

Un error habitual seria guardar només mètriques finals. Això pot ocultar col·lapses temporals, explosions poblacionals, períodes d'inestabilitat o dependència extrema d'una fase inicial.

## 10. Artefactes experimentals

Els artefactes són fitxers o objectes derivats d'una execució. No tots s'han de conservar sempre, però els més rellevants han d'estar vinculats al `run_id` i descrits en el registre.

| Artefacte | Quan conservar-lo | Comentari crític |
| --- | --- | --- |
| Configuració completa | Sempre. | És obligatòria; sense això el resultat queda orfe. |
| Mètriques temporals | Sempre en execucions experimentals. | Preferentment en Parquet per anàlisi posterior. |
| Resum final | Sempre. | Ha de permetre comparar runs sense obrir dades massives. |
| Genealogia | Sempre que hi hagi reproducció rellevant. | Necessària per estudiar llinatges i descendència. |
| Millors genomes | Quan es vulgui revisar o reexecutar llinatges destacats. | No substitueix la configuració ni les mètriques. |
| Snapshots de població | Només a intervals o en runs seleccionats. | Guardar-ho tot pot ser massa car. |
| Gràfics | Quan ajudin a revisió humana. | Són derivats, no dades primàries. |
| Logs i errors | Sempre que hi hagi avisos o fallades. | Una execució fallida també pot ser informativa. |

## 11. Persistència recomanada

El registre no ha de dependre d'un únic format. Cada tipus de dada té necessitats diferents. Forçar-ho tot dins PostgreSQL seria un error, especialment per snapshots massius o sèries temporals llargues.

| Capa | Ús recomanat | No hauria de contenir |
| --- | --- | --- |
| MLflow | Seguiment de runs, paràmetres, mètriques principals i artefactes. | Dades massives de simulació pas a pas sense necessitat. |
| PostgreSQL | Metadades consultables: runs, configuracions, genealogies resumides, índexs d'artefactes. | Arrays enormes de tots els agents a cada pas. |
| Parquet | Sèries temporals, mètriques per pas, taules d'agents, resums de llinatges. | Objectes molt jeràrquics difícils de tabular. |
| Zarr | Snapshots voluminosos o dades multidimensionals de simulació. | Petites taules relacionals que calgui consultar sovint. |
| Fitxers JSON/YAML | Configuració congelada i manifest d'artefactes. | Sèries temporals grans o dades d'alt volum. |

La decisió funcional és clara: metadades consultables a base de dades, dades massives en formats columnars o multidimensionals, i seguiment experimental centralitzat amb MLflow.

## 12. Flux funcional de registre

El registre ha d'estar integrat en el cicle de simulació, però sense contaminar la lògica evolutiva. La simulació genera dades; el sistema de registre les captura, les valida i les persisteix.

1. Crear un `run_id` abans d'inicialitzar la simulació.
2. Guardar la configuració congelada i la llavor aleatòria.
3. Registrar versió del model, codi i esquema de dades.
4. Inicialitzar món, agents i recursos segons la configuració.
5. Durant l'execució, recollir mètriques a la freqüència definida.
6. Guardar artefactes intermedis només si la configuració ho demana.
7. En acabar, registrar estat final, resum de mètriques i artefactes finals.
8. Si hi ha error, marcar el run com a fallit i conservar la informació disponible.
9. Fer que tots els fitxers resultants siguin localitzables a partir del `run_id`.

## 13. Estat de l'execució

Un registre seriós no només guarda èxits. També ha de capturar execucions fallides o interrompudes, perquè sovint revelen problemes de disseny o límits del sistema.

| Estat | Significat | Acció de registre |
| --- | --- | --- |
| `created` | El run ha estat creat però encara no ha començat. | Guardar identificador i configuració provisional si escau. |
| `running` | La simulació està en curs. | Registrar progrés i mètriques parcials segons freqüència. |
| `completed` | La simulació ha acabat correctament. | Guardar resum final i artefactes obligatoris. |
| `failed` | La simulació ha fallat per error tècnic o inconsistència. | Guardar error, pas afectat, configuració i dades parcials. |
| `stopped` | La simulació s'ha interromput manualment o per criteri d'aturada. | Guardar motiu i estat final disponible. |
| `discarded` | El run no s'utilitzarà per conclusions. | Mantenir-ne la traça mínima per evitar confusions futures. |

## 14. Validacions mínimes abans i després del run

El registre ha de fer validacions bàsiques per evitar resultats inconsistents. Aquestes validacions no han de substituir les proves del codi, però sí impedir que s'arxivin experiments clarament mal formats.

| Moment | Validació | Conseqüència si falla |
| --- | --- | --- |
| Abans | Configuració completa i compatible amb l'esquema. | No iniciar el run. |
| Abans | Llavor definida i registrada. | No iniciar el run o generar-ne una i guardar-la explícitament. |
| Abans | Versió del model i del codi identificable. | Marcar el run com a no reproduïble o bloquejar-lo. |
| Durant | Mètriques sense valors impossibles: població negativa, energia NaN, morts sense causa. | Marcar error o avisar segons gravetat. |
| Després | Artefactes obligatoris existents i accessibles. | Marcar el run com a incomplet. |
| Després | Resum final coherent amb les sèries temporals. | Marcar inconsistència i revisar. |

## 15. Repetibilitat i comparabilitat

Cal distingir repetibilitat de comparabilitat. Repetir vol dir poder obtenir el mateix resultat amb la mateixa configuració, llavor i versió. Comparar vol dir poder interpretar diferències entre execucions diferents.

- Per repetir: mateixa configuració, mateixa llavor, mateix codi, mateixa versió de dependències rellevants.
- Per comparar: configuracions documentades, mètriques equivalents, mateixa definició d'agregats i lots de llavors comparables.
- Per concloure adaptació: no n'hi ha prou amb un run favorable; cal comparar contra controls i diverses llavors.
- Per estudiar llinatges: cal conservar genealogia, no només població total.

Aquest punt és important: un resultat espectacular en una sola execució pot ser només una anècdota estadística. El registre ha de facilitar lots d'experiments i no només captures individuals.

## 16. Exclusions temporals

Per al prototip inicial, no cal implementar una plataforma completa de gestió científica. Algunes funcionalitats poden esperar fins que el motor bàsic sigui estable.

- quadres de comandament web avançats;
- sistema multiusuari amb permisos detallats;
- execució distribuïda automàtica;
- pipeline complet de publicació de resultats;
- interfície gràfica per explorar genealogies massives;
- versionat complex de datasets amb deduplicació avançada;
- auditoria criptogràfica completa dels artefactes;
- reproducció exacta garantida entre maquinari diferent.

Això no vol dir ignorar la traçabilitat. Vol dir implementar primer el mínim que impedeixi perdre el control dels experiments.

## 17. Riscos principals

| Risc | Impacte | Mesura preventiva |
| --- | --- | --- |
| Experiments orfes | Resultats sense configuració ni llavor; impossibles de revisar. | `run_id` obligatori i configuració congelada abans d'executar. |
| Guardar massa poc | No es pot distingir adaptació de soroll. | Mètriques temporals, genealogia i causes de mort. |
| Guardar massa | Cost d'emmagatzematge i anàlisi innecessari. | Freqüències de snapshot configurables i formats adequats. |
| Versions barrejades | Comparacions falses entre models diferents. | Registrar versió de model, codi i esquema. |
| Dependència d'un gràfic | Conclusions basades en visualització, no en dades. | Conservar dades primàries i resums tabulars. |
| Mètriques incompatibles | No es poden comparar runs antics i nous. | Versionar l'esquema de mètriques. |
| Runs fallits invisibles | Es repeteixen errors sense saber-ne la causa. | Registrar estats `failed` i logs mínims. |

## 18. Resultat esperat

El resultat esperat és un sistema on cada execució important del Projecte Evolució generi automàticament un registre complet. Aquest registre ha de permetre reconstruir què s'ha provat, com s'ha inicialitzat, quins paràmetres s'han utilitzat, quina versió del sistema ho ha produït i quins resultats se n'han derivat.

A la pràctica, l'equip hauria de poder seleccionar qualsevol gràfic, mètrica o genoma destacat i respondre ràpidament: de quin run surt, amb quina configuració, amb quina llavor, amb quin codi, amb quins artefactes associats i amb quines limitacions.

## 19. Criteris funcionals de tancament

1. Cada execució important rep un identificador únic abans de començar.
2. La configuració completa queda guardada i associada al `run_id`.
3. La llavor aleatòria queda registrada de manera explícita.
4. La versió del model, del codi i de l'esquema de dades queda identificada.
5. Les mètriques mínimes es registren durant l'execució i al final.
6. Els artefactes rellevants es guarden i són localitzables.
7. Les execucions fallides també deixen registre.
8. Cap gràfic o conclusió experimental queda separat del run que l'ha generat.
9. Es poden comparar execucions per configuració, llavor, versió i mètriques.
10. L'equip pot repetir o auditar una execució rellevant sense dependre de memòria informal.

## 20. Decisió funcional recomanada

La decisió recomanada per al prototip és implementar un registre híbrid i pragmàtic: MLflow per seguir execucions, PostgreSQL per metadades i consultes, Parquet o Zarr per dades massives, i fitxers YAML o JSON per configuracions congelades. Aquesta separació és més robusta que intentar guardar-ho tot en una sola eina.

El registre inicial no ha de ser sofisticat, però sí obligatori. És preferible tenir pocs camps ben definits i sempre presents que una estructura ambiciosa però aplicada de manera irregular.

## 21. Definició de tancament

Aquest punt es considerarà complet quan el projecte disposi d'una especificació funcional acceptada del registre d'experiments i quan el disseny posterior pugui implementar-la sense ambigüitat. El criteri pràctic és que ja no hi hagi experiments importants sense identitat, configuració, llavor, versió, mètriques i artefactes vinculats.

A partir d'aquest document, el següent pas tècnic serà definir l'esquema concret de configuració, el model de dades mínim, els noms dels artefactes i la integració del registre dins el bucle principal de simulació.

## 22. Annex. Checklist mínima per cada execució important

| Element | Obligatori | Comentari |
| --- | --- | --- |
| `run_id` | Sí | Identificador únic i estable. |
| `experiment_id` | Sí | Agrupa execucions comparables. |
| `config.yaml/json` | Sí | Configuració completa i congelada. |
| `seed` | Sí | Llavor mestra registrada. |
| `model_version` | Sí | Versió funcional del model. |
| `code_version` | Sí | Commit o identificador equivalent. |
| `metrics_summary` | Sí | Resum final comparable. |
| `metrics_timeseries` | Sí, per experiments | Sèrie temporal per analitzar dinàmica. |
| `lineage_data` | Sí, si hi ha reproducció | Necessari per estudiar selecció de llinatges. |
| `best_genomes` | Recomanat | Útil per reexecució i inspecció. |
| `logs` | Sí, si hi ha avisos/errors | Especialment important en runs fallits. |
| `artifact_manifest` | Sí | Llista d'artefactes i ubicacions. |
| `run_status` | Sí | `completed`, `failed`, `stopped` o `discarded`. |
