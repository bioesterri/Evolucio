# Projecte Evolució — Persistència de dades

**Especificació funcional per decidir quines dades es guarden de manera estructurada: experiments, agents, genealogies, mètriques, millors genomes i sèries temporals massives.**

## Metadades

| Camp | Valor |
| --- | --- |
| Tipus de document | Especificació funcional de persistència de dades |
| Punt del full de ruta | Punt 14 del pla funcional |
| Àmbit | Experiments, execucions, configuracions, agents, genealogies, mètriques, genomes destacats, sèries temporals massives, checkpoints i artefactes |
| Objectiu | Garantir que la informació rellevant generada per les simulacions no desaparegui quan acaba una execució i pugui ser consultada, comparada i analitzada posteriorment. |
| Resultat esperat | Una base de dades funcional per consultar resultats sense dependre de fitxers dispersos o memòria temporal. |
| Criteri de tancament | Els resultats rellevants sobreviuen al final de la simulació i poden ser analitzats posteriorment. |
| Estat | Versió funcional per a revisió interna |
| Versió | 1.0 |
| Data | Juliol de 2026 |

## Índex

1. [Objectiu de la persistència de dades](#1-objectiu-de-la-persistència-de-dades)
2. [Principi rector](#2-principi-rector)
3. [Abast funcional de la persistència](#3-abast-funcional-de-la-persistència)
4. [Nivells de persistència](#4-nivells-de-persistència)
5. [Dades estructurades principals](#5-dades-estructurades-principals)
6. [Dades massives i sèries temporals](#6-dades-massives-i-sèries-temporals)
7. [Genomes i millors individus](#7-genomes-i-millors-individus)
8. [Genealogia persistent](#8-genealogia-persistent)
9. [Política de captura temporal](#9-política-de-captura-temporal)
10. [Flux funcional de persistència](#10-flux-funcional-de-persistència)
11. [Checkpoints i recuperació](#11-checkpoints-i-recuperació)
12. [Integritat i qualitat de dades](#12-integritat-i-qualitat-de-dades)
13. [Consultes que ha de permetre el sistema](#13-consultes-que-ha-de-permetre-el-sistema)
14. [Relació amb registre d'experiments i anàlisi](#14-relació-amb-registre-dexperiments-i-anàlisi)
15. [Exclusions temporals](#15-exclusions-temporals)
16. [Riscos principals](#16-riscos-principals)
17. [Criteris funcionals d'èxit](#17-criteris-funcionals-dèxit)
18. [Decisió d'abast recomanada per al prototip](#18-decisió-dabast-recomanada-per-al-prototip)
19. [Definició de tancament](#19-definició-de-tancament)

## 1. Objectiu de la persistència de dades

L'objectiu de la persistència de dades és garantir que la informació rellevant generada pel Projecte Evolució no desaparegui quan acaba una simulació i pugui ser consultada, comparada i analitzada posteriorment. La simulació no ha de dependre de fitxers dispersos, captures puntuals o memòria temporal per conservar resultats importants.

Aquest component ha de transformar les execucions en dades estructurades. Això inclou informació d'experiments, configuracions, agents, genealogies, mètriques agregades, genomes destacats i sèries temporals massives. Sense aquesta capa, el projecte pot generar observacions interessants però no coneixement acumulable.

## 2. Principi rector

La persistència ha de ser **selectiva, estructurada i escalable**. No tot el que passa a cada pas temporal s'ha de guardar en una base de dades relacional. Guardar-ho tot indiscriminadament seria car, lent i difícil d'analitzar. El criteri correcte és conservar allò que permet reconstruir, explicar i comparar els resultats.

- Les dades petites, relacionals i consultables han de quedar en una base estructurada.
- Les dades massives i temporals s'han de guardar en formats analítics optimitzats.
- Els artefactes grans han d'estar vinculats a una execució, però no necessàriament incrustats a la base principal.
- Cada dada important ha de tenir una identitat clara i una relació amb experiment, execució i configuració.
- Cap genoma, mètrica o genealogia rellevant ha de quedar desconnectat del context que l'ha generat.

**Decisió recomanada:** separar PostgreSQL per a dades estructurades i consultables, i Parquet o Zarr per a sèries temporals massives. Intentar posar-ho tot a PostgreSQL des del primer dia seria una mala decisió: funcionaria amb proves petites, però es convertiria en un coll d'ampolla quan la població i els passos temporals creixessin.

## 3. Abast funcional de la persistència

La persistència ha de cobrir el cicle complet de dades d'una execució: inici, progrés, finalització, consulta i anàlisi posterior. No ha de limitar-se a guardar un resultat final.

| Bloc de dades | Què ha de conservar | Finalitat |
| --- | --- | --- |
| Experiments | Nom, objectiu, hipòtesi funcional, data, estat i notes. | Agrupar execucions comparables sota una pregunta comuna. |
| Execucions | Identificador, configuració, llavor, versió del model, inici, final i estat. | Saber exactament què s'ha executat. |
| Configuracions | Paràmetres del món, agents, energia, mutació, mètriques i registre. | Permetre revisió i repetició de condicions. |
| Agents | Identificador, naixement, mort, progenitor, genoma i resum vital. | Seguir individus i llinatges. |
| Genealogies | Relació progenitor-descendent i esdeveniments reproductius. | Analitzar èxit reproductiu i persistència de llinatges. |
| Mètriques | Sèries agregades de població, energia, recursos, morts, naixements i diversitat. | Comparar dinàmiques evolutives. |
| Genomes destacats | Millors genomes, genomes fundadors, genomes finals i genomes representatius. | Reexecutar, comparar o visualitzar llinatges. |
| Artefactes | Fitxers grans, gràfics, snapshots, informes, logs i exportacions. | Connectar resultats analítics amb materials derivats. |

## 4. Nivells de persistència

No totes les dades tenen el mateix valor ni el mateix volum. El sistema ha de distingir tres nivells de persistència per evitar tant la pèrdua d'informació com l'acumulació inútil de dades.

| Nivell | Dades incloses | Ús esperat |
| --- | --- | --- |
| Persistència mínima obligatòria | Experiment, execució, configuració, llavor, versió, mètriques finals i estat. | Garantir traçabilitat mínima de qualsevol execució rellevant. |
| Persistència analítica estàndard | Mètriques per pas o per intervals, agents nascuts/morts, genealogia, causes de mort i genomes destacats. | Permetre comparació evolutiva i reconstrucció de llinatges. |
| Persistència massiva opcional | Snapshots de població, mapes de recursos, trajectòries, observacions i accions per agent. | Anàlisi avançada, depuració profunda o visualitzacions posteriors. |

El nivell massiu no ha d'estar activat sempre. Pot multiplicar molt el volum de dades i alentir execucions. S'ha d'activar per mostreig, per intervals o per experiments específics.

## 5. Dades estructurades principals

La base estructurada ha de contenir la informació que necessita consultes freqüents, relacions consistents i integritat referencial. En aquesta fase, la decisió funcional recomanada és usar PostgreSQL com a font principal de dades relacionals.

| Entitat funcional | Contingut mínim | Relacions clau |
| --- | --- | --- |
| `experiment` | Nom, descripció, objectiu, etiqueta, estat i data de creació. | Conté múltiples execucions. |
| `run` | Identificador, experiment, seed, versió, estat, inici, final i durada. | Pertany a un experiment i genera mètriques, agents i artefactes. |
| `config_snapshot` | Configuració completa serialitzada i hash de configuració. | Pertany a una execució i ha de ser immutable. |
| `agent_summary` | Agent, run, progenitor, `genome_id`, `birth_step`, `death_step`, causa de mort, descendència i energia final. | Connecta agent, genoma i genealogia. |
| `genealogy_edge` | `parent_id`, `child_id`, `birth_step` i `event_id`. | Permet reconstruir arbres de descendència. |
| `life_event` | Naixement, mort, reproducció, mutació significativa o altre esdeveniment rellevant. | Connecta agents amb passos temporals. |
| `metric_series_index` | Nom de mètrica, freqüència, ubicació física i període cobert. | Apunta a dades agregades o fitxers massius. |
| `genome_registry` | Identificador de genoma, hash, origen, mida, paràmetres heretables i ubicació. | Permet localitzar i comparar genomes. |
| `artifact` | Tipus, ruta, hash, format, mida i run associat. | Vincula fitxers externs amb l'execució. |

## 6. Dades massives i sèries temporals

Les sèries temporals massives no s'han de tractar igual que les metadades. Guardar cada posició, energia, observació i acció de cada agent a cada pas dins una taula relacional pot fer que el sistema sigui innecessàriament lent i difícil de mantenir.

La solució funcional és guardar les dades massives en fitxers columnars o arrays persistents, vinculats a la base estructurada mitjançant identificadors, rutes i hashes.

| Tipus de dada massiva | Format recomanat | Granularitat recomanada |
| --- | --- | --- |
| Mètriques agregades per pas | Parquet | Cada pas o cada N passos, segons cost. |
| Estat complet de població | Parquet o Zarr | Snapshots periòdics, no necessàriament cada pas. |
| Mapes de recursos | Zarr o Parquet | Intervals configurables o moments clau. |
| Trajectòries d'agents | Parquet | Mostreig temporal o subconjunt d'agents. |
| Observacions i accions | Parquet | Només per depuració o experiments específics. |
| Genomes complets de molts agents | Parquet, Zarr o fitxers binaris amb índex | Només si cal anàlisi genètica detallada. |

**Criteri de contenció:** per defecte, cal guardar mètriques agregades i esdeveniments de vida. Els snapshots complets han de ser configurables. Una simulació llarga amb molts agents pot generar moltíssimes dades; persistir-ho tot seria més un problema que una virtut.

## 7. Genomes i millors individus

Els genomes són una peça especialment important perquè permeten reexecutar agents, comparar llinatges i analitzar si hi ha acumulació d'adaptacions. Tot i així, no cal guardar còpies completes redundants si molts genomes són idèntics o gairebé idèntics.

- Cada genoma rellevant ha de tenir un identificador estable.
- El genoma ha de tenir un hash per detectar duplicats o canvis accidentals.
- S'ha de guardar la seva relació amb l'agent, el progenitor i la mutació que l'ha generat.
- Els genomes destacats s'han de conservar encara que la resta de dades massives es redueixin.
- Els genomes fundadors i els genomes finals de llinatges importants han de ser recuperables.

| Categoria de genoma | Quan es guarda | Raó funcional |
| --- | --- | --- |
| Fundador | En inicialitzar la població. | Permet reconstruir l'origen de l'experiment. |
| Descendent mutat | Quan neix un agent amb mutacions registrables. | Permet seguir canvis hereditaris. |
| Millor per descendència | Quan un llinatge destaca reproductivament. | Permet estudiar èxit evolutiu. |
| Millor per supervivència | Quan un agent o llinatge supera llindars definits. | Permet estudiar persistència. |
| Final representatiu | Al tancament de l'execució. | Permet comparar poblacions finals. |

## 8. Genealogia persistent

La genealogia no és un detall decoratiu: és una de les fonts principals per saber si hi ha selecció acumulativa. Per això la relació entre progenitor i descendent ha de quedar registrada de manera persistent i consultable.

Cada naixement ha de generar un enllaç genealògic que permeti reconstruir com a mínim la línia directa d'ascendència i, idealment, l'arbre complet de descendència dins una execució.

| Camp funcional | Descripció | Ús |
| --- | --- | --- |
| `run_id` | Execució on es produeix el naixement. | Evita barrejar genealogies de diferents execucions. |
| `parent_agent_id` | Identificador del progenitor. | Permet reconstruir ascendència. |
| `child_agent_id` | Identificador del descendent. | Permet seguir el nou individu. |
| `birth_step` | Pas temporal del naixement. | Situa l'esdeveniment en la dinàmica temporal. |
| `parent_genome_id` | Genoma del progenitor abans de mutació. | Permet comparar herència. |
| `child_genome_id` | Genoma del descendent després de mutació. | Permet analitzar variació. |
| `mutation_summary` | Resum o referència de mutacions aplicades. | Permet estudiar canvis heretables. |

## 9. Política de captura temporal

La simulació produeix dades en cada pas temporal. Però la persistència ha de decidir amb quina freqüència es guarden. Aquesta política ha de ser explícita i formar part de la configuració de cada execució.

| Dada | Freqüència recomanada | Comentari |
| --- | --- | --- |
| Mètriques agregades | Cada pas o cada N passos petit. | Són barates i útils per detectar col·lapses o tendències. |
| Naixements i morts | Sempre. | Són esdeveniments evolutius centrals. |
| Genealogia | Sempre que hi hagi reproducció. | No s'ha de mostrejar perquè es trencaria l'arbre. |
| Genomes destacats | Segons criteris definits. | No cal guardar tots els genomes complets si el volum és alt. |
| Snapshots de població | Cada N passos o en moments clau. | Útils per visualització i anàlisi posterior. |
| Observacions i accions | Només en mode depuració o mostreig. | Poden créixer massa ràpidament. |

## 10. Flux funcional de persistència

La persistència ha d'estar integrada al bucle de simulació sense dominar-lo. El motor de simulació ha de generar dades i esdeveniments; la capa de persistència els ha de rebre, validar i escriure segons la política configurada.

1. Crear l'experiment o associar l'execució a un experiment existent.
2. Crear el registre d'execució amb seed, versió i estat inicial.
3. Guardar la configuració completa com a snapshot immutable.
4. Inicialitzar índexs d'agents fundadors, genomes inicials i condicions del món.
5. Durant la simulació, registrar mètriques agregades, naixements, morts, genealogies i artefactes segons política.
6. Guardar checkpoints si l'execució és llarga o costosa.
7. En finalitzar, marcar l'estat de l'execució i guardar resum final.
8. Si hi ha error, conservar configuració, logs i estat parcial suficient per diagnosticar-lo.

## 11. Checkpoints i recuperació

Els checkpoints són necessaris quan les simulacions són llargues, costoses o generen poblacions grans. No substitueixen les mètriques ni els resultats finals, però permeten recuperar una execució interrompuda o analitzar un estat intermedi.

- Un checkpoint ha de contenir l'estat mínim necessari per reprendre la simulació.
- Ha d'incloure estat del món, agents vius, recursos, generador aleatori i pas temporal.
- Ha d'estar vinculat al `run_id` i a la configuració original.
- Ha de tenir hash o comprovació d'integritat.
- No cal conservar tots els checkpoints indefinidament; pot aplicar-se una política de retenció.

**Risc tècnic:** un checkpoint incomplet pot donar una falsa sensació de seguretat. Si no conserva també l'estat del generador aleatori, reprendre una simulació pot produir una trajectòria diferent encara que la configuració sigui la mateixa.

## 12. Integritat i qualitat de dades

La persistència no ha de limitar-se a escriure dades. També ha de protegir-ne la coherència. Una base plena de dades inconsistents és pitjor que una base petita però fiable.

| Validació | Què comprova | Conseqüència si falla |
| --- | --- | --- |
| Integritat referencial | Cap agent, genoma o mètrica existeix sense run associat. | L'execució queda marcada com a inconsistent. |
| Unicitat d'identificadors | No hi ha agents o genomes duplicats dins un run. | S'atura o es rebutja l'escriptura conflictiva. |
| Coherència temporal | Naixement abans de mort, steps ordenats i mètriques dins rang. | Es registra error de consistència. |
| Coherència genealògica | Cap fill sense progenitor vàlid, excepte fundadors. | Es marca genealogia incompleta. |
| Hash de configuració | La configuració no canvia durant l'execució. | Es crea una nova execució o s'invalida el run. |
| Hash d'artefactes | Els fitxers vinculats no han canviat accidentalment. | Es marca l'artefacte com a no verificat. |

## 13. Consultes que ha de permetre el sistema

El disseny de persistència s'ha de validar pensant en les preguntes que el projecte haurà de respondre. Si una pregunta bàsica requereix reconstruccions manuals o fitxers solts, la persistència no està ben resolta.

- Quines execucions comparteixen la mateixa configuració excepte un paràmetre concret?
- Quin llinatge ha tingut més descendència dins una execució?
- Quins genomes finals provenen del mateix fundador?
- En quin pas temporal ha col·lapsat una població i amb quina causa principal de mort?
- Quines execucions mostren més estabilitat poblacional?
- Quina taxa de mutació ha produït més diversitat sense col·lapse?
- Quins genomes destacats es poden recuperar per reexecutar-los?
- Quins artefactes corresponen a una figura o conclusió concreta?

## 14. Relació amb registre d'experiments i anàlisi

La persistència de dades i el registre d'experiments són components relacionats però no idèntics. El registre identifica l'execució, la seva configuració i els seus resultats principals. La persistència garanteix que les dades generades queden guardades en estructures consultables i duradores.

| Component | Responsabilitat | No ha de fer |
| --- | --- | --- |
| Registre d'experiments | Identificar execucions, configuracions, versions, mètriques resum i artefactes. | Guardar totes les dades massives de simulació. |
| Persistència estructurada | Guardar entitats relacionals: runs, agents, genealogies, genomes, índexs i mètriques resum. | Substituir el motor de simulació o decidir comportaments. |
| Persistència massiva | Guardar sèries temporals, snapshots i dades d'alt volum. | Convertir-se en l'única font de metadades. |
| Anàlisi | Llegir dades persistides i generar resultats interpretables. | Modificar dades originals sense traça. |

## 15. Exclusions temporals

Per mantenir la primera versió controlable, no cal implementar encara totes les capacitats possibles de dades. Algunes funcionalitats serien útils més endavant, però ara afegirien complexitat prematura.

- No cal un data warehouse complet.
- No cal una interfície web de consulta avançada.
- No cal versionat distribuït de datasets.
- No cal guardar totes les observacions i accions de tots els agents en totes les execucions.
- No cal compressió o particionat sofisticat si encara no hi ha volum real que ho exigeixi.
- No cal un sistema de permisos complex si el projecte encara és local o d'equip reduït.
- No cal optimitzar consultes massives abans de definir quines preguntes analítiques són prioritàries.

## 16. Riscos principals

| Risc | Descripció | Resposta recomanada |
| --- | --- | --- |
| Guardar massa | Persistir-ho tot pot alentir la simulació i generar dades intractables. | Definir nivells de persistència i mostreig. |
| Guardar massa poc | Sense genealogia, genomes o mètriques suficients no es pot analitzar evolució. | Fixar un mínim obligatori no negociable. |
| Fitxers dispersos | Els resultats poden quedar desconnectats de l'execució. | Registrar rutes, hashes i `run_id` de cada artefacte. |
| Dependència de memòria temporal | Les dades es perden en acabar o fallar la simulació. | Persistència incremental i checkpoints. |
| Inconsistència de dades | Agents, genomes o mètriques no encaixen entre si. | Validacions i integritat referencial. |
| Coll d'ampolla d'escriptura | La persistència pot fer més lenta la simulació. | Escriptura per lots i formats massius fora de PostgreSQL. |

## 17. Criteris funcionals d'èxit

La implementació de persistència es considerarà funcionalment correcta quan compleixi els criteris següents:

1. Cada execució rellevant queda registrada amb identificador únic i configuració associada.
2. Els agents, naixements, morts i relacions genealògiques principals sobreviuen al final de la simulació.
3. Les mètriques agregades es poden consultar posteriorment sense rellegir logs manuals.
4. Els genomes destacats es poden recuperar i vincular amb agents i llinatges.
5. Els artefactes grans tenen ruta, format, hash i run associat.
6. Una execució fallida deixa informació suficient per entendre on i per què ha fallat.
7. Les dades massives no saturen la base relacional principal.
8. Les consultes bàsiques d'anàlisi es poden respondre amb dades persistides.
9. Les dades tenen controls mínims de coherència i integritat.
10. Els resultats importants no depenen de fitxers solts sense context.

## 18. Decisió d'abast recomanada per al prototip

Per al prototip inicial, la persistència ha de ser suficientment robusta però no excessiva. La recomanació és implementar una base estructurada per a metadades, genealogies, agents resumits, genomes destacats i índexs d'artefactes, i reservar Parquet o Zarr per a sèries temporals i snapshots de volum alt.

- **PostgreSQL:** experiments, runs, configuracions, agents resumits, genealogies, mètriques resum, genomes destacats i artefactes.
- **Parquet:** mètriques temporals, trajectòries mostrejades, esdeveniments i taules analítiques.
- **Zarr:** estats grans de món o població si calen snapshots multidimensionals.
- **Fitxers d'artefactes:** gràfics, informes, exports, checkpoints i genomes serialitzats grans.
- **Índex central:** tota peça externa ha d'estar referenciada des de la base estructurada.

## 19. Definició de tancament

Aquest punt es considerarà tancat quan existeixi una especificació acceptada que defineixi quines dades es guarden, on es guarden, amb quina granularitat i com es relacionen amb experiments i execucions.

El criteri pràctic és que una execució important pugui acabar, desaparèixer de memòria i continuar sent analitzable posteriorment. Si cal reconstruir resultats a mà a partir de fitxers dispersos, captures o logs informals, la persistència encara no està resolta.
