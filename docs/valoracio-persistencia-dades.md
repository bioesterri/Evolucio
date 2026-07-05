# Valoració d'adequació — Persistència de dades

## Objectiu de la valoració

Aquest document comprova si l'especificació **Persistència de dades** s'adequa a la resta de la documentació funcional del Projecte Evolució i identifica punts forts, encaixos, riscos i recomanacions d'implementació.

## Veredicte resumit

La valoració és **molt positiva**: el document encaixa de manera natural amb la resta del pla funcional perquè converteix els resultats evolutius en dades duradores, consultables i relacionades amb el seu context experimental. La proposta resol el pas posterior al registre d'experiments: no només identifica què s'ha executat, sinó que defineix on viuen les dades, com s'indexen i quines parts han de sobreviure al final de la simulació.

També és coherent amb l'enfocament pragmàtic del prototip. Recomana PostgreSQL per a dades relacionals i consultables, però evita saturar-lo amb sèries temporals massives, que deriva cap a Parquet o Zarr. Aquesta separació és adequada per al tipus de simulació prevista: agents, genealogies, genomes, mètriques i snapshots poden créixer a ritmes molt diferents.

## Encaix amb la documentació existent

| Document relacionat | Encaix observat | Valoració |
| --- | --- | --- |
| `document-funcional-inicial.md` | Dona continuïtat al full de ruta funcional i concreta el punt 14 com a capa que evita la pèrdua de coneixement acumulable. | Molt coherent: transforma el criteri general en un contracte de dades. |
| `arquitectura-general.md` | Respecta la separació entre motor de simulació, persistència, anàlisi i visualització. | Molt coherent: la persistència observa i conserva, però no decideix el comportament de la simulació. |
| `abast-prototip-inicial.md` | Manté una primera versió robusta però limitada, sense data warehouse ni interfície avançada prematura. | Coherent: evita infraenginyeria abans de validar volum i preguntes analítiques. |
| `bucle-principal-simulacio.md` | Defineix punts d'integració clars: iniciar run, congelar configuració, registrar mètriques i esdeveniments, tancar estat i conservar errors. | Molt coherent: encaixa amb el cicle temporal sense dominar-lo. |
| `registre-experiments.md` | Diferencia registre i persistència: el registre identifica l'execució; la persistència conserva dades consultables i artefactes indexats. | Molt coherent: és el pas funcional següent i evita duplicar responsabilitats. |
| `valoracio-registre-experiments.md` | Recull la recomanació de preparar una capa híbrida amb dades estructurades i formats massius. | Molt coherent: materialitza la persistència híbrida suggerida. |
| `metriques-evolutives.md` | Proporciona el suport perquè les mètriques agregades, temporals i de llinatge puguin consultar-se després del run. | Molt coherent: converteix mètriques en dades analitzables, no logs dispersos. |
| `reproduccio-herencia-mutacio.md` | Dona persistència a genealogies, mutacions, descendència i genomes destacats. | Molt coherent: preserva l'evidència de selecció acumulativa. |
| `genoma-xarxa-neuronal.md` | Exigeix identificadors i hashes de genoma, vinculació amb agents i recuperació de genomes fundadors o representatius. | Coherent: reforça reexecució, comparació i auditoria genètica. |
| `mort-seleccio-ambiental.md` | Inclou causes de mort i esdeveniments vitals, essencials per interpretar pressions selectives. | Coherent: evita mètriques de mortalitat sense causa o context. |
| `energia-recursos-metabolisme.md` | Permet persistir mètriques d'energia, recursos i economia metabòlica com a sèries temporals o resums. | Coherent: fa auditable la dinàmica energètica. |
| `observacions-agents.md` i `accions-agents.md` | Tracta observacions i accions com a dades massives opcionals, no com a persistència obligatòria. | Coherent: evita un volum excessiu per defecte. |

## Punts forts

- **Contracte de persistència clar:** diferencia experiments, runs, configuracions, agents, genealogies, mètriques, genomes i artefactes.
- **Separació encertada de formats:** proposa PostgreSQL per a metadades relacionals i Parquet/Zarr per a dades massives.
- **Traçabilitat de context:** tota dada important queda vinculada a `run_id`, configuració, seed, versió i artefactes.
- **Protecció de genealogia:** declara que els enllaços progenitor-descendent no s'han de mostrejar perquè trencarien l'arbre evolutiu.
- **Gestió del volum:** distingeix persistència mínima, analítica estàndard i massiva opcional.
- **Recuperació i diagnòstic:** inclou checkpoints, estat parcial i informació d'error per a execucions fallides.
- **Integritat de dades:** especifica validacions de referència, unicitat, coherència temporal, coherència genealògica i hashes.
- **Orientació a consultes:** valida el disseny a partir de preguntes analítiques concretes, com descendència per llinatge, col·lapse poblacional o recuperació de genomes destacats.

## Riscos o punts a vigilar

1. **Cost operacional de PostgreSQL en el prototip.**
   PostgreSQL és una bona direcció funcional, però pot afegir fricció si el prototip encara és local. Es pot mantenir el mateix esquema conceptual amb SQLite o fitxers estructurats inicials, sempre que no es perdi la compatibilitat futura.

2. **Definició dels límits de dades massives.**
   El document estableix que snapshots, trajectòries, observacions i accions han de ser configurables, però caldrà fixar valors per defecte perquè el sistema no generi volums inesperats.

3. **Esquemes i migracions.**
   Si les taules i formats Parquet/Zarr evolucionen, caldrà versionar esquemes i definir migracions o lectors compatibles per no perdre analitzabilitat històrica.

4. **Hash i identitat de genomes.**
   El hash del genoma ha de calcular-se sobre una serialització canònica. Si no, dues representacions equivalents podrien generar identificadors diferents o dificultar la deduplicació.

5. **Checkpoints realment reproduïbles.**
   El document detecta correctament el risc del generador aleatori. La implementació haurà de garantir que es conserva l'estat complet necessari per reprendre sense canviar la trajectòria.

6. **Risc de doble font de veritat.**
   Si els manifests del registre d'experiments i la base estructurada guarden camps semblants, caldrà definir quin component és autoritatiu per a cada dada i com es sincronitzen.

## Recomanacions d'implementació

- Definir un esquema mínim de base relacional amb `experiment`, `run`, `config_snapshot`, `agent_summary`, `genealogy_edge`, `life_event`, `metric_series_index`, `genome_registry` i `artifact`.
- Crear una interfície de persistència desacoblada del motor de simulació, de manera que el bucle emeti esdeveniments i la capa de persistència decideixi com escriure'ls.
- Implementar escriptura per lots per a mètriques i esdeveniments freqüents, evitant una operació de base de dades per cada microesdeveniment.
- Fer que tota dada externa tingui una entrada a `artifact` o `metric_series_index` amb ruta, format, hash, mida i `run_id`.
- Fixar una política per defecte: mètriques agregades i vida/genealogia sempre; snapshots i observacions només per intervals o mode depuració.
- Versionar explícitament l'esquema de base de dades, l'esquema de mètriques i el format de serialització de genomes.
- Afegir validacions post-run que detectin agents sense run, fills sense progenitor vàlid, morts abans del naixement, genomes sense hash o artefactes no verificables.
- Permetre que una execució fallida quedi en estat `failed` amb configuració, seed, step afectat, error i artefactes parcials disponibles.

## Conclusió

El document **Persistència de dades** és adequat per incorporar-se a la documentació funcional del Projecte Evolució com a punt 14. Complementa el registre d'experiments i dona una base sòlida perquè els resultats importants sobrevisquin al final de la simulació, siguin consultables i puguin alimentar anàlisis posteriors.

La recomanació és acceptar-lo com a especificació funcional inicial i utilitzar-lo com a base per dissenyar l'esquema tècnic, començant per una implementació mínima però estricta: identificació de runs, configuració immutable, agents resumits, genealogia, mètriques agregades, genomes destacats i índex central d'artefactes.
