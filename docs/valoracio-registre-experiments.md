# Valoració d'adequació — Registre d'experiments

## Objectiu de la valoració

Aquest document comprova si l'especificació **Registre d'experiments** s'adequa a la resta de la documentació funcional del Projecte Evolució i identifica punts forts, encaixos, riscos i recomanacions d'implementació.

## Veredicte resumit

La valoració és **molt positiva**: el document encaixa directament amb el principi fundacional del projecte que demana experiments repetibles, mètriques traçables i comparació basada en dades. El registre proposat converteix cada simulació rellevant en una unitat auditada amb identitat, configuració congelada, llavor, versió, mètriques, artefactes i estat final.

El document és especialment adequat com a punt 13 del pla funcional perquè dona continuïtat natural al sistema de mètriques evolutives i prepara el terreny per a la persistència de dades. També resol un risc transversal de tota la documentació: evitar que gràfics, genomes destacats o conclusions quedin separats del context experimental que els fa interpretables.

## Encaix amb la documentació existent

| Document relacionat | Encaix observat | Valoració |
| --- | --- | --- |
| `document-funcional-inicial.md` | Desenvolupa el punt 4.13 i reforça el principi que cada experiment ha de ser repetible amb configuració, llavor, resultats i genealogia registrats. | Molt coherent: transforma el criteri general en una especificació verificable. |
| `arquitectura-general.md` | Demana separar motor de simulació, registre, persistència, anàlisi i visualització, sense contaminar la lògica evolutiva. | Molt coherent: reforça modularitat i traçabilitat. |
| `abast-prototip-inicial.md` | Manté l'enfocament pragmàtic del prototip: registre obligatori però sense plataforma científica completa des del primer dia. | Coherent: evita tant infraenginyeria com experiments orfes. |
| `bucle-principal-simulacio.md` | Encaixa amb la necessitat de crear el `run_id` abans d'inicialitzar, recollir mètriques durant l'execució i tancar estat al final o en error. | Molt coherent: defineix punts d'integració clars dins el cicle temporal. |
| `metriques-evolutives.md` | Dona context experimental a les mètriques temporals i agregades, i demana versionar-ne l'esquema. | Molt coherent: les mètriques deixen de ser fitxers solts i passen a estar vinculades a runs. |
| `valoracio-metriques-evolutives.md` | Recull la recomanació de crear identificadors d'experiment que incloguin configuració, llavor i versió del model. | Molt coherent: implementa la traçabilitat recomanada en la valoració anterior. |
| `reproduccio-herencia-mutacio.md` | Exigeix conservar genealogia, millors genomes i dades de llinatge quan hi ha reproducció rellevant. | Coherent: connecta herència i descendència amb artefactes persistents. |
| `mort-seleccio-ambiental.md` | Inclou causes de mort dins les mètriques i validacions, fet necessari per interpretar pressions selectives. | Coherent: evita morts agregades sense causa funcional. |
| `energia-recursos-metabolisme.md` | Inclou energia, recursos i consum dins les mètriques temporals i resums finals. | Coherent: fa auditable la pressió energètica. |
| `genoma-xarxa-neuronal.md` | Reclama versió del model, estructura de xarxa, inicialització i artefactes de genomes destacats. | Coherent: permet saber quina arquitectura ha produït cada resultat. |

## Punts forts

- **Traçabilitat completa:** exigeix `run_id`, `experiment_id`, configuració congelada, llavor, versió, mètriques i manifest d'artefactes.
- **Captura errors i interrupcions:** no limita el registre a execucions completes; també conserva runs fallits, aturats o descartats.
- **Separa nivells d'identificació:** diferencia projecte, experiment, run, pas temporal, agent, llinatge i artefacte, reduint confusions en l'anàlisi.
- **Protegeix la comparabilitat:** demana versionar model, codi, configuració i mètriques abans de comparar resultats.
- **Evita dependència de gràfics:** estableix que les dades primàries i resums tabulars són la font d'evidència, i que les visualitzacions són derivades.
- **Adopta persistència híbrida:** combina MLflow, PostgreSQL, Parquet, Zarr i JSON/YAML segons el tipus de dada, cosa adequada per escalar sense forçar un únic format.
- **Inclou validacions funcionals:** identifica errors típics com població negativa, energia NaN, artefactes obligatoris absents o resum incoherent amb la sèrie temporal.

## Riscos o punts a vigilar

1. **Dependència prematura d'eines externes.**  
   MLflow i PostgreSQL són adequats com a direcció funcional, però el prototip pot començar amb una capa local basada en fitxers estructurats si això accelera la implementació sense perdre el contracte de traçabilitat.

2. **Cost de snapshots i genealogies massives.**  
   La documentació recomana encertadament freqüències configurables. Caldrà definir límits per defecte per evitar que una execució llarga generi un volum de dades inassumible.

3. **Reproduïbilitat exacta en JAX.**  
   Registrar la llavor mestra no serà suficient si el codi divideix claus pseudoaleatòries de manera no documentada. Caldrà una convenció tècnica clara de subllavors.

4. **Estat del repositori i canvis locals.**  
   Registrar el commit és necessari però pot ser insuficient si hi ha canvis no commitejats. El document ho detecta; la implementació haurà de decidir si bloqueja, avisa o marca el run com a no reproduïble.

5. **Definició d'execució important.**  
   El document diferencia proves exploratòries i experimentals, però caldrà una norma pràctica perquè l'equip no deixi proves rellevants fora del registre per comoditat.

6. **Compatibilitat d'esquemes al llarg del temps.**  
   Versionar esquemes és correcte, però caldrà afegir migracions o lectors compatibles quan els formats de mètriques i configuració evolucionin.

## Recomanacions d'implementació

- Definir un esquema mínim de `run_manifest.json` o `run_manifest.yaml` amb `run_id`, `experiment_id`, `status`, `seed`, versions, rutes d'artefactes i resum final.
- Crear el `run_id` abans de construir món, agents o claus aleatòries, i propagar-lo a tots els fitxers derivats.
- Guardar sempre una còpia immutable de la configuració resolta, no només dels paràmetres explícits de l'usuari.
- Registrar el commit de Git, l'estat dirty/clean del repositori i, com a mínim, versions principals de dependències que afecten resultats.
- Separar mètriques temporals en format columnar de resums finals petits i consultables.
- Establir noms d'artefacte estables: `config.yaml`, `metrics_summary.json`, `metrics_timeseries.parquet`, `lineage_summary.parquet`, `artifact_manifest.json` i `run.log`.
- Fer que les execucions fallides escriguin estat `failed`, error, pas afectat i artefactes parcials disponibles.
- Afegir validacions automàtiques abans i després del run per detectar configuracions incompletes, llavors absents, versions desconegudes i artefactes obligatoris inexistents.

## Conclusió

El document **Registre d'experiments** és adequat per incorporar-se a la documentació funcional del Projecte Evolució. Completa el pas entre mètriques evolutives i persistència de dades, i estableix el contracte necessari perquè cap resultat important quedi orfe de configuració, llavor, versió, mètriques o artefactes.

La recomanació és acceptar-lo com a especificació funcional inicial del punt 13 i utilitzar-lo com a base per dissenyar l'esquema concret de manifests, l'organització de carpetes per `run_id`, la integració amb el bucle principal i la futura capa de persistència híbrida.
