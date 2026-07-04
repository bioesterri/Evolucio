# Valoració d'adequació — Mètriques evolutives

## Objectiu de la valoració

Aquest document comprova si l'especificació **Mètriques evolutives** s'adequa a la resta de la documentació funcional del Projecte Evolució i identifica punts forts, encaixos, riscos i recomanacions d'implementació.

## Veredicte resumit

La valoració és **molt positiva**: el document encaixa amb el principi general del projecte perquè defineix un sistema d'observació i comparació sense introduir una funció de fitness externa. Les mètriques proposades fan auditable el bucle evolutiu format per observació, acció, energia, mort, reproducció, herència i mutació.

El document és especialment adequat com a punt 12 del pla funcional perquè converteix els documents previs en un conjunt d'indicadors mesurables. També delimita bé el prototip inicial: prioritza mètriques robustes i interpretables abans d'afegir estadística avançada o visualitzacions complexes.

## Encaix amb la documentació existent

| Document relacionat | Encaix observat | Valoració |
| --- | --- | --- |
| `arquitectura-general.md` | Les mètriques demanen separació entre motor de simulació, registre d'esdeveniments, agregació i anàlisi. | Molt coherent: reforça una arquitectura modular i auditable. |
| `model-agent.md` | Les mètriques individuals exigeixen identitat estable, estat vital, energia, edat, progenitor i genoma o referència al genoma. | Molt coherent: concreta quines dades de l'agent han de persistir per fer anàlisi posterior. |
| `model-mon-2d.md` | Les mètriques de recursos disponibles, consum, zones esgotades i canvis ambientals encaixen amb un món discret i local. | Coherent: converteix l'estat del món en pressió ambiental mesurable. |
| `observacions-agents.md` | La qualitat de les observacions es podrà inferir indirectament a través de supervivència, consum, energia i descendència. | Coherent: manté les observacions com a informació d'entrada, no com a puntuació. |
| `accions-agents.md` | Els costos i efectes de les accions es reflecteixen en energia, consum de recursos, mortalitat i èxit reproductiu. | Molt coherent: permet comparar comportaments sense valorar-los externament. |
| `energia-recursos-metabolisme.md` | Energia agregada, consum, recursos disponibles i eficiència energètica són continuació directa del model metabòlic. | Molt coherent: ajuda a detectar si la pressió selectiva energètica és real o massa feble. |
| `mort-seleccio-ambiental.md` | Les causes de mort agregables són imprescindibles per interpretar pressions ambientals i no només baixes poblacionals. | Molt coherent: les mètriques fan observable la selecció ambiental. |
| `reproduccio-herencia-mutacio.md` | Descendència per agent, llinatge, genealogia i diversitat genètica mesuren l'efecte real de reproducció i mutació. | Molt coherent: connecta herència i variació amb èxit reproductiu realitzat. |
| `genoma-xarxa-neuronal.md` | La diversitat genètica aproximada i les distàncies entre genomes són compatibles amb genomes vectorials de pesos i biaixos. | Coherent: evita mesures biològiques avançades innecessàries al prototip inicial. |
| `abast-prototip-inicial.md` | La decisió recomanada limita el primer lliurament a mètriques bàsiques però suficients. | Molt coherent: evita convertir l'analítica en un projecte independent. |

## Punts forts

- **No contamina la selecció:** explicita que les mètriques observen el sistema i no decideixen quins agents són millors.
- **Cobreix el nucli evolutiu complet:** inclou supervivència, descendència, genealogia, diversitat genètica, energia, recursos, estabilitat i resposta ambiental.
- **Prioritza mètriques interpretables:** evita començar amb estadística complexa abans de validar que el registre bàsic és correcte.
- **Separa unitats d'anàlisi:** diferencia clarament mètriques d'agent, llinatge, població i experiment.
- **Facilita comparació d'experiments:** demana configuració, llavor, repeticions i controls, cosa essencial per separar senyal i soroll.
- **Inclou controls de qualitat:** incorpora invariants funcionals que ajudaran a detectar errors de simulació o registre.
- **Manté una definició de tancament verificable:** les preguntes finals són concretes i orientades a dades.

## Riscos o punts a vigilar

1. **Cost de registre excessiu.**  
   Cal evitar guardar snapshots complets de tots els agents a cada pas si no és necessari. És millor combinar agregats periòdics amb esdeveniments individuals immutables.

2. **Ambigüitat en els codis de mort.**  
   Els codis funcionals del document són adequats, però caldrà alinear-los amb els codis tècnics que es defineixin a `mort-seleccio-ambiental.md` per evitar duplicats semàntics.

3. **Diversitat genètica massa costosa.**  
   Cal començar amb una distància aproximada entre vectors de genoma i evitar càlculs quadràtics massa cars si la població creix.

4. **Interpretació prematura d'adaptació.**  
   Encara que una execució mostri dominància de llinatge o recuperació post-canvi, no s'ha d'interpretar com a adaptació fins que hi hagi repeticions amb diverses llavors i controls.

5. **Mètriques derivades sense dades base fiables.**  
   Dominància, eficiència reproductiva o recuperació post-canvi només tindran valor si naixements, morts, genealogia, energia i recursos estan registrats sense inconsistències.

## Recomanacions d'implementació

- Crear un identificador d'experiment que inclogui configuració, llavor i versió del model.
- Registrar esdeveniments immutables de naixement, reproducció i mort.
- Mantenir una taula o estructura genealògica amb `agent_id`, `parent_id`, `lineage_id`, `birth_step`, `death_step` i `children_count`.
- Separar mètriques per pas o interval de les mètriques d'esdeveniment per controlar volum de dades.
- Definir un conjunt tancat i versionat de causes de mort abans de començar a comparar experiments.
- Implementar primer les mètriques mínimes recomanades: població viva, naixements, morts per causa, supervivència, descendència per llinatge, diversitat genètica aproximada, energia agregada i recursos disponibles.
- Afegir comprovacions automàtiques d'integritat: unicitat d'agents, coherència poblacional, progenitors vàlids, causes de mort vàlides i repetibilitat amb la mateixa llavor.

## Conclusió

El document **Mètriques evolutives** és adequat per incorporar-se a la documentació funcional del Projecte Evolució. Dona una base objectiva per respondre si el prototip mostra adaptació mesurable o només variació aleatòria, i ho fa sense trencar el principi central del projecte: la selecció ha d'emergir de l'entorn, l'energia, la mortalitat i la reproducció diferencial.

La recomanació és acceptar-lo com a especificació funcional inicial del sistema de mètriques i utilitzar-lo com a contracte per implementar el registre d'experiments, els agregats poblacionals, la traçabilitat genealògica i les visualitzacions mínimes.
