# Projecte Evolució — Genoma i xarxa neuronal

**Especificació funcional del vincle entre herència, decisió i selecció per al prototip inicial.**

## Metadades

| Camp | Valor |
| --- | --- |
| Fase | Prototip inicial |
| Tipus de document | Especificació funcional |
| Versió | 0.1 |
| Data | 2 de juliol de 2026 |
| Estat | Especificació funcional inicial |
| Àmbit | Definició del genoma, xarxa neuronal de decisió, mutació, herència i traçabilitat |

## Decisió principal

El prototip inicial farà servir una topologia neuronal fixa i petita. Evolucionaran els pesos, els biaixos i un nombre molt limitat de paràmetres heretables. No evolucionarà encara l'estructura de la xarxa.

## 1. Objectiu del document

Aquest document defineix funcionalment com es representarà la informació heretable dels agents i com aquesta informació controlarà les seves decisions mitjançant una xarxa neuronal petita. El propòsit és tancar el vincle entre genoma, comportament, supervivència i reproducció abans d'entrar en la implementació tècnica detallada.

La primera versió ha de demostrar evolució poblacional. Per això, el genoma no s'ha d'entendre com una còpia exhaustiva d'un organisme real, sinó com el conjunt mínim de dades heretables que permet reconstruir el sistema de decisió d'un agent i generar variació entre descendents.

## 2. Principi de disseny

El genoma ha de ser prou simple per poder simular molts agents, però prou complet perquè les diferències genètiques tinguin conseqüències funcionals. Si el genoma no afecta les accions, no hi ha selecció significativa. Si és massa complex, la primera versió serà difícil d'analitzar i de depurar.

- El genoma condiciona el sistema de decisió de l'agent.
- El sistema de decisió produeix accions a partir d'observacions locals.
- Les accions modifiquen energia, posició, alimentació, reproducció i risc de mort.
- La supervivència i la reproducció determinen quins genomes persisteixen.
- Les mutacions introdueixen variació heretable entre generacions.

## 3. Separació conceptual

Cal separar clarament quatre conceptes que sovint es confonen: genoma, xarxa neuronal, estat de l'agent i comportament observat. Aquesta separació és essencial per mantenir el sistema net i interpretable.

| Concepte | Què representa | És heretable? | Comentari funcional |
| --- | --- | --- | --- |
| Genoma | Informació que es copia del progenitor al descendent amb possibles mutacions. | Sí | Inclou pesos, biaixos i alguns paràmetres heretables. |
| Xarxa neuronal | Mecanisme de decisió reconstruït a partir del genoma. | Indirectament | No és una entitat independent del genoma; és la seva expressió funcional. |
| Estat de l'agent | Energia, edat, posició, estat vital i altres valors de vida. | No directament | Canvia durant la vida de l'individu i no s'ha de copiar com a genoma. |
| Comportament | Seqüència d'accions realitzades en un entorn concret. | No | És el resultat de genoma, estat intern, observacions i condicions ambientals. |

## 4. Definició funcional del genoma

El genoma és el paquet d'informació heretable que permet reconstruir el sistema de decisió d'un agent nou. En el prototip inicial ha de tenir una estructura fixa, serialitzable i comparable entre individus.

### 4.1. Contingut mínim del genoma

| Element | Funció | Recomanació inicial |
| --- | --- | --- |
| Pesos neuronals | Determinen la força de les connexions entre entrades, capa oculta i sortides. | Incloure'ls sempre. Són el nucli del genoma. |
| Biaixos neuronals | Desplacen l'activació de les neurones i permeten estratègies no dependents només de l'entrada. | Incloure'ls sempre. Mutació petita. |
| Paràmetres heretables simples | Permeten variar trets funcionals com metabolisme basal o llindar reproductiu. | Incloure'n pocs i només si són necessaris per a la primera fase. |
| Identificador de genoma | Permet traçar llinatges i versions mutades. | Obligatori. |
| Identificador del genoma progenitor | Permet reconstruir herència i descendència genètica. | Obligatori per a descendents. |
| Comptador o resum de mutacions | Ajuda a analitzar distància generacional i variació acumulada. | Recomanable. |

### 4.2. Què no forma part del genoma

No tot el que defineix un agent en un moment donat és genoma. Copiar massa informació pot confondre herència amb estat vital.

- La posició actual de l'agent no forma part del genoma.
- L'energia actual no forma part del genoma, tot i que l'energia inicial del descendent pot estar definida per la regla de reproducció.
- L'edat no forma part del genoma.
- La causa futura de mort no forma part del genoma.
- Els recursos vistos o consumits durant la vida no formen part del genoma.
- Cap estratègia explícita escrita a mà, com "buscar recursos" o "evitar altres agents", ha de formar part del genoma.

## 5. Paràmetres heretables recomanats

La primera versió pot funcionar només amb pesos i biaixos neuronals. Tot i això, és raonable permetre un conjunt molt petit de paràmetres heretables per estudiar compromisos evolutius. Cal evitar convertir aquesta llista en una biologia artificial massa rica abans d'hora.

| Paràmetre | Impacte funcional | Risc si s'inclou massa aviat |
| --- | --- | --- |
| Metabolisme basal | Determina el cost energètic de viure a cada pas. | Pot dominar tota la selecció i fer irrellevant la xarxa si el rang és massa ampli. |
| Cost de moviment | Afecta el preu energètic de desplaçar-se. | Pot afavorir agents immòbils si el recurs és massa abundant. |
| Llindar reproductiu | Energia mínima necessària per reproduir-se. | Pot generar estratègies degenerades si el llindar baixa massa. |
| Energia transferida al descendent | Determina la viabilitat inicial dels fills. | Pot crear descendents inviables o reproducció explosiva. |
| Sensibilitat ambiental simple | Modula la resposta a condicions locals adverses. | No s'hauria d'incloure fins que el món ambiental estigui ben calibrat. |

**Recomanació crítica:** per a la primera execució validable, és millor començar només amb pesos i biaixos. Els paràmetres heretables addicionals poden activar-se després, quan el bucle bàsic ja produeixi llinatges i mètriques fiables.

## 6. Xarxa neuronal de decisió

La xarxa neuronal és el mecanisme que transforma observacions locals en accions. En aquesta fase no aprèn durant la vida de l'agent. Els seus paràmetres són heretats i mutats entre generacions.

### 6.1. Topologia inicial

La topologia recomanada és una xarxa feedforward petita amb una capa oculta. Aquesta decisió redueix el cost computacional i facilita atribuir canvis de comportament a mutacions concretes.

| Component | Funció | Decisió inicial recomanada |
| --- | --- | --- |
| Entrada | Vector d'observacions locals normalitzades. | Mida fixa segons el model d'observació. |
| Capa oculta | Combinació no lineal de la informació observada. | Una capa petita, per exemple 8-32 unitats. |
| Activació | Introdueix no linealitat en la decisió. | `tanh` o una activació equivalent simple i estable. |
| Sortida | Scores o preferències d'acció. | Una sortida per acció disponible. |
| Selecció d'acció | Converteix sortides en una acció concreta. | Inicialment determinista per `argmax`, amb desempat reproduïble. |

### 6.2. Per què no començar amb xarxes recurrents

Una xarxa recurrent o amb memòria pot ser interessant més endavant, però no és recomanable en el primer prototip. Afegiria estat intern difícil d'interpretar i faria més complicada la distinció entre adaptació evolutiva i dinàmica interna de l'agent.

- La primera versió ha de provar el bucle evolutiu, no la capacitat cognitiva de l'agent.
- Les xarxes recurrents incrementen el cost i dificulten la depuració.
- Una xarxa feedforward ja és suficient per generar diferències de moviment, alimentació i reproducció.
- La memòria es pot afegir després com una extensió explícita, no com a requisit inicial.

## 7. Entrades de la xarxa

Les entrades han de provenir d'observacions locals i de l'estat propi mínim. No han de donar a l'agent informació global del món, perquè això reduiria la importància ecològica de l'entorn.

| Entrada funcional | Què informa | Normalització recomanada |
| --- | --- | --- |
| Energia pròpia | Capacitat de sobreviure i reproduir-se. | Valor entre 0 i 1 respecte a energia màxima o llindar definit. |
| Edat pròpia | Moment del cicle vital. | Valor entre 0 i 1 respecte a edat màxima. |
| Recurs a la posició actual | Oportunitat immediata d'alimentació. | Valor entre 0 i 1. |
| Recursos locals | Disponibilitat de recurs al voltant. | Suma o mitjana normalitzada dins un radi petit. |
| Direcció local del recurs | Gradient simple cap a zones més favorables. | Components `dx`/`dy` o valors per direcció cardinal. |
| Densitat d'agents propers | Competència local. | Valor normalitzat pel nombre màxim esperat de veïns. |
| Condició ambiental local | Qualitat o penalització del lloc. | Valor simple entre 0 i 1, si el món ambiental ja ho inclou. |

Aquest conjunt d'entrades pot ajustar-se, però ha de mantenir-se estable dins d'un experiment. Canviar les entrades canvia el significat del genoma i dificulta comparar llinatges entre execucions.

## 8. Sortides de la xarxa i accions

Les sortides de la xarxa no han de ser ordres màgiques que ignorin les regles del món. Han de ser preferències d'acció. El motor de simulació continua sent responsable de validar si l'acció és possible.

| Sortida | Acció associada | Validació externa necessària |
| --- | --- | --- |
| Quedar-se quiet | L'agent no es mou. | Sempre possible, però amb cost metabòlic. |
| Moure nord/sud/est/oest | Desplaçament a una cel·la veïna. | Cal comprovar límits, ocupació i cost energètic. |
| Alimentar-se | Consumir recurs disponible. | Només possible si hi ha recurs accessible. |
| Reproduir-se | Crear descendent. | Només possible si energia, espai i condicions compleixen els llindars. |

**Punt important:** la xarxa pot intentar reproduir-se sense energia suficient, però el món ha de rebutjar aquesta acció. Això no és un error: forma part de la pressió selectiva sobre genomes que prenen males decisions.

## 9. Interpretació de la decisió

Per reduir soroll en la primera fase, la decisió inicial hauria de ser determinista: l'acció amb puntuació més alta s'executa si és vàlida. En cas d'empat, el desempat ha de dependre d'una llavor aleatòria controlada o d'un ordre estable definit.

L'acció invàlida pot tractar-se com a inacció amb cost energètic. Aquesta opció és dura però útil, perquè penalitza xarxes que generen decisions no viables. No s'hauria de substituir automàticament per la “millor acció possible”, perquè això introduiria intel·ligència hardcodejada fora del genoma.

## 10. Mutació del genoma

La mutació és el mecanisme que introdueix variació entre progenitor i descendent. Ha de ser petita, configurada explícitament i aplicada només al genoma del nou agent.

| Tipus de mutació | Aplicació | Criteri funcional |
| --- | --- | --- |
| Pertorbació de pesos | Afegir soroll petit a alguns pesos neuronals. | Ha de permetre variació gradual sense destruir constantment xarxes útils. |
| Pertorbació de biaixos | Afegir soroll petit als biaixos. | Pot canviar tendències generals d'acció. |
| Mutació de paràmetres heretables | Modificar lleument valors com metabolisme o llindar reproductiu. | Només si aquests paràmetres estan activats en l'experiment. |
| Clipping o límits | Evitar valors extrems. | Necessari per prevenir genomes numèricament inestables. |
| Comptador de mutacions | Registrar quantes mutacions s'han aplicat o la magnitud total. | Útil per analitzar divergència de llinatges. |

La taxa de mutació i la intensitat del soroll no han de quedar ocultes en el codi. Han de formar part de la configuració registrada de cada experiment.

## 11. Transmissió hereditària

Quan un agent es reprodueix, el genoma del descendent s'ha de generar com una còpia del genoma del progenitor amb mutacions aplicades després de la còpia. El genoma del progenitor no s'ha de modificar durant aquest procés.

1. L'agent progenitor supera les condicions de reproducció.
2. El sistema crea un nou identificador d'agent i un nou identificador de genoma.
3. Es copia el genoma del progenitor.
4. S'apliquen mutacions al genoma copiat.
5. Es registra l'identificador del progenitor i del genoma progenitor.
6. Es crea el descendent amb energia inicial definida per la regla de reproducció.
7. Es descompta el cost reproductiu del progenitor.

## 12. Inicialització dels genomes

La població inicial necessita genomes inicials diversos però controlats. La inicialització ha de dependre d'una llavor aleatòria explícita per poder repetir l'experiment.

- Els pesos inicials han de ser petits per evitar sortides saturades des del primer pas.
- Els biaixos inicials poden ser zero o valors petits aleatoris.
- Tots els genomes inicials han de tenir identificador propi.
- Els genomes fundadors han de tenir `parent_genome_id` buit o marcat com a fundador.
- La distribució inicial ha de quedar registrada a la configuració de l'experiment.

## 13. Traçabilitat i reconstrucció

El criteri de tancament exigeix que el genoma sigui suficient per reconstruir el sistema de decisió d'un agent. Això vol dir que, si es carrega un genoma guardat, s'ha de poder obtenir la mateixa xarxa i la mateixa resposta davant les mateixes observacions.

| Dada registrada | Per què és necessària |
| --- | --- |
| `genome_id` | Identifica de manera única una variant heretable. |
| `parent_genome_id` | Permet reconstruir descendència genètica. |
| `agent_id` i `parent_agent_id` | Permeten relacionar genoma amb individu concret i genealogia vital. |
| `generation` o `depth` | Ajuda a saber la profunditat del llinatge. |
| Pesos i biaixos complets | Permeten reconstruir la xarxa. |
| Configuració de topologia | Evita interpretar pesos amb una arquitectura equivocada. |
| Paràmetres de mutació | Permeten reproduir o auditar la generació del descendent. |

## 14. Diversitat genètica

La diversitat genètica no ha de quedar com una idea vaga. Encara que la mètrica exacta es defineixi tècnicament més endavant, funcionalment cal poder comparar genomes.

- Cal poder representar cada genoma com un vector comparable de pesos, biaixos i paràmetres heretables.
- Cal poder calcular distàncies aproximades entre genomes.
- Cal poder detectar si la població convergeix cap a pocs llinatges dominants.
- Cal poder separar èxit reproductiu de simple abundància inicial.
- Cal poder recuperar els millors llinatges segons supervivència i descendència.

## 15. Flux funcional de dades

El flux bàsic entre genoma, xarxa, agent i entorn ha de ser estable i fàcil d'explicar.

| Pas | Entrada | Procés | Sortida |
| --- | --- | --- | --- |
| 1 | Genoma de l'agent | Reconstrucció funcional de la xarxa neuronal. | Xarxa de decisió. |
| 2 | Estat de l'agent i entorn local | Construcció del vector d'observació. | Observació normalitzada. |
| 3 | Observació i xarxa | Càlcul de scores d'acció. | Preferències d'acció. |
| 4 | Preferències d'acció | Selecció d'acció segons regla definida. | Acció proposada. |
| 5 | Acció proposada i regles del món | Validació i aplicació. | Canvi d'estat, energia, posició o reproducció. |
| 6 | Reproducció vàlida | Còpia i mutació del genoma. | Nou agent amb nou genoma. |

## 16. Exclusions temporals

Per mantenir el prototip inicial net i mesurable, queden fora d'aquesta fase els elements següents:

- Aprenentatge individual amb backpropagation o reforç.
- Actualització dels pesos durant la vida de l'agent.
- Reproducció sexual o recombinació entre genomes.
- Crossover genètic.
- Evolució de la topologia neuronal.
- Xarxes recurrents o memòria complexa.
- Neuroevolució externa com a substitut del model ecològic propi.
- Comportaments preprogramats d'alt nivell.
- Codificació manual d'estratègies com exploració intel·ligent, caça, fugida o planificació.

## 17. Riscos principals

| Risc | Conseqüència | Mesura de control |
| --- | --- | --- |
| Genoma massa ric | La selecció esdevé difícil d'interpretar. | Començar només amb pesos, biaixos i pocs paràmetres. |
| Mutacions massa fortes | Els descendents perden qualsevol continuïtat funcional. | Usar intensitats petites i registrar-les. |
| Mutacions massa febles | La població no explora variació suficient. | Analitzar diversitat i adaptar taxa després de proves. |
| Accions corregides automàticament | El sistema introdueix intel·ligència fora del genoma. | Les accions invàlides han de penalitzar-se o convertir-se en inacció. |
| Entrades globals | Els agents deixen de dependre d'interaccions locals. | Limitar l'observació a informació pròxima i pròpia. |
| Xarxa massa gran | Cost computacional elevat i menor interpretabilitat. | Mantenir una xarxa petita en la primera versió. |

## 18. Criteris funcionals d'èxit

1. Un genoma guardat permet reconstruir exactament la xarxa de decisió associada.
2. Dos agents amb genomes diferents poden produir accions diferents davant la mateixa observació.
3. Un descendent hereta el genoma del progenitor amb mutacions controlades.
4. La relació entre agent, progenitor, genoma i genoma progenitor queda registrada.
5. Les mutacions poden mesurar-se o auditar-se posteriorment.
6. Els pesos i biaixos no canvien durant la vida de l'agent.
7. Les accions de l'agent afecten energia, alimentació, reproducció i mort.
8. Es poden comparar llinatges segons èxit reproductiu i distància genètica.
9. La configuració neuronal i mutacional queda registrada amb cada experiment.
10. El sistema no conté estratègies intel·ligents hardcodejades dins les accions.

## 19. Definició de tancament

Aquest punt es considerarà tancat quan l'equip pugui explicar amb claredat què conté el genoma, com es reconstrueix la xarxa neuronal, quines observacions entren al sistema de decisió, com es trien les accions, com es transmet la informació hereditària i com es registren les mutacions.

La prova funcional mínima és poder agafar un agent qualsevol, recuperar el seu genoma, reconstruir la seva xarxa, aplicar una observació coneguda i obtenir una decisió reproduïble. També cal poder seguir el seu llinatge fins al progenitor i distingir quines diferències genètiques s'han acumulat.

## 20. Resum executiu

El prototip inicial farà servir genomes simples, mutables i completament traçables. Aquests genomes codificaran una xarxa neuronal petita que transformarà observacions locals en accions bàsiques. La selecció no s'aplicarà directament sobre els genomes, sinó sobre les conseqüències ecològiques de les decisions que aquests genomes produeixen.

La decisió més robusta és evitar aprenentatge individual, topologies canviants i paràmetres biològics excessius en aquesta fase. Primer cal demostrar que el bucle herència - comportament - supervivència - reproducció funciona i genera diferències mesurables entre llinatges.

## 21. Valoració d'adequació amb la resta de la documentació

El document és coherent amb la documentació existent i completa un buit funcional important entre el model d'agent, el món 2D i l'arquitectura general.

### Encaix amb documents existents

| Document relacionat | Valoració d'encaix |
| --- | --- |
| `document-funcional-inicial.md` | Desenvolupa el bloc d'evolució i decisió neuronal que el full de ruta defineix a alt nivell. Manté el principi de no hardcodejar estratègies i de mesurar l'adaptació amb dades. |
| `abast-prototip-inicial.md` | Respecta l'abast del prototip: món 2D, agents simples, reproducció asexual, mutació, mort, mètriques i registre. També reforça les exclusions temporals ja definides. |
| `arquitectura-general.md` | S'alinea amb la separació entre simulació, agent, genoma, xarxa neuronal, reproducció, mètriques i persistència. El flux de dades proposat és compatible amb l'arquitectura per blocs. |
| `model-mon-2d.md` | Manté l'observació local, els límits del món i la responsabilitat del món com a validador de regles físiques i ecològiques, sense introduir coneixement global als agents. |
| `model-agent.md` | Complementa la definició d'agent separant estat vital no heretable, genoma heretable, genealogia i traçabilitat de decisions. |

### Punts forts

- Tanca explícitament el vincle herència-decisió-selecció que fins ara quedava repartit entre diversos documents.
- Manté una filosofia minimalista adequada per al prototip inicial.
- Defineix criteris verificables: reconstrucció de xarxa des del genoma, decisió reproduïble, mutació auditada i comparació de llinatges.
- Evita introduir aprenentatge individual, memòria complexa, topologies evolutives o estratègies manuals abans de validar el bucle evolutiu.

### Punts a vigilar abans de la implementació

- Caldrà fixar tècnicament la mida exacta del vector d'observació i el nombre d'accions per evitar que genomes d'experiments diferents siguin incompatibles.
- Si s'activen paràmetres heretables addicionals, caldrà limitar-ne rangs i registrar-los a la configuració per evitar que dominin la selecció.
- La política d'accions invàlides ha de quedar implementada de manera uniforme perquè no introdueixi correccions intel·ligents fora del genoma.
- La persistència haurà de guardar també la configuració de topologia i mutació, no només pesos i biaixos, per garantir reconstrucció i auditoria.

### Valoració global

**Adequació alta.** El document és consistent amb l'abast, l'arquitectura i els models ja definits. La seva incorporació és recomanable perquè converteix el genoma i la xarxa neuronal en una especificació funcional pròpia, traçable i preparada per derivar criteris d'implementació i proves.
