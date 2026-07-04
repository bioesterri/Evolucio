# Projecte Evolució — Mètriques evolutives

**Especificació funcional del sistema de mètriques per distingir evolució mesurable de soroll aleatori en el prototip inicial.**

## Metadades

| Camp | Valor |
| --- | --- |
| Tipus de document | Especificació funcional de mètriques evolutives |
| Punt del full de ruta | Punt 12 del pla funcional |
| Àmbit | Supervivència, descendència, diversitat genètica, economia energètica, estabilitat poblacional i resposta a canvis ambientals |
| Objectiu | Definir quines dades permeten valorar si la simulació genera dinàmica evolutiva real o només fluctuacions aleatòries. |
| Resultat esperat | Un sistema d'avaluació objectiu per comparar experiments, configuracions i llinatges. |
| Criteri de tancament | Cada execució produeix mètriques suficients per valorar si hi ha adaptació mesurable. |
| Estat | Versió funcional per a validació interna abans de la implementació tècnica detallada |
| Versió | 1.0 |
| Data | Juliol de 2026 |

## Índex

1. [Objectiu del sistema de mètriques](#1-objectiu-del-sistema-de-mètriques)
2. [Principi rector](#2-principi-rector)
3. [Què ha de mesurar el prototip](#3-què-ha-de-mesurar-el-prototip)
4. [Unitats d'anàlisi](#4-unitats-danàlisi)
5. [Mètriques obligatòries](#5-mètriques-obligatòries)
6. [Mètriques derivades i interpretació](#6-mètriques-derivades-i-interpretació)
7. [Registre de dades necessari](#7-registre-de-dades-necessari)
8. [Comparació d'experiments](#8-comparació-dexperiments)
9. [Visualització mínima](#9-visualització-mínima)
10. [Controls de qualitat](#10-controls-de-qualitat)
11. [Exclusions temporals](#11-exclusions-temporals)
12. [Riscos principals](#12-riscos-principals)
13. [Criteris funcionals d'acceptació](#13-criteris-funcionals-dacceptació)
14. [Definició de tancament](#14-definició-de-tancament)
15. [Decisió recomanada per al prototip inicial](#15-decisió-recomanada-per-al-prototip-inicial)

## 1. Objectiu del sistema de mètriques

L'objectiu d'aquest bloc és definir el conjunt de dades que permetrà saber si el prototip del Projecte Evolució està generant una dinàmica evolutiva real o només fluctuacions aleatòries sense acumulació adaptativa.

Les mètriques han de convertir la simulació en un experiment analitzable. No n'hi ha prou amb veure agents movent-se, reproduint-se o morint. Cal poder demostrar, amb dades, si alguns genomes, agents o llinatges tenen més èxit que altres sota unes condicions ambientals determinades.

El sistema de mètriques ha de permetre respondre preguntes com aquestes:

- La població es manté, creix, oscil·la o col·lapsa?
- Hi ha llinatges que deixen més descendència que altres?
- La diversitat genètica augmenta, es manté o s'empobreix?
- Els agents consumeixen recursos de manera més eficient al llarg del temps?
- Els canvis ambientals provoquen selecció diferencial?
- Els resultats són repetibles sota la mateixa llavor aleatòria?
- Les diferències observades superen el soroll esperable entre execucions?

Aquest document no defineix encara la implementació concreta de cada càlcul, però sí que fixa quines mètriques han d'existir, quin paper funcional tenen i com s'han d'interpretar.

## 2. Principi rector

El principi rector és simple: **les mètriques han d'observar el sistema, no dirigir-lo**.

No s'ha d'introduir una puntuació externa que premiï agents segons un criteri artificial. En aquest projecte, la selecció ha de sorgir de la supervivència, l'energia, la reproducció i les condicions ambientals. Les mètriques només han de registrar i resumir el que ha passat.

Això implica una decisió important: no hi haurà una funció de fitness separada per decidir quins agents són millors. L'èxit evolutiu s'haurà d'inferir a partir de dades objectives com la descendència, la persistència del llinatge, la supervivència i la capacitat de continuar reproduint-se en un entorn canviant.

Aquest criteri evita contaminar la simulació amb objectius humans massa directes. Si es defineix una recompensa explícita, el sistema deixaria de ser una ecologia evolutiva i passaria a semblar un problema d'optimització convencional.

## 3. Què ha de mesurar el prototip

El prototip ha de mesurar sis dimensions principals. Cap d'elles, per separat, demostra adaptació. La interpretació ha de sortir de la combinació entre totes.

| Dimensió | Què indica | Per què importa |
| --- | --- | --- |
| Supervivència | Quant temps viuen els agents i quines causes els fan morir. | Permet veure si alguns agents resisteixen millor les condicions ambientals. |
| Descendència | Quants descendents produeix cada agent o llinatge. | L'evolució actua sobre l'èxit reproductiu, no sobre la simple supervivència individual. |
| Diversitat genètica | Com de diferents són els genomes presents a la població. | Evita confondre estabilitat poblacional amb dominància d'un sol clon o col·lapse genètic. |
| Economia energètica | Com entra i surt l'energia dels agents i de l'entorn. | Mostra si la supervivència té cost i si hi ha eficiència diferencial. |
| Estabilitat poblacional | Com canvia la mida de la població en el temps. | Detecta creixement descontrolat, extincions, oscil·lacions i colls d'ampolla. |
| Resposta a canvis | Com reacciona la població davant modificacions ambientals. | És el primer indici funcional d'adaptabilitat, sempre que es compari amb controls. |

Aquestes dimensions han d'estar disponibles per execució, per interval temporal i, quan sigui possible, per llinatge.

## 4. Unitats d'anàlisi

Per evitar confusions, les mètriques s'han d'organitzar segons la unitat que descriuen. No és el mateix una mètrica d'agent, de població, de llinatge o d'experiment.

### 4.1. Agent individual

L'agent individual és la unitat bàsica de vida dins la simulació. Les seves mètriques permeten reconstruir el seu cicle vital.

- Identificador de l'agent.
- Identificador del progenitor.
- Pas de naixement.
- Pas de mort.
- Edat final.
- Energia mitjana o energia final.
- Nombre de reproduccions.
- Causa de mort.
- Genoma o referència al genoma.

### 4.2. Llinatge

El llinatge és essencial per parlar d'evolució. Un agent molt longeu però sense descendència pot ser biològicament poc rellevant. En canvi, un agent de vida curta pot ser important si origina una branca persistent.

- Nombre total de descendents.
- Profunditat genealògica assolida.
- Durada del llinatge.
- Proporció de població representada pel llinatge.
- Moments d'expansió o declivi.
- Variació genètica acumulada dins la branca.

### 4.3. Població

La població resumeix l'estat global del sistema en cada pas o interval temporal.

- Nombre d'agents vius.
- Naixements i morts per interval.
- Energia mitjana, mediana i dispersió.
- Edat mitjana i distribució d'edats.
- Diversitat genètica agregada.
- Consum total de recursos.
- Nivell de recursos disponibles al món.

### 4.4. Experiment

L'experiment és una execució completa sota una configuració i una llavor aleatòria concretes. Les mètriques d'experiment permeten comparar condicions, no individus.

- Configuració utilitzada.
- Llavor aleatòria.
- Durada prevista i durada real.
- Estat final de la població.
- Existència o no d'extinció.
- Resum de llinatges dominants.
- Indicadors agregats d'estabilitat i adaptació.

## 5. Mètriques obligatòries

Les mètriques següents formen el conjunt mínim que ha d'existir en el prototip. Sense aquestes dades, el sistema pot ser executable però no serà científicament interpretable.

### 5.1. Mida poblacional

La mida poblacional indica quants agents vius hi ha en cada pas temporal o en cada interval de registre.

- **Valor principal:** nombre d'agents vius.
- **Interpretació:** permet detectar creixement, estabilització, oscil·lació, colls d'ampolla o extinció.
- **Risc:** una població gran no implica necessàriament adaptació; pot indicar recursos massa abundants o costos massa baixos.

### 5.2. Naixements i morts

El sistema ha de registrar quants agents neixen i moren, i en quin moment. Aquesta informació mostra el flux demogràfic real de la població.

- Naixements per pas o interval.
- Morts per pas o interval.
- Balanç net de població.
- Taxa de renovació generacional.
- Relació entre naixements, morts i disponibilitat de recursos.

### 5.3. Causes de mort

Cada mort ha de quedar associada a una causa funcional. Sense causes de mort, no es pot saber quina pressió ambiental està actuant.

| Causa | Significat funcional | Ús analític |
| --- | --- | --- |
| `energia_zero` | L'agent no ha pogut mantenir el cost metabòlic. | Mesura pressió energètica. |
| `edat_maxima` | L'agent ha arribat al límit de vida definit. | Controla renovació generacional. |
| `ambient_desfavorable` | Condicions locals incompatibles o penalitzadores. | Mesura pressió ambiental. |
| `competencia` | L'agent perd per saturació, bloqueig o interacció definida. | Mesura pressió densodependent. |
| `fora_limits` | Intent o resultat invàlid en el món, si el model ho permet. | Detecta errors de moviment o frontera. |

### 5.4. Supervivència

La supervivència mesura quant temps viuen els agents i com varia aquesta durada entre individus, cohorts o llinatges.

- Edat mitjana de mort.
- Edat mediana de mort.
- Distribució de durades de vida.
- Supervivència per cohort de naixement.
- Supervivència per llinatge.

La mediana és especialment important, perquè la mitjana pot quedar distorsionada per pocs individus molt longeus.

### 5.5. Descendència i èxit reproductiu realitzat

L'èxit reproductiu realitzat és una de les mètriques més importants del projecte. No mesura la capacitat teòrica d'un agent per reproduir-se, sinó quants descendents ha arribat a produir dins les condicions reals del sistema.

- Nombre de fills directes per agent.
- Nombre de descendents totals per llinatge.
- Proporció d'agents que no es reprodueixen mai.
- Distribució de descendència per agent.
- Llinatges actius al final de l'execució.

Cal vigilar amb la paraula *fitness*. En aquest projecte és millor parlar d'èxit reproductiu realitzat, perquè no és una puntuació externa sinó una conseqüència del sistema.

### 5.6. Diversitat genètica

La diversitat genètica ha de mesurar fins a quin punt els genomes presents a la població són diferents entre si. Aquesta mètrica és necessària per detectar dominància extrema, deriva genètica o pèrdua de variació.

- Distància mitjana entre genomes.
- Variància dels pesos i biaixos neuronals.
- Nombre de llinatges genèticament diferenciats.
- Diversitat dins dels llinatges dominants.
- Canvi de diversitat al llarg del temps.

En la primera versió, n'hi ha prou amb una distància numèrica aproximada entre vectors de genoma. No cal introduir encara mesures biològiques complexes si no aporten interpretabilitat.

### 5.7. Energia dels agents

Les mètriques energètiques indiquen si els agents sobreviuen gràcies a un balanç energètic real o si el sistema està permetent supervivència massa barata.

- Energia mitjana de la població.
- Energia mediana.
- Energia mínima i màxima.
- Energia en el moment de reproducció.
- Energia en el moment de mort.
- Cost energètic total per acció.

### 5.8. Consum i disponibilitat de recursos

Les mètriques de recursos són necessàries per saber si la població està realment limitada per l'entorn.

- Recursos totals disponibles al món.
- Recursos consumits per interval.
- Recursos regenerats per interval.
- Proporció de cel·les o zones amb recursos.
- Zones recurrentment esgotades.

Si la disponibilitat de recursos no baixa mai o no condiciona la mortalitat, la pressió selectiva probablement és massa feble.

### 5.9. Eficiència energètica

L'eficiència energètica relaciona l'energia consumida amb resultats biològicament rellevants.

- Descendents per unitat de recurs consumit.
- Supervivència per unitat d'energia gastada.
- Energia mitjana necessària per produir un descendent.
- Cost energètic mitjà per llinatge.

Aquesta mètrica s'ha d'interpretar amb cura. Un agent molt eficient però sense descendència no és evolutivament exitós. Per tant, l'eficiència sempre s'ha de llegir conjuntament amb reproducció i persistència del llinatge.

### 5.10. Estabilitat poblacional

L'estabilitat poblacional descriu si el sistema tendeix a un règim sostenible o si cau en extinció, explosió demogràfica o oscil·lacions extremes.

- Variació de la població al llarg del temps.
- Amplitud de les oscil·lacions.
- Freqüència de colls d'ampolla.
- Temps fins a l'extinció, si passa.
- Temps fins a estabilització aproximada.

Una població perfectament estable no és sempre millor. Pot indicar equilibri real, però també pot indicar que el sistema és massa rígid o massa fàcil.

### 5.11. Resposta a canvis ambientals

Quan el model inclogui canvis ambientals, cal mesurar com respon la població abans, durant i després del canvi.

- Mida poblacional abans del canvi.
- Mida poblacional mínima després del canvi.
- Temps de recuperació.
- Llinatges que desapareixen.
- Llinatges que augmenten la seva proporció.
- Canvi en consum de recursos.
- Canvi en diversitat genètica.

Aquesta dimensió és clau per parlar d'adaptabilitat, però només si es compara amb execucions de control sense canvi ambiental o amb canvis diferents.

## 6. Mètriques derivades i interpretació

Algunes mètriques poden derivar-se de les bàsiques i ajuden a interpretar millor el sistema. No són totes imprescindibles per al primer prototip, però convé deixar-les definides.

| Mètrica derivada | Interpretació funcional | Advertiment |
| --- | --- | --- |
| Dominància de llinatge | Percentatge de població final descendent d'una mateixa branca. | Pot indicar adaptació, però també deriva o coll d'ampolla. |
| Taxa de renovació | Relació entre naixements, morts i població viva. | Una renovació massa baixa frena l'evolució. |
| Diversitat efectiva | Nombre aproximat de llinatges amb presència significativa. | No s'ha de confondre amb nombre brut de llinatges. |
| Recuperació post-canvi | Temps necessari per recuperar població o reproducció després d'una pertorbació. | Pot no tenir sentit si la població no cau prou. |
| Eficiència reproductiva | Descendents produïts en relació amb recursos consumits. | No substitueix l'èxit reproductiu total. |

La interpretació ha de ser conservadora. Una mètrica aïllada rarament permet afirmar adaptació. Cal mirar patrons consistents entre diverses execucions.

## 7. Registre de dades necessari

Per calcular les mètriques, la simulació ha de registrar dades en diferents nivells. No cal guardar-ho tot a cada pas si el volum creix massa, però els esdeveniments clau han de quedar registrats.

### 7.1. Registre de configuració

- Identificador de l'experiment.
- Llavor aleatòria.
- Dimensions del món.
- Paràmetres de recursos.
- Paràmetres energètics.
- Taxa i intensitat de mutació.
- Nombre inicial d'agents.
- Durada prevista.
- Canvis ambientals programats.

### 7.2. Registre per pas temporal o interval

- Pas temporal.
- Nombre d'agents vius.
- Naixements.
- Morts per causa.
- Energia agregada.
- Recursos disponibles.
- Recursos consumits.
- Diversitat genètica agregada.
- Llinatges actius.

Per simulacions llargues, aquest registre pot fer-se cada N passos, no necessàriament a cada pas, sempre que els esdeveniments individuals importants quedin registrats.

### 7.3. Registre d'esdeveniments individuals

- Naixement d'un agent.
- Identificador del progenitor.
- Genoma heretat o referència al genoma.
- Mutacions aplicades.
- Reproducció executada.
- Mort i causa de mort.
- Edat i energia en morir.

### 7.4. Registre genealògic

El registre genealògic ha de permetre reconstruir l'arbre de descendència. No cal visualitzar-lo sempre, però les dades han d'existir.

- `agent_id`.
- `parent_id`.
- `birth_step`.
- `death_step`.
- `lineage_id` o ancestre inicial.
- Nombre de fills directes.
- Referència al genoma.

## 8. Comparació d'experiments

Les mètriques tenen valor sobretot quan permeten comparar execucions. Una sola simulació pot ser suggerent, però no permet concloure gaire.

La comparació ha de seguir tres criteris:

1. Repetir configuracions amb diverses llavors aleatòries.
2. Comparar variants canviant un sol factor important cada vegada.
3. Analitzar distribucions i no només mitjanes globals.

Això vol dir que una configuració no s'hauria de considerar millor només per una execució afortunada. Cal mirar si el patró es repeteix amb diferents llavors.

### 8.1. Controls recomanats

- **Control sense mutació:** serveix per veure si la variació observada depèn realment de mutacions.
- **Control amb agents aleatoris fixos:** serveix per comparar agents evolucionables contra una base sense acumulació hereditària.
- **Control sense canvi ambiental:** serveix per interpretar les respostes a pertorbacions.
- **Control amb recursos abundants:** serveix per detectar si la pressió selectiva depèn de l'escassetat.

## 9. Visualització mínima

La visualització no substitueix les mètriques, però ajuda a detectar errors i patrons. En aquesta fase ha de ser funcional, no decorativa.

- Sèrie temporal de mida poblacional.
- Sèrie temporal de recursos disponibles.
- Naixements i morts per interval.
- Morts per causa.
- Energia mitjana i dispersió.
- Diversitat genètica al llarg del temps.
- Proporció dels llinatges principals.
- Histograma de descendència per agent.
- Histograma d'edats de mort.

Un mapa visual del món pot ser útil per depurar, però no ha de ser el criteri principal per afirmar que hi ha evolució.

## 10. Controls de qualitat

El sistema de mètriques ha d'incloure comprovacions bàsiques per detectar inconsistències. Aquestes comprovacions són tan importants com les mètriques biològiques, perquè un error en el registre pot invalidar l'anàlisi.

- Cap agent mort pot tornar a actuar.
- Cap agent viu pot tenir energia negativa si el model no ho permet.
- Tot naixement ha de tenir un progenitor vàlid, excepte la població inicial.
- Tot agent ha de tenir un identificador únic.
- La suma de naixements i morts ha de quadrar amb el canvi poblacional.
- Les causes de mort han d'estar dins un conjunt tancat de categories.
- La mateixa configuració amb la mateixa llavor ha de produir el mateix resultat.
- Els recursos consumits no poden superar els recursos disponibles segons les regles del model.

## 11. Exclusions temporals

Per evitar inflar massa el primer prototip, queden fora de l'abast inicial:

- Mesures estadístiques complexes que no siguin necessàries per validar el bucle evolutiu.
- Models filogenètics biològics avançats.
- Visualitzacions interactives sofisticades.
- Inferències causals fortes sobre adaptació sense experiments controlats.
- Puntuacions externes de fitness utilitzades per seleccionar agents.
- Classificacions subjectives de comportament intel·ligent.
- Comparacions massives automatitzades abans de validar les mètriques bàsiques.

## 12. Riscos principals

### 12.1. Confondre supervivència amb adaptació

Viure més no sempre vol dir tenir més èxit evolutiu. Si un agent viu molt però no es reprodueix, el seu impacte genètic pot ser nul.

### 12.2. Confondre població estable amb sistema ben calibrat

Una població estable pot indicar equilibri, però també pot ser el resultat de regles massa restrictives o massa artificials.

### 12.3. Usar massa mitjanes

Les mitjanes amaguen desigualtats importants. En evolució, la distribució de descendència i la dominància de llinatges poden ser més informatives que una mitjana global.

### 12.4. No separar soroll de senyal

Una execució concreta pot mostrar un llinatge dominant per atzar. Cal repetir experiments amb diferents llavors abans d'interpretar-ho com a adaptació.

### 12.5. Guardar massa dades sense criteri

Registrar-ho tot pot saturar l'emmagatzematge i dificultar l'anàlisi. Cal guardar dades agregades freqüents i esdeveniments individuals rellevants.

## 13. Criteris funcionals d'acceptació

El bloc de mètriques evolutives es considerarà funcional quan compleixi aquests criteris:

1. Cada execució queda vinculada a una configuració i una llavor aleatòria.
2. Es registra la mida poblacional al llarg del temps.
3. Es registren naixements i morts.
4. Cada mort té una causa agregable.
5. Cada agent pot associar-se amb el seu progenitor.
6. Es pot calcular supervivència individual i agregada.
7. Es pot calcular descendència per agent i per llinatge.
8. Es pot estimar diversitat genètica de la població.
9. Es pot veure el consum i la disponibilitat de recursos.
10. Es poden comparar execucions amb configuracions diferents.
11. La mateixa execució amb la mateixa llavor és repetible.
12. Les dades permeten detectar extinció, explosió poblacional o estabilitat.

## 14. Definició de tancament

Aquest punt es considerarà tancat quan el projecte disposi d'una especificació funcional acceptada de les mètriques que cal registrar, calcular i visualitzar per avaluar la dinàmica evolutiva del prototip.

El resultat mínim acceptable és que cada execució produeixi un conjunt de dades suficient per respondre, com a mínim, aquestes preguntes:

- La població ha sobreviscut o s'ha extingit?
- Quines causes principals han provocat mortalitat?
- Quins llinatges han deixat més descendència?
- La diversitat genètica s'ha mantingut o ha disminuït?
- Els recursos han estat una limitació real?
- Els canvis ambientals han modificat l'èxit dels llinatges?
- Els resultats es poden repetir amb la mateixa llavor?

Si aquestes preguntes no es poden respondre amb dades, el sistema encara no té mètriques evolutives suficients. Si es poden respondre de manera clara i comparable entre execucions, el bloc funcional es pot donar per completat i passar a la implementació tècnica detallada.

## 15. Decisió recomanada per al prototip inicial

Per a la primera versió, la recomanació és implementar un conjunt reduït però robust de mètriques:

- població viva;
- naixements;
- morts per causa;
- supervivència;
- descendència per llinatge;
- diversitat genètica aproximada;
- energia agregada;
- recursos disponibles;
- resposta a canvis ambientals simples.

Aquesta selecció és suficient per detectar evolució mesurable sense convertir el sistema de mètriques en un projecte independent massa gran.
