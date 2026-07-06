# Projecte Evolució — Visualització mínima

**Especificació funcional per observar el món, els agents, els recursos i les mètriques principals sense confondre diagnòstic visual amb evidència científica.**

## Metadades

| Camp | Valor |
| --- | --- |
| Tipus de document | Especificació funcional de visualització mínima |
| Punt del full de ruta | Punt 16 del pla funcional |
| Àmbit | Vista espacial del món, mètriques temporals, llinatges destacats, resums visuals i exportació d'artefactes |
| Definició funcional | Proporcionar una representació simple del món, dels agents, dels recursos i de l'evolució de mètriques principals. |
| Resultat esperat | Una eina de diagnòstic ràpida per detectar errors grossos, col·lapses poblacionals o dinàmiques inesperades. |
| Criteri de tancament | Es pot veure una execució o revisar-ne un resum visual sense confondre visualització amb prova científica. |
| Estat | Versió funcional inicial |
| Versió | 1.0 |
| Data | Juliol de 2026 |

## Índex

1. [Objectiu del document](#1-objectiu-del-document)
2. [Principi rector](#2-principi-rector)
3. [Abast de la visualització mínima](#3-abast-de-la-visualització-mínima)
4. [Elements que s'han de representar](#4-elements-que-shan-de-representar)
5. [Vista espacial del món](#5-vista-espacial-del-món)
6. [Visualització de mètriques temporals](#6-visualització-de-mètriques-temporals)
7. [Visualització de llinatges i genomes destacats](#7-visualització-de-llinatges-i-genomes-destacats)
8. [Modes de visualització](#8-modes-de-visualització)
9. [Relació amb el motor de simulació](#9-relació-amb-el-motor-de-simulació)
10. [Dades necessàries per visualitzar](#10-dades-necessàries-per-visualitzar)
11. [Requisits de rendiment](#11-requisits-de-rendiment)
12. [Errors que la visualització ha d'ajudar a detectar](#12-errors-que-la-visualització-ha-dajudar-a-detectar)
13. [Exclusions temporals](#13-exclusions-temporals)
14. [Criteris funcionals d'èxit](#14-criteris-funcionals-dèxit)
15. [Riscos principals](#15-riscos-principals)
16. [Resultat esperat](#16-resultat-esperat)
17. [Definició de tancament](#17-definició-de-tancament)

## 1. Objectiu del document

Aquest document defineix l'abast funcional de la visualització mínima del Projecte Evolució. La seva funció principal és ajudar a entendre què passa dins la simulació, detectar errors evidents i revisar de manera ràpida l'evolució de la població i dels recursos.

La visualització no substitueix les mètriques ni les proves de validació. Una simulació pot semblar interessant visualment i, alhora, estar produint resultats equivocats. Per això, aquesta capa s'ha de tractar com una eina de diagnòstic i comunicació, no com una demostració científica d'adaptació.

## 2. Principi rector

La primera visualització ha de ser simple, ràpida i robusta. No ha de condicionar el disseny del motor de simulació ni introduir dependències fortes dins el nucli. El motor ha de poder executar-se sense visualització, i la visualització ha de poder llegir estats o resultats sense alterar-los.

- La visualització ha d'ajudar a depurar, no a embellir el projecte.
- Ha de mostrar informació suficient per detectar problemes grossos de dinàmica.
- No ha d'incloure funcionalitats d'interfície avançada en aquesta fase.
- No ha de modificar l'estat del món ni dels agents.
- No ha de convertir-se en una dependència obligatòria per executar experiments llargs.

**Decisió recomanada:** començar amb sortides estàtiques i animacions simples generades a partir de dades de simulació. Una interfície interactiva completa ha de quedar fora de la primera versió.

## 3. Abast de la visualització mínima

La visualització mínima ha de cobrir dues necessitats diferents: observar l'estat espacial del món i revisar l'evolució temporal de les mètriques. Aquests dos nivells no s'han de barrejar en una única pantalla massa carregada.

| Nivell visual | Què mostra | Finalitat |
| --- | --- | --- |
| Vista espacial del món | Agents, recursos, zones buides i, si escau, condicions ambientals locals. | Detectar distribucions absurdes, acumulacions, buidatge de recursos o agents atrapats. |
| Vista temporal de mètriques | Població, energia mitjana, naixements, morts, recursos disponibles i diversitat aproximada. | Detectar col·lapse, explosió poblacional, estabilitat falsa o canvis de règim. |
| Vista de llinatges | Èxit reproductiu de llinatges destacats o genomes seleccionats. | Revisar si hi ha diferències acumulatives entre descendències. |
| Resum final d'execució | Indicadors agregats i gràfics principals d'una simulació acabada. | Comparar ràpidament experiments sense obrir dades crues. |
| Exportació visual | Imatges o animacions curtes associades a un experiment registrat. | Documentar resultats i facilitar revisió posterior. |

## 4. Elements que s'han de representar

La visualització no ha de mostrar tot el que existeix al sistema. Ha de seleccionar els elements que ajuden a entendre la dinàmica ecològica i evolutiva sense saturar l'usuari.

| Element | Representació mínima | Comentari funcional |
| --- | --- | --- |
| Món 2D | Graella o espai amb límits visibles. | Ha de quedar clar si el món és tancat, periòdic o amb frontera dura. |
| Recursos | Intensitat per cel·la o punts de recurs segons abundància. | Cal detectar si els recursos no es regeneren, es regeneren massa o apareixen en patrons erronis. |
| Agents vius | Punts o marcadors sobre el món. | No cal representar cada atribut; energia o edat poden codificar-se només en modes de diagnòstic. |
| Agents morts | No s'han de mantenir a la vista principal. | Les morts s'han de veure en mètriques o en una capa temporal opcional, no com a soroll permanent. |
| Naixements | Opcionalment, marcador breu o recompte temporal. | Útil per veure zones reproductives, però no imprescindible en cada fotograma. |
| Condició ambiental | Mapa de fons simple si el model inclou variació ambiental. | No s'ha d'afegir si encara no hi ha variable ambiental funcional. |
| Llinatges destacats | Color o etiqueta només per a un subconjunt seleccionat. | Mostrar tots els llinatges alhora pot fer la visualització il·legible. |
| Mètriques temporals | Gràfics de línia separats. | Han de permetre veure tendències sense interpretar-les com a prova directa d'adaptació. |

## 5. Vista espacial del món

La vista espacial ha de respondre una pregunta senzilla: què està passant ara mateix en el món simulat? Aquesta vista és especialment útil durant el desenvolupament perquè permet detectar errors que les mètriques agregades poden amagar.

- Els agents han d'aparèixer en la seva posició real dins el món.
- Els recursos han d'aparèixer amb una escala visual clara d'abundància.
- Els límits del món han de ser visibles o inferibles.
- La visualització ha de permetre distingir zones buides, zones saturades i zones riques en recursos.
- No ha de mostrar informació que l'agent no percep com si fos part de la seva observació; la vista global és per al desenvolupador, no per a l'agent.

| Problema detectable | Símptoma visual | Possible causa |
| --- | --- | --- |
| Agents acumulats en una vora | La població queda enganxada als límits. | Regla de moviment o frontera mal definida. |
| Recursos uniformement buits | El mapa queda sense recursos massa aviat. | Regeneració insuficient o consum excessiu. |
| Recursos sempre plens | No hi ha zones esgotades. | Pressió selectiva massa baixa o consum mal aplicat. |
| Agents immòbils | Els punts gairebé no canvien de posició. | Cost de moviment massa alt, acció mal resolta o xarxa saturada. |
| Explosió local | Molts agents neixen en una zona sense límit aparent. | Cost reproductiu massa baix o manca de restricció espacial. |
| Patrons geomètrics artificials | Formes massa regulars sense causa ecològica. | Error d'indexació, actualització síncrona o regeneració mal formulada. |

## 6. Visualització de mètriques temporals

La visualització temporal ha de mostrar l'evolució de les variables principals al llarg dels passos de simulació. Aquesta és la part més útil per entendre si la població s'estabilitza, col·lapsa, explota o respon a canvis ambientals.

| Mètrica | Forma visual recomanada | Ús diagnòstic |
| --- | --- | --- |
| Mida poblacional | Línia temporal. | Detectar col·lapse, explosió o estabilització. |
| Energia mitjana | Línia temporal amb possible rang o percentils. | Veure si la població viu al límit energètic o acumula energia massa fàcilment. |
| Naixements per pas | Línia o barres agregades per finestra temporal. | Identificar episodis reproductius i canvis de règim. |
| Morts per pas | Línia o barres, separable per causa. | Detectar pressions dominants: fam, edat, ambient o competència. |
| Recursos totals | Línia temporal. | Relacionar disponibilitat de recursos amb població i reproducció. |
| Diversitat genètica aproximada | Línia temporal. | Detectar convergència excessiva, deriva o pèrdua de variabilitat. |
| Descendència per llinatge | Rànquing o gràfic final. | Identificar llinatges dominants sense assumir que són adaptatius sense més proves. |
| Resposta a canvi ambiental | Línies abans i després del canvi. | Veure si el sistema mostra recuperació, col·lapse o substitució de llinatges. |

## 7. Visualització de llinatges i genomes destacats

La visualització de llinatges no ha de convertir-se en un arbre genealògic complet de tota la població, perquè això pot ser intractable i poc llegible. En la primera versió, només cal representar els llinatges més rellevants segons criteris funcionals.

- Mostrar els llinatges amb més descendència acumulada.
- Mostrar els llinatges que sobreviuen més passos temporals.
- Mostrar els llinatges que creixen després d'un canvi ambiental.
- Evitar representar tots els individus si la població és gran.
- Permetre vincular un llinatge visualitzat amb el seu identificador d'experiment i dades persistides.

| Visualització | Què aporta | Risc si s'abusa |
| --- | --- | --- |
| Rànquing de llinatges | Mostra qui deixa més descendència. | Pot confondre oportunisme local amb adaptació general. |
| Trajectòria d'un llinatge | Mostra persistència i expansió al llarg del temps. | Pot ignorar diversitat de llinatges petits però importants. |
| Mapa amb llinatges destacats | Relaciona ubicació i èxit reproductiu. | Pot ser il·legible si es mostren massa colors. |
| Comparació de genomes seleccionats | Permet revisar diferències entre genomes dominants. | No ha de substituir anàlisi genètica quantitativa. |

## 8. Modes de visualització

La primera versió pot tenir més d'un mode, però tots han de ser simples. No cal una interfície completa; és preferible generar visualitzacions consistents i reproduïbles a partir de dades guardades.

| Mode | Descripció | Prioritat |
| --- | --- | --- |
| Fotograma d'estat | Imatge d'un pas concret de la simulació amb món, recursos i agents. | Alta |
| Animació curta | Seqüència de fotogrames cada N passos per veure dinàmica espacial. | Alta |
| Dashboard estàtic | Conjunt de gràfics finals d'una execució. | Alta |
| Comparació d'experiments | Gràfics de mètriques de dues o més execucions. | Mitjana |
| Vista de llinatges | Rànquing o trajectòria dels llinatges principals. | Mitjana |
| Interfície interactiva | Panell navegable amb filtres i controls. | Baixa en el prototip inicial |

La prioritat baixa de la interfície interactiva no vol dir que no sigui útil, sinó que encara no és necessària per validar el motor evolutiu.

## 9. Relació amb el motor de simulació

La visualització ha d'estar desacoblada del motor. El motor produeix estats, mètriques i artefactes; la visualització els consumeix. Aquesta separació evita que la simulació depengui d'un entorn gràfic i permet executar experiments llargs sense cost visual innecessari.

| Component | Responsabilitat | No ha de fer |
| --- | --- | --- |
| Motor de simulació | Executar passos, actualitzar món, agents, recursos, naixements, morts i mètriques. | No ha de dibuixar ni decidir formats visuals. |
| Exportador d'estat | Produir snapshots o dades resumides per visualitzar. | No ha de modificar l'estat de la simulació. |
| Mòdul de visualització | Convertir dades en imatges, gràfics o animacions. | No ha de recalcular regles evolutives. |
| Persistència | Guardar dades i artefactes visuals associats a l'experiment. | No ha de ser una carpeta desordenada de fitxers sense identificador. |
| Anàlisi | Interpretar mètriques amb criteri estadístic. | No ha de dependre d'una inspecció visual subjectiva. |

## 10. Dades necessàries per visualitzar

La visualització mínima no necessita totes les dades internes. Ha de consumir un subconjunt clar i estable, suficient per reconstruir l'estat visual sense exposar detalls innecessaris del motor.

| Dada | Origen | Ús visual |
| --- | --- | --- |
| Dimensions del món | Configuració de l'experiment. | Definir escala i límits de la vista espacial. |
| Mapa de recursos | Estat de l'entorn. | Mostrar abundància i esgotament. |
| Posicions dels agents vius | Estat poblacional. | Representar distribució espacial. |
| Energia o edat dels agents | Estat dels agents. | Opcionalment codificar mida, intensitat o etiqueta diagnòstica. |
| Identificador de llinatge | Genealogia. | Destacar llinatges seleccionats. |
| Mètriques per pas | Sistema de mètriques. | Construir gràfics temporals. |
| Esdeveniments de naixement i mort | Registre del bucle de simulació. | Generar capes temporals o resums finals. |
| Identificador d'experiment | Registre d'experiments. | Vincular imatges i gràfics amb dades persistides. |

## 11. Requisits de rendiment

La visualització no ha de penalitzar la simulació. En experiments llargs, dibuixar cada pas pot ser molt més car que simular-lo. Per això cal distingir entre visualització en directe, mostreig periòdic i visualització posterior.

- La simulació ha de poder executar-se amb visualització desactivada.
- Els fotogrames s'han de generar cada N passos, no necessàriament a cada pas.
- Les animacions han de ser mostres resumides, no còpies completes de totes les dades.
- La generació d'imatges no ha d'alterar la llavor aleatòria ni l'ordre funcional del bucle.
- Els artefactes visuals han de tenir una mida raonable i estar vinculats a l'experiment.

| Escenari | Estratègia recomanada | Motiu |
| --- | --- | --- |
| Depuració curta | Visualització freqüent o animació densa. | Ajuda a veure errors immediats. |
| Experiment llarg | Snapshots cada N passos i gràfics finals. | Redueix cost i volum de dades. |
| Comparació de configuracions | Dashboards estàtics homogenis. | Facilita comparar sense biaix visual. |
| Presentació de resultats | Animació curta i gràfics seleccionats. | Comunica dinàmica sense sobrecarregar. |

## 12. Errors que la visualització ha d'ajudar a detectar

Una bona visualització mínima no és la més espectacular, sinó la que fa evidents els errors grossos abans que contaminin experiments llargs.

| Error potencial | Indicador visual | Acció posterior |
| --- | --- | --- |
| Població creix indefinidament | La línia poblacional puja sense estabilitzar-se i el mapa es satura. | Revisar energia, reproducció i límits espacials. |
| Extinció immediata | La població cau a zero en pocs passos. | Revisar energia inicial, recursos i costos metabòlics. |
| Cap moviment efectiu | Els agents queden immòbils tot i tenir acció de moviment. | Revisar sortida de xarxa, aplicació d'accions i costos. |
| Recursos mal regenerats | Patrons rígids o valors sempre extrems. | Revisar regla de regeneració i actualització del món. |
| Naixements sense cost | La població augmenta sense caiguda energètica associada. | Revisar cost reproductiu i energia del descendent. |
| Mort sense causa clara | Caigudes de població no reflectides en mètriques de mort. | Revisar registre de causes i ordre del bucle. |
| Llinatges impossibles | Un llinatge apareix sense progenitor o amb creixement incoherent. | Revisar genealogia i persistència. |
| Canvi ambiental invisible | Les mètriques indiquen canvi però el mapa no el reflecteix. | Revisar exportació de capes ambientals. |

## 13. Exclusions temporals

Per evitar que la visualització consumeixi massa temps de desenvolupament, queden fora de la primera versió les funcionalitats següents:

- Interfície web completa.
- Panells interactius avançats amb filtres complexos.
- Renderització 3D.
- Edició manual de paràmetres durant una execució.
- Reproducció interactiva pas a pas amb control complet del motor.
- Visualització exhaustiva de tots els genomes.
- Arbres genealògics complets per poblacions grans.
- Visualització en temps real obligatòria per a tots els experiments.
- Animacions d'alta qualitat pensades per divulgació externa.
- Inferències automàtiques d'adaptació basades només en gràfics.

## 14. Criteris funcionals d'èxit

La visualització mínima es considerarà implementada correctament si compleix aquests criteris:

1. Es pot generar una imatge del món en un pas determinat.
2. Els agents vius i els recursos es representen de manera clara.
3. Es poden generar gràfics temporals de les mètriques principals.
4. Es pot crear un resum visual final d'una execució.
5. Els artefactes visuals queden associats a l'identificador de l'experiment.
6. La simulació pot executar-se sense visualització activada.
7. La visualització no modifica l'estat del motor ni consumeix aleatorietat funcional.
8. Els gràfics ajuden a detectar col·lapse, explosió poblacional o escassetat de recursos.
9. La visualització de llinatges és limitada i interpretable.
10. El sistema evita presentar una imatge atractiva com si fos una prova científica.

## 15. Riscos principals

| Risc | Impacte | Mesura preventiva |
| --- | --- | --- |
| Confondre visualització amb validació | S'accepten resultats perquè “semblen bons”. | Exigir proves i mètriques abans d'interpretar adaptació. |
| Sobrecarregar la primera versió | La interfície retarda el motor evolutiu. | Limitar-se a imatges, animacions simples i gràfics finals. |
| Dependència forta del motor | La simulació no pot executar-se sense gràfics. | Separar motor, exportació i visualització. |
| Massa informació en pantalla | La visualització es torna il·legible. | Separar vista espacial, mètriques i llinatges. |
| Cost excessiu | Els experiments llargs es tornen lents. | Mostrejar snapshots i generar visualització posterior. |
| Biaix narratiu | Es seleccionen només gràfics bonics. | Guardar també mètriques completes i criteris objectius. |

## 16. Resultat esperat

El resultat d'aquest punt serà una capa de visualització mínima capaç de representar el món, els agents, els recursos i les mètriques principals d'una execució. Aquesta capa servirà per depurar el sistema, revisar dinàmiques generals i comunicar resultats preliminars de manera clara.

La visualització ha de permetre veure ràpidament si una execució té sentit funcional: si la població sobreviu, si els recursos s'esgoten o es regeneren, si hi ha naixements i morts, si apareixen col·lapses o si alguns llinatges destaquen.

Tanmateix, cap conclusió evolutiva s'haurà de basar només en la inspecció visual. La visualització és una porta d'entrada al diagnòstic; la interpretació final ha de dependre de mètriques, proves i comparacions entre experiments.

## 17. Definició de tancament

Aquest punt es considerarà tancat quan el projecte disposi d'una visualització mínima que permeti generar, com a mínim, una vista espacial del món i un resum temporal de mètriques per a cada execució rellevant.

- La vista espacial mostra món, agents i recursos.
- Els gràfics temporals mostren població, energia, naixements, morts i recursos.
- Els artefactes visuals queden vinculats a l'experiment registrat.
- La visualització pot executar-se sobre dades guardades, no només en temps real.
- La capa visual queda separada del nucli de simulació.
- El sistema manté explícit que la visualització és diagnòstica, no prova científica.
