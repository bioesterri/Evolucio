# Projecte Evolució — Analitzar resultats

**Especificació funcional per convertir mètriques, genealogies i repeticions experimentals en conclusions evolutives prudents i basades en dades.**

## Metadades

| Camp | Valor |
| --- | --- |
| Nom del document | Analitzar resultats |
| Tipus de document | Especificació funcional per al prototip inicial |
| Punt del full de ruta | Punt 18 del pla funcional |
| Àmbit | Prototip funcional de vida artificial evolutiva |
| Fase | Anàlisi experimental inicial |
| Propòsit | Examinar mètriques i genealogies per identificar avantatges de llinatge, diversitat genètica i resposta a canvis ambientals. |
| Nivell de detall | Funcional, sense estadística avançada ni implementació de codi. |
| Estat | Versió funcional per a revisió interna |
| Versió | 1.0 |
| Data | Juliol de 2026 |

## Índex

1. [Objectiu](#1-objectiu)
2. [Principi rector](#2-principi-rector)
3. [Abast de l'anàlisi](#3-abast-de-lanàlisi)
4. [Dades mínimes necessàries](#4-dades-mínimes-necessàries)
5. [Preguntes analítiques bàsiques](#5-preguntes-analítiques-bàsiques)
6. [Mètriques principals d'anàlisi](#6-mètriques-principals-danàlisi)
7. [Interpretació de genealogies](#7-interpretació-de-genealogies)
8. [Adaptació, deriva, error o atzar](#8-adaptació-deriva-error-o-atzar)
9. [Comparació entre experiments](#9-comparació-entre-experiments)
10. [Resposta a canvis ambientals](#10-resposta-a-canvis-ambientals)
11. [Visualitzacions analítiques mínimes](#11-visualitzacions-analítiques-mínimes)
12. [Procediment funcional d'anàlisi](#12-procediment-funcional-danàlisi)
13. [Criteris per parlar d'adaptació](#13-criteris-per-parlar-dadaptació)
14. [Senyals d'alerta](#14-senyals-dalerta)
15. [Resultat esperat de l'anàlisi](#15-resultat-esperat-de-lanàlisi)
16. [Exclusions temporals](#16-exclusions-temporals)
17. [Criteris de tancament](#17-criteris-de-tancament)
18. [Decisió funcional recomanada](#18-decisió-funcional-recomanada)

## 1. Objectiu

Aquest document defineix com s'han d'analitzar els resultats dels experiments inicials del Projecte Evolució. L'objectiu no és només descriure què ha passat en una simulació, sinó determinar si les dinàmiques observades tenen significat evolutiu o si són conseqüència de soroll, deriva, mala parametrització o errors del model.

L'anàlisi de resultats ha de convertir les dades generades pel motor de simulació en conclusions funcionals: quins llinatges prosperen, quins desapareixen, si la diversitat genètica es manté, si el consum energètic és coherent i si la població respon de manera mesurable als canvis ambientals.

## 2. Principi rector

Cap conclusió evolutiva s'ha d'acceptar només perquè una visualització sembli interessant. La visualització pot ajudar a detectar patrons, però la interpretació ha de basar-se en mètriques, genealogies, repeticions i comparacions controlades.

La pregunta central no és si els agents semblen intel·ligents, sinó si hi ha diferències heretables que afecten la supervivència i la reproducció de manera consistent sota unes condicions ambientals concretes.

## 3. Abast de l'anàlisi

L'anàlisi funcional dels resultats ha d'incloure cinc nivells principals:

- **Execució individual:** què ha passat en una simulació concreta i si les dades són coherents.
- **Comparació entre llavors:** si el patró es repeteix o depèn d'una inicialització afortunada.
- **Comparació entre configuracions:** com canvien els resultats quan es modifica un paràmetre del món, del metabolisme o de la mutació.
- **Anàlisi de llinatges:** quines branques genealògiques acumulen descendència i persistència temporal.
- **Anàlisi genètica i funcional:** si els genomes que prosperen són diferents dels que desapareixen i si això es relaciona amb comportaments observables.

## 4. Dades mínimes necessàries

Per poder analitzar resultats amb garanties, cada execució ha d'haver registrat com a mínim les dades següents:

- configuració completa de l'experiment;
- llavor aleatòria utilitzada;
- versió del model o del codi;
- sèrie temporal de mida poblacional;
- naixements, morts i causes de mort per pas temporal;
- energia mitjana, distribució energètica i consum de recursos;
- identificador de cada agent i del seu progenitor;
- genoma o resum genètic dels agents rellevants;
- mètriques de diversitat genètica;
- registre dels canvis ambientals aplicats;
- artefactes visuals o resums gràfics associats.

Si aquestes dades no existeixen o són incompletes, l'anàlisi només pot considerar-se exploratòria. No s'haurien d'extreure conclusions fortes d'un experiment mal registrat.

## 5. Preguntes analítiques bàsiques

L'anàlisi ha de respondre, com a mínim, aquestes preguntes:

1. La població sobreviu, col·lapsa o creix sense control?
2. El sistema manté una pressió selectiva real o els agents sobreviuen de manera gairebé gratuïta?
3. Hi ha llinatges que produeixen més descendència que altres?
4. Els llinatges exitosos persisteixen en el temps o només apareixen per fluctuacions curtes?
5. La diversitat genètica augmenta, es manté o s'esgota ràpidament?
6. Els canvis ambientals modifiquen la supervivència i la reproducció?
7. Les mateixes configuracions produeixen patrons semblants amb llavors diferents?
8. Hi ha indicis d'errors de model, com energia creada del no-res o naixements impossibles?
9. Els resultats permeten ajustar paràmetres o cal revisar la definició del model?

## 6. Mètriques principals d'anàlisi

Les mètriques no s'han d'interpretar aïlladament. Una població estable pot amagar manca de selecció, i un llinatge molt abundant pot ser resultat d'un coll d'ampolla inicial. Per això cal analitzar les mètriques en conjunt.

### 6.1. Mètriques poblacionals

- Mida poblacional: nombre d'agents vius per pas temporal.
- Taxa de creixement: canvi relatiu de població entre intervals.
- Estabilitat poblacional: capacitat de mantenir-se dins d'un rang sense col·lapse ni explosió.
- Temps fins al col·lapse: pas temporal en què la població s'extingeix, si passa.
- Variabilitat entre llavors: grau de diferència entre execucions equivalents.

### 6.2. Mètriques reproductives

- nombre total de naixements;
- naixements per pas temporal;
- descendència acumulada per agent;
- descendència acumulada per llinatge;
- edat mitjana en el moment de la primera reproducció;
- proporció d'agents que es reprodueixen almenys una vegada;
- distribució de l'èxit reproductiu.

### 6.3. Mètriques de supervivència

- vida mitjana dels agents;
- distribució d'edats de mort;
- causes de mort agregades;
- supervivència mitjana per llinatge;
- supervivència abans i després de canvis ambientals;
- proporció d'agents morts abans de reproduir-se.

### 6.4. Mètriques energètiques

- energia mitjana i mediana de la població;
- distribució d'energia entre agents;
- energia consumida per moviment, metabolisme i reproducció;
- recursos consumits per pas temporal;
- recursos disponibles al món;
- eficiència aproximada: descendència generada per energia consumida.

### 6.5. Mètriques genètiques

- distància genètica mitjana entre agents vius;
- diversitat genètica per generació o finestra temporal;
- nombre de llinatges vius;
- dominància dels llinatges més abundants;
- canvi genètic acumulat respecte a la població inicial;
- persistència de variants genètiques al llarg del temps.

## 7. Interpretació de genealogies

La genealogia és una peça central de l'anàlisi perquè permet separar l'èxit individual puntual de l'èxit evolutiu acumulat. Un agent que viu molt però no deixa descendència té una importància evolutiva limitada. En canvi, un agent amb vida curta però molts descendents pot fundar un llinatge rellevant.

L'anàlisi genealògica ha d'identificar:

- llinatges fundadors i branques derivades;
- llinatges extingits i moment d'extinció;
- llinatges dominants per nombre de descendents vius;
- llinatges resilients després de canvis ambientals;
- colls d'ampolla genealògics;
- substitució d'un llinatge dominant per un altre.

## 8. Adaptació, deriva, error o atzar

El criteri de tancament d'aquest bloc exigeix poder distingir situacions que poden semblar semblants si només es mira una gràfica de població.

| Situació | Indicadors | Interpretació funcional |
| --- | --- | --- |
| Adaptació probable | Un o més llinatges mostren avantatge reproductiu o de supervivència repetible entre llavors i sota pressió ambiental comparable. | Pot haver-hi selecció acumulativa. Cal confirmar amb comparació genètica i controls. |
| Deriva genètica | Canvis de freqüència de llinatges sense avantatge consistent; resultats molt dependents de la llavor. | El patró pot ser real però no necessàriament adaptatiu. |
| Error de model | Energia impossible, població explosiva sense recursos, morts incoherents, naixements sense cost o mètriques contradictòries. | No s'han d'interpretar resultats evolutius fins a corregir el motor. |
| Atzar estadístic | Un efecte apareix en una execució però desapareix en repeticions equivalents. | Cal augmentar repeticions o reduir conclusions. |
| Mala parametrització | Col·lapse immediat o saturació permanent en gairebé totes les execucions. | El model pot ser correcte però els paràmetres no permeten observar evolució útil. |

## 9. Comparació entre experiments

Les comparacions han de fer-se canviant pocs paràmetres alhora. Si es modifiquen massa elements simultàniament, no es podrà atribuir cap diferència a una causa concreta.

- **Comparació base:** execució amb configuració estàndard i diverses llavors.
- **Comparació de recursos:** canvis en abundància, regeneració o distribució espacial.
- **Comparació de mutació:** taxes de mutació baixes, mitjanes i altes.
- **Comparació metabòlica:** cost basal, cost de moviment i cost de reproducció.
- **Comparació ambiental:** presència o absència de canvis ambientals simples.
- **Comparació de percepció:** radi perceptiu o quantitat d'informació local disponible.

Per a cada configuració s'haurien d'executar diverses llavors. Com a recomanació inicial, cinc llavors poden servir per detectar errors grossos, però deu o més són preferibles quan es vulguin comparar tendències amb més confiança.

## 10. Resposta a canvis ambientals

Si l'entorn incorpora canvis ambientals, l'anàlisi ha de distingir tres fases: abans del canvi, durant la transició i després del canvi. No n'hi ha prou amb mirar el resultat final.

- mida poblacional abans, durant i després del canvi;
- mortalitat associada al canvi;
- llinatges que desapareixen durant la transició;
- llinatges que incrementen la seva freqüència després del canvi;
- temps de recuperació poblacional;
- pèrdua o manteniment de diversitat genètica;
- canvis en consum de recursos i energia mitjana.

Una resposta adaptativa no s'ha de confondre amb simple supervivència. La població pot sobreviure a un canvi ambiental per abundància inicial, no perquè hagi evolucionat una estratègia millor.

## 11. Visualitzacions analítiques mínimes

Les visualitzacions recomanades per a l'anàlisi no són les mateixes que la visualització del món en temps real. Aquí l'objectiu és interpretar dades acumulades.

- gràfica de mida poblacional al llarg del temps;
- gràfica de naixements i morts per pas temporal;
- distribució de causes de mort;
- energia mitjana i dispersió al llarg del temps;
- recursos disponibles i recursos consumits;
- abundància relativa dels llinatges principals;
- diversitat genètica al llarg del temps;
- arbre genealògic simplificat dels llinatges dominants;
- comparació de mètriques entre configuracions i llavors.

## 12. Procediment funcional d'anàlisi

L'anàlisi ha de seguir un ordre estable per evitar conclusions precipitades.

1. Validar que l'execució té configuració, llavor i versió registrades.
2. Comprovar que no hi ha incoherències bàsiques: energia negativa impossible, naixements sense progenitor o agents vius duplicats.
3. Analitzar la dinàmica poblacional global.
4. Examinar naixements, morts i causes de mort.
5. Analitzar consum energètic i disponibilitat de recursos.
6. Revisar genealogies i llinatges dominants.
7. Examinar diversitat genètica i canvis de freqüència genètica.
8. Comparar amb altres llavors de la mateixa configuració.
9. Comparar amb configuracions alternatives.
10. Formular conclusions amb grau de confiança i limitacions explícites.

## 13. Criteris per parlar d'adaptació

Només s'hauria de parlar d'adaptació quan es compleixin diversos indicadors alhora. Un sol indicador no és suficient.

- els llinatges exitosos tenen més descendència o més supervivència que la mitjana;
- l'avantatge es manté durant més d'una finestra temporal curta;
- el patró apareix en diverses llavors o configuracions comparables;
- hi ha una diferència genètica detectable entre llinatges exitosos i no exitosos;
- el patró canvia quan canvia la pressió ambiental;
- no hi ha errors de registre o de regles que expliquin el resultat.

Encara que aquests criteris es compleixin, la conclusió ha de ser prudent. En aquesta fase inicial és preferible parlar d'indicis d'adaptació que no pas d'adaptació demostrada de manera forta.

## 14. Senyals d'alerta

Hi ha resultats que han d'aturar la interpretació i obligar a revisar el model o les dades:

- població que creix indefinidament sense limitació de recursos;
- extinció immediata en gairebé totes les execucions;
- morts sense causa registrada;
- naixements sense cost energètic aplicat;
- descendents sense progenitor vàlid;
- agents amb energia fora dels límits permesos;
- diversitat genètica nul·la quan hi ha mutació activa;
- diversitat explosiva sense cap persistència de llinatges;
- canvis massius de població que no coincideixen amb cap regla del sistema;
- resultats completament diferents entre llavors sense explicació estructural.

## 15. Resultat esperat de l'anàlisi

Cada bloc d'anàlisi ha de produir una sortida clara i revisable. El resultat no hauria de ser només una col·lecció de gràfiques, sinó una interpretació funcional amb evidències i límits.

- resum de configuracions analitzades;
- taula de mètriques principals per execució;
- comparació entre llavors;
- identificació de llinatges dominants o persistents;
- avaluació de diversitat genètica;
- diagnòstic de col·lapse, estabilitat o creixement excessiu;
- indicis o absència d'adaptació;
- possibles errors o punts febles del model;
- recomanacions per als següents experiments.

## 16. Exclusions temporals

En aquesta fase no s'ha d'intentar encara:

- demostrar intel·ligència general dels agents;
- atribuir intencions o estratègies conscients als comportaments observats;
- fer anàlisis estadístiques massa sofisticades abans de validar el motor;
- interpretar una sola execució com a evidència forta;
- optimitzar automàticament paràmetres sense entendre primer les dinàmiques bàsiques;
- comparar resultats si no s'han conservat configuració, llavor i versió del model.

## 17. Criteris de tancament

Aquest bloc es considerarà complet quan l'equip pugui analitzar un conjunt d'experiments i produir conclusions funcionals basades en dades. Concretament, s'ha de poder:

- validar que les dades de cada execució són completes i coherents;
- comparar diverses llavors d'una mateixa configuració;
- identificar estabilitat, col·lapse o creixement excessiu;
- reconèixer llinatges amb avantatge reproductiu o de supervivència;
- avaluar si la diversitat genètica es manté o s'esgota;
- mesurar la resposta poblacional i genealògica a canvis ambientals;
- distingir adaptació probable, deriva, error de model i atzar estadístic;
- generar un informe d'anàlisi que indiqui resultats, limitacions i pròxims passos.

## 18. Decisió funcional recomanada

La primera fase d'anàlisi ha de ser conservadora: poques configuracions, diverses llavors, mètriques clares i conclusions modestes. És millor detectar aviat que el model encara no produeix adaptació mesurable que no pas construir una narrativa evolutiva sobre dades febles.

La recomanació principal és establir una plantilla fixa d'informe per a cada lot d'experiments. Aquesta plantilla hauria d'incloure configuració, llavors, gràfiques bàsiques, mètriques resum, interpretació de llinatges, diagnòstic de qualitat i decisió sobre el següent experiment.
