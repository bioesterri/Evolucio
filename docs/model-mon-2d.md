# Projecte Evolució — Model del món 2D

**Definició funcional d'un entorn simple, repetible i ecològicament significatiu que generi pressió selectiva real en el prototip inicial.**

## Metadades

| Camp | Valor |
| --- | --- |
| Tipus de document | Especificació funcional del món 2D |
| Punt del full de ruta | Punt 3 — Definir el model del món 2D |
| Àmbit | Definició funcional del món 2D del prototip inicial |
| Objectiu | Definir un entorn simple, repetible i ecològicament significatiu que generi pressió selectiva real. |
| Nivell de detall | Funcional; no entra encara en classes, APIs ni implementació JAX detallada |
| Decisió recomanada | Món 2D discret, amb recursos escassos i distribució espacial irregular |
| Criteri de tancament | El món es pot inicialitzar, evolucionar i afectar els agents de manera clara i mesurable. |
| Estat | Especificació funcional inicial |
| Data | 2 de juliol de 2026 |

## Decisió funcional principal

El primer món ha de ser una graella 2D discreta, limitada i sense física realista. Aquesta decisió redueix el cost tècnic, facilita la depuració i permet mesurar clarament la relació entre recursos, energia, mort i reproducció.

## 1. Objectiu del document

Aquest document defineix funcionalment el món 2D del prototip inicial del Projecte Evolució. El món és l'espai ecològic on els agents viuen, es desplacen, consumeixen recursos, competeixen indirectament, pateixen condicions ambientals i deixen descendència si aconsegueixen prou energia.

La funció principal del món no és decorar la simulació, sinó crear pressió selectiva. Si l'entorn no limita els recursos, no canvia mai o no penalitza decisions dolentes, els agents poden sobreviure per inèrcia i l'evolució observada serà feble o falsa.

## 2. Decisió de model per al prototip inicial

Per a la primera versió es recomana un món 2D discret representat com una graella de cel·les. Cada cel·la pot contenir una quantitat de recurs, un valor ambiental i, segons les regles d'ocupació, cap, un o diversos agents.

| Aspecte | Decisió recomanada | Motiu funcional |
| --- | --- | --- |
| Tipus d'espai | Graella 2D discreta | És més fàcil de validar, visualitzar i vectoritzar en una primera versió. |
| Límits | Límits tancats, no toroidals | Evita artefactes ecològics com sortir per un costat i aparèixer per l'altre. |
| Recursos | Un sol tipus de recurs energètic | Permet provar selecció sense afegir química o dietes complexes. |
| Canvi ambiental | Simple, controlat i configurable | Permet provar adaptació sense convertir l'entorn en un simulador climàtic. |
| Pressió selectiva | Escassetat, distribució irregular i cost energètic | Obliga els agents a diferir en supervivència i reproducció. |

## 3. Responsabilitats funcionals del món

El món 2D ha de tenir responsabilitats clares i limitades. No ha de decidir la conducta dels agents ni contenir lògica neuronal.

| Responsabilitat | Descripció funcional |
| --- | --- |
| Representar l'espai | Mantenir les dimensions, coordenades, límits i cel·les disponibles. |
| Mantenir recursos | Guardar la quantitat de recurs disponible a cada cel·la i actualitzar-la en el temps. |
| Mantenir condicions ambientals | Guardar valors ambientals simples que puguin afectar costos, mortalitat o regeneració de recursos. |
| Aplicar regles físiques mínimes | Validar moviments, impedir posicions fora del món i resoldre ocupació local. |
| Proporcionar observacions locals | Permetre extreure informació parcial de l'entorn per alimentar la xarxa neuronal dels agents. |
| Registrar canvis ambientals | Exposar dades suficients perquè les mètriques puguin mesurar l'estat ecològic del sistema. |

**Límit important:** el món no ha de seleccionar directament els agents. La selecció ha d'emergir del fet que alguns agents aconsegueixen més energia, viuen més temps i es reprodueixen més en unes condicions ambientals determinades.

## 4. Dimensions i coordenades

El món tindrà una amplada i una alçada configurables. Cada agent ocuparà una posició definida per coordenades enteres dins la graella. La mida inicial no s'ha de triar per realisme biològic, sinó per equilibri entre cost computacional, densitat poblacional i capacitat d'observar patrons.

| Paràmetre | Funció | Recomanació inicial |
| --- | --- | --- |
| Amplada del món | Nombre de cel·les horitzontals | Configurable; començar amb una mida moderada, per exemple 64-128 cel·les. |
| Alçada del món | Nombre de cel·les verticals | Configurable; preferiblement igual o semblant a l'amplada. |
| Coordenades | Posició espacial dels agents i recursos | Enteres: `x`, `y`. |
| Escala temporal | Unitat d'actualització del sistema | Passos discrets o ticks. |

## 5. Límits del món

La primera versió ha d'utilitzar límits tancats. Quan un agent intenta sortir del món, el moviment es considera invàlid o queda bloquejat. Aquesta opció és menys elegant que un món toroidal, però és més interpretable ecològicament.

| Opció | Decisió per al prototip | Comentari crític |
| --- | --- | --- |
| Límit tancat | Inclòs | És el comportament recomanat per a la primera versió. |
| Món toroidal | Exclòs temporalment | Pot crear dinàmiques artificials difícils d'interpretar ecològicament. |
| Barreres internes | Excloses temporalment | Afegeixen complexitat espacial abans de validar el bucle evolutiu. |
| Terreny continu | Exclòs temporalment | Incrementa el cost sense aportar gaire al primer prototip. |

## 6. Capes funcionals del món

El món es pot entendre com un conjunt de capes superposades. Aquesta separació facilita que més endavant es puguin afegir nous factors sense reescriure tota la simulació.

| Capa | Contingut | Ús funcional |
| --- | --- | --- |
| Capa espacial | Dimensions, coordenades i límits | Defineix on poden existir agents i recursos. |
| Capa de recursos | Quantitat de recurs per cel·la | Determina oportunitats d'alimentació i competència. |
| Capa ambiental | Valor simple de qualitat, estrès o penalització local | Modula regeneració, consum energètic o risc de mort. |
| Capa d'ocupació | Mapa de presència o densitat d'agents | Permet detectar saturació, competència local i observacions de veïnat. |
| Capa de mètriques | Agregats ecològics del pas temporal | Permet analitzar recursos, consum, regeneració i pressió ambiental. |

## 7. Recursos

Els recursos són el mecanisme principal que connecta món i evolució. Han de ser limitats, localitzats i regenerables. Si els recursos són uniformes i abundants, moure's o decidir bé tindrà poc valor adaptatiu.

### 7.1. Representació del recurs

Cada cel·la tindrà una quantitat numèrica de recurs. Aquest recurs representa energia potencial. Quan un agent s'alimenta, part o tota aquesta quantitat es transfereix a la seva reserva energètica segons les regles definides pel sistema.

| Element | Descripció funcional |
| --- | --- |
| Quantitat actual | Recurs disponible en una cel·la en un moment concret. |
| Capacitat màxima | Límit superior de recurs que una cel·la pot acumular. |
| Valor energètic | Conversió funcional entre recurs consumit i energia obtinguda per l'agent. |
| Regeneració | Procés pel qual el recurs torna a créixer amb el temps. |

### 7.2. Distribució inicial

La distribució inicial dels recursos no ha de ser completament uniforme. Es recomana una distribució irregular amb taques o gradients suaus. Això crea zones favorables i desfavorables, i obliga els agents a respondre a l'estructura espacial.

| Distribució | Ús en el prototip | Valoració |
| --- | --- | --- |
| Uniforme | Només per proves de control | Útil per depurar, però ecològicament pobra. |
| Aleatòria simple | Acceptable com a punt de partida | Genera variabilitat, però pot ser massa sorollosa. |
| Taques de recursos | Recomanada | Crea gradients ecològics i competència espacial. |
| Gradient global | Opcional | Permet estudiar migració o preferència espacial. |

### 7.3. Regeneració

Els recursos s'han de regenerar amb una regla simple i configurable. La regeneració ha de ser prou lenta perquè el consum dels agents tingui conseqüències, però no tan lenta que la població col·lapsi sempre per manca absoluta d'energia.

- Cada cel·la té una capacitat màxima de recurs.
- Cada pas temporal, el recurs pot augmentar segons una taxa de regeneració.
- La regeneració pot dependre de la qualitat ambiental local.
- La regeneració no pot superar la capacitat màxima de la cel·la.
- El consum dels agents redueix el recurs disponible.

## 8. Condicions ambientals

El prototip ha d'incloure condicions ambientals simples, però no un clima complet. L'objectiu és que el món pugui canviar i que aquests canvis modifiquin les possibilitats de supervivència dels agents.

| Factor ambiental | Funció | Recomanació inicial |
| --- | --- | --- |
| Qualitat local | Valor que indica si una zona és favorable o desfavorable | Incloure com a camp numèric simple. |
| Estrès ambiental | Penalització sobre metabolisme o supervivència | Incloure només si es pot mesurar clarament. |
| Canvi temporal | Variació de la qualitat o dels recursos al llarg del temps | Incloure com a escenari controlat. |
| Esdeveniment brusc | Pertorbació ambiental puntual | Opcional per a proves d'adaptabilitat. |

### 8.1. Tipus de canvi ambiental

Per a la primera versió, el canvi ambiental ha de ser simple i reproduïble. Es recomanen tres escenaris, no tots obligatoris alhora.

| Escenari | Descripció | Ús experimental |
| --- | --- | --- |
| Entorn estable | Els recursos es regeneren amb les mateixes regles durant tota la simulació. | Control de base. |
| Entorn amb canvi gradual | La zona més favorable es desplaça o varia lentament. | Mesura adaptació a canvis progressius. |
| Entorn amb pertorbació | En un pas temporal definit, una part del món perd recursos o qualitat. | Mesura robustesa i recuperació poblacional. |

**Criteri científic:** els canvis ambientals han de ser registrats i reproduïbles. Un entorn que canvia de manera massa arbitrària pot impedir distingir adaptació real de simple sort.

## 9. Ocupació espacial i densitat

El món ha de permetre saber quins agents es troben a cada zona. Aquesta informació és necessària per calcular observacions locals, competència i possibles bloquejos de moviment.

| Regla | Decisió recomanada | Motiu |
| --- | --- | --- |
| Ocupació per cel·la | Permetre com a mínim detectar densitat local | La competència pot existir encara que diversos agents comparteixin zona. |
| Moviment invàlid | Bloquejar sortides del món | Manté coherència espacial. |
| Saturació local | Opcional en la primera versió | Pot afegir competència, però no és imprescindible al principi. |
| Col·lisions complexes | Excloses temporalment | No cal simular física d'interacció entre cossos. |

Una decisió pràctica és permetre més d'un agent per cel·la durant el prototip inicial i representar la competència mitjançant consum de recursos i densitat local. Limitar estrictament una cel·la a un sol agent pot complicar el moviment i la reproducció abans d'hora.

## 10. Inicialització del món

El món s'ha de poder inicialitzar sempre a partir d'una configuració explícita i una llavor aleatòria. Això és imprescindible per repetir experiments i comparar resultats.

| Element inicial | Ha de quedar definit |
| --- | --- |
| Dimensions | Amplada i alçada del món. |
| Llavor aleatòria | Valor que permet reproduir la mateixa inicialització. |
| Distribució de recursos | Tipus de patró inicial i quantitat total aproximada. |
| Capacitat màxima de recurs | Límit per cel·la. |
| Taxa de regeneració | Velocitat de recuperació dels recursos. |
| Mapa ambiental | Valor inicial de qualitat o estrès per cel·la. |
| Posicions inicials dels agents | Distribució espacial inicial, encara que la creació dels agents pertanyi al bloc de simulació. |

## 11. Evolució temporal del món

El món evoluciona en passos discrets. En cada pas temporal, cal aplicar de manera consistent les accions dels agents, el consum de recursos, la regeneració i els possibles canvis ambientals.

| Fase del pas temporal | Descripció funcional |
| --- | --- |
| 1. Estat inicial del pas | El sistema disposa del mapa de recursos, condicions ambientals i posicions dels agents. |
| 2. Observacions locals | Cada agent rep informació limitada derivada del món proper. |
| 3. Accions dels agents | Els agents intenten moure's, alimentar-se o reproduir-se segons la sortida neuronal. |
| 4. Resolució espacial | El món valida moviments, límits i consum local. |
| 5. Actualització de recursos | El recurs consumit es resta i la regeneració s'aplica segons la regla definida. |
| 6. Canvi ambiental | S'apliquen variacions ambientals si l'escenari ho preveu. |
| 7. Mètriques ecològiques | Es registren totals, gradients, consum i estat ambiental. |

**Ordre de càlcul:** l'ordre exacte del pas temporal s'haurà de fixar tècnicament més endavant. Funcionalment, el més important és que sigui estable, documentat i idèntic entre execucions amb la mateixa llavor.

## 12. Observacions que el món proporciona als agents

Els agents no han de veure tot el món. El món només ha de proporcionar observacions locals, suficients perquè les decisions tinguin sentit però no tan completes que eliminin la pressió evolutiva.

| Observació local | Descripció | Incloure al prototip |
| --- | --- | --- |
| Recurs a la cel·la actual | Quantitat disponible on és l'agent. | Sí |
| Recursos al veïnat proper | Informació agregada o direccional dins un radi petit. | Sí |
| Densitat local d'agents | Nombre aproximat d'agents propers. | Sí |
| Qualitat ambiental local | Valor ambiental de la posició actual o del veïnat. | Sí, si s'activa la capa ambiental |
| Mapa global | Coneixement complet del món. | No |
| Genealogia d'altres agents | Informació sobre llinatges veïns. | No |

## 13. Com el món afecta la supervivència

El món afecta la supervivència dels agents de manera indirecta però mesurable. No cal que el món “decideixi” qui viu; n'hi ha prou que imposi condicions que facin més o menys probable obtenir energia i reproduir-se.

| Mecanisme | Efecte sobre agents |
| --- | --- |
| Escassetat de recursos | Agents mal posicionats o ineficients acumulen menys energia. |
| Distribució irregular | Agents capaços d'aproximar-se a zones favorables tenen avantatge. |
| Regeneració limitada | El sobreconsum local pot esgotar una zona. |
| Densitat local alta | Més agents competeixen pels mateixos recursos. |
| Ambient desfavorable | Pot incrementar el cost energètic o reduir recursos disponibles. |
| Canvi temporal | Estratègies rígides poden fallar quan l'entorn canvia. |

## 14. Pressió selectiva mínima esperada

El model del món 2D ha de generar com a mínim quatre pressions selectives bàsiques:

1. **Pressió per localitzar recursos:** els agents que es mouen cap a zones més riques obtenen més energia.
2. **Pressió per evitar moviment inútil:** moure's sense guany energètic ha de ser costós.
3. **Pressió per reproduir-se en el moment adequat:** reproduir-se massa aviat o massa tard pot ser perjudicial.
4. **Pressió per tolerar canvis ambientals:** els llinatges massa dependents d'una situació estable poden perdre avantatge quan l'entorn canvia.

## 15. Mètriques pròpies del món

A més de les mètriques dels agents, el món ha de generar mètriques ecològiques pròpies. Aquestes dades permeten saber si la població evoluciona en resposta a una pressió real o només fluctua aleatòriament.

| Mètrica | Interpretació |
| --- | --- |
| Recurs total disponible | Energia potencial present al món en cada pas. |
| Recurs mitjà per cel·la | Abundància global del sistema. |
| Variància espacial del recurs | Grau d'irregularitat o formació de taques. |
| Recurs consumit per pas | Pressió de consum exercida pels agents. |
| Recurs regenerat per pas | Capacitat de recuperació del món. |
| Percentatge de cel·les esgotades | Indicador de sobreexplotació. |
| Densitat mitjana d'agents | Pressió poblacional sobre l'espai. |
| Densitat en zones riques | Mesura si els agents s'acumulen en zones favorables. |
| Intensitat ambiental mitjana | Estat general de la capa ambiental. |
| Impacte d'una pertorbació | Canvi en recursos, morts o reproducció després d'un canvi ambiental. |

## 16. Paràmetres configurables

Els paràmetres del món han de viure en una configuració explícita. No s'han d'amagar com a constants internes perquè això impediria comparar experiments.

| Grup | Paràmetres |
| --- | --- |
| Geometria | Amplada, alçada, tipus de límit. |
| Recursos | Quantitat inicial, capacitat màxima, taxa de regeneració, patró espacial. |
| Ambient | Activació de capa ambiental, intensitat, taxa de canvi, calendari de pertorbacions. |
| Ocupació | Regla de densitat, radi de veïnat, resolució de moviments invàlids. |
| Observació | Radi perceptiu, variables observables, agregació local. |
| Experiment | Llavor aleatòria, durada, escenari ambiental. |

## 17. Escenaris mínims de prova

Per validar el món, no n'hi ha prou amb una sola simulació. Calen escenaris simples que permetin detectar errors i comparar comportaments.

| Escenari | Finalitat |
| --- | --- |
| Món buit sense agents | Comprovar regeneració i estabilitat de recursos. |
| Agents sense reproducció | Mesurar consum, moviment i mortalitat sense evolució. |
| Recursos uniformes | Control per veure si el sistema no força patrons artificials. |
| Recursos en taques | Comprovar si apareix competència espacial. |
| Canvi ambiental gradual | Mesurar si la població respon a una alteració progressiva. |
| Pertorbació sobtada | Mesurar resiliència i recuperació després d'un xoc. |

## 18. Exclusions temporals

Queden fora del model inicial del món 2D:

- món continu amb coordenades reals;
- física realista de moviment;
- obstacles complexos i topografia;
- múltiples nutrients o recursos químicament diferenciats;
- clima avançat amb moltes variables;
- estacionalitat biològicament detallada;
- catàstrofes aleatòries no controlades;
- interaccions espacials complexes com empentes, col·lisions o territoris;
- visualització interactiva com a part del món;
- decisions intel·ligents hardcodejades dins l'entorn.

## 19. Riscos de disseny

| Risc | Conseqüència | Mesura preventiva |
| --- | --- | --- |
| Recursos massa abundants | La supervivència deixa de dependre de la conducta. | Ajustar quantitat, regeneració i cost energètic. |
| Recursos massa escassos | La població col·lapsa abans que hi hagi evolució. | Fer proves de viabilitat amb agents aleatoris. |
| Entorn massa uniforme | No hi ha avantatge espacial clar. | Usar taques o gradients moderats. |
| Canvis ambientals massa caòtics | La selecció queda dominada per soroll. | Fer canvis controlats i registrats. |
| Massa variables ambientals | Difícil saber què causa cada resultat. | Començar amb una sola dimensió ambiental. |
| Observació massa global | Els agents tenen massa informació. | Limitar la percepció a un radi local. |

## 20. Criteris de tancament

La definició del model del món 2D es considerarà tancada quan es compleixin aquests criteris:

1. Les dimensions i límits del món estan definits funcionalment.
2. El sistema de recursos està definit: quantitat, distribució, consum i regeneració.
3. La capa ambiental té una decisió clara: inclosa de manera simple o ajornada explícitament.
4. Les regles de moviment i ocupació espacial són interpretables.
5. Els agents poden obtenir observacions locals del món.
6. El món pot afectar energia, supervivència i reproducció de manera mesurable.
7. Els canvis ambientals, si s'activen, són configurables i reproduïbles.
8. Hi ha mètriques ecològiques suficients per interpretar els resultats.
9. Els escenaris de prova permeten detectar si el món és massa fàcil, massa hostil o massa sorollós.
10. Tot el model pot inicialitzar-se a partir d'una configuració i una llavor aleatòria.

## 21. Resultat funcional esperat

El resultat d'aquest punt ha de ser un model de món 2D que pugui actuar com a entorn ecològic mínim. Ha de permetre inicialitzar una graella, distribuir recursos, actualitzar-los al llarg del temps, aplicar condicions ambientals simples i oferir observacions locals als agents.

El món no ha de ser biològicament complet, però sí prou restrictiu per generar diferències reals entre agents. Els individus que consumeixin millor, es moguin amb menys malbaratament i es reprodueixin en condicions favorables haurien de deixar més descendència que els individus menys eficients.

## 22. Adequació amb la resta de documentació

El document és coherent amb la resta de la documentació funcional del projecte:

- **Alineació amb el document funcional inicial:** desenvolupa el punt del full de ruta dedicat a definir el model del món 2D i manté el criteri que l'entorn ha d'afectar supervivència i reproducció de manera mesurable.
- **Alineació amb l'abast del prototip inicial:** concreta el component “Món 2D” sense ampliar l'abast més enllà del prototip mínim; manté recursos limitats, observacions locals, energia i condicions ambientals simples.
- **Alineació amb l'arquitectura general:** respecta la separació de responsabilitats entre entorn, agents, xarxa neuronal, simulació i mètriques. El món proporciona condicions i observacions, però no decideix estratègies ni selecciona agents directament.

### Valoració funcional

La proposta és adequada per al prototip inicial perquè és simple, mesurable i prou restrictiva per generar pressió selectiva. La decisió de començar amb una graella discreta, límits tancats, un únic recurs energètic i canvis ambientals controlats redueix risc tècnic i facilita proves repetibles.

Els punts més forts són:

- delimita clarament què fa i què no fa el món;
- evita introduir física, clima o interaccions complexes abans de validar el bucle evolutiu;
- reforça la importància de recursos escassos, distribució irregular i regeneració limitada;
- defineix mètriques ecològiques específiques, necessàries per interpretar si hi ha selecció real;
- manté la compatibilitat amb un futur nucli vectoritzable i amb experiments reproduïbles per configuració i llavor.

Els principals punts a vigilar en la fase tècnica posterior són:

- fixar l'ordre exacte d'actualització del pas temporal per evitar ambigüitats experimentals;
- traduir les observacions locals en una interfície estable per a la xarxa neuronal;
- calibrar recursos, regeneració i costos energètics per evitar tant supervivència trivial com col·lapse sistemàtic;
- decidir si la capa ambiental entra en la primera implementació o queda darrere d'un paràmetre d'activació;
- assegurar que les mètriques del món es registren sense introduir dependències cap a persistència o visualització.

**Conclusió operativa:** el món 2D inicial ha de ser simple, discret i mesurable. La complexitat ecològica vindrà més endavant. Ara cal garantir que l'entorn crea pressió selectiva suficient per fer visible l'evolució poblacional.
