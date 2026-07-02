# Projecte Evolució — Observacions disponibles per als agents

**Especificació funcional del sistema perceptiu local dels agents: informació parcial, local, normalitzada i de mida fixa abans de decidir una acció.**

## Metadades

| Camp | Valor |
| --- | --- |
| Tipus de document | Especificació funcional del sistema perceptiu local dels agents |
| Punt del full de ruta | Punt 6 — Definir les observacions disponibles per als agents |
| Àmbit | Definició de la informació observable pels agents abans de la decisió neuronal |
| Objectiu | Definir quina informació pot percebre cada agent abans de decidir una acció. |
| Resultat esperat | Agents que decideixen amb informació parcial, local i mesurable, sense accés a avantatges artificials. |
| Decisió recomanada | Observacions locals, normalitzades, de mida fixa i calculades a partir de l'estat del món abans de l'acció. |
| Exclusió principal | No donar mapa global, població total, posicions absolutes, genealogia aliena ni informació futura. |
| Criteri de tancament | El vector d'observació és estable, limitat, repetible i suficient per alimentar la xarxa neuronal de cada agent. |
| Estat | Especificació funcional inicial |
| Data | 2 de juliol de 2026 |

## Resum executiu

El sistema perceptiu del prototip inicial ha de donar a cada agent un vector petit, fix i normalitzat d'observacions locals. Aquest vector ha d'incloure estat intern bàsic, recursos propers, presència agregada d'altres agents, informació mínima de bloqueig i un senyal ambiental local. No ha d'incloure informació global, futura, genealògica aliena ni mètriques d'anàlisi.

## 1. Objectiu del document

Aquest document defineix funcionalment el sistema d'observacions dels agents dins del prototip inicial del Projecte Evolució. La seva funció és establir quina informació pot rebre cada agent de si mateix i del seu entorn immediat abans que la xarxa neuronal produeixi una acció.

La decisió és important perquè les observacions condicionen directament el tipus d'estratègies que poden aparèixer. Si l'agent rep massa informació, el problema es torna artificialment fàcil. Si en rep massa poca, pot ser impossible que emergeixi cap comportament adaptatiu. Per tant, cal un equilibri: informació local suficient per actuar, però no prou completa per convertir l'agent en un controlador omniscient.

## 2. Principi funcional de percepció

El principi rector és que cada agent només pot percebre informació local, parcial i immediata. La xarxa neuronal no ha de rebre dades globals de la simulació, sinó un resum limitat del que l'individu podria detectar en una proximitat raonable dins del món 2D.

- L'agent coneix el seu propi estat energètic i vital bàsic.
- L'agent percep recursos propers, però no el mapa complet de recursos.
- L'agent percep presència o densitat d'altres agents propers, però no la identitat completa ni el seu genoma.
- L'agent pot percebre condicions ambientals locals, però no prediccions futures.
- Les observacions han de tenir mida fixa perquè la xarxa neuronal sigui estable i heretable.
- Les observacions s'han de normalitzar per evitar escales numèriques descompensades.

## 3. Paper de les observacions dins del bucle de simulació

Les observacions són el pont entre l'estat del món i la decisió neuronal de l'agent. En cada pas temporal, el sistema construeix una observació per a cada agent viu, aquesta observació entra a la xarxa neuronal i la sortida de la xarxa es tradueix en una acció candidata.

1. El món i els agents tenen un estat actual.
2. El mòdul d'observació extreu informació local per a cada agent viu.
3. La informació es transforma en un vector numèric de mida fixa.
4. El vector entra a la xarxa neuronal codificada pel genoma de l'agent.
5. La xarxa produeix una acció candidata.
6. La simulació resol les accions i actualitza món, energia, recursos, naixements i morts.

Per evitar inconsistències, totes les observacions d'un pas s'han de calcular a partir del mateix estat del món, abans d'aplicar les accions d'aquell pas. Això evita que els primers agents actualitzats tinguin avantatge sobre els últims.

## 4. Categories d'informació observables

El prototip inicial hauria de treballar amb cinc categories d'observació: estat propi, recursos locals, presència d'altres agents, límits o obstacles i senyal ambiental local. Aquesta divisió és suficient per generar decisions no trivials sense carregar el sistema.

| Categoria | Contingut | Funció ecològica |
| --- | --- | --- |
| Estat propi | Energia, edat i marge reproductiu de l'agent. | Permet que l'agent condicioni el comportament al seu estat vital. |
| Recursos locals | Disponibilitat de recurs a la cel·la actual i en direccions properes. | Crea la base funcional per a la cerca d'aliment i la competència. |
| Altres agents | Densitat o presència d'agents propers, sense identitat completa. | Permet evitar saturació, competència local o aprofitament de zones ocupades. |
| Límits/obstacles | Direccions bloquejades o no transitables, si el món no és toroidal. | Evita que l'agent intenti accions físicament impossibles de manera sistemàtica. |
| Ambient local | Estrès ambiental o qualitat local del medi. | Permet que l'entorn canviant afecti decisions i supervivència. |

## 5. Observacions internes de l'agent

L'agent ha de rebre informació mínima sobre el seu propi estat. Aquesta informació no és una trampa perceptiva: en termes funcionals, equival a l'estat fisiològic intern de l'individu.

| Observació | Definició funcional | Justificació |
| --- | --- | --- |
| Energia relativa | Energia actual dividida pel màxim o per una escala de referència. | Necessària per decidir si cal buscar recursos, estalviar moviment o reproduir-se. |
| Edat relativa | Edat normalitzada respecte a l'edat màxima o escala esperada. | Permet diferenciar estratègies en fases vitals diferents. |
| Marge reproductiu | Distància normalitzada fins al llindar de reproducció. | Evita que la xarxa hagi d'inferir indirectament si reproduir-se és viable. |
| Estat de viabilitat | Indicador simple de si l'agent pot actuar, reproduir-se o està prop de morir. | Útil per estabilitzar decisions en situacions extremes. |

No és recomanable incloure mètriques com descendència acumulada, rang dins la població o èxit del llinatge com a entrada de la xarxa. Aquestes dades són útils per a l'anàlisi, però no haurien de formar part de la percepció de l'agent.

## 6. Observacions de recursos

Els recursos són el senyal ecològic més important del prototip. L'agent ha de percebre disponibilitat de recurs en el seu entorn proper, però no la distribució completa del món. Això força decisions locals i evita estratègies trivials basades en coneixement global.

| Observació | Definició funcional | Funció |
| --- | --- | --- |
| Recurs a la posició actual | Quantitat de recurs disponible on es troba l'agent. | Permet decidir si alimentar-se és una acció útil. |
| Recurs al nord/sud/est/oest | Quantitat agregada de recurs en direccions cardinals dins un radi limitat. | Permet orientar moviment sense donar el mapa complet. |
| Densitat local de recursos | Recurs total proper dividit per la capacitat màxima local. | Informa si l'agent es troba en una zona rica o pobra. |
| Gradient local aproximat | Diferència de recurs entre direccions oposades, si es decideix incloure'l. | Ajuda a detectar direcció favorable amb poques entrades. |

Per a la primera versió, la decisió més robusta és representar els recursos per direccions agregades, no per una finestra completa de cel·les. Una finestra 5x5 dona més detall, però augmenta el nombre d'entrades i pot complicar la interpretació sense necessitat inicial.

## 7. Observacions sobre altres agents

La presència d'altres agents és rellevant perquè afecta la competència pels recursos i l'espai. Ara bé, la primera versió no ha de donar informació social sofisticada. L'agent no ha de saber qui és l'altre agent, quin genoma té, quin llinatge representa ni quin èxit reproductiu acumula.

| Observació | Definició funcional | Funció |
| --- | --- | --- |
| Ocupació immediata | Indicador de si hi ha agents en posicions adjacents o direccions properes. | Ajuda a evitar moviments inútils cap a espais ocupats. |
| Densitat local d'agents | Nombre d'agents propers normalitzat pel màxim possible. | Permet que emergeixin estratègies d'evitació o concentració. |
| Pressió competitiva local | Relació simple entre agents propers i recursos propers. | Pot indicar si una zona és rica però saturada. |

En aquesta fase no és necessari distingir entre progenitor, descendents o agents del mateix llinatge dins de l'observació. La genealogia s'ha de registrar per a l'anàlisi, no per donar avantatges cognitius als agents.

## 8. Observacions ambientals

Si el món 2D incorpora variació ambiental, l'agent ha de percebre com a mínim el senyal ambiental local. Aquesta informació pot representar qualitat del medi, estrès ambiental, toxicitat, temperatura abstracta o penalització energètica local.

| Observació | Definició funcional | Funció |
| --- | --- | --- |
| Estrès ambiental local | Valor normalitzat que indica com de desfavorable és la posició actual. | Pot modificar costos energètics, supervivència o regeneració de recursos. |
| Gradient ambiental proper | Diferència aproximada d'estrès entre direccions, si és necessari. | Permet fugir o apropar-se a zones millors sense conèixer tot el mapa. |
| Canvi ambiental percebut | Variació recent local, només si el model inclou canvis temporals. | Pot afavorir estratègies adaptatives davant entorns canviants. |

Per al prototip inicial, n'hi ha prou amb un únic senyal ambiental local. Afegir múltiples variables ambientals massa aviat pot fer difícil saber quina pressió selectiva està actuant.

## 9. Limitacions perceptives recomanades

Les limitacions perceptives no són una restricció arbitrària: són el mecanisme que impedeix que el problema sigui trivial. El sistema ha de simular agents que decideixen sota incertesa, no algoritmes amb accés complet a l'estat del món.

| Limitació | Definició |
| --- | --- |
| Radi local limitat | Només informació dins un radi petit, recomanat entre 1 i 3 cel·les en el prototip inicial. |
| Sense mapa global | Cap agent rep la distribució completa de recursos, agents o condicions ambientals. |
| Sense futur | Cap agent rep informació sobre regeneració futura, canvis ambientals programats o accions pendents d'altres agents. |
| Sense identitat aliena | No es dona l'identificador, genoma, llinatge o energia exacta dels altres agents. |
| Sense mètriques poblacionals | No es dona població total, rànquing, fitness ni èxit històric. |
| Sense coordenades absolutes inicialment | No incloure x/y absolutes en la primera versió, tret que es justifiqui per un experiment espacial concret. |

## 10. Vector d'observació inicial recomanat

La primera versió hauria d'utilitzar un vector d'observació petit, fix i interpretable. La proposta següent és funcional, no una obligació d'implementació exacta, però és una bona base per tancar l'abast.

| # | Entrada | Categoria | Definició |
| --- | --- | --- | --- |
| 1 | `energia_relativa` | Estat propi | Energia actual normalitzada. |
| 2 | `edat_relativa` | Estat propi | Edat actual normalitzada. |
| 3 | `marge_reproductiu` | Estat propi | Distància normalitzada al llindar de reproducció. |
| 4 | `recurs_actual` | Recursos | Recurs disponible a la posició actual. |
| 5 | `recurs_nord` | Recursos | Recurs agregat cap al nord dins el radi perceptiu. |
| 6 | `recurs_sud` | Recursos | Recurs agregat cap al sud dins el radi perceptiu. |
| 7 | `recurs_est` | Recursos | Recurs agregat cap a l'est dins el radi perceptiu. |
| 8 | `recurs_oest` | Recursos | Recurs agregat cap a l'oest dins el radi perceptiu. |
| 9 | `agents_nord` | Altres agents | Presència o densitat d'agents cap al nord. |
| 10 | `agents_sud` | Altres agents | Presència o densitat d'agents cap al sud. |
| 11 | `agents_est` | Altres agents | Presència o densitat d'agents cap a l'est. |
| 12 | `agents_oest` | Altres agents | Presència o densitat d'agents cap a l'oest. |
| 13 | `densitat_local` | Altres agents | Densitat agregada d'agents propers. |
| 14 | `estres_ambiental` | Ambient | Condició ambiental local normalitzada. |
| 15 | `bloqueig_moviment` | Límits/obstacles | Valor agregat o codificat de direccions no transitables. |

Aquesta proposta manté la xarxa petita i fa que el comportament sigui analitzable. Si en proves inicials els agents no mostren cap capacitat d'orientació, abans d'afegir moltes entrades convé revisar l'escala de recursos, costos energètics i taxa de regeneració.

## 11. Normalització i codificació

Totes les observacions han de ser valors numèrics comparables. Si una entrada pot tenir una escala molt diferent de les altres, la xarxa neuronal pot quedar dominada per aquesta variable i produir comportaments poc interpretables.

- Les magnituds positives, com energia o recursos, s'han de normalitzar preferentment entre 0 i 1.
- Els gradients o diferències direccionals poden normalitzar-se entre -1 i 1.
- Els indicadors booleans, com presència d'obstacle, poden codificar-se com 0 o 1.
- Els valors impossibles o no aplicables no han d'introduir `NaN`; s'han de substituir per un valor neutre definit.
- Les escales de normalització han de formar part de la configuració de l'experiment.

Cal evitar canviar la mida o el significat del vector d'observació entre generacions dins d'un mateix experiment. Si es modifica el vector, s'està modificant l'espai evolutiu i l'experiment deixa de ser comparable.

## 12. Consistència temporal

La construcció de les observacions ha de ser temporalment consistent. En un pas de simulació, tots els agents han de percebre el mateix estat base del món. Després, les accions es poden resoldre de manera simultània o amb una regla d'ordre definida, però les observacions no han de dependre de l'ordre d'actualització.

- No calcular observacions després que alguns agents ja hagin actuat.
- No permetre que un agent vegi recursos que ja han estat consumits en el mateix pas per un agent anterior.
- No permetre que un agent vegi naixements o morts que encara no s'han resolt formalment.
- Registrar clarament si les accions es resolen de manera simultània o seqüencial controlada.

## 13. Què no s'ha d'incloure encara

Per mantenir el prototip net, cal excloure observacions que introdueixin complexitat cognitiva o informació biològicament poc justificada.

| Exclusió | Motiu |
| --- | --- |
| Mapa complet del món | Convertiria la decisió en un problema de navegació global massa artificial. |
| Posició exacta de tots els agents | Augmenta molt la dimensió i dona informació social irrealista. |
| Genoma o xarxa d'altres agents | No és una percepció ecològica raonable en aquesta fase. |
| Fitness històric | El fitness és una mètrica d'anàlisi, no una percepció de l'individu. |
| Genealogia aliena | La relació de parentiu s'ha de registrar, no ser observada automàticament. |
| Predicció de regeneració futura | Donaria coneixement anticipat del model ambiental. |
| Objectius explícits | L'agent no ha de rebre instruccions com buscar menjar o reproduir-se; ho ha d'inferir evolutivament. |

## 14. Relació amb la xarxa neuronal

El vector d'observació defineix la capa d'entrada de la xarxa neuronal. Per tant, qualsevol canvi en les observacions modifica també el genoma funcional de l'agent. Aquesta relació ha d'estar controlada i versionada.

- La mida del vector d'entrada ha de quedar fixada per a cada experiment.
- El significat de cada entrada ha d'estar documentat.
- Les escales de normalització han de ser constants dins l'experiment.
- Un genoma només és interpretable si es coneix la versió del vector d'observació que alimentava la xarxa.
- Els millors genomes guardats han d'anar associats a la definició exacta de les observacions.

## 15. Validació funcional de les observacions

Abans de considerar tancat aquest bloc, cal validar que les observacions són correctes i que no introdueixen informació accidental. Aquesta validació és tan important com la definició mateixa.

| Validació | Criteri |
| --- | --- |
| Prova de mida fixa | Tots els agents vius reben vectors de la mateixa longitud en tots els passos. |
| Prova de rang | Totes les entrades es mantenen dins els rangs normalitzats previstos. |
| Prova de localitat | Cap entrada canvia per modificacions fora del radi perceptiu definit. |
| Prova de simetria | Situacions equivalents rotades o desplaçades produeixen observacions coherents. |
| Prova de límits | Agents propers a vores o obstacles reben informació de bloqueig correcta. |
| Prova de no omniscència | L'agent no pot inferir directament el mapa global, població total o futur ambiental. |
| Prova de reproduïbilitat | Amb la mateixa llavor i estat inicial, les observacions generades són idèntiques. |

## 16. Riscos principals

| Risc | Conseqüència |
| --- | --- |
| Observacions massa riques | Poden generar comportaments aparentment bons però dependents d'informació artificial. |
| Observacions massa pobres | Poden impedir qualsevol adaptació, fent que l'evolució només seleccioni paràmetres energètics. |
| Escales mal normalitzades | Poden fer que una variable domini la decisió neuronal. |
| Canvis no versionats | Poden fer incomparables experiments o genomes guardats. |
| Dependència de l'ordre d'actualització | Pot introduir avantatges artificials no ecològics. |

## 17. Decisió funcional recomanada

La primera versió del Projecte Evolució hauria d'utilitzar observacions locals agregades, no una finestra completa del món. La proposta recomanada és un vector petit, d'unes quinze entrades, amb estat intern bàsic, recursos direccionals, densitat d'agents, estrès ambiental i informació mínima de bloqueig.

Aquesta decisió manté la simulació simple, eficient i interpretable. També redueix el risc que l'agent aprengui o evolucioni estratègies que depenguin de detalls artificials del disseny perceptiu.

## 18. Criteri de tancament

Aquest bloc es considerarà tancat quan existeixi una definició estable i documentada de les observacions disponibles per als agents, amb les seves categories, escales, límits i proves de validació.

- El vector d'observació té mida fixa.
- Cada entrada té un significat funcional clar.
- Les observacions són locals i no globals.
- Les dades estan normalitzades.
- Les observacions es calculen abans de les accions de cada pas.
- No s'inclou informació futura ni mètriques d'anàlisi com a percepció.
- La definició de les observacions queda versionada amb cada experiment.
- Les proves funcionals confirmen localitat, consistència i reproduïbilitat.

Amb aquest criteri complert, el sistema perceptiu serà prou limitat per generar pressió adaptativa real i prou complet perquè els agents puguin prendre decisions significatives dins del món 2D.

## 19. Valoració d'adequació amb la resta de la documentació

El document és coherent amb la resta de l'especificació funcional del projecte i encaixa especialment amb els documents de model d'agent, genoma/xarxa neuronal, món 2D i arquitectura general.

| Àrea revisada | Adequació | Valoració |
| --- | --- | --- |
| Model d'agent | Alta | Reforça que l'agent només rep observacions locals i que la genealogia és informació d'anàlisi, no una percepció disponible per decidir. |
| Genoma i xarxa neuronal | Alta | Fixa la mida i el significat de la capa d'entrada, cosa necessària perquè el genoma neuronal sigui interpretable i comparable. |
| Model de món 2D | Alta | Assumeix recursos, agents, límits i ambient local com a fonts d'informació del món sense exposar el mapa complet. |
| Arquitectura general | Alta | Manté la separació de responsabilitats: l'entorn construeix observacions, l'agent decideix i la simulació resol accions. |
| Abast del prototip inicial | Alta | Manté una proposta petita i analitzable, adequada per a una primera versió evolutiva. |

**Valoració global:** el document s'adequa bé a la línia funcional existent. La decisió d'utilitzar un vector local d'unes quinze entrades és consistent amb la necessitat de mantenir una xarxa neuronal petita, heretable i interpretable. La principal recomanació per a fases posteriors és versionar explícitament l'esquema d'observacions —per exemple, `observations_v1`— i vincular-lo als genomes i experiments persistits.
