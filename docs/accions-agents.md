# Projecte Evolució — Accions possibles dels agents

**Especificació funcional del conjunt mínim d'accions disponibles per als agents: moviment cardinal, alimentació local, reproducció asexual, inacció i resolució neutral de la competència indirecta.**

## Metadades

| Camp | Valor |
| --- | --- |
| Tipus de document | Especificació funcional de les accions possibles dels agents |
| Punt del full de ruta | Punt 7 — Definir les accions possibles |
| Àmbit | Contracte funcional entre decisió neuronal, estat de l'agent, món 2D, energia, reproducció, conflictes i mètriques |
| Objectiu | Establir el conjunt mínim d'accions disponibles per als agents: moure's, alimentar-se, reproduir-se i resoldre la competència bàsica per espai o recursos quan sigui necessari. |
| Resultat esperat | Un espai d'acció reduït però suficient perquè apareguin diferències de supervivència entre agents i llinatges, sense codificar estratègies intel·ligents dins les accions. |
| Decisió recomanada | Quatre accions explícites: moviment cardinal, alimentació local, reproducció asexual i inacció; competència limitada a conflictes per recursos i espai. |
| Exclusió principal | No incloure agressió, cooperació explícita, comunicació, depredació, robatori d'energia ni plans compostos en el prototip inicial. |
| Criteri de tancament | Cada acció té un efecte clar sobre el món, l'energia de l'agent o altres agents, i cap acció incorpora objectius d'alt nivell com “buscar menjar” o “evitar competidors”. |
| Estat | Especificació funcional inicial |
| Versió | 1.0 |
| Data | Juliol de 2026 |

## Resum executiu

El prototip inicial ha de limitar l'espai d'accions a un conjunt petit, discret i interpretable: moure's en direccions cardinals, alimentar-se del recurs local, reproduir-se asexualment i esperar. La competència no s'ha de modelar com a combat, sinó com a conflicte indirecte per recursos, posicions i espai de naixement. Les accions s'han de proposar a partir de la xarxa neuronal, validar-se amb l'estat del món i resoldre's de manera simultània o neutral per evitar biaixos d'ordre intern.

## 1. Objectiu del document

Aquest document defineix funcionalment quines accions poden executar els agents dins el prototip inicial del Projecte Evolució. L'objectiu no és descriure encara la implementació tècnica, sinó fixar el contracte funcional entre decisió neuronal, estat de l'agent, món 2D, energia, reproducció i mètriques.

Les accions han de ser prou simples per simular poblacions grans, però prou expressives perquè diferents genomes puguin produir estratègies de supervivència diferents. Una acció no ha de contenir intel·ligència pròpia: la intel·ligència, si emergeix, ha de provenir de la xarxa neuronal i de la selecció ambiental.

## 2. Principi de disseny

El conjunt d'accions ha de generar dilemes ecològics reals. Un agent no pot fer-ho tot alhora: moure's pot acostar-lo a recursos, però costa energia; alimentar-se pot recuperar energia, però consumeix un torn; reproduir-se pot transmetre el genoma, però redueix l'energia del progenitor i pot fallar si l'entorn està saturat.

- Les accions han de ser locals i concretes, no objectius abstractes.
- Cada acció ha de tenir conseqüències energètiques o ecològiques mesurables.
- Cap acció ha de resoldre automàticament un problema adaptatiu complex.
- Les fallades d'acció han de ser possibles i informatives.
- Les regles han de ser simètriques per a tots els agents.
- La resolució d'accions no ha d'afavorir individus per ordre intern de càlcul.

> Nota funcional: una acció com “anar cap al recurs més proper” seria una mala decisió d'abast perquè ja incorpora una estratègia. En canvi, una acció com “moure's al nord” o “moure's segons una direcció produïda per la xarxa” manté la responsabilitat adaptativa dins el genoma i la xarxa neuronal.

## 3. Conjunt mínim d'accions

Per a la primera versió, el conjunt d'accions recomanat és reduït i tancat. No cal afegir agressió, comunicació ni cooperació explícita. La competència inicial ha de sorgir de la limitació de recursos i de l'ocupació de l'espai.

| Acció | Efecte principal | Cost energètic | Condicions mínimes | Risc de disseny |
| --- | --- | --- | --- | --- |
| Moure's | Canvia la posició de l'agent dins el món. | Cost addicional de moviment. | Direcció vàlida i cel·la accessible. | Convertir el moviment en una acció massa informada, com “anar al recurs”. |
| Alimentar-se | Transforma recurs local en energia de l'agent. | Sense cost extra o cost molt baix. | Hi ha recurs disponible dins l'abast definit. | Fer que alimentar-se sigui automàtic i eliminar el dilema entre explorar i consumir. |
| Reproduir-se | Crea un descendent amb genoma heretat i mutat. | Cost elevat per al progenitor i energia inicial del descendent. | Energia suficient i espai de naixement disponible. | Permetre reproducció massa fàcil i saturar el món. |
| Inacció | L'agent no modifica el món, però continua viu un pas més. | Cost metabòlic basal. | Cap. | Que sigui inútil si el cost basal és massa alt o dominant si és massa baix. |
| Competència indirecta | Resol conflictes per espai o recursos quan diversos agents coincideixen. | Depèn de l'acció implicada. | Conflicte real pel mateix recurs o cel·la. | Introduir agressió o jerarquies abans d'hora. |

## 4. Flux funcional d'una acció

Cada pas de simulació ha de seguir un procés estable. Primer es calculen les observacions, després els agents proposen accions, el sistema valida i resol conflictes, i finalment s'actualitzen el món i les mètriques. Aquesta separació evita que l'ordre de processament introdueixi avantatges artificials.

| Fase | Descripció funcional |
| --- | --- |
| 1. Observació | Cada agent rep només les observacions locals definides per al pas actual. |
| 2. Decisió neuronal | La xarxa neuronal transforma les observacions en una proposta d'acció. |
| 3. Validació | El motor comprova si l'acció és possible segons energia, espai, recursos i estat vital. |
| 4. Resolució simultània | Les accions dels agents es resolen evitant avantatges per ordre fix d'iteració. |
| 5. Actualització | S'apliquen canvis sobre energia, posició, recursos, naixements i morts. |
| 6. Registre | Es guarden mètriques d'intents, èxits, fallades, costos i efectes. |

## 5. Acció de moviment

El moviment permet que l'agent canviï de posició dins el món 2D. És l'acció principal per explorar, escapar de zones pobres o acostar-se a zones amb recursos. El moviment no ha de saber on hi ha el millor recurs; només executa la direcció decidida per la xarxa neuronal.

Efectes funcionals:

- Actualitza la posició de l'agent si la destinació és vàlida.
- Consumeix energia addicional respecte al metabolisme basal.
- Pot fallar si la destinació és fora del món, està bloquejada o supera la capacitat local.
- Pot generar conflictes amb altres agents que intenten ocupar el mateix espai.

Decisió recomanada:

En un món discret, el moviment inicial hauria de limitar-se a quatre direccions cardinals i, opcionalment, quedar-se quiet. En un món continu, la xarxa podria produir un vector direccional, però això és menys recomanable per a la primera versió perquè complica la resolució de col·lisions i la visualització.

> Nota funcional: per coherència amb els documents anteriors, la primera versió hauria de començar amb món discret i moviment cardinal. És menys elegant que el moviment continu, però molt més fàcil de validar i d'analitzar.

## 6. Acció d'alimentació

L'alimentació transforma recursos locals en energia de l'agent. Aquesta acció és central perquè connecta directament percepció, decisió, energia, supervivència i reproducció.

Efectes funcionals:

- Redueix la quantitat de recurs disponible a la posició de l'agent o a l'abast immediat definit.
- Incrementa l'energia de l'agent segons una eficiència configurable.
- Pot fallar si no hi ha recurs disponible.
- Pot generar competència quan diversos agents intenten consumir el mateix recurs.

Decisió funcional important:

L'alimentació hauria de ser una acció explícita, no automàtica. Si alimentar-se és automàtic sempre que hi ha recurs, s'elimina una part del dilema adaptatiu. L'agent ha de decidir si dedica el pas a consumir, moure's o reproduir-se.

Aquesta decisió fa que el genoma pugui afavorir estratègies diferents: agents que exploren més, agents que exploten ràpidament recursos locals, agents que es reprodueixen tan aviat com poden o agents que acumulen energia abans de reproduir-se.

## 7. Acció de reproducció

La reproducció és l'acció que connecta comportament i evolució. Un agent només transmet el seu genoma si aconsegueix sobreviure, obtenir energia i executar amb èxit una acció reproductiva.

Condicions mínimes:

- L'agent està viu.
- L'energia és igual o superior al llindar reproductiu.
- Hi ha una posició de naixement disponible.
- La població no supera, si s'ha definit, una capacitat màxima funcional.
- L'agent no està bloquejat per cap restricció ambiental activa.

Efectes funcionals:

- Crea un nou agent amb identificador propi.
- Registra l'identificador del progenitor.
- Copia el genoma del progenitor.
- Aplica mutacions segons les regles definides al mòdul de genoma.
- Assigna energia inicial al descendent.
- Redueix l'energia del progenitor segons el cost reproductiu.
- Situa el descendent en una cel·la propera o posició vàlida definida.

> Nota funcional: la reproducció no ha de seleccionar “bons” genomes. Només ha de copiar i mutar. La selecció es produeix perquè alguns genomes generen decisions que permeten viure i reproduir-se més que altres.

## 8. Acció d'inacció o espera

La inacció és necessària perquè la xarxa neuronal pugui expressar una decisió de no intervenir. En alguns contextos, no moure's pot ser adaptatiu si el moviment costa energia o si el recurs local es regenera.

- L'agent conserva la posició.
- No consumeix recursos ni genera descendència.
- Continua pagant el cost metabòlic basal.
- Pot ser una sortida funcional quan cap altra acció és viable.

Cal vigilar que la inacció no es converteixi en una estratègia dominant per defecte. Si el cost basal és massa baix i els recursos són abundants, agents passius poden sobreviure sense pressió selectiva suficient.

## 9. Competència i interacció entre agents

En el prototip inicial, la competència hauria de ser principalment indirecta. Això vol dir que els agents competeixen pels mateixos recursos, per l'espai i per les oportunitats de reproducció, però no ataquen ni cooperen explícitament.

Competència inclosa:

- Dos o més agents intenten consumir el mateix recurs.
- Dos o més agents intenten ocupar la mateixa posició.
- Un agent intenta reproduir-se però no hi ha espai disponible.
- La densitat local redueix l'eficiència ecològica d'una zona.

Competència exclosa inicialment:

- Atac directe.
- Robatori d'energia.
- Depredació.
- Cooperació explícita.
- Senyals socials.
- Jerarquies o dominància programada.

Aquesta exclusió és important. Afegir agressió massa aviat pot fer que el sistema evolucioni al voltant d'una regla artificial de combat, en lloc de posar a prova el bucle bàsic energia-recursos-reproducció.

## 10. Resolució de conflictes

Quan diverses accions entren en conflicte, el sistema ha d'aplicar regles neutrals, simples i repetibles. No es pot resoldre sempre a favor de l'agent que apareix primer en una llista interna, perquè això introduiria un biaix no ecològic.

- Els conflictes s'han de resoldre després de recollir totes les accions proposades.
- Els empats poden resoldre's amb una clau aleatòria reproduïble per llavor.
- Quan sigui possible, es pot repartir un recurs entre agents competidors, però això ha de quedar registrat.
- Els conflictes d'espai poden resoldre's fent fallar tots els moviments incompatibles o adjudicant la posició a un agent segons regla neutral.
- Les regles de conflicte han de ser iguals per a tots els agents.

> Nota funcional: per al primer prototip, la solució més neta és una resolució simultània amb empats pseudoaleatoris controlats per la llavor de l'experiment. És simple, reproduïble i evita privilegis amagats.

## 11. Accions invàlides o fallides

Les accions fallides són informatives perquè mostren decisions poc eficients o condicions ambientals adverses. Però no s'han de castigar de manera exagerada, perquè això pot eliminar massa ràpidament variació genètica i convertir la simulació en soroll.

| Situació | Resposta funcional recomanada |
| --- | --- |
| Moviment fora dels límits | L'agent no es mou. Paga només el cost basal, o com a màxim un cost d'intent molt baix si es vol penalitzar la mala decisió. |
| Moviment cap a una cel·la ocupada | Es resol com a conflicte d'espai. Pot fallar o adjudicar-se segons una regla neutral i reproduïble. |
| Alimentar-se sense recurs | No obté energia. Perd el torn i paga el cost basal. |
| Reproduir-se sense energia suficient | No neix cap descendent. No s'aplica el cost complet de reproducció. |
| Reproduir-se sense espai disponible | No neix cap descendent. Es registra com a fallada per saturació local. |
| Agent mort o amb energia zero | No executa cap acció. Es retira segons el mòdul de mortalitat. |

## 12. Relació entre xarxa neuronal i accions

La xarxa neuronal no ha de produir directament un pla complex. Ha de produir una decisió simple dins l'espai d'accions definit. Aquesta decisió pot representar-se funcionalment com una categoria d'acció i, en el cas del moviment, una direcció.

- Entrada: vector d'observacions locals normalitzades.
- Sortida: puntuacions o preferències per accions possibles.
- Selecció: acció amb puntuació més alta o mostra probabilística controlada.
- Validació: el motor comprova si l'acció es pot executar.
- Efecte: el món i l'agent s'actualitzen segons la regla de l'acció.

En aquesta fase, és recomanable que l'acció final sigui discreta. Això facilita saber quants agents intenten moure's, alimentar-se o reproduir-se en cada pas i permet detectar patrons evolutius amb mètriques simples.

## 13. Costos energètics de les accions

Els costos energètics són el principal mecanisme que converteix una acció en una decisió amb conseqüències evolutives. Si totes les accions són massa barates, el sistema perd pressió selectiva. Si totes són massa cares, la població col·lapsa abans que hi hagi evolució observable.

Costos que cal definir:

- Cost metabòlic basal per pas.
- Cost addicional de moviment.
- Guany energètic per unitat de recurs consumit.
- Cost de reproducció per al progenitor.
- Energia inicial assignada al descendent.
- Possible cost d'intent per accions fallides, si s'activa.

Criteri recomanat:

La primera versió hauria d'incloure cost basal, cost de moviment i cost de reproducció. El cost d'intent per accions fallides pot quedar desactivat al principi, perquè afegeix complexitat i pot castigar massa el soroll inicial dels genomes.

## 14. Mètriques associades a accions

Per saber si l'espai d'accions funciona, cal registrar no només naixements i morts, sinó també quines accions s'intenten i amb quin resultat. Aquest registre permetrà detectar si la població queda atrapada en patrons pobres, com moure's sempre, intentar reproduir-se massa aviat o no alimentar-se mai.

- Nombre d'intents per tipus d'acció.
- Percentatge d'accions executades amb èxit.
- Percentatge d'accions fallides i causa de fallada.
- Energia gastada per tipus d'acció.
- Energia obtinguda per alimentació.
- Naixements per intent de reproducció.
- Conflictes per espai.
- Conflictes per recurs.
- Distribució d'accions per llinatge.
- Relació entre patró d'accions i supervivència o descendència.

## 15. Exclusions temporals

Per mantenir el prototip controlable, queden fora de la primera versió les accions següents:

- Atacar altres agents.
- Defensar-se activament.
- Robar energia o recursos.
- Comunicar-se amb altres agents.
- Cooperar de manera explícita.
- Construir estructures.
- Modificar el terreny.
- Emmagatzemar recursos externament.
- Aprendre una acció nova durant la vida.
- Tenir plans compostos o seqüències d'accions programades.

Aquestes accions no són descartades per sempre, però introduir-les ara faria molt més difícil interpretar les causes de supervivència i reproducció diferencial.

## 16. Criteris funcionals d'èxit

1. Cada agent pot proposar una acció en cada pas de simulació.
2. Cada acció proposada és validada abans d'executar-se.
3. Les accions tenen efectes clars sobre posició, energia, recursos, naixements o conflictes.
4. Les accions fallides es registren amb una causa comprensible.
5. Cap acció incorpora informació global ni una estratègia intel·ligent explícita.
6. Els conflictes entre agents es resolen sense biaix d'ordre intern.
7. Les accions generen costos i beneficis suficients per crear pressió selectiva.
8. Els patrons d'acció es poden relacionar amb supervivència i reproducció per agent o llinatge.

## 17. Riscos principals

| Risc | Conseqüència |
| --- | --- |
| Accions massa intel·ligents | Si una acció ja conté una estratègia, com buscar automàticament el millor recurs, la xarxa neuronal deixa de ser el centre de decisió i es falseja l'evolució del comportament. |
| Accions massa pobres | Si les accions només permeten moure's i menjar sense reproducció activa o sense costos energètics rellevants, no apareixerà un bucle evolutiu interessant. |
| Costos mal calibrats | Els costos poden fer col·lapsar la població o eliminar tota pressió selectiva. Aquesta calibració haurà de ser experimental i registrada. |
| Biaix d'ordre de càlcul | Processar agents un darrere l'altre i donar sempre prioritat als primers pot crear avantatges artificials no heretables o vinculats a l'identificador. |
| Massa interacció directa | Afegir combat, depredació o cooperació abans de validar el sistema bàsic pot amagar errors fonamentals del model energètic. |

## 18. Decisió d'abast recomanada

La decisió recomanada per al prototip inicial és definir quatre accions explícites: moviment cardinal, alimentació local, reproducció asexual i inacció. La competència s'ha de limitar a conflictes per recursos i espai, resolts per regles neutrals i reproduïbles.

Aquesta tria és austera, però és la més robusta per demostrar el primer objectiu del projecte: que diferències heretables en la xarxa neuronal poden produir diferències mesurables en supervivència i descendència.

## 19. Resultat esperat

Un cop aquest bloc estigui implementat, els agents podran actuar sobre el món de manera simple però significativa. Hauran de triar entre moure's, alimentar-se, reproduir-se o esperar, i aquestes decisions afectaran directament la seva energia, la seva supervivència i la seva capacitat de deixar descendència.

El sistema haurà de permetre observar patrons com exploració excessiva, consum local eficient, reproducció prematura, acumulació energètica, saturació espacial o col·lapse poblacional. Aquests patrons no hauran estat programats explícitament, sinó que hauran emergit de la relació entre genoma, percepció, acció i entorn.

## 20. Definició de tancament

Aquest punt es considerarà tancat quan existeixi una especificació funcional acceptada que defineixi:

- Quines accions formen part del prototip inicial.
- Quins efectes té cada acció.
- Quins costos energètics s'han d'aplicar.
- Com es tracten les accions invàlides o fallides.
- Com es resolen conflictes per espai i recursos.
- Quines accions queden explícitament excloses.
- Quines mètriques cal registrar per acció.
- Quins criteris indiquen que l'espai d'accions és útil per estudiar evolució poblacional.

Amb aquesta definició, el projecte pot avançar cap a la implementació del mòdul de decisió i resolució d'accions sense haver de rediscutir constantment què pot fer un agent i què no hauria de fer encara.

## 21. Valoració d'adequació amb la resta de la documentació

El document és coherent amb la resta de l'especificació funcional del projecte i encaixa especialment amb els blocs ja definits de món 2D, agent, genoma, xarxa neuronal i observacions:

- **Coherència amb el món 2D:** la preferència per moviment discret i cardinal s'alinea amb un prototip de món discret, més fàcil de validar, visualitzar i sotmetre a resolució de conflictes reproduïble.
- **Coherència amb el model d'agent:** les accions afecten directament posició, energia, reproducció i estat vital, que són les dimensions bàsiques que ha de mantenir cada agent.
- **Coherència amb genoma i xarxa neuronal:** el document evita codificar estratègies dins les accions i manté la xarxa neuronal com a font de la decisió. Això preserva el vincle entre variació heretable, comportament i selecció.
- **Coherència amb observacions locals:** el flux observació → decisió neuronal → validació → resolució simultània reforça que tots els agents decideixin a partir d'informació local i del mateix estat temporal del món.
- **Coherència amb mètriques:** la proposta amplia les mètriques necessàries amb intents, èxits, fallades, costos, energia obtinguda i conflictes, cosa que facilitarà relacionar patrons d'acció amb supervivència i descendència.

La valoració general és **positiva**. El document és adequat per avançar cap a la implementació perquè tanca l'espai d'acció sense introduir intel·ligència preprogramada. Els únics punts que convindrà concretar en documents tècnics posteriors són: la regla exacta de desempat pseudoaleatori, si els conflictes d'espai fallen tots o adjudiquen una cel·la, el rang dels costos energètics inicials i l'esquema exacte de sortides de la xarxa neuronal per representar acció i direcció.
