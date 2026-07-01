# Projecte Evolució — Abast del prototip inicial

**Definició funcional del sistema mínim necessari per validar evolució poblacional amb agents neuronals, recursos, energia, reproducció, mort, mutació, mètriques i registre experimental.**

## Decisió principal d'abast

La primera versió ha de demostrar selecció diferencial observable, no intel·ligència complexa. El sistema ha de ser petit, executable, repetible i mesurable.

## Metadades

| Camp | Valor |
| --- | --- |
| Versió | 0.1 funcional |
| Data | 1 de juliol de 2026 |
| Estat | Especificació inicial per validació |
| Destinatari | Direcció tècnica / CTO |
| Àmbit | Prototip mínim funcional de vida artificial |

## 1. Resum executiu

Aquest document defineix l'abast funcional del prototip inicial del Projecte Evolució. L'objectiu és construir un sistema mínim però complet de vida artificial on una població d'agents, controlats per xarxes neuronals simples, interactuï amb un entorn 2D, consumeixi energia, competeixi per recursos, mori i es reprodueixi asexualment transmetent descendència amb mutacions.

El prototip no pretén simular biologia real amb detall ni generar comportaments intel·ligents sofisticats. El seu objectiu és més concret: comprovar si, sota pressió ambiental, apareixen diferències mesurables entre individus i llinatges pel que fa a supervivència, reproducció i ús dels recursos.

**Resultat esperat:** una simulació executable, repetible i mesurable que permeti comparar poblacions, llinatges i configuracions ambientals al llarg del temps.

## 2. Objectiu del prototip inicial

El prototip inicial ha de fixar el mínim sistema complet que ha de funcionar: món, agents, recursos, energia, reproducció, mort, mutació, mètriques i registre. També ha de deixar explícit què no es farà encara.

La primera versió serà útil si permet observar evolució poblacional en un entorn controlat. No cal que els agents semblin intel·ligents; cal que el sistema generi selecció diferencial i que aquesta es pugui mesurar.

## 3. Principi rector

La primera versió ha de ser deliberadament limitada. El risc principal és introduir massa complexitat abans de tenir un bucle evolutiu verificable.

Per tant, el prototip ha de prioritzar:

- agents simples;
- entorn comprensible;
- regles energètiques clares;
- reproducció i mort funcionals;
- mutacions traçables;
- mètriques bàsiques però útils;
- experiments repetibles amb llavor aleatòria.

**Criteri tècnic:** no s'ha de buscar complexitat emergent abans de demostrar que el sistema selecciona llinatges de manera mesurable.

## 4. Abast funcional inclòs

El prototip inicial inclourà els components funcionals següents.

| Component | Funció dins el prototip |
| --- | --- |
| Món 2D | Espai on viuen els agents, amb dimensions configurables, posicions i actualització per passos. |
| Recursos | Font d'energia limitada que crea competència espacial i pressió selectiva. |
| Agents | Individus amb posició, energia, edat, estat vital, genoma neuronal i genealogia. |
| Observacions locals | Informació limitada de l'entorn proper, sense coneixement global del món. |
| Xarxa neuronal | Controlador petit que transforma observacions en accions. |
| Accions | Moviment, alimentació, reproducció i inacció. |
| Energia | Variable central que determina supervivència, cost d'accions i reproducció. |
| Mort | Mecanisme per manca d'energia, edat màxima o condicions simples. |
| Reproducció asexual | Còpia del genoma, mutació, cost energètic i registre del progenitor. |
| Mètriques i registre | Dades mínimes per comparar execucions, llinatges i poblacions. |

## 5. Components detallats

### 5.1. Món 2D

El sistema tindrà un món bidimensional on viuen els agents. Per a aquesta primera fase es recomana una graella discreta, perquè facilita la depuració, la visualització, el càlcul de recursos locals i la validació de regles.

- dimensions configurables;
- posicions ocupables pels agents;
- distribució de recursos;
- actualització temporal per passos;
- regeneració de recursos;
- condicions ambientals simples.

**Nota funcional:** no s'ha de simular encara clima complex, física realista ni topografia elaborada.

### 5.2. Recursos

Els recursos seran la font principal d'energia dels agents. Han d'estar distribuïts en el món i regenerar-se segons una regla simple.

- alimentar els agents;
- generar competència espacial;
- crear zones més o menys favorables;
- permetre canvis ambientals simples;
- limitar el creixement poblacional.

**Nota funcional:** la quantitat de recursos ha de ser limitada. Si són massa abundants, no hi haurà pressió selectiva real.

### 5.3. Agents

Cada agent representarà un individu de la població. Ha de ser simple, però prou definit per poder seguir la seva vida i el seu llinatge.

- identificador únic;
- posició;
- energia;
- edat;
- estat vital;
- identificador del progenitor;
- genoma neuronal;
- paràmetres heretables bàsics.

**Nota funcional:** les conductes no s'han de programar directament. Les accions han de sortir de la xarxa neuronal a partir de les observacions.

### 5.4. Observacions locals

Cada agent rebrà informació limitada del seu entorn proper. Això evita que l'agent disposi d'un coneixement global artificial del món.

- energia pròpia;
- edat pròpia;
- presència de recursos a prop;
- densitat local d'altres agents;
- direcció relativa del recurs més proper dins un radi limitat;
- condició ambiental local, si s'inclou.

**Nota funcional:** no s'ha d'incloure encara memòria complexa, comunicació simbòlica ni mapa global.

### 5.5. Xarxa neuronal

Cada agent serà controlat per una xarxa neuronal petita. Els pesos i biaixos formen part del genoma i evolucionen per reproducció, mutació i selecció ambiental.

- entrada basada en observacions locals;
- sortida en forma d'accions possibles;
- mida reduïda per permetre molts agents;
- sense entrenament individual durant la vida.

**Nota funcional:** no s'ha d'afegir aprenentatge per reforç ni aprenentatge individual en aquesta fase.

### 5.6. Accions bàsiques

El conjunt d'accions ha de ser reduït per evitar comportaments difícils d'interpretar.

- moviment en una direcció vàlida;
- alimentació si hi ha recurs disponible;
- reproducció si hi ha energia suficient;
- inacció amb consum energètic basal.

**Nota funcional:** la competència directa pot quedar limitada inicialment a recursos i espai. L'agressió explícita no és necessària.

### 5.7. Energia

L'energia serà el mecanisme central del prototip. Determina la viabilitat de l'agent, el cost de les accions i la possibilitat de reproducció.

- viure consumeix energia;
- moure's consumeix energia addicional;
- reproduir-se té un cost elevat;
- alimentar-se incrementa energia;
- l'agent mor quan l'energia cau per sota del llindar definit.

**Nota funcional:** el sistema energètic ha de ser prou restrictiu per crear pressió evolutiva real.

### 5.8. Mort

La mort és imprescindible per evitar acumulació indefinida d'agents i per generar renovació evolutiva.

- manca d'energia;
- edat màxima;
- condicions ambientals desfavorables, si s'inclouen;
- saturació local, només si es defineix una regla clara.

**Nota funcional:** la mort per manca d'energia ha de ser obligatòria. La mort per edat és recomanable.

### 5.9. Reproducció asexual

Un agent amb prou energia podrà generar un descendent. Aquesta reproducció serà el mecanisme bàsic d'herència i renovació poblacional.

- creació d'un nou agent;
- assignació d'un identificador nou;
- registre del progenitor;
- còpia del genoma;
- aplicació de mutacions;
- cost energètic per al progenitor;
- energia inicial per al descendent.

**Nota funcional:** no hi haurà reproducció sexual en aquesta fase.

### 5.10. Mutació

Les mutacions afectaran el genoma neuronal i, si cal, alguns paràmetres heretables simples.

- petits canvis en pesos neuronals;
- petits canvis en biaixos;
- taxa de mutació configurable;
- magnitud de mutació configurable;
- acumulació gradual al llarg dels llinatges.

**Nota funcional:** les mutacions massa grans poden convertir l'evolució en soroll aleatori. Cal començar amb mutacions petites.

### 5.11. Genealogia

Cada individu ha de mantenir una relació explícita amb el seu progenitor. Això permet reconstruir llinatges i èxit reproductiu.

- identificador d'agent;
- identificador de progenitor;
- generació o profunditat de llinatge;
- descendència acumulada;
- persistència o extinció de branques.

**Nota funcional:** aquest punt és essencial: sense genealogia, només tindrem dinàmica poblacional, però no anàlisi evolutiva de llinatges.

### 5.12. Mètriques

El prototip ha de registrar mètriques mínimes per saber si hi ha evolució mesurable i no només fluctuació aleatòria.

- mida poblacional per pas temporal;
- energia mitjana;
- edat mitjana;
- nombre de naixements;
- nombre de morts;
- causa de mort;
- descendència per agent o llinatge;
- supervivència mitjana;
- consum total de recursos;
- diversitat genètica aproximada;
- estabilitat o col·lapse poblacional.

**Nota funcional:** sense mètriques, la simulació pot semblar interessant visualment però no serà científicament útil.

### 5.13. Registre d'experiments

Cada execució ha de quedar registrada amb la seva configuració. La repetibilitat és un requisit funcional, no un afegit posterior.

- llavor aleatòria;
- mida del món;
- nombre inicial d'agents;
- quantitat inicial de recursos;
- taxa de regeneració de recursos;
- paràmetres energètics;
- taxa i magnitud de mutació;
- durada de la simulació;
- mètriques principals;
- millors llinatges segons criteris definits.

**Nota funcional:** el sistema ha de permetre repetir un experiment i comparar execucions amb paràmetres diferents.

## 6. Exclusions temporals

Per mantenir el prototip controlable, queden fora de la primera versió les funcionalitats següents:

| Exclusió | Motiu |
| --- | --- |
| Reproducció sexual | Afegiria complexitat genètica abans de validar l'herència bàsica. |
| Aprenentatge individual | Confondria evolució poblacional amb aprenentatge durant la vida. |
| Memòria complexa | Augmentaria el cost i dificultaria interpretar resultats. |
| Comunicació entre agents | Pot ser interessant després, però ara complicaria la causalitat. |
| Múltiples espècies | Primer cal validar una sola població evolutiva. |
| Depredació sofisticada | La competència pels recursos ja és pressió selectiva suficient. |
| Física realista | No aporta valor al primer objectiu evolutiu. |
| Clima avançat | Cal començar amb variació ambiental simple i controlable. |
| Interfície web | La visualització avançada no és prioritària per validar el model. |
| Optimització distribuïda | Prematur abans de conèixer el cost real del prototip. |
| Estratègies evolutives com a motor principal | El motor principal ha de ser ecològic: supervivència i reproducció dins l'entorn. |
| Conductes hardcodejades | Anirien contra l'objectiu d'emergència de patrons. |

## 7. Criteris funcionals d'èxit

El prototip inicial es considerarà útil si compleix aquests criteris:

1. Es pot executar una simulació completa de principi a fi.
2. Els agents consumeixen energia i poden morir.
3. Els agents poden alimentar-se de recursos limitats.
4. Els agents poden reproduir-se asexualment.
5. Els descendents hereten genomes amb mutacions.
6. Es pot seguir la relació entre progenitor i descendent.
7. La població mostra variacions de supervivència entre individus o llinatges.
8. Es registren mètriques bàsiques de població, energia, reproducció i mort.
9. Es pot repetir un experiment amb la mateixa llavor aleatòria.
10. Es poden comparar diferents execucions canviant paràmetres ambientals o mutacionals.

**Criteri clau:** el prototip no s'avaluarà per si els agents semblen intel·ligents, sinó per si hi ha selecció diferencial observable i mesurable.

## 8. Riscos principals

| Risc | Impacte | Mitigació |
| --- | --- | --- |
| Massa complexitat inicial | Pot impedir entendre per què el sistema funciona o falla. | Retallar funcionalitats i protegir el bucle evolutiu mínim. |
| Absència de pressió selectiva | Recursos massa abundants o mort massa rara poden anul·lar l'evolució. | Ajustar recursos, costos energètics i llindars de mort. |
| Soroll mutacional excessiu | Mutacions massa grans poden impedir acumulació adaptativa. | Començar amb mutacions petites i configurables. |
| Agents massa potents | Observacions globals o xarxes grans poden distorsionar el model. | Limitar observacions locals i xarxes petites. |
| Mètriques insuficients | No es podrà distingir evolució real de fluctuació aleatòria. | Registrar població, energia, naixements, morts, genealogia i diversitat. |

## 9. Decisió d'abast recomanada

La primera versió hauria de limitar-se a:

- món 2D discret;
- una sola mena de recurs;
- agents amb xarxa neuronal petita;
- observació local;
- accions bàsiques;
- energia, mort i reproducció;
- mutació simple de pesos i biaixos;
- genealogia mínima;
- mètriques poblacionals i genètiques bàsiques;
- execucions repetibles amb llavor aleatòria.

Aquesta decisió redueix el risc tècnic i permet arribar ràpidament a una simulació funcional que després es podrà ampliar.

**Decisió crítica:** començar amb món 2D discret és preferible a un món continu. El món continu és viable més endavant, però ara afegeix cost sense aportar gaire al primer objectiu evolutiu.

## 10. Resultat esperat del prototip

El resultat final d'aquesta fase serà un prototip executable capaç de mostrar una població d'agents que viu en un entorn limitat, competeix indirectament pels recursos, es reprodueix, acumula mutacions i genera llinatges amb diferent èxit reproductiu.

El sistema haurà de permetre respondre preguntes com:

- quins llinatges sobreviuen més?
- quins produeixen més descendència?
- la població s'estabilitza o col·lapsa?
- les mutacions generen variació útil?
- els canvis ambientals modifiquen l'èxit dels llinatges?
- hi ha diferències mesurables entre configuracions?

## 11. Definició de tancament

Aquest punt es considerarà complet quan existeixi una especificació funcional acceptada que defineixi:

- què inclou el prototip;
- què queda fora;
- quins components mínims s'han d'implementar;
- quines mètriques s'han de registrar;
- quins criteris indiquen que el prototip és útil;
- quins riscos s'han d'evitar;
- quin és el resultat mínim acceptable.

A partir d'aquest document es podrà passar a la definició tècnica del sistema sense haver de rediscutir constantment l'abast funcional.

## 12. Llista mínima d'acceptació

| Element | Condició | Prioritat |
| --- | --- | --- |
| Simulació executable | La simulació corre de principi a fi sense intervenció manual. | Obligatori |
| Llavor aleatòria | La mateixa configuració i llavor reprodueixen el mateix resultat. | Obligatori |
| Agents vius i morts | Els agents neixen, consumeixen energia i moren. | Obligatori |
| Recursos limitats | Els recursos condicionen supervivència i reproducció. | Obligatori |
| Herència amb mutació | Els descendents copien el genoma amb variacions petites. | Obligatori |
| Genealogia | Cada agent pot vincular-se al seu progenitor. | Obligatori |
| Mètriques | Es registren població, energia, naixements, morts i llinatges. | Obligatori |
| Comparació | Es poden comparar dues execucions amb configuracions diferents. | Obligatori |

## 13. Valoració d'adequació amb la documentació existent

El document és coherent amb el document funcional inicial del projecte i en concreta especialment la tasca d'alt nivell de definir l'abast del prototip inicial. Manté la mateixa orientació de producte: una primera fase petita, repetible i mesurable, centrada en selecció diferencial i no en intel·ligència complexa.

### Encaix amb el document funcional inicial

- **Visió i propòsit:** l'abast proposat reforça la visió d'una simulació de vida artificial on la pressió selectiva neix de l'entorn i no de conductes preprogramades.
- **Principis de decisió:** el document desenvolupa els principis ja definits: agents simples, món comprensible, herència mutable, mètriques traçables i experiments repetibles.
- **Mapa de treball:** concreta el bloc d'abast i dona una base clara per als blocs següents de model ecològic, evolució, execució de dades i validació.
- **Exclusions:** és consistent amb les decisions que cal evitar en aquesta fase, especialment aprenentatge individual, conductes hardcodejades, complexitat arquitectònica prematura i valoració basada només en visualització.

### Valoració

La valoració és **positiva**. El document és adequat per incorporar-se a la documentació del projecte perquè transforma la visió funcional inicial en una especificació d'abast més accionable. També redueix riscos de desviació, ja que defineix components inclosos, exclusions, mètriques, registre experimental i criteris d'acceptació.

Com a recomanació per a la fase següent, aquest document s'hauria de convertir en una especificació tècnica amb decisions concretes sobre estructura de dades, ordre del bucle de simulació, format de persistència, esquema de mètriques i paràmetres per defecte dels experiments inicials.
