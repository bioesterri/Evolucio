# Projecte Evolució — Reproducció, herència i mutació

**Especificació funcional de la reproducció asexual, la transmissió hereditària, la mutació configurable i el registre genealògic del prototip inicial.**

## Metadades

| Camp | Valor |
| --- | --- |
| Tipus de document | Especificació funcional de reproducció, herència i mutació |
| Punt del full de ruta | Nucli evolutiu del prototip inicial |
| Àmbit | Contracte funcional entre acció reproductiva, energia, genoma, mutació, naixement, genealogia, mètriques i selecció ambiental |
| Objectiu | Establir quan un agent pot reproduir-se, quin cost té fer-ho, què hereta el descendent, com s'introdueix variació i com es registra cada naixement. |
| Resultat esperat | Una mecànica evolutiva mesurable on els descendents s'assemblen als progenitors, però incorporen variacions petites que poden afectar supervivència i reproducció. |
| Decisió recomanada | Reproducció asexual amb mutació de pesos, biaixos i, com a màxim, un conjunt molt reduït de paràmetres heretables. |
| Exclusió principal | No incloure reproducció sexual, recombinació, mutacions estructurals de xarxa, aprenentatge individual ni funcions de fitness externes en el prototip inicial. |
| Criteri de tancament | Cada naixement és energèticament costós, genealògicament traçable, reproduïble sota la mateixa llavor i basat en un genoma derivat del progenitor amb mutacions configurables. |
| Estat | Especificació funcional inicial |
| Versió | 1.0 |
| Data | Juliol de 2026 |

## Resum executiu

La reproducció, l'herència i la mutació formen el nucli evolutiu del prototip inicial. La reproducció ha de ser una conseqüència funcional de sobreviure i acumular energia, no una acció gratuïta ni garantida. L'agent només pot deixar descendència si està viu, té prou energia, ha superat l'edat mínima, disposa d'espai local i la seva xarxa ha seleccionat l'acció reproductiva. El descendent rep una còpia mutada del genoma del progenitor, neix amb estat vital propi i queda registrat dins una traça genealògica auditable.

## 1. Objectiu del document

Aquest document defineix funcionalment la reproducció, l'herència i la mutació dins del prototip inicial del Projecte Evolució. Aquests tres elements permeten continuïtat poblacional, formació de llinatges i acumulació de variació.

La finalitat no és crear un model genètic biològicament complet, sinó una mecànica mínima, clara i mesurable que permeti observar diferències d'èxit entre agents i llinatges al llarg del temps.

El document fixa:

- quan un agent pot intentar reproduir-se;
- quin cost energètic assumeix el progenitor;
- què rep el descendent;
- com s'introdueixen mutacions petites i configurables;
- com es registren naixements, genealogia i resum de mutacions;
- com s'evita substituir la selecció ambiental per una funció de fitness externa.

## 2. Principi rector

La reproducció ha de ser una conseqüència funcional de la supervivència. Un agent només ha de poder deixar descendència si ha mantingut prou energia, ha viscut prou temps i troba condicions mínimes dins l'entorn.

- La reproducció és asexual en el prototip inicial.
- El descendent s'assembla funcionalment al progenitor, però no és una còpia perfecta.
- Les mutacions són petites, configurables i registrables.
- El cost reproductiu impedeix el creixement poblacional indefinit.
- La selecció sorgeix de la supervivència i la reproducció diferencials, no d'una funció de fitness externa.
- Cada naixement deixa una traça genealògica clara.

**Decisió recomanada:** la primera versió s'ha de limitar a reproducció asexual amb mutació de pesos, biaixos i un conjunt molt reduït de paràmetres heretables. La reproducció sexual, la recombinació i les mutacions estructurals de la xarxa queden fora del prototip inicial.

## 3. Rol evolutiu de la reproducció

La reproducció converteix diferències individuals en diferències poblacionals. Un agent que pren decisions més eficients pot acumular més energia, sobreviure més temps i produir més descendents. Si els descendents hereten part d'aquesta configuració, el llinatge pot augmentar la seva presència en la població.

El sistema no avalua explícitament si un agent és bo o dolent. La qualitat funcional queda mesurada indirectament per la seva capacitat de continuar viu i deixar descendència en un entorn amb recursos limitats.

- Un agent no rep premi extern per estar ben adaptat.
- Un agent no es reprodueix perquè tingui una puntuació alta.
- Un agent es reprodueix només si el seu estat energètic i ambiental ho permet.
- Els llinatges exitosos són els que persisteixen i acumulen descendència viable.

## 4. Condicions mínimes de reproducció

La reproducció requereix diverses condicions simultànies. Això evita multiplicació sense restricció i introdueix pressió selectiva directa sobre la gestió energètica.

| Condició | Funció dins del sistema | Risc si no s'inclou |
| --- | --- | --- |
| Agent viu | Només els agents actius poden reproduir-se. | Podrien aparèixer naixements incoherents després de la mort. |
| Energia mínima | Assegura que l'agent ha acumulat recursos abans de reproduir-se. | La població pot créixer sense límit ni selecció real. |
| Edat mínima | Evita reproduccions immediates en cadena just després del naixement. | Pot aparèixer explosió demogràfica artificial. |
| Espai disponible | Permet ubicar el descendent sense trencar les regles del món. | Poden aparèixer solapaments o agents sense posició vàlida. |
| Acció reproductiva seleccionada | La reproducció depèn de la decisió de l'agent. | La reproducció seria automàtica i reduiria el paper del comportament. |
| Límit o cost suficient | Manté la reproducció com una decisió cara. | La selecció queda afeblida perquè reproduir-se sempre surt a compte. |

Aquestes condicions no han de codificar una estratègia intel·ligent. Només defineixen el marc físic i energètic dins del qual la xarxa neuronal de l'agent pot decidir intentar reproduir-se.

## 5. Cost funcional de la reproducció

La reproducció ha de tenir un cost elevat. Si generar un descendent és massa barat, el sistema pot derivar cap a creixement explosiu; si és massa car, la població pot extingir-se abans que la selecció tingui temps d'actuar.

| Component | Descripció funcional | Efecte esperat |
| --- | --- | --- |
| Cost fix | Quantitat mínima d'energia que el progenitor perd quan es reprodueix. | Evita reproduccions massa freqüents. |
| Energia inicial del descendent | Reserva energètica amb què neix el nou agent. | Determina la probabilitat inicial de supervivència. |
| Pèrdua del progenitor | Reducció efectiva de l'energia del progenitor després del naixement. | Introdueix el compromís entre reproduir-se i continuar viu. |
| Saturació energètica | El descendent no pot néixer amb energia superior al màxim permès. | Evita avantatges artificials fora dels límits del model. |
| Fracàs reproductiu | Si les condicions no es compleixen, l'acció falla i pot aplicar o no un petit cost. | Permet estudiar errors de decisió sense regalar reproducció. |

**Criteri energètic:** la reproducció no ha de crear energia gratuïta. El naixement redistribueix o consumeix energia ja acumulada pel progenitor.

## 6. Procés funcional de naixement

Cada naixement ha de seguir un procés consistent i reconstruïble:

1. L'agent selecciona o intenta l'acció de reproducció.
2. El sistema comprova si l'agent compleix les condicions mínimes.
3. Si la reproducció és vàlida, es calcula el cost energètic.
4. Es copia el genoma del progenitor.
5. S'apliquen mutacions segons la configuració de l'experiment.
6. Es crea un identificador únic per al descendent.
7. S'assigna el progenitor i la generació corresponent.
8. Es col·loca el descendent en una posició vàlida propera al progenitor.
9. S'inicialitzen energia, edat, estat vital i estat intern del descendent.
10. Es registra l'esdeveniment de naixement amb la informació necessària.

Aquest procés ha de ser determinista sota la mateixa llavor aleatòria i la mateixa configuració per poder repetir experiments i depurar resultats.

## 7. Informació heretable

El genoma és el conjunt d'informació que pot passar del progenitor al descendent. Cal separar clarament informació genètica i informació adquirida durant la vida.

| Element | Heretable? | Motiu |
| --- | --- | --- |
| Pesos de la xarxa neuronal | Sí | Determinen part del comportament i són el principal substrat evolutiu. |
| Biaixos de la xarxa neuronal | Sí | Afecten la decisió de l'agent i poden acumular variació. |
| Paràmetres metabòlics simples | Opcional | Poden permetre evolució de ritmes energètics, però cal limitar-los en la primera fase. |
| Llindar reproductiu heretable | Opcional | Pot generar estratègies reproductives diferents, però pot desestabilitzar el sistema si es deixa massa lliure. |
| Energia actual | No | És estat vital del progenitor, no informació genètica. |
| Edat | No | És història de vida, no genoma. |
| Posició | No | Depèn del món i de la col·locació del naixement. |
| Identificador i genealogia | No com a genoma | Són metadades de traça, no informació que controli l'agent. |

Per al prototip inicial, el més robust és començar només amb pesos i biaixos neuronals com a genoma principal. Els paràmetres metabòlics heretables poden afegir-se si el sistema bàsic ja és estable.

## 8. Descendent: estat inicial

El descendent ha de néixer amb un estat funcional complet però mínim. No ha d'arrossegar l'estat vital del progenitor, excepte allò que la regla reproductiva defineixi explícitament.

| Camp inicial | Valor funcional esperat |
| --- | --- |
| Identificador | Nou identificador únic. |
| Progenitor | Identificador de l'agent que ha generat el descendent. |
| Generació | Generació del progenitor més una unitat, o valor equivalent de profunditat genealògica. |
| Genoma | Còpia mutada del genoma del progenitor. |
| Energia inicial | Quantitat definida per la regla reproductiva i restada del cost del progenitor. |
| Edat | Zero o valor inicial equivalent. |
| Estat vital | Viu, si el naixement és vàlid. |
| Posició | Cel·la o punt vàlid proper al progenitor. |
| Estat intern | Inicialitzat a zero o a un valor neutre definit per configuració. |

La posició inicial ha de ser local i controlada. Si els descendents apareixen en qualsevol punt del món, reben un avantatge artificial; si sempre apareixen damunt del progenitor, poden aparèixer problemes d'ocupació o densitat.

## 9. Mutació

La mutació introdueix variació sobre la informació heretable. En aquesta fase ha de ser simple, petita i parametritzable. Les mutacions s'apliquen en crear el descendent, després de copiar el genoma del progenitor i abans d'assignar el genoma definitiu al nou agent.

| Tipus de mutació | Incloure al prototip? | Comentari funcional |
| --- | --- | --- |
| Soroll petit en pesos | Sí | Canvi principal per generar variació conductual gradual. |
| Soroll petit en biaixos | Sí | Permet variacions en llindars de decisió. |
| Mutació de paràmetres metabòlics | Opcional | Útil però potencialment desestabilitzadora si no es limita. |
| Canvi de mida de la xarxa | No | Afegiria complexitat innecessària en la primera versió. |
| Afegir o eliminar neurones | No | Complica comparació de genomes i vectorització. |
| Recombinació entre dos progenitors | No | Pertany a reproducció sexual i queda fora de l'abast inicial. |
| Mutacions massives aleatòries | No | Destrueixen continuïtat de llinatge i converteixen l'evolució en soroll. |

## 10. Paràmetres configurables de mutació

La mutació no ha de quedar amagada en constants internes. Els seus paràmetres han de formar part de la configuració de l'experiment.

| Paràmetre | Funció | Criteri funcional |
| --- | --- | --- |
| Taxa de mutació | Probabilitat que cada element heretable rebi una alteració. | Ha de ser prou baixa per conservar continuïtat entre generacions. |
| Intensitat de mutació | Magnitud típica del canvi aplicat. | Ha de generar variació sense destruir el comportament heretat. |
| Límit de valors | Rang permès per pesos, biaixos o paràmetres. | Evita valors extrems que trenquin la simulació. |
| Mutació per grup | Permet diferenciar pesos, biaixos i paràmetres heretables. | Evita tractar tots els elements com si tinguessin el mateix efecte. |
| Llavor aleatòria | Controla la reproductibilitat de les mutacions. | Ha de quedar registrada sempre. |
| Resum de mutacions | Nombre i magnitud agregada de canvis aplicats. | Permet auditar l'origen de diferències entre llinatges. |

És preferible una mutació gaussiana petita o una alteració equivalent de baixa magnitud. La propietat essencial és que la majoria de descendents siguin semblants al progenitor.

## 11. Registre genealògic

Cada naixement ha de registrar-se com un esdeveniment. Aquest registre permet estudiar llinatges, èxit reproductiu i acumulació de canvis.

| Dada registrada | Motiu |
| --- | --- |
| Identificador del descendent | Permet seguir l'individu durant tota la seva vida. |
| Identificador del progenitor | Permet reconstruir genealogia directa. |
| Pas temporal del naixement | Situa l'esdeveniment dins la simulació. |
| Generació o profunditat | Facilita comparar branques genealògiques. |
| Genoma heretat o referència al genoma | Permet reconstruir el sistema de decisió del descendent. |
| Resum de mutacions aplicades | Permet saber què ha canviat respecte del progenitor. |
| Energia del progenitor abans i després | Permet auditar el cost reproductiu. |
| Energia inicial del descendent | Permet estudiar supervivència primerenca. |
| Posició de naixement | Permet relacionar èxit reproductiu amb condicions locals. |
| Configuració i llavor de l'experiment | Permet repetir i validar resultats. |

No cal guardar en cada pas tots els pesos de tots els agents si això és massa costós, però el sistema ha de poder reconstruir o referenciar el genoma d'un individu nascut.

## 12. Relació amb selecció ambiental

La selecció no ha de ser una funció externa que tria els millors agents. Ha d'aparèixer perquè alguns agents aconsegueixen reproduir-se i altres no. Les condicions ambientals, els recursos, el metabolisme i les decisions de la xarxa neuronal determinen aquesta diferència.

- Els agents amb decisions ineficients consumiran energia sense reproduir-se o moriran abans.
- Els agents que localitzin millor els recursos podran arribar més sovint al llindar reproductiu.
- Els llinatges amb mutacions perjudicials tendiran a desaparèixer.
- Els llinatges amb mutacions avantatjoses poden persistir, sempre que el context ambiental les afavoreixi.
- Els canvis ambientals poden alterar quines configuracions resulten viables.

El sistema no ha de saber què és una bona estratègia; ho ha de revelar la dinàmica poblacional.

## 13. Exclusions temporals

Per protegir la simplicitat del prototip inicial, queden fora d'aquest punt:

- Reproducció sexual.
- Recombinació genètica entre dos progenitors.
- Selecció artificial directa basada en una funció de fitness externa.
- Aprenentatge individual durant la vida de l'agent.
- Mutacions que canviïn la topologia de la xarxa neuronal.
- Herència de memòries adquirides durant la vida.
- Epigenètica o regulació genòmica avançada.
- Estratègies reproductives hardcodejades.
- Cures parentals o protecció activa del descendent.
- Espècies múltiples amb compatibilitat reproductiva diferenciada.

Aquestes exclusions no descarten les idees per sempre; simplement no són necessàries per demostrar el primer bucle evolutiu.

## 14. Riscos principals

| Risc | Conseqüència | Mesura preventiva |
| --- | --- | --- |
| Cost reproductiu massa baix | Creixement explosiu i pèrdua de pressió selectiva. | Fer que reproduir-se redueixi clarament l'energia del progenitor. |
| Cost reproductiu massa alt | Extinció ràpida de la població. | Ajustar llindars i recursos amb experiments curts de calibratge. |
| Mutació massa forta | Els descendents no s'assemblen al progenitor i no hi ha acumulació adaptativa. | Usar mutacions petites i límits de rang. |
| Mutació massa feble | La població queda estancada sense variació útil. | Permetre taxa i intensitat configurables. |
| Heretar massa paràmetres | El sistema es torna difícil d'interpretar. | Començar només amb pesos i biaixos; afegir paràmetres després. |
| Reproducció automàtica | Es redueix el paper del comportament neuronal. | Exigir que la reproducció sigui una acció decidida per l'agent. |
| Traça genealògica incompleta | No es poden analitzar llinatges ni èxit reproductiu. | Registrar cada naixement com a esdeveniment auditable. |

## 15. Criteris funcionals d'èxit

Aquest bloc es considerarà funcionalment resolt quan el sistema compleixi els criteris següents:

1. Un agent només pot reproduir-se si és viu i compleix les condicions mínimes definides.
2. La reproducció té un cost energètic clar per al progenitor.
3. El descendent rep un genoma derivat del progenitor.
4. El genoma del descendent pot incorporar mutacions petites i configurables.
5. El descendent neix amb identificador propi, progenitor registrat i estat inicial complet.
6. Cada naixement queda registrat amb pas temporal, condicions energètiques i resum de mutacions.
7. La població no pot créixer indefinidament sense consumir recursos i energia.
8. Els llinatges poden comparar-se segons descendència, supervivència i persistència.
9. La mateixa configuració i llavor permeten repetir el procés reproductiu.
10. El sistema no introdueix una funció de fitness externa que substitueixi la selecció ambiental.

## 16. Definició de tancament

La definició de reproducció, herència i mutació es considerarà tancada quan l'equip pugui explicar:

- quan un agent pot intentar reproduir-se;
- quines condicions fan que la reproducció sigui vàlida o fallida;
- quin cost energètic assumeix el progenitor;
- quin estat inicial rep el descendent;
- quina informació és heretable i quina no;
- com s'apliquen les mutacions sobre el genoma;
- quines dades es registren en cada naixement;
- com es pot reconstruir la genealogia;
- com la dinàmica reproductiva contribueix a la selecció ambiental.

## 17. Valoració d'adequació amb la resta de la documentació

El document és coherent amb la resta de la documentació funcional i completa un bloc que fins ara apareixia repartit entre accions, energia i genoma.

### Encaix amb documents existents

| Document relacionat | Valoració d'encaix |
| --- | --- |
| `document-funcional-inicial.md` | Desenvolupa el nucli evolutiu del projecte mantenint el principi d'evolució emergent sense objectius externs. |
| `abast-prototip-inicial.md` | Respecta l'abast del prototip: reproducció asexual, mutació simple, agents simples, energia limitada i mètriques auditables. |
| `arquitectura-general.md` | S'alinea amb una arquitectura modular que separa agent, genoma, món, reproducció, mètriques i persistència. |
| `model-agent.md` | Reforça la separació entre estat vital no heretable, identificació de l'agent, genealogia i genoma. |
| `model-mon-2d.md` | Manté la necessitat d'ubicar descendents en posicions locals vàlides i d'evitar solapaments o privilegis espacials artificials. |
| `energia-recursos-metabolisme.md` | Encaixa amb la regla que la reproducció depèn d'energia acumulada, té cost elevat i no pot crear energia gratuïta. |
| `accions-agents.md` | Dona detall funcional a l'acció de reproduir-se, que continua sent una decisió proposada per la xarxa i validada pel motor. |
| `genoma-xarxa-neuronal.md` | És totalment consistent amb la decisió de començar amb pesos i biaixos com a substrat hereditari principal i mutacions petites. |

### Punts forts

- Defineix condicions reproductives simultànies i evita la reproducció automàtica.
- Connecta de manera clara energia, acció reproductiva, herència, mutació i genealogia.
- Preserva la filosofia del projecte: la selecció emergeix de supervivència i reproducció diferencials, no d'una funció de fitness externa.
- Insisteix en traçabilitat, llavor aleatòria i registres suficients per repetir experiments.
- Manté l'abast prudent del prototip inicial i exclou complexitats que podrien distorsionar l'anàlisi.

### Punts a concretar abans de la implementació

- Caldrà fixar valors inicials de calibratge per al llindar reproductiu, l'edat mínima, el cost fix, l'energia inicial del descendent i l'eventual cost d'intent fallit.
- Caldrà definir la regla exacta per trobar una posició local vàlida de naixement i resoldre conflictes simultanis per espai de reproducció.
- Caldrà decidir si els paràmetres metabòlics heretables s'activen des del principi o queden completament desactivats fins que el bucle bàsic sigui estable.
- Caldrà especificar el format persistent dels esdeveniments de naixement i de les referències a genomes per no duplicar dades de manera innecessària.

### Valoració global

**Adequació alta.** El document encaixa amb la resta de l'especificació, reforça el bucle evolutiu mínim i és prou concret per orientar una implementació posterior. La recomanació és incorporar-lo com a document funcional propi i utilitzar-lo com a contracte entre els mòduls d'accions, energia, genoma, món 2D i mètriques.
