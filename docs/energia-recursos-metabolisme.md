# Projecte Evolució — Energia, recursos i metabolisme

**Especificació funcional de l'economia energètica del prototip inicial: energia interna limitada, recursos ambientals finits i regenerables, costos metabòlics i d'acció, mort per manca d'energia i reproducció condicionada per energia acumulada.**

## Metadades

| Camp | Valor |
| --- | --- |
| Tipus de document | Especificació funcional d'energia, recursos i metabolisme |
| Punt del full de ruta | Punt 8 — Definir energia, recursos i metabolisme |
| Àmbit | Contracte funcional entre recursos del món, estat energètic dels agents, accions, reproducció, mort, canvi ambiental i mètriques |
| Objectiu | Formalitzar com els agents guanyen energia, com la gasten i com els recursos apareixen, desapareixen o es regeneren dins l'entorn. |
| Resultat esperat | Una economia energètica que faci impossible la supervivència gratuïta i obligui els agents a competir indirectament per recursos limitats. |
| Decisió recomanada | Un únic recurs energètic en un món 2D discret, distribuït irregularment, amb capacitat local màxima, regeneració gradual, energia interna limitada, cost basal, cost de moviment, cost reproductiu i mort per energia insuficient. |
| Exclusió principal | No incloure múltiples nutrients, magatzems externs, intercanvi o robatori d'energia, fisiologia complexa, fotosíntesi, depredació ni cadenes tròfiques en el prototip inicial. |
| Criteri de tancament | La població no creix indefinidament sense restricció i la manca d'energia genera conseqüències funcionals mesurables. |
| Estat | Especificació funcional inicial |
| Versió | 1.0 |
| Data | Juliol de 2026 |

## Resum executiu

L'energia és la moneda funcional que connecta recursos, accions, supervivència, reproducció, mort i selecció ambiental. Cada agent ha de perdre energia pel simple fet d'existir, gastar energia quan actua i només reproduir-se si abans ha aconseguit prou energia del món. Els recursos han de ser finits en cada moment, esgotables localment, regenerables de manera gradual i distribuïts de forma heterogènia per generar competència indirecta i pressió selectiva real.

## 1. Objectiu del document

Aquest document defineix funcionalment l'economia energètica del prototip inicial del Projecte Evolució. L'energia ha de ser el mecanisme que connecta recursos, accions, supervivència, reproducció, mort i selecció ambiental.

La finalitat no és crear un model bioquímic realista, sinó un sistema simple, mesurable i prou restrictiu perquè els agents no puguin sobreviure ni reproduir-se sense prendre decisions eficients dins un entorn limitat.

## 2. Principi rector

La supervivència no pot ser gratuïta. Cada agent ha de perdre energia pel simple fet d'existir, ha de gastar energia quan actua i només ha de poder transmetre el genoma si abans ha aconseguit prou energia del món.

- L'energia ha de ser finita, acumulable només fins a un límit i consumida de manera contínua.
- Els recursos han de ser limitats, regenerables i distribuïts de manera no completament homogènia.
- L'alimentació ha de transformar recursos ambientals en energia interna.
- La reproducció ha de tenir un cost prou alt per impedir creixement explosiu.
- La mort per manca d'energia ha de ser una conseqüència funcional inevitable.
- Les regles han de ser iguals per a tots els agents i configurables per experiment.

> Nota funcional: si el sistema permet que una població creixi indefinidament sense esgotar recursos ni patir mortalitat, no estem davant d'una ecologia evolutiva funcional, sinó d'una animació sense pressió selectiva real.

## 3. Rol funcional de l'energia

L'energia és l'estat intern més important de l'agent. Representa la seva capacitat de continuar viu, executar accions costoses i reproduir-se. No s'ha d'interpretar com una magnitud biològica literal, sinó com una moneda funcional comuna dins la simulació.

- Si l'energia baixa, l'agent s'apropa a la mort.
- Si l'energia puja, l'agent guanya marge d'acció i pot arribar al llindar reproductiu.
- Si l'energia arriba al màxim, l'excés obtingut es perd o queda saturat.
- Si l'energia arriba a zero o sota el llindar de mort, l'agent mor.

Aquesta simplicitat és una virtut. Permet que les conseqüències de cada decisió siguin analitzables i que la selecció actuï sobre diferències heretables en comportament.

## 4. Elements de l'economia energètica

El sistema energètic queda definit per la relació entre energia interna, recursos externs, costos metabòlics i costos d'acció.

| Element | Funció ecològica | Efecte sobre l'agent | Risc si està mal calibrat |
| --- | --- | --- | --- |
| Energia interna | Mesura la capacitat immediata de continuar viu i actuar. | Permet moure's, alimentar-se, reproduir-se i resistir períodes pobres. | Si és massa abundant, no hi ha selecció; si cau massa ràpid, hi ha extinció prematura. |
| Recurs ambiental | Font externa d'energia limitada dins el món 2D. | Incrementa energia quan és consumit amb èxit. | Si es regenera massa ràpid, desapareix la competència; si massa lent, col·lapse global. |
| Metabolisme basal | Cost mínim de mantenir-se viu per pas temporal. | Redueix energia encara que l'agent no faci res. | Si és massa baix, sobreviure és gratuït; si és massa alt, l'exploració és inviable. |
| Cost de moviment | Penalitza l'exploració i crea dilema entre buscar i conservar. | Redueix energia quan l'agent es mou. | Si és nul, moure's sempre domina; si és massa alt, els agents queden immòbils. |
| Cost reproductiu | Fa que reproduir-se sigui una decisió arriscada. | Redueix energia del progenitor i dota el descendent. | Si és massa baix, explosió poblacional; si és massa alt, no hi ha descendència. |
| Límit d'energia | Evita acumulacions il·limitades i estratègies trivials d'emmagatzematge. | Satura l'energia màxima de cada agent. | Sense límit, poden aparèixer individus pràcticament invulnerables. |

## 5. Model funcional de recursos

El recurs inicial del prototip hauria de ser únic. Afegir múltiples nutrients, aigua, minerals o tipus d'aliment pot semblar més ecològic, però en aquesta fase complicaria la interpretació dels resultats sense aportar una pressió selectiva imprescindible.

Cada posició del món pot contenir una quantitat de recurs entre zero i una capacitat local màxima. El recurs pot ser consumit pels agents i regenerar-se amb el temps segons una taxa configurada.

| Fase | Descripció funcional | Dades que cal registrar |
| --- | --- | --- |
| Inicialització | El món crea una distribució inicial de recursos segons una regla configurada. | Quantitat total, patró espacial, llavor aleatòria. |
| Disponibilitat | Cada cel·la o zona manté una quantitat de recurs entre zero i una capacitat local màxima. | Recurs per cel·la, recurs total disponible. |
| Consum | Els agents que executen alimentació poden transformar part del recurs local en energia pròpia. | Unitats consumides, energia guanyada, conflictes de consum. |
| Esgotament | Una cel·la pot quedar temporalment sense recurs després del consum. | Nombre de cel·les esgotades, durada mitjana de l'esgotament. |
| Regeneració | El recurs torna a aparèixer gradualment segons una taxa i un sostre local. | Recurs regenerat per pas, zones recuperades. |
| Canvi ambiental | La productivitat pot variar per fase ambiental simple, sense clima complex. | Multiplicador ambiental, durada, impacte sobre població. |

## 6. Distribució espacial dels recursos

La distribució de recursos ha de crear heterogeneïtat ambiental. Un món completament uniforme redueix la necessitat d'exploració i pot fer que moltes estratègies siguin equivalents.

Distribució recomanada per al prototip inicial:

- zones amb concentració moderada de recursos;
- zones pobres o temporalment buides;
- variació espacial generada per llavor aleatòria;
- capacitat màxima local per evitar acumulació infinita;
- regeneració gradual, no aparició instantània completa.

No cal introduir encara biomes complexos. N'hi ha prou amb una distribució irregular i persistent que faci que moure's, quedar-se o consumir siguin decisions amb conseqüències.

## 7. Regeneració dels recursos

La regeneració és el mecanisme que evita que el món s'esgoti completament després de pocs passos, però també pot destruir la pressió selectiva si és massa generosa.

- Cada cel·la pot regenerar una fracció fixa o estocàstica del seu recurs per pas.
- La regeneració no pot superar la capacitat màxima local.
- La taxa de regeneració ha de ser un paràmetre explícit de l'experiment.
- La regeneració pot estar modulada per una condició ambiental global simple.
- Les zones esgotades han de trigar diversos passos a recuperar-se si han estat sobreexplotades.

> Nota funcional: la regeneració instantània completa és una mala opció per al primer prototip: converteix els recursos en una font gairebé infinita i debilita la selecció per eficiència o moviment.

## 8. Metabolisme basal

El metabolisme basal és el cost energètic de continuar viu durant un pas de simulació. S'aplica a tots els agents vius, independentment de l'acció que facin.

Funcionalment, aquest cost obliga els agents a obtenir recursos de manera recurrent. Sense cost basal, un agent podria quedar-se quiet indefinidament i continuar viu, cosa que trencaria el vincle entre comportament i supervivència.

Regla funcional recomanada:

- aplicar cost basal a cada pas;
- aplicar-lo abans o després de les accions, però sempre de manera consistent;
- registrar energia abans i després del cost;
- mantenir-lo igual per a tots els agents en la primera versió, tret que es decideixi incloure metabolisme com a paràmetre heretable en una fase posterior.

## 9. Costos d'acció

Les accions han de tenir costos diferents perquè l'agent afronti dilemes reals. En particular, el moviment i la reproducció han de tenir conseqüències energètiques clares.

Costos mínims inclosos:

- cost basal: s'aplica sempre;
- cost de moviment: s'aplica quan l'agent es desplaça amb èxit;
- cost de reproducció: s'aplica quan neix un descendent;
- cost d'alimentació: pot ser nul o molt baix en el prototip inicial;
- cost d'intent fallit: millor deixar-lo desactivat inicialment o molt baix.

Penalitzar massa les accions fallides pot fer que els primers genomes aleatoris morin abans que aparegui variació útil. Per això, la primera versió ha de prioritzar costos clars però no excessivament punitius.

## 10. Alimentació i conversió energètica

L'alimentació converteix recurs ambiental en energia interna de l'agent. Aquesta conversió ha de ser explícita, limitada per la quantitat de recurs disponible i saturada pel màxim energètic de l'agent.

- L'agent només pot alimentar-se si hi ha recurs dins l'abast definit.
- El recurs consumit desapareix parcialment o totalment de la cel·la.
- L'energia guanyada depèn d'una eficiència o factor de conversió.
- Si l'agent ja està al màxim d'energia, l'excés no s'acumula.
- Si diversos agents consumeixen el mateix recurs, cal una regla neutral de repartiment o adjudicació.

> Nota funcional: és millor que alimentar-se sigui una acció explícita, no automàtica. Això manté el dilema entre explorar, consumir i reproduir-se.

## 11. Reproducció i transferència d'energia

La reproducció ha de convertir energia acumulada en descendència. Aquesta conversió és el pont entre èxit ecològic i èxit evolutiu: només els agents que obtenen prou energia poden deixar còpies mutades del seu genoma.

- L'agent ha de superar un llindar mínim d'energia per intentar reproduir-se.
- El progenitor paga un cost reproductiu definit.
- El descendent rep una energia inicial definida.
- La suma de cost reproductiu i energia del descendent ha de ser coherent, encara que no cal simular conservació física exacta.
- Si no hi ha espai disponible, la reproducció falla i s'ha de registrar la causa.
- La reproducció massa barata ha de considerar-se un error de calibració.

La reproducció no ha d'incloure cap criteri artificial de qualitat genètica. La selecció ja s'expressa a través de la capacitat prèvia d'obtenir energia i sobreviure.

## 12. Mort per manca d'energia

La mort per manca d'energia és obligatòria en el prototip inicial. És la conseqüència funcional que dona sentit al metabolisme i a la competència pels recursos.

- Un agent mor si la seva energia arriba a zero o baixa sota el llindar definit.
- La causa de mort s'ha de registrar com a manca d'energia.
- Un agent mort no executa accions posteriors.
- El seu espai queda alliberat segons la regla del món.
- La seva genealogia i mètriques acumulades es conserven per a l'anàlisi.

No és recomanable introduir encara estats intermedis complexos com malaltia, fatiga o dany parcial. Primer cal validar que el cicle energia-recurs-mort funciona sense ambigüitats.

## 13. Balanç energètic per pas temporal

Cada pas de simulació ha de tancar un balanç energètic comprensible. Això no vol dir conservar energia física exacta, sinó poder explicar d'on ve l'energia dels agents, on es gasta i quina part del recurs ambiental s'ha consumit o regenerat.

| Ordre | Operació | Raó funcional |
| --- | --- | --- |
| 1 | Aplicar metabolisme basal als agents vius. | La supervivència mai és gratuïta. |
| 2 | Calcular observacions locals segons estat actual. | Les decisions parteixen d'informació parcial i consistent. |
| 3 | La xarxa neuronal proposa accions. | El genoma condiciona el comportament. |
| 4 | Validar accions i costos mínims. | Evita accions impossibles o sense energia suficient. |
| 5 | Resoldre moviment, alimentació, reproducció i conflictes. | Actualitza món i agents amb regles neutrals. |
| 6 | Aplicar guanys i costos energètics finals. | Tanca el balanç energètic del pas. |
| 7 | Regenerar recursos i aplicar condició ambiental. | El món evoluciona i crea pressió temporal. |
| 8 | Determinar morts, naixements i mètriques. | La simulació deixa traça analitzable. |

## 14. Paràmetres funcionals mínims

Els paràmetres energètics han de quedar explícits a la configuració de cada experiment. No s'han d'amagar dins constants màgiques, perquè petits canvis poden alterar completament la dinàmica evolutiva.

| Paràmetre funcional | Descripció | Criteri de calibració inicial |
| --- | --- | --- |
| `energia_inicial_agent` | Energia assignada als agents fundadors. | Ha de permetre explorar uns quants passos, però no reproduir-se immediatament sense alimentar-se. |
| `energia_maxima_agent` | Sostre d'energia acumulable. | Evita acumulació il·limitada i estabilitza el rang de valors neuronals observables. |
| `cost_basal` | Energia perduda per pas temporal. | Ha de ser prou alt per obligar a buscar recursos. |
| `cost_moviment` | Cost addicional per desplaçar-se. | Ha de crear dilema entre explorar i conservar energia. |
| `guany_per_recurs` | Energia obtinguda per unitat consumida. | Ha de compensar el cost de buscar menjar, però no fer trivial la supervivència. |
| `taxa_regeneracio` | Velocitat de recuperació dels recursos. | Ha de permetre recuperació local, però amb escassetat sostinguda. |
| `capacitat_recurs_local` | Màxim recurs acumulable per cel·la o zona. | Ha de crear heterogeneïtat espacial sense crear oasis infinits. |
| `llindar_reproduccio` | Energia mínima per intentar reproduir-se. | Ha d'exigir èxit previ d'alimentació. |
| `cost_reproduccio` | Energia perduda pel progenitor. | Ha d'impedir creixement explosiu. |
| `energia_descendent` | Energia inicial del nou agent. | Ha de permetre viabilitat inicial sense garantir supervivència. |

## 15. Control del creixement poblacional

El sistema no ha de dependre només d'un límit artificial de població per evitar el creixement indefinit. La regulació principal ha de venir de recursos limitats, metabolisme basal i cost reproductiu.

- Els recursos limitats redueixen la capacitat de mantenir massa agents.
- El metabolisme basal elimina individus que no s'alimenten prou.
- El cost reproductiu impedeix reproducció gratuïta.
- La saturació espacial pot fer fallar reproduccions en zones massa denses.
- Un límit màxim tècnic de població pot existir com a protecció computacional, però no ha de ser el mecanisme ecològic principal.

> Nota funcional: un límit dur de població és útil per seguretat del programa, però si és el que regula realment la simulació, la pressió selectiva queda parcialment artificialitzada.

## 16. Canvis ambientals simples

El prototip inicial pot incloure una variació ambiental simple, però no ha de convertir-se en un model climàtic. La funció del canvi ambiental és comprovar si els llinatges responen diferent quan canvia la disponibilitat energètica.

Opció recomanada: aplicar fases temporals on la taxa de regeneració dels recursos augmenta o disminueix mitjançant un multiplicador ambiental global o regional. Per exemple, períodes de productivitat normal alternats amb períodes pobres.

- El canvi ambiental ha de ser registrat amb inici, durada i intensitat.
- Ha d'afectar recursos o costos, no directament la qualitat del genoma.
- No ha de ser tan extrem que extingeixi sempre la població.
- Ha de permetre comparar llinatges abans, durant i després del canvi.

## 17. Mètriques energètiques

Sense mètriques energètiques, no es podrà saber si l'economia del sistema funciona o si els resultats són simples fluctuacions. Aquest bloc ha de registrar dades globals, individuals i per llinatge.

- Energia mitjana, mínima i màxima de la població.
- Distribució d'energia per edat o llinatge.
- Recurs total disponible al món.
- Recurs consumit per pas.
- Recurs regenerat per pas.
- Energia guanyada per alimentació.
- Energia gastada en metabolisme basal.
- Energia gastada en moviment.
- Energia gastada en reproducció.
- Morts per manca d'energia.
- Naixements per energia disponible.
- Eficiència energètica per llinatge: descendència o supervivència per energia consumida.

## 18. Validació i calibració

La calibració energètica s'ha de fer amb experiments petits i repetibles abans d'executar simulacions llargues. El primer objectiu no és trobar paràmetres perfectes, sinó evitar règims clarament inútils: extinció immediata, explosió poblacional o supervivència passiva sense selecció.

| Problema observat | Interpretació probable | Resposta recomanada |
| --- | --- | --- |
| La població creix sense límit. | Recursos massa abundants, cost basal baix o reproducció massa barata. | Reduir regeneració, augmentar cost reproductiu o introduir límit energètic més estricte. |
| La població s'extingeix sempre molt aviat. | Costos massa alts o recursos massa escassos. | Baixar cost basal o moviment, augmentar recurs inicial o regeneració. |
| Els agents no es mouen. | Moviment massa car o recursos locals massa suficients. | Reduir cost de moviment o fer els recursos més irregulars espacialment. |
| Els agents només es mouen i no mengen. | Alimentació poc rendible o observacions insuficients. | Revisar guany per recurs i senyal local de recurs. |
| Reproducció gairebé nul·la. | Llindar o cost reproductiu excessius. | Baixar llindar reproductiu o energia necessària del descendent. |
| Tots els llinatges semblen equivalents. | Pressió selectiva feble o mètriques insuficients. | Augmentar escassetat, heterogeneïtat o canvi ambiental simple. |

## 19. Exclusions temporals

Per mantenir el prototip inicial net i interpretable, queden fora d'aquesta fase:

- múltiples tipus de nutrients;
- magatzems externs de recursos;
- intercanvi d'energia entre agents;
- robatori d'energia;
- malalties metabòliques;
- fisiologia interna complexa;
- costos dependents d'òrgans o morfologia avançada;
- fotosíntesi, depredació o cadenes tròfiques;
- adaptació individual del metabolisme durant la vida;
- optimització manual de dietes o estratègies alimentàries.

Aquestes ampliacions poden ser útils més endavant, però ara farien més difícil saber si el bucle evolutiu bàsic funciona.

## 20. Criteris funcionals d'èxit

1. Cada agent té una energia interna que canvia al llarg del temps.
2. Cada pas temporal aplica un cost basal de supervivència.
3. El moviment i la reproducció tenen costos energètics explícits.
4. Els recursos del món són finits en cada moment i poden esgotar-se localment.
5. Els recursos es regeneren segons una regla configurada i mesurable.
6. Els agents poden morir per manca d'energia.
7. La reproducció només és possible si l'agent ha acumulat prou energia.
8. La població no creix indefinidament sense limitació ecològica.
9. Les mètriques permeten explicar guanys, costos, morts, naixements i consum de recursos.
10. La mateixa configuració amb la mateixa llavor pot repetir la dinàmica energètica de manera reproduïble.

## 21. Decisió d'abast recomanada

La decisió recomanada per a la primera versió és utilitzar un únic recurs energètic, distribuït de manera irregular en un món 2D discret, amb regeneració gradual i capacitat local màxima. Cada agent tindrà energia interna limitada, cost basal per pas, cost de moviment, cost reproductiu i mort per energia insuficient.

Aquesta solució és simple, però suficient per crear competència indirecta, pressió selectiva i diferències mesurables entre llinatges. Afegir més fisiologia ara seria una complicació prematura.

## 22. Resultat esperat

Un cop definit i implementat aquest bloc, la simulació haurà de mostrar agents que no poden sobreviure sense alimentar-se, que han de gestionar el cost de moure's i que només poden reproduir-se si han obtingut prou energia del món.

El sistema haurà de permetre observar si alguns genomes generen patrons més eficients de consum, exploració i reproducció. La selecció no vindrà d'una puntuació externa, sinó de la capacitat de mantenir energia suficient fins a deixar descendència viable.

## 23. Definició de tancament

Aquest punt es considerarà tancat quan existeixi una especificació funcional acceptada que defineixi:

- com es representa l'energia interna de l'agent;
- com es representen els recursos del món;
- com es consumeixen i regeneren els recursos;
- quins costos energètics té cada acció rellevant;
- quin metabolisme basal s'aplica;
- com es produeix la mort per manca d'energia;
- com la reproducció transforma energia en descendència;
- quins paràmetres han de quedar a la configuració experimental;
- quines mètriques energètiques s'han de registrar;
- quins règims de calibració indiquen que el model és útil o defectuós.

Amb aquesta definició, el projecte pot avançar cap a la implementació del bucle energètic sense confondre el model ecològic amb una simple animació d'agents que es mouen pel món.

## 24. Adequació amb la resta de documentació

La proposta és coherent amb el full de ruta funcional perquè cobreix el punt 8 i manté el criteri central que el prototip ha de tenir regles energètiques coherents, mètriques traçables i experiments repetibles.

També encaixa amb l'arquitectura general: l'entorn manté recursos i regeneració, l'agent manté energia i estat vital, la reproducció aplica costos i naixements, mort i selecció ambiental registren defuncions, i mètriques calcula indicadors energètics i poblacionals.

Pel que fa al model del món 2D, el document reforça que els recursos siguin irregulars, locals, finits, regenerables i afectats per una condició ambiental simple. Pel que fa al model d'agent, concreta l'energia com a estat intern central. Pel que fa a observacions i accions, preserva que l'alimentació sigui una acció explícita i que els agents observin informació local, no un mapa global.

### Valoració funcional

La valoració és positiva: el document és adequat per incorporar-se a la documentació del projecte perquè completa una peça essencial del bucle evolutiu sense introduir complexitat prematura. El seu punt més fort és que converteix la supervivència, l'exploració, l'alimentació i la reproducció en decisions amb conseqüències energètiques mesurables.

Hi ha tres punts que convé vigilar en la fase d'implementació:

1. **Ordre del pas temporal:** cal mantenir un ordre únic i testejable perquè metabolisme, observacions, accions, regeneració i morts no generin ambigüitats.
2. **Calibració inicial:** els paràmetres proposats són funcionals, però necessitaran experiments petits per evitar extinció immediata o explosió poblacional.
3. **Traçabilitat de mètriques:** el model depèn molt de poder explicar energia guanyada, energia gastada, recurs consumit, recurs regenerat, morts i naixements per pas.

No s'han detectat contradiccions importants amb els documents existents. L'única decisió que caldrà deixar especialment clara quan s'implementi és si el metabolisme basal s'aplica abans o després de la decisió; aquest document recomana consistència i proposa aplicar-lo al començament del balanç per reforçar que la supervivència mai és gratuïta.
