# Projecte Evolució — Mort i selecció ambiental

**Especificació funcional de les causes de mort i del mecanisme de selecció ambiental emergent del prototip inicial, sense puntuació de fitness externa.**

## Metadades

| Camp | Valor |
| --- | --- |
| Tipus de document | Especificació funcional de mort i selecció ambiental |
| Punt del full de ruta | Nucli evolutiu del prototip inicial |
| Àmbit | Contracte funcional entre energia, recursos, metabolisme, edat, ambient, competència indirecta, reproducció, genealogia i mètriques de mortalitat |
| Objectiu | Definir causes de mort clares, mesurables i registrables, i establir que la selecció ambiental emergeix de la supervivència i la reproducció efectiva en un entorn limitat. |
| Resultat esperat | Un model on els agents moren per causes internes al sistema, aquestes morts generen pressió selectiva real i els llinatges poden comparar-se per persistència, descendència i mortalitat. |
| Decisió recomanada | Basar la primera versió principalment en manca d'energia, competència indirecta pels recursos i límit d'edat; deixar l'ambient com a modificador gradual fins validar metabolisme i reproducció. |
| Exclusió principal | No incloure fitness extern, depredació explícita, combat directe, malalties, catàstrofes massives ni mort probabilística sense causa funcional interpretable en el prototip inicial. |
| Criteri de tancament | Les causes de mort tenen codis estables, cada mort és explicable i registrable, la població queda limitada funcionalment i la selecció diferencial pot estudiar-se de manera repetible. |
| Estat | Especificació funcional inicial |
| Versió | 1.0 |
| Data | Juliol de 2026 |

## Resum executiu

La mort és un mecanisme central del Projecte Evolució: impedeix poblacions immortals, fa rellevants els recursos i permet que els llinatges menys eficients desapareguin o es redueixin amb el temps. La selecció ambiental no s'ha de calcular amb una puntuació externa, sinó que ha d'emergir del balanç entre energia obtinguda, energia gastada, temps de vida, competència indirecta i reproducció efectiva.

## 1. Objectiu del document

Aquest document defineix funcionalment la mort i la selecció ambiental dins del prototip inicial del Projecte Evolució. La mort no és només una condició de finalització d'un agent: és un mecanisme essencial perquè la població no creixi indefinidament, perquè els recursos siguin rellevants i perquè els llinatges amb comportaments menys eficients desapareguin o es redueixin amb el temps.

L'objectiu és establir causes de mort clares, mesurables i relacionades amb el funcionament intern del sistema. Cada mort ha de poder ser explicada per una causa concreta, registrada com a esdeveniment i agregada posteriorment com a mètrica d'anàlisi.

Aquest document també fixa com s'entén la selecció ambiental en aquesta primera fase: no com una puntuació assignada des de fora, sinó com el resultat acumulat de sobreviure prou temps, obtenir prou energia i reproduir-se amb èxit en un entorn limitat.

## 2. Principi rector

La selecció ha de ser conseqüència de les regles del món, no una regla explícita que decideixi quins agents són millors. Un agent no ha de morir perquè el sistema el consideri poc apte, sinó perquè no ha aconseguit mantenir energia, superar condicions ambientals, evitar els efectes de la competència o arribar a una reproducció viable.

- La mort ha de ser funcional, no ornamental.
- Les causes de mort han de ser poques, però ben definides.
- La pressió selectiva ha de sorgir de recursos limitats, costos energètics i condicions ambientals.
- La simulació no ha d'utilitzar una funció de fitness separada per decidir supervivència.
- Cada mort ha de deixar una traça analitzable.
- La selecció ha d'actuar sobre individus, però s'ha de poder estudiar en llinatges.

**Decisió recomanada:** en la primera versió, la selecció s'ha de basar principalment en manca d'energia, competència indirecta pels recursos i límit d'edat. Les condicions ambientals poden afegir pressió, però no haurien de dominar la simulació abans de comprovar que el metabolisme i la reproducció funcionen correctament.

## 3. Definició funcional de mort

La mort és el pas d'un agent de l'estat viu a l'estat mort. Un agent mort deixa d'actuar, deixa de consumir energia, deixa de percebre l'entorn i deixa de poder reproduir-se. El seu registre, però, s'ha de conservar per reconstruir la genealogia i calcular mètriques.

La mort s'ha de produir quan es compleix almenys una condició terminal definida pel model. Si diverses condicions es compleixen al mateix pas temporal, el sistema ha d'assignar una causa principal segons una prioritat establerta, però també pot registrar causes secundàries per a anàlisi posterior.

| Element | Definició funcional |
| --- | --- |
| Agent viu | Individu actiu que pot observar, decidir, gastar energia, alimentar-se, moure's i reproduir-se. |
| Agent mort | Individu inactiu que ja no participa en la simulació, però continua existint com a registre històric. |
| Causa de mort | Motiu principal pel qual l'agent ha passat a estat mort en un pas temporal concret. |
| Moment de mort | Pas temporal exacte en què l'agent és retirat del conjunt d'agents vius. |
| Traça postmortem | Dades conservades: identificador, progenitor, edat, energia final, genoma o referència al genoma, causa de mort i descendència acumulada. |

## 4. Causes de mort incloses en el prototip inicial

El prototip inicial ha d'incloure causes de mort suficients per evitar poblacions immortals i per generar pressió selectiva real. No cal introduir encara depredació complexa ni malalties, perquè afegirien soroll abans de validar el sistema energètic bàsic.

### 4.1. Mort per manca d'energia

És la causa de mort principal i obligatòria. Un agent mor quan la seva energia arriba a zero o cau per sota del llindar mínim definit. Aquesta causa connecta directament comportament, recursos, metabolisme i supervivència.

- Ha de ser la primera causa activa en la versió inicial.
- Ha de dependre del balanç entre energia obtinguda i energia gastada.
- Ha de penalitzar agents que es mouen sense obtenir recursos o que es reprodueixen massa aviat.
- Ha de permetre diferències entre llinatges sense necessitat d'una puntuació externa.

### 4.2. Mort per edat màxima

La mort per edat màxima evita que individus afortunats o molt eficients bloquegin indefinidament l'espai poblacional. També força renovació generacional i facilita que la selecció actuï sobre descendents.

Aquest límit no s'ha de fer servir com a mecanisme principal de selecció, sinó com a control estructural. Si la majoria d'agents moren per edat, la pressió energètica probablement és massa feble.

### 4.3. Mort per condició ambiental desfavorable

Les condicions ambientals poden afectar la supervivència quan el món introdueix variació temporal o espacial: zones més costoses, períodes de baixa regeneració de recursos, gradients de toxicitat o penalitzacions metabòliques locals.

En aquesta fase, és millor modelar l'ambient com un modificador de costos o disponibilitat de recursos, no com una eliminació aleatòria directa. Això manté la selecció vinculada al comportament de l'agent.

### 4.4. Mort per competència indirecta

La competència inicial hauria de ser indirecta: diversos agents consumeixen els mateixos recursos o ocupen zones favorables, i alguns acaben morint per manca d'energia. No cal implementar encara combat explícit.

Aquesta forma de competència és suficient per al prototip perquè genera selecció sense afegir regles agressives que podrien codificar estratègies artificials.

### 4.5. Mort per estat inviable o inconsistència funcional

El sistema pot necessitar una causa tècnica per retirar agents que arriben a estats invàlids: posició impossible, valors energètics corruptes o errors de simulació. Aquesta causa no hauria d'aparèixer en experiments normals; si apareix sovint, indica un problema d'implementació.

| Codi | Causa | Condició funcional | Comentari |
| --- | --- | --- | --- |
| `ENERGY_DEPLETION` | Manca d'energia | Energia <= llindar mínim | Causa principal esperada en una simulació ben calibrada. |
| `MAX_AGE` | Edat màxima | Edat >= límit configurat | Controla la renovació generacional; no hauria de dominar sempre. |
| `ENV_STRESS` | Estrès ambiental | Penalització ambiental acumulada o condició extrema | Millor com a cost o risc derivat, no com a eliminació arbitrària. |
| `COMPETITIVE_EXCL` | Exclusió competitiva | L'agent queda sense accés funcional a recursos o espai viable | Pot registrar-se com a causa secundària quan la mort final és energètica. |
| `INVALID_STATE` | Estat inviable | Valors físicament o lògicament impossibles | Ha de servir per depuració, no com a mecanisme ecològic. |

## 5. Prioritat de causes de mort

Quan diverses condicions terminals es compleixen al mateix pas temporal, el sistema ha d'assignar una causa principal de manera consistent. Aquesta prioritat evita registres ambigus i facilita comparar experiments.

1. **Estat inviable:** si l'agent té dades corruptes o impossibles, la causa principal ha de ser `INVALID_STATE`.
2. **Manca d'energia:** si l'energia és igual o inferior al llindar mínim, la causa principal ha de ser `ENERGY_DEPLETION`.
3. **Condició ambiental extrema:** si el model ambiental defineix una situació terminal directa, la causa principal ha de ser `ENV_STRESS`.
4. **Edat màxima:** si l'agent supera el límit d'edat sense cap altra causa terminal més directa, la causa principal ha de ser `MAX_AGE`.
5. **Exclusió competitiva:** s'ha de registrar preferentment com a causa secundària, tret que en una versió posterior es defineixi una regla explícita d'expulsió o desplaçament letal.

Aquesta prioritat pot ajustar-se tècnicament més endavant, però el criteri funcional ha de quedar estable: les morts han de ser interpretables i comparables entre execucions.

## 6. Seqüència dins d'un pas temporal

La mort s'ha d'avaluar en una posició clara dins del cicle de simulació. Si l'ordre és ambigu, poden aparèixer avantatges artificials: agents que es reprodueixen tot i estar morts, agents que consumeixen recursos amb energia negativa o descendents creats per progenitors inviables.

1. Calcular observacions dels agents vius.
2. Generar accions a partir de la xarxa neuronal.
3. Aplicar accions amb els seus costos energètics.
4. Aplicar alimentació i canvis de recursos.
5. Aplicar metabolisme basal i efectes ambientals del pas.
6. Avaluar condicions de mort.
7. Registrar morts i retirar agents del conjunt actiu.
8. Aplicar reproduccions vàlides només si el progenitor continua viu i compleix els llindars necessaris.
9. Registrar naixements, genealogia i mutacions.
10. Calcular mètriques del pas temporal.

> Advertiment funcional: no és recomanable permetre que un agent es reprodueixi després d'haver caigut per sota del llindar de mort en el mateix pas. Això premiaria estratègies suïcides que podrien dominar artificialment la simulació.

## 7. Selecció ambiental

La selecció ambiental és el procés pel qual determinats agents i llinatges tenen més probabilitat de sobreviure i reproduir-se perquè les seves accions els permeten mantenir energia, accedir a recursos i evitar condicions perjudicials. No és una entitat separada ni una funció de puntuació.

En el prototip inicial, la selecció s'ha d'observar a través de diferències estadístiques: alguns genomes deixen més descendència, alguns llinatges persisteixen més temps i altres desapareixen. Aquesta diferència ha de sorgir del sistema, no d'una classificació manual dels agents.

| Pressió selectiva | Com actua | Efecte esperat |
| --- | --- | --- |
| Escassetat de recursos | Els agents que no localitzen o consumeixen recursos moren abans. | Diferències en supervivència i descendència. |
| Cost de moviment | Moure's sense benefici redueix energia. | Penalització de comportaments erràtics o massa costosos. |
| Cost de reproducció | Reproduir-se massa aviat pot deixar el progenitor inviable. | Selecció sobre el moment i la freqüència reproductiva. |
| Edat màxima | Cap agent pot viure indefinidament. | Renovació generacional i més oportunitat per a mutacions. |
| Variació ambiental | Canvia el valor relatiu de comportaments i zones. | Prova d'adaptabilitat i robustesa dels llinatges. |
| Densitat local | Molts agents en una zona exhaureixen recursos ràpidament. | Competència indirecta i dispersió emergent. |

## 8. Relació entre mort, reproducció i llinatge

La mort individual només té sentit evolutiu si es relaciona amb la reproducció anterior i amb la persistència del llinatge. Un agent que mor jove però ha deixat molts descendents pot ser evolutivament rellevant; un agent que viu molt però no es reprodueix pot no aportar continuïtat genètica.

Per això, el sistema ha de registrar la mort juntament amb dades reproductives i genealògiques. La selecció no es pot interpretar només amb l'edat de mort; cal mirar també descendència, energia acumulada, causes de mort i èxit del llinatge.

| Pregunta d'anàlisi | Dada necessària |
| --- | --- |
| Quins llinatges desapareixen més sovint? | Causa de mort per agent i identificador de llinatge. |
| Quins genomes generen més descendència? | Nombre de fills directes i descendents acumulats. |
| Els agents més longeus són també els més reproductius? | Edat de mort i nombre de naixements associats. |
| La pressió ambiental afecta tots els llinatges igual? | Mortalitat per període ambiental i genealogia. |
| La població està col·lapsant per manca de recursos? | Morts per energia, recursos disponibles i densitat poblacional. |

## 9. Dades mínimes que cal registrar en cada mort

Cada mort ha de generar un esdeveniment registrable. Aquest esdeveniment ha de permetre reconstruir què ha passat sense haver de tornar a executar la simulació completa.

| Camp | Descripció funcional |
| --- | --- |
| `event_id` | Identificador únic de l'esdeveniment de mort. |
| `step` | Pas temporal en què s'ha produït la mort. |
| `agent_id` | Identificador de l'agent mort. |
| `parent_id` | Identificador del progenitor, si existeix. |
| `lineage_id` | Identificador del llinatge o arrel genealògica. |
| `age_at_death` | Edat de l'agent en el moment de morir. |
| `energy_at_death` | Energia final registrada abans de retirar l'agent. |
| `death_cause` | Causa principal de mort segons la taxonomia definida. |
| `secondary_causes` | Causes secundàries o factors contribuents, si s'escau. |
| `position` | Posició de l'agent en el moment de la mort. |
| `offspring_count` | Nombre de descendents directes produïts abans de morir. |
| `genome_id` | Referència al genoma o versió genètica de l'agent. |
| `environment_state` | Estat ambiental rellevant en el moment de la mort. |

## 10. Mètriques agregades de mortalitat

Les morts individuals han de poder agregar-se per produir mètriques útils. Aquestes mètriques són imprescindibles per saber si la selecció és massa feble, massa forta o simplement sorollosa.

- Nombre total de morts per pas temporal.
- Percentatge de morts per causa.
- Edat mitjana de mort.
- Distribució d'edats de mort.
- Energia mitjana en el moment de morir.
- Mortalitat per llinatge.
- Mortalitat abans i després de la primera reproducció.
- Nombre d'agents que moren sense descendència.
- Morts per manca d'energia en relació amb recursos disponibles.
- Mortalitat durant canvis ambientals.
- Taxa d'extinció de llinatges.
- Relació entre densitat poblacional i mortalitat.

## 11. Condicions de calibratge

Un sistema de mort i selecció pot fallar tant per excés com per defecte. Si gairebé ningú mor, no hi ha selecció. Si moren gairebé tots abans de reproduir-se, no hi ha evolució acumulativa. Cal definir proves de calibratge abans d'interpretar resultats.

| Símptoma | Interpretació probable | Ajust funcional |
| --- | --- | --- |
| La població creix sense límit | Recursos massa abundants, costos massa baixos o reproducció massa barata. | Reduir regeneració, pujar metabolisme o augmentar cost reproductiu. |
| La població s'extingeix sempre molt ràpid | Pressió massa forta o recursos massa escassos. | Augmentar recursos inicials, reduir costos o retardar edat reproductiva mínima. |
| Gairebé totes les morts són per edat | La selecció energètica és massa feble. | Reduir recursos o augmentar costos de vida i moviment. |
| Gairebé totes les morts són ambientals | L'ambient està dominant el model. | Convertir l'ambient en modificador gradual, no causa terminal massiva. |
| Els llinatges canvien només per atzar | Mutació o mortalitat massa sorollosa. | Reduir mutació, estabilitzar ambient o ampliar durada experimental. |
| Agents suïcides es reprodueixen abans de morir | Ordre temporal mal definit o cost reproductiu massa baix. | Aplicar mort abans de reproducció o exigir energia postreproductiva mínima. |

## 12. Exclusions temporals

Per mantenir la primera versió clara i analitzable, queden fora del prototip inicial diversos mecanismes de mortalitat que podrien ser interessants més endavant però que ara afegirien soroll o complexitat prematura.

- Depredació explícita entre agents.
- Combat directe amb dany i defensa.
- Malalties transmissibles.
- Envelliment fisiològic complex.
- Diferenciació per sexes.
- Catàstrofes aleatòries massives com a mecanisme principal.
- Mort probabilística sense causa funcional interpretable.
- Fitness extern que elimini agents segons una puntuació abstracta.
- Intervenció manual per preservar o eliminar llinatges.
- Reciclatge ecològic complex de cadàvers en nutrients.

## 13. Riscos principals

### 13.1. Convertir la mort en una funció de fitness amagada

Si el sistema elimina agents perquè tenen una puntuació baixa, s'està substituint l'ecologia per una optimització externa. Això pot ser útil en altres contextos, però no és coherent amb l'objectiu del Projecte Evolució.

### 13.2. Fer la mortalitat massa aleatòria

Una mica d'estocasticitat pot ser acceptable, però si la mort depèn massa de l'atzar, la selecció acumulativa queda debilitada. Els agents han de poder influir indirectament en la seva supervivència mitjançant accions.

### 13.3. Fer l'entorn massa dur

Si la mortalitat és extrema, cap llinatge té temps d'acumular variacions útils. L'extinció constant no és evolució; és un sistema mal calibrat.

### 13.4. Fer l'entorn massa fàcil

Si la majoria d'agents sobreviuen i es reprodueixen sense dificultat, les diferències de genoma no importaran gaire. El sistema pot créixer, però no seleccionar.

### 13.5. No registrar prou informació

Sense causa de mort, edat, energia i llinatge, no es pot interpretar la dinàmica. La visualització pot semblar convincent, però l'anàlisi seria feble.

## 14. Proves funcionals mínimes

Abans de considerar tancat aquest bloc, caldrà comprovar que les regles de mort produeixen efectes coherents i mesurables.

1. Amb recursos zero o molt baixos, els agents han de morir per manca d'energia.
2. Amb recursos abundants però edat màxima activa, els agents no han de viure indefinidament.
3. Amb costos de moviment elevats, els agents que es mouen molt sense alimentar-se han de morir abans.
4. Amb regeneració de recursos limitada, la població no ha de créixer indefinidament.
5. Amb entorn canviant, la distribució de causes de mort ha de variar de manera mesurable.
6. Cada mort ha de quedar registrada amb agent, pas temporal, causa, edat, energia i llinatge.
7. Cap agent mort ha de poder executar accions ni reproduir-se en passos posteriors.
8. La repetició amb la mateixa llavor ha de produir el mateix patró de morts.

## 15. Criteri de tancament

Aquest bloc es considerarà tancat quan el model defineixi clarament les causes de mort, l'ordre temporal d'avaluació, la taxonomia de registres i les mètriques agregables. També cal que la selecció ambiental quedi descrita com un resultat emergent de la supervivència i la reproducció, no com una funció externa de puntuació.

- Les causes de mort estan definides i tenen codis estables.
- Cada mort és explicable per una regla del sistema.
- Les morts es poden registrar com a esdeveniments analitzables.
- La població no pot créixer indefinidament sense restriccions.
- La manca d'energia té conseqüències funcionals clares.
- La mort per edat evita immortalitat sense substituir la selecció energètica.
- La competència inicial és indirecta i vinculada als recursos.
- No hi ha cap fitness extern que decideixi quins agents sobreviuen.
- Les mètriques permeten comparar mortalitat entre llinatges i configuracions.
- El model permet estudiar selecció diferencial de manera repetible.

## 16. Resultat esperat

El resultat esperat és un model on els agents moren per causes internes al sistema i on aquestes morts generen pressió selectiva real. Alguns agents moriran abans de reproduir-se, altres deixaran descendència i alguns llinatges persistiran més que altres. Aquesta diferència, acumulada al llarg del temps, és el nucli de la dinàmica evolutiva del prototip.

Si el sistema permet explicar per què moren els agents, agregar aquestes morts en mètriques i relacionar-les amb supervivència, reproducció i llinatge, el model de mort i selecció ambiental estarà funcionalment definit.
