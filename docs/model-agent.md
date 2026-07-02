# Projecte Evolució — Model d'agent

**Definició funcional de l'individu dins el prototip inicial: estat mínim, identitat, energia, genoma neuronal, cicle vital i traçabilitat genealògica.**

## Metadades

| Camp | Valor |
| --- | --- |
| Tipus de document | Especificació funcional del model d'agent |
| Punt del full de ruta | Punt 4 — Definir el model d'agent |
| Àmbit | Definició funcional de l'individu dins el prototip inicial |
| Objectiu | Establir quin estat ha de tenir un agent per viure, actuar, reproduir-se, morir i deixar traça genealògica. |
| Nivell de detall | Funcional; no defineix encara classes, mòduls concrets ni esquemes físics de base de dades |
| Decisió recomanada | Agent simple, sense aprenentatge individual, amb estat mínim vectoritzable i genoma neuronal heretable |
| Criteri de tancament | Cada agent pot ser seguit des del naixement fins a la mort i vinculat al seu progenitor. |
| Estat | Especificació funcional inicial |
| Data | 2 de juliol de 2026 |

## Resum executiu

L'agent del prototip inicial ha de ser una unitat biològica abstracta: té una posició, una energia, una edat, un genoma, una xarxa neuronal associada i una relació genealògica amb el seu progenitor. No ha de contenir estratègies programades ni intel·ligència explícita. La seva conducta emergeix de la xarxa neuronal i és seleccionada indirectament per supervivència i reproducció.

## 1. Objectiu del document

Aquest document defineix què és un agent dins el prototip inicial del Projecte Evolució. L'objectiu és establir l'estat funcional mínim que necessita cada individu per participar en el bucle ecològic i evolutiu: néixer, observar l'entorn, actuar, consumir energia, reproduir-se, mutar, morir i deixar registre.

La definició de l'agent és central perquè condiciona tot el sistema. Si l'agent és massa pobre, no hi haurà variació evolutiva útil. Si és massa complex, el prototip serà difícil de depurar i massa car computacionalment.

## 2. Principi rector

L'agent ha de ser simple. La complexitat no s'ha d'introduir dins l'agent en forma de regles intel·ligents, sinó que ha d'aparèixer a escala poblacional per selecció ambiental.

- L'agent no sap què és sobreviure ni què és reproduir-se en termes conceptuals.
- L'agent només rep observacions locals i produeix una acció mitjançant la seva xarxa neuronal.
- La supervivència i la reproducció depenen de l'encaix entre la seva conducta, el seu cost energètic i l'entorn.
- El sistema ha de poder simular molts agents; per tant, l'estat individual ha de ser compacte.
- Cada agent ha de ser traçable al llarg de tota la seva vida i dins la genealogia poblacional.

## 3. Definició funcional d'agent

Un agent és un individu autònom de la simulació. No és una classe psicològica ni una entitat amb objectius explícits. És una estructura funcional que concentra l'estat necessari per prendre accions i participar en el procés evolutiu.

En el prototip inicial, un agent es defineix per quatre dimensions: identitat, estat corporal, genoma i estat vital.

| Dimensió | Contingut funcional | Funció dins la simulació |
| --- | --- | --- |
| Identitat | Identificador únic, progenitor, generació i llinatge. | Permet traçabilitat, genealogia i anàlisi de descendència. |
| Estat corporal | Posició, energia, edat i estat intern mínim. | Determina on és, quant pot viure i quin context fisiològic té. |
| Genoma | Pesos, biaixos i paràmetres heretables. | Codifica la xarxa neuronal i la variació evolutiva. |
| Estat vital | Viu, mort, causa de mort i moment de naixement/mort. | Permet gestionar el cicle vital i registrar mètriques. |

## 4. Estat mínim obligatori

El prototip inicial hauria de considerar obligatori el conjunt d'estat següent. Qualsevol implementació tècnica posterior pot organitzar aquestes dades de manera diferent, però funcionalment han d'existir.

| Camp funcional | Definició | Justificació |
| --- | --- | --- |
| `agent_id` | Identificador únic | Permet distingir cada individu de manera estable durant tota la seva vida. |
| `parent_id` | Identificador del progenitor | Permet reconstruir genealogies i èxit reproductiu. Pot ser nul en agents inicials. |
| `lineage_id` | Identificador de llinatge | Agrupa descendents d'un fundador o branca rellevant. Recomanable des del principi. |
| `generation` | Generació evolutiva | Nombre de passos reproductius respecte al fundador. |
| `birth_step` | Pas de naixement | Permet calcular edat real i supervivència. |
| `death_step` | Pas de mort | Permet tancar el registre vital. Nul mentre és viu. |
| `position` | Posició al món 2D | Ubica l'agent i determina observacions, moviment i accés a recursos. |
| `energy` | Reserva energètica | Variable central de supervivència, reproducció i mort. |
| `age` | Edat | Pot calcular-se des de `birth_step` o guardar-se com a estat explícit. |
| `internal_state` | Estat intern mínim | Espai reduït per informació fisiològica o memòria molt simple, si es decideix incloure. |
| `genome` | Genoma neuronal | Conté pesos, biaixos i possibles paràmetres heretables. |
| `alive` | Estat vital | Indica si l'agent participa en la simulació activa. |
| `death_cause` | Causa de mort | Necessària per analitzar pressions selectives. |

## 5. Identitat i genealogia

Cada agent ha de tenir una identitat estable i no reutilitzable. Un identificador no s'ha de reciclar després de la mort, perquè això trencaria la traçabilitat genealògica.

Els agents inicials de la simulació no tenen progenitor real. En aquest cas, `parent_id` pot quedar buit o rebre un valor reservat. A partir de la primera reproducció, tot descendent ha de guardar l'identificador del seu progenitor.

- `agent_id`: identitat individual única.
- `parent_id`: vincle directe amb el progenitor.
- `lineage_id`: branca genealògica o fundador principal.
- `generation`: profunditat reproductiva respecte al fundador.
- `offspring_count`: mètrica derivada, no necessàriament estat intern de l'agent.

**Decisió recomanada:** la genealogia no s'ha de deixar per més endavant. És barata d'incloure i molt cara de reconstruir si no s'ha registrat bé des del principi.

## 6. Posició i relació amb el món

La posició indica on es troba l'agent dins el món 2D. En la primera versió, aquesta posició ha d'estar vinculada a una cel·la o coordenada discreta del món.

La posició no és només una dada geomètrica: determina quins recursos pot consumir, quines observacions rep i amb quins altres agents competeix localment.

| Aspecte | Regla funcional recomanada | Motiu |
| --- | --- | --- |
| Sistema espacial | Coordenades discretes `x`, `y`. | Facilita depuració, visualització i càlcul local. |
| Ocupació | Una cel·la pot tenir cap, un o diversos agents segons la regla d'ocupació decidida. | Permet modular competència espacial. |
| Moviment | La xarxa neuronal genera una intenció de moviment; el món valida si és possible. | Evita que l'agent controli directament les regles físiques. |
| Límits | El món resol què passa als marges: bloqueig, rebot o món toroidal. | La responsabilitat és de l'entorn, no de l'agent. |

## 7. Energia

L'energia és la variable fisiològica principal. Sense energia no hi ha supervivència, moviment ni reproducció. Per tant, el model d'agent ha d'exposar clarament la seva reserva energètica.

L'energia no ha de ser un simple marcador visual. Ha de tenir conseqüències directes i inevitables.

- Cada pas temporal ha de tenir un cost basal.
- El moviment ha de tenir un cost addicional.
- La reproducció ha de tenir un cost substancial.
- L'alimentació incrementa energia fins a un límit màxim o segons una regla definida.
- Quan l'energia cau a zero o sota el llindar definit, l'agent mor.

**Advertiment funcional:** si el cost energètic és massa baix o els recursos són massa abundants, l'agent no estarà sotmès a pressió selectiva real. El model d'agent depèn directament del model energètic.

## 8. Edat i cicle vital

L'edat permet introduir senescència funcional i evita que individus afortunats bloquegin indefinidament l'espai poblacional. En el prototip inicial, l'edat pot ser simple: nombre de passos des del naixement.

No cal modelar desenvolupament, creixement ni etapes vitals complexes. Però sí és recomanable tenir una edat màxima o una probabilitat de mort associada a l'edat per mantenir renovació evolutiva.

| Variable | Ús funcional | Recomanació |
| --- | --- | --- |
| `birth_step` | Marca el pas en què neix l'agent. | Guardar-lo sempre. |
| `age` | Mesura durada de vida actual. | Pot derivar-se del pas actual menys `birth_step`. |
| `max_age` | Límit superior de vida. | Configurable. Recomanable per evitar immortalitat. |
| `death_step` | Marca el pas en què mor. | Guardar-lo per tancar trajectòria vital. |

## 9. Estat intern

L'estat intern ha de tractar-se amb prudència. Pot ser útil per permetre formes mínimes de persistència temporal, però també pot complicar el sistema abans d'hora.

Per a la primera versió, la decisió més robusta és començar sense memòria complexa. Si es vol incloure estat intern, hauria de ser un vector petit, actualitzat de manera simple i limitat a informació fisiològica o recurrent mínima.

| Opció | Valor funcional | Risc |
| --- | --- | --- |
| Sense estat intern explícit | Agent molt simple i fàcil de depurar. | Pot limitar conductes temporals. |
| Vector intern petit | Permet una memòria mínima o estat recurrent. | Pot dificultar interpretar l'origen de la conducta. |
| Memòria complexa | Pot generar conductes més riques. | No recomanable en el prototip inicial. |

**Decisió recomanada:** començar sense memòria complexa. Si més endavant cal, afegir un estat intern petit i explícitament mesurat.

## 10. Genoma de l'agent

El genoma és la part heretable de l'agent. En el prototip inicial ha de contenir, com a mínim, els pesos i biaixos de la xarxa neuronal que transforma observacions en accions.

El genoma no s'ha de confondre amb el rendiment de l'agent. Un genoma pot funcionar bé en un entorn i malament en un altre. La selecció ha de sorgir de la interacció amb el món.

- Pesos neuronals.
- Biaixos neuronals.
- Paràmetres heretables simples, només si són necessaris i mesurables.
- Identificador o hash del genoma per comparar similituds.
- Metadades de mutació aplicades en el naixement.

Els paràmetres heretables addicionals poden ser temptadors, però cal limitar-los. Afegir metabolisme, llindar reproductiu, energia inicial o radi sensorial com a trets evolutius pot ser interessant; fer-ho tot alhora pot amagar què està seleccionant realment el sistema.

## 11. Xarxa neuronal associada

La xarxa neuronal no ha de ser un objecte independent del procés evolutiu. Funcionalment és l'expressió del genoma de l'agent. Rep observacions locals i genera una acció o una distribució d'intencions d'acció.

| Entrada | Processament | Sortida |
| --- | --- | --- |
| Observacions locals: energia pròpia, recursos propers, densitat local, edat i senyals ambientals simples. | Xarxa neuronal petita definida pel genoma heretable. | Intenció d'acció: moure's, alimentar-se, reproduir-se o quedar-se quiet. |

L'agent no hauria de decidir directament si una acció és legal. Ha de proposar una acció; la simulació i l'entorn resolen si es pot executar, quin cost té i quin resultat produeix.

## 12. Interfície funcional agent-entorn

Per mantenir l'arquitectura neta, l'agent no ha de contenir el món. L'entorn proporciona observacions i resol les conseqüències de les accions.

| Flux | Responsable | Descripció |
| --- | --- | --- |
| Observació | Entorn / simulació | Construeix el vector d'observacions locals per a cada agent viu. |
| Decisió | Agent / xarxa neuronal | Transforma observacions en una intenció d'acció. |
| Validació | Entorn / simulació | Determina si l'acció és possible segons límits, recursos i ocupació. |
| Cost energètic | Simulació | Aplica cost basal, cost d'acció i guanys per alimentació. |
| Naixement o mort | Reproducció / simulació | Crea descendents o marca morts segons condicions. |
| Registre | Mètriques / persistència | Guarda esdeveniments i estat agregat. |

## 13. Cicle vital funcional

Cada agent ha de passar per un cicle vital clar. Aquest cicle no és necessàriament una classe o un mètode concret, sinó una seqüència funcional que el sistema ha de respectar.

1. **Naixement:** es crea un agent amb identificador únic, energia inicial, posició i genoma.
2. **Activació:** l'agent entra al conjunt d'agents vius.
3. **Observació:** rep informació local del món i del seu estat fisiològic.
4. **Decisió:** la xarxa neuronal produeix una intenció d'acció.
5. **Resolució:** el món i la simulació apliquen regles físiques, energètiques i ecològiques.
6. **Actualització:** canvien energia, posició, edat i estat vital.
7. **Reproducció:** si compleix condicions, pot crear descendència amb mutació.
8. **Mort:** si incompleix condicions vitals, es marca com a mort i es registra la causa.
9. **Persistència:** es conserven les dades necessàries per mètriques i genealogia.

## 14. Naixement

El naixement pot produir-se per inicialització de població o per reproducció asexual. En tots dos casos, l'agent resultant ha de tenir estat complet des del primer pas en què pot actuar.

| Origen | Condicions | Dades assignades |
| --- | --- | --- |
| Població inicial | Creació al començament de l'experiment. | `agent_id`, `parent_id` nul, `lineage_id` propi, genoma inicial aleatori o configurat, energia inicial i posició. |
| Reproducció | Un progenitor viu compleix condicions energètiques i espacials. | `agent_id` nou, `parent_id` del progenitor, `lineage_id` heretat, `generation + 1`, genoma copiat i mutat. |

## 15. Reproducció

La reproducció és l'esdeveniment central del model evolutiu. Un agent només s'hauria de reproduir si té prou energia i si el món pot allotjar el descendent segons la regla espacial definida.

- El progenitor ha d'estar viu.
- L'energia del progenitor ha de superar un llindar reproductiu.
- La reproducció ha de tenir un cost energètic real.
- El descendent ha de rebre energia inicial.
- El genoma del descendent ha de ser una còpia mutada del genoma del progenitor.
- La relació `parent_id` ha de quedar registrada sempre.

No és recomanable permetre reproducció sense cost o amb cost massa baix. Això generaria explosions poblacionals artificials i dificultaria interpretar l'evolució.

## 16. Mort

La mort ha de ser un estat definitiu dins una execució. Un agent mort no observa, no actua, no consumeix recursos i no es reprodueix. Però el seu registre no s'ha d'eliminar, perquè és necessari per analitzar supervivència, causes de mort i genealogia.

| Causa de mort | Condició funcional | Valor analític |
| --- | --- | --- |
| Energia | L'energia cau a zero o sota un llindar definit. | Mesura pressió per eficiència energètica i accés a recursos. |
| Edat | L'agent supera l'edat màxima. | Evita immortalitat i manté renovació generacional. |
| Ambient | Condició ambiental local incompatible amb supervivència. | Permet estudiar adaptació a canvis ambientals. |
| Competència espacial | El sistema resol eliminació per saturació o conflicte local, si s'inclou. | Mesura pressió per ocupació i competència. |

## 17. Representació poblacional

Encara que funcionalment parlem d'agents individuals, tècnicament convé pensar en poblacions. El prototip haurà de simular molts agents, i això fa perillós dissenyar l'agent com una entitat pesada amb molta lògica interna.

La decisió funcional recomanada és considerar l'agent com un conjunt de dades compacte i vectoritzable. La lògica principal viu al sistema de simulació, no dins de cada agent individual.

| Enfocament | Avantatge | Risc |
| --- | --- | --- |
| Agent com a objecte ric | Natural d'entendre en codi orientat a objectes. | Pot escalar malament i barrejar responsabilitats. |
| Agent com a registre compacte | Més fàcil de vectoritzar i analitzar. | Requereix disciplina arquitectònica. |
| Població com a matrius d'estat | Adequat per JAX i simulacions grans. | Menys intuïtiu al principi, però més robust. |

**Decisió recomanada:** modelar funcionalment agents individuals, però preparar la implementació perquè l'estat es pugui representar com arrays poblacionals. Això és més coherent amb JAX i amb l'objectiu de simular poblacions grans.

## 18. Què no ha de contenir l'agent encara

Per evitar sobrecarregar el prototip, l'agent inicial no ha d'incloure funcionalitats que confonguin evolució amb intel·ligència programada o aprenentatge individual.

- Objectius explícits com “buscar menjar” o “evitar morir” codificats manualment.
- Aprenentatge durant la vida de l'individu.
- Memòria complexa o planificació a llarg termini.
- Comunicació simbòlica entre agents.
- Rols socials, jerarquies o cooperació programada.
- Sexe, gènere o reproducció sexual.
- Morfologia complexa o creixement corporal.
- Personalitat o emocions simulades.
- Coneixement global del mapa.

## 19. Mètriques derivades de l'agent

El model d'agent ha de permetre extreure mètriques sense haver d'inferir-les posteriorment de manera fràgil. Les dades individuals han d'alimentar mètriques poblacionals i genealògiques.

| Mètrica | Dades necessàries | Ús |
| --- | --- | --- |
| Supervivència individual | `birth_step`, `death_step`, `death_cause`. | Comparar durada de vida entre agents i llinatges. |
| Èxit reproductiu | `parent_id` i recompte de descendents. | Detectar llinatges més exitosos. |
| Energia mitjana vital | `energy` al llarg del temps. | Mesurar eficiència energètica. |
| Edat reproductiva | `birth_step` i pas de reproducció. | Saber si l'evolució afavoreix reproducció precoç o tardana. |
| Diversitat genètica | `genome` o hash de genoma. | Mesurar variació i convergència. |
| Causes de mort | `death_cause`. | Identificar pressions selectives dominants. |

## 20. Criteris funcionals de tancament

La definició del model d'agent es considerarà tancada quan es compleixin els criteris següents:

1. Hi ha una llista clara dels camps funcionals obligatoris de cada agent.
2. Cada agent pot ser identificat de manera única i persistent.
3. Cada descendent queda vinculat al seu progenitor.
4. L'agent té energia, edat, posició i estat vital definits.
5. El genoma de l'agent és heretable i mutable.
6. La xarxa neuronal associada pot generar accions a partir d'observacions locals.
7. La mort queda registrada amb causa i moment.
8. El model permet calcular mètriques de supervivència, reproducció i genealogia.
9. No hi ha conductes intel·ligents hardcodejades dins l'agent.
10. El model és prou compacte per escalar a poblacions grans.

## 21. Riscos principals

| Risc | Conseqüència | Mesura preventiva |
| --- | --- | --- |
| Agent massa complex | Simulació lenta i difícil d'interpretar. | Limitar estat i funcionalitats inicials. |
| Conducta hardcodejada | L'evolució queda falsejada. | Fer que les accions surtin de la xarxa neuronal. |
| Genealogia insuficient | No es poden estudiar llinatges. | Registrar `parent_id`, `lineage_id` i `generation` des del principi. |
| Energia mal calibrada | No hi ha selecció o hi ha extinció immediata. | Fer proves amb paràmetres simples i mètriques clares. |
| Massa paràmetres heretables | No se sap què està evolucionant. | Començar només amb genoma neuronal i afegir trets gradualment. |

## 22. Decisió d'abast recomanada

Per al prototip inicial, l'agent hauria de quedar definit de manera minimalista però completa:

- Identificador únic i no reutilitzable.
- Progenitor, llinatge i generació.
- Posició discreta al món 2D.
- Energia i edat.
- Estat vital i causa de mort.
- Genoma neuronal amb pesos i biaixos.
- Sense aprenentatge individual.
- Sense memòria complexa inicial.
- Accions produïdes per xarxa neuronal a partir d'observacions locals.
- Registre complet de naixement, reproducció i mort.

## 23. Resultat esperat

El resultat d'aquest punt ha de ser una definició funcional d'agent prou clara perquè l'equip pugui explicar què representa cada individu, quines dades conserva, com participa en la simulació i com pot ser analitzat després.

Un agent correcte per a aquesta fase no és aquell que sembla intel·ligent, sinó aquell que pot formar part d'una població evolutiva mesurable. Ha de poder viure, actuar, reproduir-se, morir i deixar dades suficients perquè el sistema permeti comparar llinatges.

## 24. Valoració d'adequació amb la documentació existent

El model d'agent és coherent amb la resta de la documentació funcional del projecte per aquests motius:

- Manté el principi d'agents simples, sense objectius hardcodejats ni aprenentatge individual, ja fixat al document funcional inicial i a l'arquitectura general.
- Encaixa amb el model del món 2D perquè assumeix posicions discretes, observacions locals, validació ambiental del moviment i pressió selectiva basada en recursos i energia.
- Reforça la separació de responsabilitats: l'agent conserva estat i genoma, mentre que l'entorn, la simulació, la reproducció, la mort, les mètriques i la persistència resolen les conseqüències del cicle vital.
- Afegeix concreció al punt 4 del full de ruta sense avançar-se a decisions d'implementació, classes, esquemes físics o APIs.
- Prioritza genealogia, identificadors persistents i causes de mort des del principi, cosa compatible amb l'objectiu del projecte de mesurar supervivència, reproducció i llinatges.

**Valoració global:** el document és adequat i complementari. No introdueix contradiccions funcionals rellevants amb els documents existents. L'únic punt que caldrà concretar en documents posteriors és quins camps seran estat explícit i quins seran derivats o persistits com a mètriques, especialment `age`, `offspring_count`, hash de genoma i possibles metadades de mutació.

## Conclusió funcional

El model d'agent queda tancat quan l'individu és simple, complet, traçable i evolutivament operatiu. Qualsevol complexitat addicional hauria d'entrar només després de demostrar que aquest agent mínim genera diferències mesurables entre llinatges.
