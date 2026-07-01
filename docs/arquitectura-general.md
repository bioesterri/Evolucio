# Projecte Evolució — Arquitectura funcional general

**Arquitectura funcional per separar responsabilitats, estabilitzar el nucli de simulació i preparar el projecte per a proves, persistència, anàlisi i ampliacions futures.**

## Metadades

| Camp | Valor |
| --- | --- |
| Tipus de document | Especificació funcional d'arquitectura |
| Àmbit | Prototip inicial de vida artificial evolutiva |
| Nivell de detall | Blocs, responsabilitats i fluxos de dades; sense implementació detallada |
| Destinatari | Equip tècnic, direcció de producte i CTO |
| Versió | 1.0 |
| Data | 1 de juliol de 2026 |
| Estat | Especificació funcional d'arquitectura |

## 1. Objectiu del document

Aquest document defineix l'arquitectura general del Projecte Evolució a nivell funcional. L'objectiu és establir on viu cada responsabilitat del sistema, quines dades circulen entre blocs i quins límits cal respectar per evitar acoblaments prematurs.

No és encara un document d'implementació. No fixa classes definitives, signatures de funcions, esquemes SQL ni detalls interns de JAX o Equinox. La seva funció és donar una estructura clara perquè el prototip inicial sigui modificable, verificable i ampliable.

## 2. Principi arquitectònic rector

La decisió principal és separar el projecte en un nucli de simulació pur i unes capes externes d'orquestració, persistència, anàlisi i visualització.

El nucli ha de poder executar una simulació completa sense dependre d'una base de dades, d'una interfície gràfica, d'una API web ni d'eines d'anàlisi. Aquest criteri és clau per facilitar proves, reproduïbilitat i substitució de components.

- El nucli de simulació conté la lògica evolutiva i ecològica mínima.
- La persistència desa configuracions, resultats, genealogies i artefactes, però no decideix res de la simulació.
- L'anàlisi consumeix dades produïdes per la simulació; no altera l'execució.
- La visualització representa estats i mètriques; no ha de contenir regles del món.

## 3. Decisions d'arquitectura inicial

| Decisió | Descripció funcional | Motiu |
| --- | --- | --- |
| Nucli separat de les capes externes | La simulació ha de poder córrer sense PostgreSQL, MLflow, FastAPI ni visualització. | Redueix acoblament i facilita proves unitàries i experiments repetibles. |
| Flux de dades explícit | Configuració, estat, accions, esdeveniments i mètriques han de circular en estructures identificables. | Permet auditar què passa en cada pas temporal. |
| Agents com a dades, sistemes com a lògica | L'agent guarda estat i genoma; la lògica d'actualització viu en blocs funcionals. | Evita objectes massa carregats i facilita vectorització amb JAX. |
| Evolució poblacional, no aprenentatge individual | Els pesos neuronals canvien per herència i mutació, no per entrenament dins la vida de l'agent. | Manté net el mecanisme evolutiu del prototip. |
| Persistència desacoblada | El registre rep dades ja generades pel nucli, però no condiciona les regles de simulació. | Permet començar simple i créixer cap a PostgreSQL, Parquet/Zarr i MLflow. |

## 4. Blocs funcionals principals

L'arquitectura general es divideix en deu blocs. Cada bloc té una responsabilitat clara i unes dades d'entrada i sortida definides.

| Bloc | Responsabilitat | Entrada principal | Sortida principal |
| --- | --- | --- | --- |
| Simulació | Coordina el pas temporal, aplica l'ordre d'actualització i connecta entorn, agents, accions, reproducció, mort i mètriques. | Configuració inicial, estat global, llavor aleatòria. | Nou estat global, esdeveniments, mètriques per pas. |
| Entorn | Manté el món 2D, els recursos, les condicions locals i la regeneració ambiental. | Estat ambiental anterior, accions que afecten recursos, paràmetres ambientals. | Nou estat ambiental, observacions locals, recursos consumibles. |
| Agent | Representa l'estat individual: identitat, posició, energia, edat, estat vital, progenitor i genoma associat. | Estat anterior de l'agent, acció escollida, canvis d'energia, reproducció o mort. | Estat actualitzat de l'agent. |
| Genoma | Codifica els pesos, biaixos i paràmetres heretables que defineixen el comportament potencial de l'agent. | Genoma parental, paràmetres de mutació, clau aleatòria. | Genoma fill, distància genètica o resum de mutació. |
| Xarxa neuronal | Transforma observacions locals en preferències d'acció. No conté objectius manuals ni estratègies hardcodejades. | Observacions, genoma/pesos, estat intern simple si existeix. | Acció o distribució de puntuacions d'acció. |
| Reproducció | Decideix si una reproducció és possible i crea descendència amb cost energètic i mutació. | Agents vius, energia, llindars reproductius, espai disponible, genoma parental. | Nous agents, vincles genealògics, esdeveniments de naixement. |
| Mort i selecció ambiental | Determina quins agents deixen d'estar vius per energia, edat o condicions ambientals. | Estat dels agents, energia, edat, entorn local. | Agents marcats com morts, causa de mort, esdeveniments. |
| Mètriques | Calcula indicadors poblacionals, energètics, reproductius, genètics i d'estabilitat. | Estat global, esdeveniments, genealogia, recursos. | Sèries temporals, resums per execució, dades per anàlisi. |
| Persistència | Desa configuració, llavors, resultats, genealogia, genomes destacats i artefactes massius. | Configuració, mètriques, esdeveniments, snapshots seleccionats. | Registres consultables i fitxers d'anàlisi. |
| Anàlisi i visualització | Permet interpretar resultats i representar poblacions, trajectòries, genealogies i mètriques. | Dades persistides o sortides directes del simulador. | Gràfics, taules, comparatives i informes. |

## 5. Flux funcional de dades

El flux general ha de ser simple i traçable. Cada execució comença amb una configuració i una llavor aleatòria, genera un estat inicial, executa passos temporals i emet mètriques i esdeveniments.

- Configuració de l'experiment: defineix món, agents inicials, recursos, energia, mutació i durada.
- Inicialització: crea estat ambiental, població inicial, genomes i identificadors.
- Observació: cada agent rep informació local del món i del seu propi estat.
- Decisió neuronal: la xarxa transforma observacions en accions candidates.
- Aplicació d'accions: el simulador resol moviment, alimentació, costos energètics i conflictes simples.
- Reproducció i mutació: els agents que compleixen condicions generen descendents mutats.
- Mort: el sistema elimina o marca agents que no compleixen condicions de viabilitat.
- Actualització ambiental: els recursos es consumeixen, es regeneren i poden variar ambientalment.
- Mètriques i esdeveniments: es registren naixements, morts, energia, població, recursos i genealogia.
- Persistència i anàlisi: es desen resultats i es preparen per comparació posterior.

## 6. Responsabilitats per bloc

### 6.1. Bloc de simulació

És l'orquestrador del sistema. No ha de contenir detalls de base de dades ni visualització. La seva responsabilitat és aplicar l'ordre correcte de cada pas temporal i garantir que el mateix experiment amb la mateixa configuració i llavor produeixi el mateix resultat.

- Manté l'estat global de la simulació.
- Controla el pas temporal.
- Coordina observació, decisió, acció, reproducció, mort i mètriques.
- Gestiona la llavor aleatòria i la divisió de claus pseudoaleatòries.
- Emet sortides estructurades per persistència i anàlisi.

No ha de decidir estratègies dels agents ni interpretar resultats científics. Aquestes responsabilitats pertanyen respectivament a la xarxa/genoma i al bloc d'anàlisi.

### 6.2. Bloc d'entorn

Defineix el món on existeixen els agents. En el prototip inicial es recomana un món 2D discret amb recursos limitats, perquè és més fàcil de depurar i suficient per generar pressió selectiva.

- Guarda dimensions i estat del món.
- Gestiona recursos disponibles i regeneració.
- Proporciona observacions locals als agents.
- Aplica canvis ambientals simples quan estiguin configurats.
- Informa de restriccions espacials, com vores, ocupació o saturació.

No ha de conèixer la genealogia ni els criteris d'èxit evolutiu. Només ofereix condicions i restriccions.

### 6.3. Bloc d'agent

L'agent és la unitat individual de la població. Ha de ser una estructura de dades clara, no un objecte amb lògica amagada. Aquesta decisió és important per mantenir el sistema vectoritzable i auditable.

- Identificador únic.
- Identificador del progenitor.
- Posició.
- Energia.
- Edat.
- Estat vital.
- Genoma associat.
- Paràmetres heretables simples, si s'inclouen.

El comportament de l'agent no s'ha d'escriure manualment com a regles intel·ligents. Ha d'emergir de la xarxa neuronal i de la selecció ambiental.

### 6.4. Bloc de genoma

El genoma és la part heretable. En la primera versió ha d'incloure sobretot els pesos i biaixos de la xarxa neuronal, i només alguns paràmetres heretables addicionals si són imprescindibles.

- Codifica pesos i biaixos neuronals.
- Permet còpia parental.
- Permet mutació configurable.
- Permet calcular distància genètica aproximada.
- Permet vincular un descendent amb el seu progenitor.

Cal evitar genomes massa complexos al principi. Si hi ha massa paràmetres heretables, serà difícil interpretar què està seleccionant l'entorn.

### 6.5. Bloc de xarxa neuronal

La xarxa neuronal converteix observacions locals en accions. Ha de ser petita, estable i computacionalment barata. No ha d'aprendre durant la vida de l'agent.

- Rep observacions locals.
- Utilitza el genoma com a conjunt de paràmetres.
- Produeix puntuacions o probabilitats d'acció.
- No conté objectius explícits com “buscar menjar” o “fugir”.
- Ha de ser substituïble en versions futures sense canviar tot el sistema.

Equinox pot ser una bona eina per representar aquesta xarxa dins l'ecosistema JAX, però aquesta decisió pertany al document tècnic d'implementació.

### 6.6. Bloc de reproducció

La reproducció defineix com apareixen nous individus. En aquesta fase ha de ser asexual, amb cost energètic i mutació del genoma parental.

- Comprova condicions mínimes de reproducció.
- Aplica cost energètic al progenitor.
- Assigna energia inicial al descendent.
- Copia i muta el genoma.
- Assigna identificador nou al descendent.
- Registra el vincle progenitor-descendent.

No ha de seleccionar manualment els millors agents. La selecció ha de sorgir de qui sobreviu i es reprodueix sota les condicions de l'entorn.

### 6.7. Bloc de mort i selecció ambiental

Aquest bloc determina quan un agent deixa d'estar viu. És un component central de la selecció, perquè transforma les condicions ambientals i energètiques en diferències de supervivència.

- Mort per energia insuficient.
- Mort per edat màxima, si es configura.
- Mort per condicions ambientals adverses, si existeixen.
- Registre de causa de mort.
- Actualització de l'estat vital de la població.

La causa de mort s'ha de registrar sempre que sigui possible. Sense aquest registre, una població que col·lapsa seria difícil d'interpretar.

### 6.8. Bloc de mètriques

Les mètriques converteixen la simulació en un experiment analitzable. Han d'estar separades de la lògica de simulació per evitar que els indicadors modifiquin el comportament del sistema.

- Població total i població viva.
- Naixements i morts per pas.
- Causes de mort.
- Energia mitjana i distribució energètica.
- Edat mitjana i supervivència.
- Consum i regeneració de recursos.
- Descendència per llinatge.
- Diversitat genètica aproximada.
- Estabilitat o col·lapse poblacional.

Les mètriques no han de ser decoratives. Han de permetre decidir si hi ha evolució diferencial o només soroll.

### 6.9. Bloc de persistència

La persistència desa allò necessari per repetir, auditar i comparar experiments. No ha de ser obligatòria per executar el nucli, però sí per considerar el projecte científicament útil.

- Configuracions completes.
- Llavors aleatòries.
- Mètriques agregades.
- Esdeveniments principals: naixements, morts i mutacions.
- Genealogies.
- Genomes destacats.
- Snapshots o dades massives quan calgui.

La recomanació funcional és separar dades relacionals i dades massives: PostgreSQL per experiments, configuracions i genealogies; Parquet o Zarr per sèries grans i snapshots; MLflow per seguiment d'execucions i artefactes.

### 6.10. Bloc d'anàlisi i visualització

Aquest bloc serveix per entendre què ha passat. Ha de consumir dades del simulador o de la persistència, però no ha de condicionar la simulació.

- Comparar execucions.
- Analitzar estabilitat poblacional.
- Detectar llinatges exitosos.
- Mesurar diversitat genètica.
- Representar evolució temporal de mètriques.
- Visualitzar distribució espacial d'agents i recursos.

La visualització inicial pot ser simple. Una interfície interactiva avançada no és necessària per validar l'arquitectura.

## 7. Límits entre blocs

Els límits són tan importants com els blocs. La taula següent fixa què no ha de fer cada component.

| Bloc | No ha de fer |
| --- | --- |
| Simulació | No ha de contenir consultes SQL, codi de visualització ni regles intel·ligents dels agents. |
| Entorn | No ha de decidir quins agents són evolutivament millors. |
| Agent | No ha d'amagar lògica complexa de decisió dins mètodes individuals. |
| Genoma | No ha de conèixer el món ni les mètriques globals. |
| Xarxa neuronal | No ha de contenir estratègies manuals hardcodejades. |
| Reproducció | No ha de fer selecció artificial dels millors genomes. |
| Mètriques | No ha de modificar l'estat de la simulació. |
| Persistència | No ha de ser necessària per executar proves bàsiques del nucli. |
| Visualització | No ha de contenir lògica de simulació. |

## 8. Estructura funcional recomanada del projecte

Sense entrar encara en implementació detallada, l'estructura funcional del projecte hauria de reflectir les responsabilitats següents.

| Àrea funcional | Contingut esperat |
| --- | --- |
| Configuració | Definició d'experiments, paràmetres del món, energia, mutació, població inicial i llavors. |
| Core/simulació | Bucle temporal, estat global, ordre d'actualització i execució del prototip. |
| Core/entorn | Món 2D, recursos, observacions locals i canvis ambientals. |
| Core/agents | Estructures d'estat individual i operacions bàsiques sobre població. |
| Core/genoma | Codificació heretable, còpia, mutació i distància genètica. |
| Core/xarxa | Model neuronal petit compatible amb execució vectoritzada. |
| Core/evolució | Reproducció, naixement, mort, genealogia i selecció ambiental emergent. |
| Mètriques | Càlcul de sèries temporals, agregats, diversitat i resultats per llinatge. |
| Persistència | Adaptadors per PostgreSQL, Parquet/Zarr i MLflow. |
| Anàlisi | Lectura de resultats, comparació d'experiments i informes. |
| Visualització | Gràfics i representacions espacials simples. |
| Proves | Validació de regles energètiques, reproducció, mutació, mètriques i repetibilitat. |

## 9. Dades principals del sistema

Perquè l'equip pugui parlar amb precisió, cal identificar els grans tipus de dades que circularan pel sistema.

| Dada | Descripció funcional | On s'origina | On es consumeix |
| --- | --- | --- | --- |
| Configuració | Paràmetres d'un experiment. | Usuari o fitxer de configuració. | Inicialització, simulació, persistència. |
| Estat ambiental | Mapa del món, recursos i condicions locals. | Entorn. | Observació, accions, mètriques. |
| Estat d'agent | Posició, energia, edat, vitalitat i identificadors. | Inicialització i simulació. | Xarxa, reproducció, mort, mètriques. |
| Genoma | Pesos, biaixos i paràmetres heretables. | Inicialització o reproducció. | Xarxa neuronal, mutació, genealogia. |
| Observació | Informació local rebuda per l'agent. | Entorn i estat de l'agent. | Xarxa neuronal. |
| Acció | Decisió generada per la xarxa. | Xarxa neuronal. | Simulació i entorn. |
| Esdeveniment | Naixement, mort, mutació o consum rellevant. | Simulació. | Mètriques, persistència, anàlisi. |
| Mètrica | Indicador agregat o temporal. | Bloc de mètriques. | Persistència, anàlisi, visualització. |
| Genealogia | Relació progenitor-descendent. | Reproducció. | Persistència, anàlisi evolutiva. |

## 10. Ordre recomanat d'execució dins un pas temporal

Cal fixar un ordre funcional per evitar ambigüitats. L'ordre concret es podrà ajustar tècnicament, però ha de ser estable i documentat.

- Actualitzar edat i costos basals dels agents vius.
- Generar observacions locals a partir de l'estat actual.
- Executar la xarxa neuronal per obtenir accions.
- Resoldre moviment i restriccions espacials.
- Resoldre alimentació i consum de recursos.
- Aplicar costos energètics d'accions.
- Resoldre intents de reproducció.
- Aplicar mutacions i crear descendents.
- Determinar morts i causes.
- Regenerar recursos o actualitzar condicions ambientals.
- Calcular mètriques i registrar esdeveniments.

Aquest ordre evita que la reproducció, la mort o el consum de recursos quedin indefinits. Qualsevol canvi futur en l'ordre haurà de quedar registrat perquè pot alterar els resultats evolutius.

## 11. Persistència i traçabilitat

La persistència s'ha de pensar des del principi, però no ha d'envair el nucli. El sistema hauria de poder executar-se en memòria per a proves petites i després enviar resultats a les capes de registre.

- PostgreSQL: experiments, configuracions, llavors, genealogies, resums i referències a artefactes.
- Parquet o Zarr: sèries temporals grans, snapshots, estats poblacionals i dades d'anàlisi massiva.
- MLflow: seguiment d'execucions, paràmetres, mètriques, artefactes i comparació d'experiments.

La traçabilitat mínima exigeix que qualsevol resultat important es pugui vincular a una configuració, una llavor, una versió del codi i un conjunt de mètriques.

## 12. Proves i mantenibilitat

L'arquitectura ha de facilitar proves petites i específiques. Si una regla bàsica només es pot validar executant tota la simulació, el disseny estarà massa acoblat.

- Proves de consum energètic.
- Proves de mort per energia.
- Proves de reproducció amb cost correcte.
- Proves de mutació dins rang esperat.
- Proves de genealogia progenitor-descendent.
- Proves de regeneració de recursos.
- Proves de repetibilitat amb la mateixa llavor.
- Proves de mètriques bàsiques.

Aquest criteri condiciona l'arquitectura: cada bloc ha de poder ser validat amb entrades petites i resultats esperats.

## 13. Exclusions arquitectòniques temporals

Per evitar sobreenginyeria, queden fora de l'arquitectura inicial com a dependències centrals:

- FastAPI com a part del nucli de simulació.
- Interfície gràfica avançada.
- Sistema distribuït o multi-node.
- Reproducció sexual.
- Aprenentatge per reforç.
- Memòria neuronal complexa.
- Comunicació entre agents.
- Múltiples espècies.
- Evosax com a motor principal del sistema ecològic.

Aquestes peces poden aparèixer més endavant, però no han de determinar l'arquitectura de la primera versió.

## 14. Riscos tècnics principals

| Risc | Impacte | Mesura preventiva |
| --- | --- | --- |
| Acoblament entre simulació i persistència | Dificulta proves i canvis de base de dades. | Fer que el nucli retorni dades estructurades i que la persistència sigui un adaptador. |
| Agents massa orientats a objectes | Complica vectorització i execució eficient amb JAX. | Tractar agents com a estructures de dades i moure la lògica a sistemes. |
| Mètriques integrades dins la lògica | Pot contaminar la simulació i fer-la difícil d'auditar. | Separar càlcul de mètriques del canvi d'estat. |
| Arquitectura massa genèrica | Retarda el prototip i introdueix abstraccions inútils. | Dissenyar per al prototip real, no per a un ecosistema futur imaginari. |
| Flux aleatori poc controlat | Impedeix repetir experiments. | Centralitzar configuració de llavors i registre de l'execució. |

## 15. Criteris de tancament

Aquest punt es considerarà tancat quan l'equip pugui explicar, sense entrar en implementació detallada:

- quins blocs formen el projecte;
- quina responsabilitat té cada bloc;
- quines dades entren i surten de cada bloc;
- quin és el flux general d'un pas temporal;
- què pertany al nucli de simulació i què és una capa externa;
- com es preserven repetibilitat i traçabilitat;
- quines parts queden excloses temporalment;
- quins riscos arquitectònics s'han d'evitar.

## 16. Decisió final recomanada

L'arquitectura inicial del Projecte Evolució ha de ser modular, però no excessivament abstracta. La millor opció és un nucli de simulació pur, vectoritzable i verificable, envoltat per adaptadors de persistència, anàlisi i visualització.

Aquesta arquitectura permet començar amb un prototip petit, però deixa el camí obert per ampliar recursos, entorns, xarxes, mètriques i eines d'anàlisi sense haver de reescriure el sistema sencer.

La decisió més important és no barrejar la lògica evolutiva amb la infraestructura. La simulació ha de ser el centre; la resta de peces han de servir per executar-la, registrar-la i entendre-la.

## 17. Adequació amb la resta de la documentació

Aquest document s'adequa bé a la documentació existent perquè desenvolupa la tasca d'alt nivell de dissenyar l'arquitectura general del projecte i manté el mateix criteri que el document d'abast: construir primer un prototip petit, complet, repetible i mesurable.

### Encaix amb el document funcional inicial

- **Continuïtat de visió:** conserva l'objectiu de vida artificial evolutiva basada en selecció ambiental emergent, no en conductes preprogramades.
- **Separació de responsabilitats:** concreta el principi ja indicat de separar món, agents, genoma, xarxa neuronal, reproducció, mètriques, persistència i visualització.
- **Repetibilitat i traçabilitat:** reforça la necessitat de registrar configuració, llavor, resultats i genealogia perquè els experiments siguin comparables.
- **Nivell adequat:** es manté en el pla funcional i evita tancar encara detalls d'implementació com signatures, esquemes SQL o decisions internes de JAX/Equinox.

### Encaix amb l'abast del prototip inicial

- **Coherència d'abast:** respecta les exclusions temporals ja definides: reproducció sexual, aprenentatge individual, memòria complexa, comunicació entre agents, múltiples espècies i interfície avançada.
- **Pressió selectiva:** manté el món 2D discret, els recursos limitats, l'energia, la mort i la reproducció asexual com a base del prototip.
- **Mètriques:** amplia el criteri de mesura sense convertir les mètriques en part de la lògica de simulació.
- **Persistència:** introdueix una visió escalable de registre experimental sense fer-la obligatòria per executar el nucli.

### Valoració

La valoració és **positiva**. El document és adequat per incorporar-se com a peça d'arquitectura funcional perquè dona límits clars entre blocs, redueix risc d'acoblament prematur i prepara el projecte per a proves, persistència i anàlisi sense sobreenginyeria.

L'únic punt que caldrà vigilar en fases posteriors és convertir aquesta arquitectura en una especificació tècnica concreta sense trencar-ne el principi rector: el nucli de simulació ha de continuar sent pur, verificable i independent de les capes externes.
