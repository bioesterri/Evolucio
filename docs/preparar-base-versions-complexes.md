# Projecte Evolució — Preparar la base per a versions més complexes

**Especificació funcional per deixar el sistema llest per incorporar complexitat posterior sense reescriure el nucli del prototip.**

## Metadades

| Camp | Valor |
| --- | --- |
| Nom del document | Preparar la base per a versions més complexes |
| Tipus de document | Especificació funcional de tancament i extensibilitat |
| Punt del full de ruta | Punt 20 del pla funcional |
| Àmbit | Arquitectura extensible, contractes de dades, versionat, priorització de millores i criteris d'acceptació futura |
| Fase | Tancament de la primera versió funcional |
| Propòsit | Deixar el prototip preparat perquè la complexitat futura es pugui incorporar sense reescriure el nucli de simulació. |
| Resultat esperat | Una plataforma extensible que no obligui a reescriure el prototip quan el projecte avanci. |
| Criteri de tancament | La primera versió queda tancada amb arquitectura clara, resultats mesurables i una llista prioritzada de millores futures. |
| Estat | Versió funcional inicial |
| Versió | 1.0 |
| Data | Juliol de 2026 |

## Índex

1. [Objectiu del document](#1-objectiu-del-document)
2. [Principi rector](#2-principi-rector)
3. [Abast d'aquesta preparació](#3-abast-daquesta-preparació)
4. [Condicions mínimes abans d'ampliar](#4-condicions-mínimes-abans-dampliar)
5. [Blocs que han de quedar extensibles](#5-blocs-que-han-de-quedar-extensibles)
6. [Contractes funcionals de dades](#6-contractes-funcionals-de-dades)
7. [Versionat del model](#7-versionat-del-model)
8. [Millores futures prioritzades](#8-millores-futures-prioritzades)
9. [Millores que convé posposar](#9-millores-que-convé-posposar)
10. [Estratègia tècnica d'extensibilitat](#10-estratègia-tècnica-dextensibilitat)
11. [Full de ruta recomanat](#11-full-de-ruta-recomanat)
12. [Criteris per acceptar una ampliació futura](#12-criteris-per-acceptar-una-ampliació-futura)
13. [Riscos principals](#13-riscos-principals)
14. [Resultat esperat](#14-resultat-esperat)
15. [Definició de tancament](#15-definició-de-tancament)
16. [Decisió recomanada](#16-decisió-recomanada)

## 1. Objectiu del document

Aquest document defineix com s'ha de tancar la primera versió del Projecte Evolució perquè serveixi de base sòlida per a versions més complexes. L'objectiu no és afegir complexitat immediatament, sinó deixar el prototip preparat perquè aquesta complexitat pugui incorporar-se sense reescriure el nucli de simulació.

La primera versió ha de quedar tancada amb una arquitectura clara, dades registrades, mètriques interpretables i punts d'extensió explícits. Només així les ampliacions futures podran millorar el model en lloc de desordenar-lo.

## 2. Principi rector

Una plataforma extensible no és una plataforma carregada d'opcions prematures. És una plataforma amb separacions netes, contractes de dades estables i decisions documentades. En aquesta fase, preparar el futur vol dir evitar dependències rígides, constants amagades, formats improvisats i acoblaments innecessaris entre simulació, persistència, anàlisi i visualització.

El criteri crític és aquest: qualsevol ampliació futura ha de poder substituir o ampliar un bloc concret sense alterar de manera imprevisible la resta del sistema.

## 3. Abast d'aquesta preparació

Aquest punt no consisteix a implementar ja totes les millores futures. Consisteix a deixar la primera versió tancada de manera ordenada i amb una ruta evolutiva del mateix projecte.

- **S'inclou:** definir punts d'extensió, estabilitzar formats de dades, documentar interfícies funcionals, prioritzar millores i identificar riscos d'escalabilitat.
- **No s'inclou:** afegir reproducció sexual, agents socials, memòria complexa, entorns multirecurs, visualització avançada o entrenament individual abans de validar el prototip base.
- **Finalitat:** garantir que la segona fase parteixi d'una base mesurable i no d'un conjunt d'experiments dispersos.

## 4. Condicions mínimes abans d'ampliar

No s'hauria d'incorporar complexitat nova fins que la primera versió compleixi uns requisits de maduresa. Si aquests requisits no es compleixen, qualsevol ampliació pot ocultar errors bàsics i fer més difícil interpretar els resultats.

| Condició | Què ha d'estar resolt | Risc si no es compleix |
| --- | --- | --- |
| Simulació estable | El bucle principal executa molts passos sense incoherències de població, energia, naixements o morts. | Els errors de motor es poden confondre amb dinàmiques evolutives. |
| Registre complet | Cada execució guarda configuració, llavor, versió, paràmetres, mètriques i artefactes rellevants. | No es poden repetir ni comparar experiments. |
| Mètriques útils | Les mètriques permeten distingir supervivència, descendència, diversitat, col·lapse i estabilitat. | La interpretació queda basada en impressions visuals. |
| Proves bàsiques | Els tests validen energia, recursos, mutació, reproducció, mort i coherència poblacional. | Una ampliació pot consolidar errors silenciosos. |
| Separació funcional | Entorn, agents, genoma, xarxa, reproducció, mètriques, persistència i visualització tenen responsabilitats separades. | Qualsevol canvi trenca parts no relacionades. |
| Resultats inicials | Hi ha diverses execucions comparables amb llavors i configuracions registrades. | No es coneix la zona funcional del model. |

## 5. Blocs que han de quedar extensibles

La primera versió ha de tancar-se amb blocs funcionals que puguin créixer. Això no vol dir fer abstraccions massa generals, sinó evitar que una decisió local impedeixi canvis previsibles.

| Bloc | Extensió futura prevista | Preparació necessària |
| --- | --- | --- |
| Entorn | Afegir diversos recursos, zones ambientals, obstacles, gradients, estacions o pertorbacions. | Separar l'estat del món de les regles d'actualització i mantenir paràmetres ambientals configurables. |
| Agent | Afegir estat intern, memòria simple, sensors especialitzats o diferents fenotips. | Mantenir un model d'estat clar i distingir dades de vida, dades heretables i dades derivades. |
| Observacions | Incorporar nous canals perceptius o canviar el radi de percepció. | Definir l'espai d'observació com una interfície estable entre entorn i xarxa neuronal. |
| Accions | Afegir interaccions socials, senyals, competència directa o construcció ambiental. | Tractar les accions com sortides explícites de la xarxa i resoldre-les en una fase separada. |
| Genoma | Afegir arquitectura neuronal variable, paràmetres heretables o mutacions estructurals. | Guardar una representació reconstruïble i versionada del genoma. |
| Xarxa neuronal | Passar de xarxes petites a xarxes recurrents, modulars o amb més capes. | No lligar la simulació a una sola implementació concreta de xarxa. |
| Mètriques | Afegir índexs de diversitat, anàlisi de llinatges, resposta adaptativa i robustesa. | Centralitzar el càlcul de mètriques i mantenir noms i unitats consistents. |
| Persistència | Guardar grans sèries temporals, snapshots, genomes destacats i genealogies completes. | Separar dades estructurades de dades massives i versionar esquemes. |
| Visualització | Afegir panells, animacions, exploració de genealogies i comparació d'experiments. | Evitar que la visualització governi la simulació o modifiqui l'estat. |

## 6. Contractes funcionals de dades

Perquè el sistema sigui ampliable, cal estabilitzar els tipus de dades que circulen entre blocs. No cal definir encara totes les classes definitives, però sí deixar clar quina informació travessa cada frontera funcional.

| Flux de dades | Contingut mínim | Criteri d'estabilitat |
| --- | --- | --- |
| Configuració -> simulació | Mida del món, població inicial, recursos, paràmetres energètics, mutació, llavor i durada. | Una execució ha de poder reconstruir-se a partir de la configuració registrada. |
| Entorn -> observacions | Recursos locals, densitat d'agents, senyals ambientals i límits perceptius. | Les observacions no poden contenir informació global no disponible per l'agent. |
| Agent + genoma -> xarxa | Pesos, biaixos i paràmetres heretables necessaris per decidir accions. | El genoma ha de ser suficient per reconstruir el sistema de decisió. |
| Xarxa -> accions | Acció escollida o vector d'intencions, amb intensitat o direcció si escau. | Les accions han de ser interpretables pel motor sense codificar estratègies. |
| Simulació -> mètriques | Població, energia, naixements, morts, causes, descendència, diversitat i recursos. | Les mètriques han de poder agregar-se per pas, execució i llinatge. |
| Simulació -> persistència | Snapshots, resum d'execució, genealogies, genomes destacats i artefactes. | Les dades crítiques han de sobreviure al final de la simulació. |
| Persistència -> anàlisi | Consultes sobre experiments, llinatges, mètriques i configuracions. | L'anàlisi no ha de dependre de memòria temporal ni de fitxers dispersos. |

## 7. Versionat del model

A mesura que el projecte avanci, els resultats només seran comparables si queda clar amb quina versió del model s'han produït. Cal versionar tant el codi com la configuració funcional i els esquemes de dades.

- **Versió del motor:** identifica la lògica de simulació, resolució d'accions, energia, reproducció i mort.
- **Versió de configuració:** identifica els camps disponibles, valors per defecte i rangs funcionals de cada paràmetre.
- **Versió de genoma:** identifica com es representen pesos, biaixos, arquitectura neuronal i paràmetres heretables.
- **Versió de mètriques:** identifica definicions, unitats i agregacions per evitar comparar magnituds incompatibles.
- **Versió d'esquema de dades:** identifica estructura de taules, arxius massius i relacions persistents.

Sense versionat, una millora futura pot invalidar silenciosament resultats anteriors. Això és especialment perillós en un projecte evolutiu, on petites diferències de regles poden canviar tota la dinàmica poblacional.

## 8. Millores futures prioritzades

Les ampliacions futures s'han de prioritzar segons valor científic, risc tècnic i dependències. No totes les idees interessants són bones candidates per a la segona fase.

| Prioritat | Millora | Motiu | Condició prèvia |
| --- | --- | --- | --- |
| Alta | Comparació sistemàtica entre configuracions | Permet entendre sensibilitat del model abans d'augmentar complexitat. | Registre i mètriques estables. |
| Alta | Canvis ambientals més rics però controlats | Prova adaptabilitat sense alterar l'agent de manera prematura. | Entorn base validat. |
| Alta | Anàlisi de genealogies i llinatges dominants | Ajuda a distingir adaptació acumulativa de deriva o coll d'ampolla. | Genealogia completa i genomes guardats. |
| Mitjana | Nous canals d'observació | Pot generar estratègies més fines sense codificar comportament. | Observacions actuals validades i normalitzades. |
| Mitjana | Múltiples tipus de recursos | Introdueix nínxols ecològics i possibles trade-offs. | Economia energètica simple estable. |
| Mitjana | Interaccions socials simples | Permet cooperació, competència directa o senyals, però augmenta molt la interpretació. | Accions bàsiques i mètriques robustes. |
| Mitjana | Xarxes neuronals més grans o recurrents | Pot permetre conducta temporal més rica. | Cost computacional i mètriques de comportament controlats. |
| Baixa inicial | Reproducció sexual | Pot aportar recombinació però complica genealogia, herència i interpretació. | Model asexual ben caracteritzat. |
| Baixa inicial | Aprenentatge individual | Barreja evolució i aprenentatge; pot ser interessant però confon el mecanisme principal. | Evolució poblacional demostrada sense aprenentatge. |
| Baixa inicial | Interfície web avançada | Pot ajudar a explorar resultats, però no millora el model si arriba massa aviat. | Persistència i anàlisi consolidades. |

## 9. Millores que convé posposar

Algunes ampliacions són atractives però perilloses si s'afegeixen massa aviat. Poden augmentar el cost computacional, fer opac el sistema o confondre el mecanisme evolutiu amb regles externes.

- **Reproducció sexual:** no s'hauria d'afegir fins que la genealogia asexual, la mutació i la selecció acumulativa estiguin ben interpretades.
- **Aprenentatge durant la vida:** pot ser una línia futura, però ara barrejaria adaptació individual i evolució poblacional.
- **Conductes socials complexes:** són difícils d'interpretar si encara no s'ha validat competència indirecta per recursos.
- **Xarxes massa grans:** augmenten cost i opacitat sense garantir més evolució útil.
- **Visualització avançada:** pot donar sensació de progrés sense millorar la validesa científica del model.
- **Funcions de fitness externes:** s'han d'evitar com a motor principal perquè substituirien la selecció ambiental emergent.

## 10. Estratègia tècnica d'extensibilitat

La base per a versions més complexes ha de recolzar-se en decisions tècniques prudents. La prioritat és mantenir el nucli de simulació net, vectoritzable i independent de cap interfície o sistema d'anàlisi concret.

| Decisió | Recomanació funcional | Risc que evita |
| --- | --- | --- |
| Separar nucli i persistència | El motor ha de poder executar-se sense base de dades i després escriure resultats mitjançant una capa separada. | Evita que problemes d'I/O condicionin la lògica evolutiva. |
| Configuració explícita | Tot paràmetre rellevant ha d'estar en configuració, no amagat en constants internes. | Evita experiments irrepetibles. |
| Formats d'exportació estables | Dades tabulars per mètriques i formats massius per sèries llargues. | Evita fitxers dispersos i incompatibles. |
| Interfícies funcionals | Observacions, accions, actualització d'entorn i mètriques han de tenir entrades i sortides clares. | Evita acoblaments difícils de trencar. |
| Tests abans d'extensions | Cada ampliació ha d'anar precedida de proves sobre el comportament base. | Evita atribuir errors nous a dinàmiques evolutives. |
| Compatibilitat amb execucions petites | Qualsevol versió nova ha de poder executar escenaris petits de diagnòstic. | Evita dependre només d'experiments llargs i cars. |

## 11. Full de ruta recomanat

Després de tancar el prototip inicial, la progressió recomanada és incremental. Cada fase ha de produir resultats comparables abans de passar a la següent.

1. Congelar una versió base del model amb configuració, codi, esquemes i mètriques identificables.
2. Executar un conjunt de referència amb diverses llavors i paràmetres coneguts.
3. Validar estabilitat poblacional, variació genètica i diferències entre llinatges.
4. Afegir una sola millora ambiental controlada i comparar-la contra la versió base.
5. Afegir anàlisi més rica de genealogia i persistència de genomes destacats.
6. Introduir nous canals d'observació només si les mètriques actuals justifiquen la necessitat.
7. Explorar múltiples recursos o nínxols només quan l'economia energètica simple sigui estable.
8. Posposar aprenentatge individual, reproducció sexual i interaccions socials complexes fins que la dinàmica evolutiva bàsica estigui ben demostrada.

## 12. Criteris per acceptar una ampliació futura

No tota ampliació ha de ser acceptada pel fet de ser possible. Cada nova funcionalitat hauria de superar un filtre funcional abans d'entrar al model principal.

| Pregunta | Resposta acceptable |
| --- | --- |
| Quin problema resol? | Ha de millorar interpretació, control experimental, capacitat ecològica o anàlisi de resultats. |
| Quin bloc modifica? | Ha de quedar clar si afecta entorn, agent, genoma, accions, mètriques, persistència o visualització. |
| Trenca comparabilitat? | Si canvia mètriques o regles bàsiques, cal versionar i no barrejar resultats. |
| És mesurable? | Ha de tenir mètriques associades o una hipòtesi funcional comprovable. |
| Afegeix estratègies hardcodejades? | Si codifica conducta intel·ligent directament, s'ha de rebutjar o reformular. |
| Augmenta molt el cost? | Si el cost puja massa, cal justificar-ho amb valor científic clar. |
| Pot provar-se en petit? | Ha de poder validar-se amb escenaris curts abans d'experiments llargs. |

## 13. Riscos principals

Preparar versions més complexes no elimina riscos; només els fa visibles. Cal deixar-los documentats per no confondre creixement del projecte amb millora real del model.

| Risc | Descripció | Mesura preventiva |
| --- | --- | --- |
| Complexitat prematura | Afegir massa mecanismes abans d'entendre el prototip base. | Exigir comparacions contra una versió base congelada. |
| Acoblament tècnic | Fer que visualització, base de dades o anàlisi condicionin el motor de simulació. | Mantenir capes separades i fluxos de dades explícits. |
| Pèrdua de traçabilitat | Canviar regles sense versionar configuracions i esquemes. | Registrar versió del model, llavor, configuració i mètriques. |
| Interpretació exagerada | Atribuir adaptació a patrons visuals o a poques execucions. | Treballar amb mètriques, replicació i anàlisi estadística mínima. |
| Agents massa sofisticats | Fer xarxes o observacions tan potents que el sistema sigui opac i car. | Incrementar capacitat neuronal només quan hi hagi una hipòtesi clara. |
| Desviació de l'objectiu | Convertir el projecte en optimització d'una puntuació en lloc d'ecologia artificial. | Evitar fitness extern com a motor principal. |

## 14. Resultat esperat

El resultat d'aquest punt ha de ser una primera versió tancada, no perfecta, però sòlida. Ha de ser possible executar-la, repetir experiments, consultar resultats, seguir genealogies i explicar quina part del sistema s'hauria de modificar per introduir noves capacitats.

Una plataforma preparada per créixer és aquella que permet dir: aquesta és la versió base, aquests són els seus resultats, aquests són els seus límits i aquesta és la propera millora prioritària. Si això no es pot dir amb claredat, encara no s'ha acabat la primera fase.

## 15. Definició de tancament

Aquest punt es considerarà complet quan la primera versió del Projecte Evolució quedi documentada i preparada per evolucionar sense perdre traçabilitat ni coherència funcional.

- L'arquitectura funcional està clara i separada per responsabilitats.
- El prototip base té resultats mesurables i experiments registrats.
- Les dades principals sobreviuen al final de la simulació i poden analitzar-se posteriorment.
- Els esquemes, configuracions, mètriques i genomes estan versionats o preparats per versionar-se.
- Hi ha una llista prioritzada de millores futures amb criteris d'acceptació.
- Les millores prematures o confuses queden explícitament posposades.
- L'equip pot explicar com ampliar el sistema sense reescriure el nucli del prototip.

## 16. Decisió recomanada

La decisió recomanada és tancar la primera versió com a línia base estable abans d'afegir funcionalitats noves. La següent fase hauria de començar amb experiments de comparació i refinament ambiental, no amb reproducció sexual, aprenentatge individual ni agents socials complexos.

Aquesta és una restricció deliberada. Si el projecte vol ser científicament interpretable, primer ha de demostrar evolució poblacional simple, mesurable i repetible. La complexitat ha de ser una extensió justificada, no una fugida endavant.
