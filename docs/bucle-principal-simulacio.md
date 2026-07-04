# Projecte Evolució — Bucle principal de simulació

**Especificació funcional del cicle temporal discret que coordina entorn, agents, observacions, decisions neuronals, accions, reproducció, mort, mètriques i registre.**

## Metadades

| Camp | Valor |
| --- | --- |
| Tipus de document | Especificació funcional del bucle principal de simulació |
| Punt del full de ruta | 4.11 — Implementar el bucle principal de simulació |
| Àrea | Simulació i coordinació del sistema |
| Versió | Especificació funcional inicial |
| Abast | Ordre de pas temporal, flux de dades, resolució d'efectes, naixements, morts i registre |
| Estat | Preparat per passar a disseny tècnic |
| Data | Juliol de 2026 |

## Propòsit del document

Aquest document defineix l'ordre funcional de cada pas temporal de la simulació, els criteris de consistència de dades i les condicions perquè el sistema pugui executar-se durant molts passos sense intervenció manual.

Defineix el comportament funcional del bucle, no la seva implementació concreta. Encara no fixa classes, signatures de funcions ni optimitzacions específiques amb JAX, però sí estableix quines responsabilitats ha de complir el cicle de simulació.

## Índex funcional

- [1. Objectiu del bucle principal](#1-objectiu-del-bucle-principal)
- [2. Funció dins l'arquitectura general](#2-funció-dins-larquitectura-general)
- [3. Principis de disseny del cicle de simulació](#3-principis-de-disseny-del-cicle-de-simulació)
- [4. Estat d'entrada i estat de sortida de cada pas](#4-estat-dentrada-i-estat-de-sortida-de-cada-pas)
- [5. Ordre funcional recomanat del pas temporal](#5-ordre-funcional-recomanat-del-pas-temporal)
- [6. Fases del bucle principal](#6-fases-del-bucle-principal)
- [7. Resolució de conflictes i simultaneïtat](#7-resolució-de-conflictes-i-simultaneïtat)
- [8. Naixements, morts i actualització poblacional](#8-naixements-morts-i-actualització-poblacional)
- [9. Registre, mètriques i traçabilitat](#9-registre-mètriques-i-traçabilitat)
- [10. Repetibilitat i control de llavors aleatòries](#10-repetibilitat-i-control-de-llavors-aleatòries)
- [11. Validacions internes i invariants](#11-validacions-internes-i-invariants)
- [12. Errors típics que cal evitar](#12-errors-típics-que-cal-evitar)
- [13. Exclusions temporals](#13-exclusions-temporals)
- [14. Criteris de tancament](#14-criteris-de-tancament)
- [15. Valoració d'adequació amb la resta de la documentació](#15-valoració-dadequació-amb-la-resta-de-la-documentació)

## 1. Objectiu del bucle principal

El bucle principal de simulació és el mecanisme que fa avançar el món artificial en passos temporals discrets. La seva funció és coordinar l'entorn, els agents, les xarxes neuronals, les accions, la reproducció, la mort i el registre de dades dins un ordre únic, repetible i verificable.

**Decisió funcional principal:** la simulació ha de funcionar per passos temporals discrets. En cada pas es llegeix l'estat actual, es calculen decisions, s'apliquen efectes i es produeix un nou estat coherent. No s'han d'actualitzar agents individualment de manera improvisada ni amb efectes laterals difícils de reproduir.

## 2. Funció dins l'arquitectura general

El bucle principal actua com a coordinador del sistema. No hauria de contenir la lògica detallada de tots els blocs, sinó orquestrar-los en l'ordre correcte.

| Bloc funcional | Responsabilitat dins el bucle |
| --- | --- |
| Entorn | Actualitzar recursos, condicions ambientals i estat espacial del món. |
| Agents | Mantenir l'estat vital, posició, energia, edat i genealogia dels individus. |
| Observacions | Construir la informació local disponible per cada agent a partir de l'estat actual. |
| Xarxa neuronal | Convertir observacions en accions candidates. |
| Accions | Traduir decisions en efectes sobre posició, energia, recursos o reproducció. |
| Reproducció | Crear descendents quan les condicions es compleixen. |
| Mort | Eliminar o marcar agents morts segons causes definides. |
| Mètriques | Agregació de dades del pas temporal. |
| Persistència | Desar configuració, esdeveniments, mètriques i estats seleccionats. |

## 3. Principis de disseny del cicle de simulació

- **Determinisme:** amb la mateixa configuració i la mateixa llavor aleatòria, l'execució ha de produir el mateix resultat.
- **Separació de fases:** observar, decidir i aplicar efectes no s'han de barrejar sense control.
- **Absència de fitness extern:** la selecció no prové d'una puntuació afegida, sinó de supervivència i reproducció.
- **Consistència poblacional:** cap agent mort no pot actuar, alimentar-se ni reproduir-se dins el mateix pas després de morir.
- **Traçabilitat:** cada naixement, mort i mutació rellevant ha de poder registrar-se.
- **Escalabilitat:** el disseny ha de permetre evolucionar cap a càlcul vectoritzat sense reescriure tota la lògica.
- **Simplicitat:** el bucle no ha de convertir-se en un contenidor de regles especials per arreglar casos puntuals.

## 4. Estat d'entrada i estat de sortida de cada pas

Cada pas temporal ha de transformar un estat complet de simulació en un nou estat complet. Aquesta idea facilita proves, depuració, vectorització i repetibilitat.

| Element | A l'entrada del pas | A la sortida del pas |
| --- | --- | --- |
| Món | Recursos, condicions ambientals i ocupació espacial actuals. | Recursos actualitzats, condicions aplicades i ocupació resultant. |
| Agents vius | Estat funcional abans de decidir. | Estat modificat per accions, costos, alimentació, edat i supervivència. |
| Agents morts | No han d'intervenir en decisions. | Poden quedar registrats com a esdeveniments o arxiu històric. |
| Genomes | Genomes vigents dels agents vius. | Genomes copiats o mutats en descendents nous. |
| Mètriques | Acumulats previs. | Mètriques del pas i agregats actualitzats. |
| Llavors aleatòries | Clau o estat RNG disponible per al pas. | Clau o estat RNG avançat de manera controlada. |

## 5. Ordre funcional recomanat del pas temporal

| Ordre | Fase | Descripció funcional |
| --- | --- | --- |
| 1 | Validar estat inicial del pas | Comprovar que no hi ha inconsistències greus abans d'avançar. |
| 2 | Actualitzar entorn | Regenerar recursos i aplicar canvis ambientals programats o estocàstics. |
| 3 | Incrementar edat i aplicar metabolisme basal | Fer que tots els agents vius paguin el cost mínim d'existir. |
| 4 | Calcular observacions | Construir la percepció local de cada agent viu. |
| 5 | Decidir accions | Executar la xarxa neuronal de cada agent i obtenir accions candidates. |
| 6 | Validar accions candidates | Descartar o convertir accions impossibles segons energia, posició o estat vital. |
| 7 | Aplicar efectes d'acció | Resoldre moviment, alimentació, costos, competència i intents de reproducció. |
| 8 | Resoldre naixements | Crear descendents vàlids, copiar i mutar genomes, registrar genealogia. |
| 9 | Resoldre morts | Marcar agents morts i assignar causa de mort. |
| 10 | Actualitzar estat global | Construir el nou estat coherent de món i població. |
| 11 | Registrar mètriques i esdeveniments | Guardar dades del pas, agregats i esdeveniments rellevants. |
| 12 | Validar estat final del pas | Comprovar invariants abans del pas següent. |

**Nota crítica sobre l'ordre:** l'ordre exacte pot ajustar-se en la implementació, però no ha de ser ambigu. Una simulació on uns agents actuen abans que altres sense regla explícita pot introduir avantatges artificials i fer difícil interpretar els resultats.

## 6. Fases del bucle principal

### 6.1. Validació inicial del pas

Abans d'aplicar cap canvi, el sistema ha de comprovar que l'estat actual és coherent. Aquesta fase no ha de corregir silenciosament errors greus; ha de detectar-los i, si cal, aturar l'execució o registrar una incidència.

- No hi pot haver agents vius amb energia negativa.
- No hi pot haver agents vius fora dels límits del món.
- No hi pot haver identificadors duplicats.
- Els recursos han d'estar dins dels valors permesos.
- Les referències genealògiques han de ser vàlides o explícitament nul·les en agents inicials.

### 6.2. Actualització de l'entorn

L'entorn s'actualitza abans que els agents observin. Això fa que tots els agents percebin el mateix estat ambiental actualitzat del pas.

- Regenerar recursos segons la regla definida.
- Aplicar límits de capacitat de recursos per cel·la o zona.
- Actualitzar condicions ambientals simples, com zones de penalització o canvis estacionals.
- Registrar canvis ambientals rellevants si afecten la interpretació de resultats.

### 6.3. Metabolisme basal i envelliment

Cada agent viu ha de pagar el cost mínim d'existir. Això evita que quedar-se quiet sigui una forma de supervivència gratuïta. També s'incrementa l'edat de cada agent.

- Incrementar l'edat en una unitat temporal.
- Restar cost metabòlic basal.
- Aplicar penalitzacions ambientals generals si pertoquen.
- Marcar com a candidats a mort els agents que ja no compleixen energia mínima o edat màxima.

### 6.4. Càlcul d'observacions

Les observacions es calculen només per als agents vius i aptes per actuar. Han de basar-se en l'estat del món després de l'actualització ambiental i abans de les accions del pas.

- Energia pròpia normalitzada.
- Edat pròpia normalitzada o codificada.
- Recursos locals dins el radi perceptiu.
- Densitat local d'agents.
- Senyals ambientals locals.
- Cap informació global de població, mapa complet o futur del sistema.

### 6.5. Decisió d'accions

La xarxa neuronal de cada agent transforma observacions en accions candidates. Aquesta fase no ha d'aplicar encara els efectes; només ha de produir una decisió interpretable pel sistema d'accions.

- Tots els agents decideixen a partir del mateix estat observat del pas.
- Les accions resultants poden ser discretes o parametritzades, però han de pertànyer a l'espai d'accions permès.
- Les decisions impossibles no s'han d'executar directament; s'han de validar a la fase següent.

### 6.6. Validació d'accions candidates

Una acció pot ser triada per la xarxa però no ser executable. El sistema ha de resoldre aquests casos amb una política clara.

| Cas | Resolució funcional recomanada |
| --- | --- |
| Moviment fora del món | Convertir en inacció o impedir el moviment amb cost reduït o nul, segons configuració. |
| Alimentar-se sense recurs disponible | Aplicar l'acció sense guany energètic; opcionalment amb cost mínim. |
| Reproduir-se sense energia suficient | Convertir en inacció o intent fallit amb cost definit. |
| Agent ja mort o invàlid | No executar cap acció. |
| Cel·la de destí saturada | Resoldre per regla de conflicte, no per ordre accidental de l'array. |

### 6.7. Aplicació d'efectes

Els efectes de les accions s'apliquen de manera ordenada i consistent. Aquest és un dels punts més sensibles del bucle, perquè una aplicació seqüencial mal definida pot generar biaixos.

- Aplicar costos energètics de les accions.
- Resoldre moviments i ocupació espacial.
- Consumir recursos i transferir energia als agents.
- Registrar intents de reproducció que compleixen condicions.
- Aplicar penalitzacions per competència o saturació, si el model les inclou.

## 7. Resolució de conflictes i simultaneïtat

Quan diversos agents intenten actuar sobre el mateix recurs, la mateixa cel·la o la mateixa oportunitat espacial, el sistema ha de tenir una regla explícita. No és acceptable que el resultat depengui de l'ordre intern de memòria o de la posició accidental en una llista.

| Conflicte | Regla funcional recomanada | Motiu |
| --- | --- | --- |
| Diversos agents volen el mateix recurs | Repartiment proporcional o elecció aleatòria controlada per llavor. | Evita privilegiar agents per ordre de processament. |
| Diversos agents volen entrar a la mateixa cel·la | Permetre densitat limitada o resoldre amb sorteig determinista. | Manté coherència espacial. |
| Diversos naixements en zona saturada | Acceptar fins a capacitat màxima i rebutjar la resta amb regla explícita. | Evita creixement impossible. |
| Agent mor i es reproduiria al mateix pas | Prioritzar la mort si ja no compleix condicions mínimes abans de resoldre naixement. | Evita descendència generada per agents inviables. |

**Recomanació per al prototip:** per al primer prototip és millor usar regles simples i deterministes sempre que sigui possible. Quan calgui aleatorietat, ha d'estar controlada per la llavor de l'experiment i registrada indirectament per poder reproduir l'execució.

## 8. Naixements, morts i actualització poblacional

### 8.1. Resolució de naixements

Els naixements s'han de resoldre després d'aplicar els costos i efectes de les accions. Això garanteix que només es reprodueixen agents que realment mantenen les condicions mínimes dins el pas.

- Comprovar energia mínima reproductiva després dels costos aplicables.
- Crear un identificador únic per al descendent.
- Assignar progenitor explícit.
- Copiar genoma del progenitor.
- Aplicar mutacions petites i configurables.
- Assignar energia inicial al descendent.
- Restar el cost reproductiu al progenitor.
- Col·locar el descendent en una posició vàlida.
- Registrar l'esdeveniment de naixement.

### 8.2. Resolució de morts

La mort s'ha de resoldre amb causa explícita. Quan un agent compleix més d'una causa de mort al mateix pas, cal una prioritat definida per registrar una causa principal i, si es vol, causes secundàries.

| Prioritat recomanada | Causa de mort | Justificació |
| --- | --- | --- |
| 1 | Energia insuficient | És la causa central del model energètic. |
| 2 | Edat màxima | Evita immortalitat i bloqueig poblacional. |
| 3 | Condició ambiental letal | Permet analitzar impacte ambiental. |
| 4 | Competència o saturació | Només si el model ho activa explícitament. |

### 8.3. Construcció del nou estat poblacional

Al final del pas, el sistema ha de construir la població activa del pas següent. Els agents morts poden quedar fora de la població activa, però han de quedar disponibles com a registres històrics o esdeveniments.

- Agents vius supervivents.
- Nous descendents vàlids.
- Agents morts registrats però no actius.
- Comptadors d'identificadors actualitzats.
- Referències genealògiques preservades.

## 9. Registre, mètriques i traçabilitat

El bucle principal ha de registrar dades en cada pas o a intervals configurables. No cal guardar sempre l'estat complet de tots els agents en cada pas, però sí cal garantir que les mètriques bàsiques i els esdeveniments evolutius principals no es perden.

| Tipus de registre | Contingut mínim |
| --- | --- |
| Mètriques per pas | Població viva, energia mitjana, recursos totals, naixements, morts i causes de mort. |
| Esdeveniments de naixement | Pas temporal, nou identificador, progenitor, energia inicial, mutacions aplicades o resum de mutació. |
| Esdeveniments de mort | Pas temporal, agent, edat, energia final, causa principal i possible causa secundària. |
| Snapshots | Estat parcial o complet cada cert interval per anàlisi posterior. |
| Configuració | Paràmetres de món, agents, energia, mutació, llavors i durada. |

**Criteri pràctic de registre:** per al prototip és millor registrar pocs camps però fiables que una gran quantitat de dades difícil d'interpretar. La traçabilitat genealògica, les causes de mort i les mètriques poblacionals són prioritàries.

## 10. Repetibilitat i control de llavors aleatòries

El bucle ha de tractar l'aleatorietat com una part explícita de l'estat experimental. Això és especialment important per a inicialització, regeneració de recursos, resolució de conflictes, reproducció i mutació.

- Cada execució ha de tenir una llavor principal registrada.
- Els subcomponents poden derivar claus o subllavors de manera controlada.
- La mutació no ha de dependre d'aleatorietat externa no registrada.
- Els empats o conflictes resolts aleatòriament han de ser reproduïbles.
- Canviar el nombre d'agents no hauria de generar efectes caòtics per consum d'aleatorietat mal gestionat, en la mesura que el disseny ho permeti.

## 11. Validacions internes i invariants

Els invariants són condicions que sempre haurien de ser certes si la simulació és coherent. S'han de comprovar durant el desenvolupament i, si cal, en mode diagnòstic durant execucions llargues.

| Invariant | Motiu |
| --- | --- |
| Cap agent viu amb energia inferior al mínim vital després de resoldre morts. | Evita agents zombis que continuïn actuant. |
| Cap agent viu fora dels límits del món. | Manté coherència espacial. |
| Cap identificador duplicat. | Preserva genealogia i persistència. |
| Cap recurs negatiu. | Evita errors en economia energètica. |
| La suma de naixements registrats coincideix amb descendents creats. | Garanteix traçabilitat. |
| La suma de morts registrades coincideix amb agents retirats. | Garanteix mètriques fiables. |
| Els descendents tenen progenitor existent o registrat. | Permet reconstruir llinatges. |
| Les claus aleatòries avancen de manera controlada. | Garanteix repetibilitat. |

## 12. Errors típics que cal evitar

- Aplicar accions agent per agent i deixar que els primers de la llista tinguin avantatge sistemàtic.
- Permetre que agents morts actuïn dins el mateix pas.
- Fer que la reproducció passi abans dels costos energètics reals.
- No registrar les causes de mort.
- Fer servir una funció de fitness externa que substitueixi la selecció ambiental.
- Guardar només gràfics visuals sense dades analitzables.
- Corregir errors d'estabilitat amb constants arbitràries no documentades.
- Introduir massa excepcions especials fins que el bucle deixi de ser interpretable.

## 13. Exclusions temporals

Per mantenir el bucle inicial net i verificable, queden fora d'aquesta fase:

- Simulació asíncrona contínua sense passos discrets.
- Planificació d'accions a llarg termini.
- Interaccions socials complexes.
- Sistema de combat elaborat.
- Aprenentatge individual dins la vida de l'agent.
- Reproducció sexual.
- Funció de fitness externa.
- Visualització interactiva acoblada al nucli de simulació.
- Persistència completa de cada estat si encara no és necessària.
- Execució distribuïda o paral·lelització complexa abans de tenir consistència funcional.

## 14. Criteris de tancament

| Criteri | Condició de compliment |
| --- | --- |
| Execució autònoma | La simulació pot córrer molts passos sense intervenció manual. |
| Ordre definit | Cada pas temporal té una seqüència clara i documentada. |
| Consistència de dades | No apareixen agents impossibles, recursos negatius ni identificadors duplicats. |
| Repetibilitat | La mateixa configuració i llavor reprodueixen el resultat. |
| Selecció ambiental | La supervivència i reproducció depenen de conseqüències del sistema, no d'una puntuació externa. |
| Registre suficient | Naixements, morts, mètriques i configuració queden registrats. |
| Traçabilitat | Cada agent pot seguir-se des del naixement fins a la mort. |
| Preparació tècnica | El disseny permet una implementació vectoritzable i modular. |

## 15. Valoració d'adequació amb la resta de la documentació

### Veredicte resumit

La valoració és **positiva**: el document encaixa amb el full de ruta funcional i completa el punt 4.11 del projecte. La seva aportació principal és convertir les peces ja descrites —món, agents, genoma, observacions, accions, energia, reproducció i mort— en una seqüència temporal auditable i repetible.

### Encaix amb documents existents

| Document relacionat | Encaix observat | Valoració |
| --- | --- | --- |
| `document-funcional-inicial.md` | Desenvolupa el punt 4.11 i manté els criteris de simulació repetible, mètriques traçables i selecció sense fitness extern. | Molt coherent. |
| `arquitectura-general.md` | Respecta la separació entre nucli de simulació, persistència, anàlisi i visualització; el bucle queda com a orquestrador, no com a contenidor de totes les regles. | Molt coherent. |
| `model-mon-2d.md` | Situa l'actualització ambiental i la regeneració de recursos abans de les observacions, cosa que dona un estat percebut comú per a tots els agents. | Coherent. |
| `model-agent.md` | Reforça que l'agent és estat identificable amb energia, edat, posició, estat vital i genealogia, i que els morts no han de seguir actuant. | Coherent. |
| `observacions-agents.md` | Manté les observacions com a informació local calculada abans de les accions i sense informació global impossible. | Coherent. |
| `accions-agents.md` | Defineix una fase específica de validació i aplicació d'accions, evitant que una decisió neuronal impossible es converteixi directament en efecte. | Molt coherent. |
| `energia-recursos-metabolisme.md` | Integra metabolisme basal, costos d'acció, alimentació i límits de recursos dins un ordre que evita supervivència gratuïta. | Molt coherent. |
| `reproduccio-herencia-mutacio.md` | Situa els naixements després dels costos i efectes d'acció, de manera que la reproducció continua sent una conseqüència d'estat viable i energia suficient. | Coherent, amb un punt de vigilància indicat més avall. |
| `mort-seleccio-ambiental.md` | Exigeix causa de mort explícita, prioritats i retirada d'agents morts de la població activa. | Coherent, amb un punt de vigilància indicat més avall. |

### Punts forts

- **Defineix un contracte temporal complet:** cada pas té entrada, sortida, ordre i validacions.
- **Redueix biaixos d'ordre:** separa observació, decisió, validació i efectes, i demana regles explícites de conflicte.
- **Preserva el principi evolutiu central:** la selecció continua depenent de supervivència, energia i reproducció, no d'una puntuació externa.
- **Millora la traçabilitat:** exigeix registrar naixements, morts, causes, mètriques, configuració i llavor.
- **Prepara la vectorització:** planteja transformacions d'estat completes i fases separades, compatibles amb una evolució cap a JAX o altres execucions vectoritzades.

### Riscos o punts a vigilar

1. **Ordre exacte entre morts i naixements.** El document situa naixements abans de morts en la taula principal, però també indica que cal prioritzar la mort si un agent ja no compleix condicions mínimes. En el disseny tècnic caldrà separar clarament agents inviables després de metabolisme/costos abans d'autoritzar descendència, o bé fer una fase de filtratge de viabilitat prèvia als naixements.
2. **Resolució de conflictes amb aleatorietat.** Si es fan sortejos per recursos, espai o naixements en zona saturada, el consum de RNG ha de quedar estructurat per pas i per subfase per preservar repetibilitat.
3. **Validacions massa costoses en execucions llargues.** Les invariants són imprescindibles en desenvolupament, però convé preveure modes de diagnòstic o freqüències configurables per no penalitzar experiments llargs.
4. **Mètriques mínimes insuficients per anàlisi evolutiva.** El document cobreix població, energia, recursos, naixements i morts; caldrà connectar-lo amb mètriques de diversitat genètica i llinatges quan s'implementi el punt de mètriques evolutives.

### Recomanacions d'implementació

- Implementar el pas temporal com una transformació explícita `state + rng -> state + events + metrics + rng`, encara que la signatura final pugui variar.
- Crear subfases pures i testables per validació, actualització ambiental, metabolisme, observació, decisió, validació d'accions, efectes, naixements, morts, mètriques i invariants.
- Definir una política única de conflictes per recursos i ocupació abans d'executar experiments comparatius.
- Registrar esdeveniments com a dades estructurades i no només com a text de log.
- Fer proves de determinisme: mateixa configuració i mateixa llavor han de produir les mateixes mètriques i els mateixos esdeveniments.

### Conclusió de la valoració

El document és adequat per incorporar-se a la documentació funcional del Projecte Evolució. Completa una peça central que connecta els documents ja existents i proporciona el contracte necessari perquè el prototip pugui avançar cap a una implementació modular, repetible i auditable.

La recomanació és acceptar-lo com a especificació funcional inicial del bucle principal, amb una nota tècnica pendent sobre la resolució exacta de viabilitat abans de naixements quan un agent pot morir i reproduir-se dins el mateix pas.
