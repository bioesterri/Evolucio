# Projecte Evolució — Proves bàsiques de validació

**Especificació funcional per validar el comportament mínim del motor abans d'interpretar resultats evolutius.**

## Metadades

| Camp | Valor |
| --- | --- |
| Tipus de document | Especificació funcional de validació bàsica del motor |
| Punt del full de ruta | Punt 15 del pla funcional |
| Àmbit | Configuració, món 2D, agents, energia, accions, reproducció, mutació, mort, mètriques, persistència i repetibilitat |
| Objectiu | Definir comprovacions mínimes que evitin errors silenciosos abans d'executar experiments llargs o interpretar resultats evolutius. |
| Resultat esperat | Confiança en el comportament del sistema abans d'interpretar qualsevol resultat evolutiu. |
| Criteri de tancament | Els casos bàsics passen proves automàtiques o verificacions sistemàtiques abans d'executar experiments llargs. |
| Estat | Versió funcional per a revisió interna |
| Versió | 1.0 |
| Data | Juliol de 2026 |

## Índex

1. [Objectiu del document](#1-objectiu-del-document)
2. [Principi rector](#2-principi-rector)
3. [Abast de la validació inicial](#3-abast-de-la-validació-inicial)
4. [Invariants mínims del sistema](#4-invariants-mínims-del-sistema)
5. [Proves de configuració i inicialització](#5-proves-de-configuració-i-inicialització)
6. [Proves del món 2D i dels recursos](#6-proves-del-món-2d-i-dels-recursos)
7. [Proves d'agent i estat vital](#7-proves-dagent-i-estat-vital)
8. [Proves d'energia i metabolisme](#8-proves-denergia-i-metabolisme)
9. [Proves d'accions](#9-proves-daccions)
10. [Proves de reproducció, herència i mutació](#10-proves-de-reproducció-herència-i-mutació)
11. [Proves de mort i selecció ambiental](#11-proves-de-mort-i-selecció-ambiental)
12. [Proves de coherència poblacional](#12-proves-de-coherència-poblacional)
13. [Proves del bucle principal](#13-proves-del-bucle-principal)
14. [Proves de mètriques i registre](#14-proves-de-mètriques-i-registre)
15. [Escenaris controlats de validació](#15-escenaris-controlats-de-validació)
16. [Proves automàtiques i verificacions sistemàtiques](#16-proves-automàtiques-i-verificacions-sistemàtiques)
17. [Criteris mínims abans d'experiments llargs](#17-criteris-mínims-abans-dexperiments-llargs)
18. [Exclusions temporals](#18-exclusions-temporals)
19. [Riscos principals](#19-riscos-principals)
20. [Resultat esperat](#20-resultat-esperat)
21. [Definició de tancament](#21-definició-de-tancament)

## 1. Objectiu del document

Aquest document defineix el conjunt mínim de proves i verificacions que ha de tenir el prototip del Projecte Evolució abans d'executar experiments llargs o interpretar resultats com si fossin evolució real.

La finalitat no és demostrar que els agents evolucionen, sinó garantir que el motor respecta les regles bàsiques: energia, recursos, accions, reproducció, mutació, mort, genealogia, mètriques i persistència. Sense aquesta validació, qualsevol patró observat podria ser conseqüència d'un error silenciós.

## 2. Principi rector

Una simulació evolutiva només és interpretable si primer sabem que les seves regles funcionen de manera coherent. Les proves bàsiques han d'actuar com una barrera entre el desenvolupament del motor i la lectura científica dels resultats.

Per tant, cap experiment llarg s'hauria de considerar vàlid si abans no han passat les proves mínimes del sistema.

- Les proves han de ser petites, repetibles i fàcils d'entendre.
- Han de cobrir els errors que poden alterar la dinàmica evolutiva sense ser visibles a simple vista.
- Han de validar invariants del sistema, no només exemples feliços.
- Han de distingir entre errors de simulació i resultats biològicament interessants.
- Han de poder executar-se sovint durant el desenvolupament.

## 3. Abast de la validació inicial

La primera bateria de proves ha de cobrir el comportament funcional mínim del prototip. No cal validar encara funcionalitats futures, però sí tot allò que pot afectar directament la supervivència, la reproducció o la interpretació de mètriques.

| Bloc validat | Què s'ha de comprovar | Risc que evita |
| --- | --- | --- |
| Configuració | Els paràmetres essencials existeixen, són coherents i es poden repetir amb una llavor. | Experiments no reproduïbles o configuracions impossibles. |
| Món 2D | El món s'inicialitza, conserva límits i actualitza recursos de manera controlada. | Agents fora del món, recursos infinits o entorn incoherent. |
| Agents | Cada agent té identificador, posició, energia, edat, genoma, progenitor i estat vital vàlids. | Individus corruptes o genealogies trencades. |
| Energia | Tota acció té el cost o guany esperat i la supervivència gratuïta no és possible. | Creixement artificial de població o morts inexplicables. |
| Accions | Moviment, alimentació, reproducció i inacció modifiquen el sistema com s'ha definit. | Estratègies aparents causades per efectes erronis. |
| Reproducció | Els naixements compleixen condicions, cost i registre genealògic. | Descendència espontània o herència mal aplicada. |
| Mutació | Les mutacions són configurables, petites i afecten només camps heretables. | Soroll excessiu o mutacions sobre estat no heretable. |
| Mort | Les causes de mort són detectables i registrables. | Pèrdua d'agents sense explicació analítica. |
| Mètriques | Els indicadors agregats coincideixen amb l'estat real del sistema. | Interpretació estadística falsa. |
| Persistència | Els resultats rellevants sobreviuen al final de l'execució. | Experiments orfes o impossibles de revisar. |

## 4. Invariants mínims del sistema

Els invariants són condicions que sempre han de ser certes. Si es trenquen, el resultat de la simulació no és fiable, encara que visualment sembli correcte.

| Invariant | Descripció funcional | Conseqüència si falla |
| --- | --- | --- |
| Identitat única | Cap dos agents vius o registrats han de compartir el mateix identificador. | Genealogia i mètriques de descendència queden contaminades. |
| Posició vàlida | Tot agent viu ha d'estar dins dels límits del món. | Observacions, recursos i moviment deixen de tenir sentit. |
| Energia finita | L'energia d'un agent no pot ser indefinida, negativa sense mort associada ni créixer sense causa. | La pressió selectiva energètica desapareix. |
| Mort irreversible | Un agent mort no pot actuar, alimentar-se ni reproduir-se. | La població deixa de representar individus vius. |
| Naixement registrat | Tot descendent ha de tenir progenitor conegut o una marca clara d'individu inicial. | No es poden reconstruir llinatges. |
| Herència separada de vida | Només s'hereta el genoma i els paràmetres heretables, no l'edat ni l'historial vital. | Es confon evolució amb transferència artificial d'estat. |
| Mutació acotada | Les mutacions han de respectar taxes i magnituds configurades. | L'adaptació acumulativa es converteix en soroll. |
| Mètrica coherent | Els totals de naixements, morts i població han de quadrar pas a pas. | Les conclusions poblacionals són invàlides. |
| Llavor repetible | La mateixa configuració amb la mateixa llavor ha de reproduir el mateix resultat. | No es poden comparar experiments amb rigor. |

## 5. Proves de configuració i inicialització

Abans de simular cap agent, el sistema ha de validar que la configuració inicial és coherent. Molts errors evolutius provenen de paràmetres impossibles o contradictoris.

### 5.1. Configuració mínima obligatòria

- La mida del món ha de ser positiva i suficient per allotjar la població inicial.
- La població inicial no pot superar la capacitat espacial definida si el món és discret i no permet solapament.
- L'energia inicial dels agents ha de ser positiva.
- Els costos energètics no poden ser negatius.
- La taxa de mutació ha d'estar dins un rang vàlid.
- La taxa de regeneració de recursos no pot generar recursos il·limitats fora del màxim definit.
- La llavor aleatòria ha d'estar registrada abans de començar l'execució.

### 5.2. Casos de prova recomanats

| Cas | Condició inicial | Resultat esperat |
| --- | --- | --- |
| Configuració vàlida mínima | Món petit, pocs agents, pocs recursos i llavor fixa. | La simulació s'inicialitza sense errors i registra la configuració. |
| Mida de món invàlida | Amplada o alçada igual a zero. | El sistema rebutja la configuració abans d'executar-se. |
| Energia inicial invàlida | Agents creats amb energia negativa. | El sistema impedeix la inicialització o normalitza segons regla explícita. |
| Taxa de mutació impossible | Taxa negativa o superior al rang admès. | El sistema rebutja la configuració. |
| Població inicial excessiva | Més agents que posicions disponibles en un món sense solapament. | El sistema falla de manera controlada i explicable. |

## 6. Proves del món 2D i dels recursos

El món és la font principal de pressió ambiental. Cal verificar que els recursos apareixen, desapareixen i es regeneren segons les regles definides, i que els agents no poden aprofitar incoherències espacials.

| Prova | Què comprova | Resultat esperat |
| --- | --- | --- |
| Límits espacials | Cap agent pot sortir del món després d'una acció de moviment. | Les posicions finals són sempre vàlides. |
| Recurs consumit | Un agent sobre un recurs pot consumir-lo segons la regla definida. | El recurs baixa o desapareix i l'agent guanya energia. |
| Recurs no duplicat | Dos agents no poden consumir íntegrament el mateix recurs si la regla no ho permet. | El repartiment o la prioritat queda resolt de manera determinista. |
| Regeneració controlada | Els recursos reapareixen dins dels límits establerts. | No se supera la capacitat màxima del món. |
| Escassetat real | Amb pocs recursos i costos energètics actius, alguns agents acaben morint. | La manca de recursos té conseqüències funcionals. |
| Entorn sense recursos | El món no conté recursos durant diversos passos. | La població perd energia i tendeix al col·lapse. |
| Entorn abundant | Els recursos són molt freqüents. | La població pot créixer, però limitada per energia, espai o regles de reproducció. |

## 7. Proves d'agent i estat vital

Cada agent ha de poder seguir-se des del naixement fins a la mort. Les proves han de garantir que l'estat de l'individu és complet, consistent i no es barreja amb informació global indeguda.

- Un agent viu ha de tenir sempre identificador, posició vàlida, energia, edat, genoma i estat vital actiu.
- Un agent inicial ha de tenir una marca clara que indiqui que no té progenitor.
- Un agent nascut durant la simulació ha de tenir un progenitor registrat.
- L'edat ha d'augmentar segons el pas temporal definit.
- Un agent mort ha de conservar-se com a registre històric si cal per a mètriques, però no pot participar en el bucle d'acció.
- Els camps de vida no han de modificar el genoma excepte en el moment explícit de mutació heretable.

## 8. Proves d'energia i metabolisme

Aquest és un dels blocs més importants. Si l'energia no està ben validada, la simulació pot produir supervivència gratuïta o col·lapses artificials.

| Cas | Condició | Resultat esperat |
| --- | --- | --- |
| Cost basal | Agent viu que no fa res durant un pas. | La seva energia disminueix pel metabolisme basal. |
| Cost de moviment | Agent que executa moviment vàlid. | La seva energia disminueix pel cost basal i el cost de moviment, si s'apliquen tots dos. |
| Moviment invàlid | Agent intenta sortir del món o ocupar una posició no permesa. | S'aplica la regla definida: bloqueig amb cost, bloqueig sense cost o correcció explícita. |
| Alimentació | Agent accedeix a recurs disponible. | Augmenta energia sense superar el màxim definit, si existeix. |
| Cap alimentació | Agent intenta alimentar-se sense recurs. | No guanya energia i pot pagar el cost de l'acció si s'ha definit. |
| Mort energètica | Agent arriba a energia zero o sota el llindar de mort. | Canvia a estat mort i registra causa de mort energètica. |
| Energia màxima | Agent consumeix molt recurs. | L'energia queda limitada pel màxim o es gestiona segons regla explícita. |
| Cap energia negativa viva | Agent viu després d'un pas amb cost superior a la seva energia. | No queda viu amb energia negativa; mor o es clampa segons regla definida. |

## 9. Proves d'accions

Les accions han de tenir efectes simples i verificables. Cap acció ha d'incorporar una estratègia intel·ligent oculta; només ha de modificar el sistema segons les regles.

| Acció | Validació mínima | Error silenciós que cal evitar |
| --- | --- | --- |
| Inacció | Consumeix energia basal i no altera posició ni recursos. | Agents que sobreviuen sense cost. |
| Moviment | Canvia posició només si la destinació és vàlida. | Agents que travessen límits o ignoren ocupació. |
| Alimentació | Converteix recurs disponible en energia segons taxa definida. | Energia creada sense recurs consumit. |
| Reproducció | Només passa si hi ha energia, espai i condicions mínimes. | Naixements sense cost o sense progenitor. |
| Competència indirecta | El consum d'un agent redueix recursos disponibles per altres. | Recursos consumibles infinitament per tothom. |
| Interacció directa, si existeix | Té cost, efecte i registre explícits. | Atacs o bloquejos que codifiquen estratègies massa complexes. |

## 10. Proves de reproducció, herència i mutació

La reproducció és el pont entre comportament i evolució. Cal validar que només es reprodueixen els agents que compleixen condicions i que la descendència és semblant, però no necessàriament idèntica, al progenitor.

### 10.1. Reproducció

- Un agent amb energia insuficient no pot reproduir-se.
- Un agent mort no pot reproduir-se.
- Un agent amb energia suficient pot reproduir-se només si l'espai o la regla de col·locació ho permet.
- La reproducció redueix l'energia del progenitor segons el cost definit.
- El descendent rep energia inicial segons la regla definida.
- El descendent rep un identificador nou i un enllaç explícit al progenitor.

### 10.2. Herència

- El genoma del descendent deriva del genoma del progenitor.
- Els pesos i biaixos neuronals es poden reconstruir a partir del genoma heretat.
- Els paràmetres heretables es copien i, si toca, muten.
- L'edat, l'energia acumulada, la posició històrica i el nombre de descendents no s'hereten.

### 10.3. Mutació

| Cas | Configuració | Resultat esperat |
| --- | --- | --- |
| Mutació desactivada | Taxa de mutació igual a zero. | El descendent té el mateix genoma que el progenitor, excepte identificadors. |
| Mutació activada | Taxa petita i magnitud controlada. | Alguns valors genètics canvien dins dels límits establerts. |
| Mutació extrema rebutjada | Magnitud fora del rang permès. | La configuració es rebutja o queda limitada explícitament. |
| Mutació només heretable | Es revisen camps afectats. | No muten edat, energia vital, causa de mort ni mètriques històriques. |
| Registre de mutació | Naixement amb mutació aplicada. | Queda constància de si hi ha hagut mutació i, idealment, de la magnitud agregada. |

## 11. Proves de mort i selecció ambiental

La mort ha de ser explicable. Si els agents desapareixen sense causa, les mètriques de selecció perden valor. Si no moren mai, no hi ha pressió selectiva suficient.

| Causa de mort | Prova mínima | Resultat esperat |
| --- | --- | --- |
| Energia | Agent sense energia després del cost basal o d'acció. | Mort registrada com a manca d'energia. |
| Edat | Agent arriba a l'edat màxima definida. | Mort registrada com a edat màxima. |
| Ambient | Agent exposat a condició ambiental letal o penalitzadora. | Mort o penalització registrada segons regla definida. |
| Competència | Dos o més agents competeixen per espai o recurs limitat. | La resolució queda registrada i no genera duplicats incoherents. |
| Mort múltiple potencial | Un agent compleix diverses causes alhora. | El sistema assigna causa principal o múltiples causes segons criteri explícit. |
| Agent mort | Agent mort roman en dades històriques. | No actua, no consumeix, no es reprodueix i no torna a viu. |

## 12. Proves de coherència poblacional

La coherència poblacional és una comprovació agregada. Serveix per detectar errors que no es veuen en un sol agent però alteren tota la simulació.

| Comprovació | Regla esperada |
| --- | --- |
| Balanç de població | Població final del pas = població inicial + naixements - morts. |
| Agents vius | El conjunt d'agents vius coincideix amb els agents amb estat vital actiu. |
| Agents morts | Els agents morts no apareixen al conjunt d'acció del pas següent. |
| Naixements únics | Cada naixement crea exactament un agent nou amb identificador nou. |
| Genealogia | Cada descendent apunta a un progenitor existent o registrat històricament. |
| Cap agent fantasma | No hi ha agents sense estat, sense genoma o sense posició si són vius. |
| Cap població impossible | La població no supera els límits físics o configurats sense una regla explícita. |

## 13. Proves del bucle principal

El bucle principal ha d'executar els passos en un ordre estable i repetible. Les proves han de garantir que les observacions, decisions i efectes no es barregen temporalment de manera incoherent.

1. Inicialitzar estat del món, agents i recursos.
2. Actualitzar l'entorn segons la regla temporal definida.
3. Calcular observacions locals només per agents vius.
4. Generar accions a partir de la xarxa neuronal o d'un controlador de prova.
5. Aplicar costos i efectes de les accions.
6. Resoldre conflictes de recursos, espai o reproducció.
7. Crear descendents vàlids i registrar-ne genealogia.
8. Determinar morts i causes.
9. Actualitzar mètriques agregades.
10. Persistir o preparar dades per al registre experimental.

En proves controlades, és acceptable substituir temporalment la xarxa neuronal per accions fixes. Això no altera el model evolutiu; només permet comprovar que les regles del motor funcionen.

## 14. Proves de mètriques i registre

Les mètriques no s'han de donar per bones només perquè es calculen. Han de contrastar-se contra estats petits on el resultat esperat es pot comptar manualment.

| Mètrica | Validació mínima |
| --- | --- |
| Població viva | Ha de coincidir amb el nombre real d'agents vius. |
| Naixements per pas | Ha de coincidir amb els descendents creats durant aquell pas. |
| Morts per causa | La suma de morts per causa ha de quadrar amb el total de morts, segons el criteri de causa única o múltiple. |
| Energia mitjana | S'ha de calcular només sobre el conjunt definit: vius, tots o cohort concreta. |
| Recursos consumits | Ha de coincidir amb la diferència entre recursos abans i després, ajustada per regeneració. |
| Descendència per llinatge | Ha de coincidir amb la genealogia registrada. |
| Diversitat genètica | Ha de ser zero o mínima quan la mutació està desactivada i tots els genomes inicials són iguals. |
| Repetibilitat | Dues execucions amb mateixa llavor i configuració han de generar mètriques idèntiques. |

## 15. Escenaris controlats de validació

Aquests escenaris no són experiments evolutius. Són proves de sanitat del motor. Han de ser petits, ràpids i amb resultat esperat clar.

| Escenari | Configuració | Què hauria de passar |
| --- | --- | --- |
| Món buit | Un agent, cap recurs, metabolisme actiu. | L'agent perd energia i acaba morint. |
| Recurs únic | Un agent situat sobre un recurs. | L'agent consumeix el recurs i guanya energia. |
| Moviment bloquejat | Agent al límit del món intenta sortir. | La posició final continua sent vàlida. |
| Reproducció possible | Agent amb energia suficient i espai disponible. | Neix un descendent, baixa l'energia del progenitor i es registra genealogia. |
| Reproducció impossible | Agent amb energia insuficient. | No hi ha naixement i queda registrat o verificable el bloqueig. |
| Mutació zero | Reproducció amb taxa de mutació zero. | El genoma del descendent és idèntic al del progenitor. |
| Mutació positiva | Reproducció amb taxa de mutació controlada. | El genoma pot variar dins dels límits establerts. |
| Mort per edat | Agent prop de l'edat màxima. | Mor quan arriba al límit. |
| Població inicial petita | Dos o tres agents amb recursos limitats. | Els balanços de població, energia i recursos es poden comprovar manualment. |
| Llavor repetida | Mateixa configuració executada dues vegades. | Resultats idèntics si no hi ha fonts externes de no determinisme. |

## 16. Proves automàtiques i verificacions sistemàtiques

La primera versió hauria de combinar proves automàtiques amb verificacions sistemàtiques. No tot cal automatitzar immediatament, però les parts crítiques no haurien de dependre de mirar gràfics a mà.

| Tipus | Ús recomanat | Prioritat |
| --- | --- | --- |
| Proves unitàries | Validar funcions petites: cost energètic, mutació, reproducció, mort, límits espacials. | Alta |
| Proves d'integració | Executar diversos passos amb un món petit i comprovar balanços globals. | Alta |
| Proves de regressió | Assegurar que una correcció no canvia resultats esperats amb llavor fixa. | Alta |
| Verificacions estadístiques simples | Comprovar que moltes repeticions no produeixen valors impossibles. | Mitjana |
| Revisió visual | Confirmar que la simulació representa el que diuen les dades. | Baixa com a prova principal; útil com a suport. |
| Validació manual documentada | Acceptable per prototip quan encara no hi ha automatització completa. | Temporal. |

## 17. Criteris mínims abans d'experiments llargs

Abans de considerar vàlid un experiment de moltes iteracions, s'han de complir aquestes condicions:

- Les proves de configuració passen sense errors.
- Els escenaris controlats produeixen els resultats esperats.
- El balanç de població quadra pas a pas.
- Els agents morts no actuen en passos posteriors.
- Els naixements creen descendents amb progenitor i genoma vàlids.
- Les mutacions respecten taxes i magnituds configurades.
- Les causes de mort es registren de manera agregable.
- Les mètriques principals coincideixen amb recomptes manuals en mons petits.
- La mateixa llavor reprodueix la mateixa execució.
- El registre experimental guarda configuració, llavor, versió, mètriques i artefactes mínims.

## 18. Exclusions temporals

Queden fora de la primera bateria de validació funcional:

- Proves de rendiment a gran escala.
- Validació estadística profunda d'adaptació evolutiva.
- Comparació formal entre estratègies evolutives alternatives.
- Proves de visualització avançada.
- Proves de concurrència o execució distribuïda.
- Validació de models multi-espècie.
- Validació de comunicació entre agents.
- Proves de reproducció sexual o aprenentatge individual.

Aquestes proves seran necessàries més endavant, però afegir-les ara desviaria el focus del prototip inicial.

## 19. Riscos principals

| Risc | Impacte | Mesura de control |
| --- | --- | --- |
| Confiar en la visualització | Una simulació pot semblar correcta mentre les dades són incoherents. | Prioritzar proves de dades i balanços interns. |
| No validar l'energia | La selecció ambiental queda distorsionada. | Proves específiques de costos, guanys, mort i límits energètics. |
| Mutacions massa opaques | No es pot saber si la variació és heretable o un error. | Registrar mutacions i comparar casos amb mutació zero. |
| Mètriques mal definides | Es poden interpretar com adaptació patrons que són soroll. | Validar mètriques contra mons petits comptables manualment. |
| Proves massa complexes | Les proves mateixes es tornen difícils de depurar. | Començar amb escenaris mínims i deterministes. |
| No repetir amb llavor fixa | No es poden reproduir errors ni resultats. | Incloure proves obligatòries de determinisme. |
| Acceptar errors menors | Petites incoherències acumulades poden dominar l'evolució. | Bloquejar experiments llargs si fallen invariants bàsics. |

## 20. Resultat esperat

El resultat d'aquest bloc ha de ser un sistema de validació funcional que permeti executar proves ràpides abans de cada etapa important de desenvolupament o abans d'un experiment llarg. Aquest sistema no ha de demostrar que hi ha evolució, sinó que el motor és prou coherent perquè una evolució observada pugui ser interpretada amb prudència.

La confiança en els resultats del Projecte Evolució dependrà menys de gràfics atractius i més de la capacitat de demostrar que els naixements, morts, costos energètics, mutacions i mètriques tenen una base verificable.

## 21. Definició de tancament

Aquest punt es considerarà complet quan existeixi una bateria mínima de proves o verificacions documentades que cobreixi:

- inicialització de configuració i llavor;
- validesa del món 2D i dels límits espacials;
- coherència de recursos i regeneració;
- costos energètics i metabolisme;
- execució correcta de les accions bàsiques;
- naixement, herència i mutació;
- mort i causes de mort;
- balanç poblacional;
- mètriques evolutives bàsiques;
- persistència mínima de resultats rellevants;
- repetibilitat amb mateixa llavor.

Quan aquests punts passin de manera consistent, el prototip podrà executar experiments més llargs amb una base mínima de confiança. Si fallen, no s'haurien d'interpretar els resultats evolutius, encara que semblin interessants.
