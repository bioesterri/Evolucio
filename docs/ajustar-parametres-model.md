# Projecte Evolució — Ajustar paràmetres del model

**Especificació funcional per modificar de manera controlada taxes de mutació, costos energètics, densitat de recursos, ritme ambiental, mida poblacional i condicions de reproducció.**

## Metadades

| Camp | Valor |
| --- | --- |
| Nom del document | Ajustar paràmetres del model |
| Tipus de document | Especificació funcional per al calibratge del model |
| Punt del full de ruta | Punt 19 del pla funcional |
| Àmbit | Calibratge controlat de paràmetres ecològics, genètics i demogràfics |
| Fase | Ajust posterior als primers experiments inicials |
| Propòsit | Establir un procediment controlat per situar la simulació en una zona funcional: prou exigent per generar selecció, però no tan hostil que provoqui col·lapse sistemàtic. |
| Resultat esperat | Un model calibrat prou bé perquè no sigui ni trivial ni impossible: ha de permetre evolució observable sense col·lapse permanent. |
| Criteri de tancament | Els ajustos es fan comparant experiments registrats, no canviant valors a cegues. |
| Estat | Versió funcional inicial |
| Versió | 1.0 |
| Data | Juliol de 2026 |

## Índex

1. [Objectiu del document](#1-objectiu-del-document)
2. [Principi rector](#2-principi-rector)
3. [Abast de l'ajust](#3-abast-de-lajust)
4. [Paràmetres principals ajustables](#4-paràmetres-principals-ajustables)
5. [Estat funcional desitjat](#5-estat-funcional-desitjat)
6. [Protocol d'ajust](#6-protocol-dajust)
7. [Ordre recomanat de calibratge](#7-ordre-recomanat-de-calibratge)
8. [Criteris quantitatius de comparació](#8-criteris-quantitatius-de-comparació)
9. [Ajust de la mutació](#9-ajust-de-la-mutació)
10. [Ajust de recursos i metabolisme](#10-ajust-de-recursos-i-metabolisme)
11. [Ajust de reproducció i mida poblacional](#11-ajust-de-reproducció-i-mida-poblacional)
12. [Ajust del ritme ambiental](#12-ajust-del-ritme-ambiental)
13. [Registre obligatori dels ajustos](#13-registre-obligatori-dels-ajustos)
14. [Antipatrons que s'han d'evitar](#14-antipatrons-que-shan-devitar)
15. [Resultat esperat del calibratge](#15-resultat-esperat-del-calibratge)
16. [Criteris de tancament](#16-criteris-de-tancament)
17. [Decisió funcional recomanada](#17-decisió-funcional-recomanada)

## 1. Objectiu del document

Aquest document defineix com s'han d'ajustar els paràmetres del model del Projecte Evolució després dels primers experiments inicials. L'objectiu no és trobar una configuració perfecta, sinó establir un procediment controlat per situar la simulació en una zona funcional: prou exigent per generar selecció, però no tan hostil que provoqui col·lapse sistemàtic.

L'ajust de paràmetres ha de permetre comparar experiments, detectar configuracions inviables, reduir comportaments trivials i preparar el sistema per a experiments evolutius més llargs i interpretables.

## 2. Principi rector

Cap paràmetre s'ha de modificar només perquè una execució concreta sembli estranya. Una sola simulació pot fallar per soroll, per la llavor aleatòria o per una condició inicial poc representativa. Per això, qualsevol ajust ha de basar-se en comparacions entre execucions registrades, idealment amb diverses llavors i amb canvis petits.

El criteri funcional és simple: el model ha de sostenir poblacions que competeixen per recursos limitats, acumulen variació hereditària i mostren diferències mesurables entre llinatges. Si la població creix sense fre, no hi ha pressió selectiva suficient. Si col·lapsa sempre, no hi ha temps evolutiu útil.

## 3. Abast de l'ajust

L'ajust de paràmetres inclou els elements que modulen la dificultat ecològica, la variabilitat genètica i la capacitat de renovació poblacional. No inclou redissenyar el model de comportament ni afegir estratègies intel·ligents codificades manualment.

- **S'inclou:** densitat i regeneració de recursos, costos energètics, llindars reproductius, taxa i magnitud de mutació, mida inicial de població, ritme de canvi ambiental i límits demogràfics funcionals.
- **No s'inclou:** afegir conductes fixes, introduir una funció de fitness externa, seleccionar manualment individus, alterar resultats posteriors o canviar múltiples regles sense registre.
- **Finalitat:** trobar un règim dinàmic on la supervivència i la reproducció siguin possibles però competitives.

## 4. Paràmetres principals ajustables

Els paràmetres s'han d'organitzar per famílies funcionals. Això facilita entendre quin mecanisme s'està modificant i evita barrejar efectes ecològics, genètics i demogràfics.

| Família | Paràmetres | Efecte esperat | Risc si s'ajusta malament |
| --- | --- | --- | --- |
| Recursos | Densitat inicial, taxa de regeneració, capacitat màxima per cel·la, distribució espacial. | Controla la disponibilitat energètica i la competència indirecta. | Excés de recursos: població trivial. Massa pocs recursos: extinció sistemàtica. |
| Energia i metabolisme | Cost basal, cost de moviment, energia obtinguda per aliment, límit energètic. | Determina si viure, moure's i alimentar-se tenen conseqüències funcionals. | Costos massa baixos eliminen pressió selectiva; costos massa alts impedeixen reproducció. |
| Reproducció | Llindar reproductiu, cost reproductiu, energia inicial del descendent, espai requerit. | Controla renovació generacional i pressió per acumular energia. | Llindar massa baix causa explosió; massa alt bloqueja el procés evolutiu. |
| Mutació | Probabilitat de mutació, desviació típica, límits de pesos, mutació de paràmetres heretables. | Genera variació heretable perquè la selecció pugui actuar. | Mutació massa baixa congela la població; massa alta converteix l'evolució en soroll. |
| Demografia | Població inicial, capacitat màxima operativa, criteris d'inicialització. | Afecta competència inicial, deriva i cost computacional. | Poblacions massa petites amplifiquen l'atzar; massa grans poden saturar recursos i càlcul. |
| Entorn | Freqüència de canvis, intensitat de canvis, zones favorables/desfavorables. | Permet provar adaptabilitat i resposta a pertorbacions. | Canvis massa freqüents impedeixen acumulació; canvis massa febles no seleccionen res. |

## 5. Estat funcional desitjat

El calibratge ha de buscar una zona intermèdia entre tres estats dolents: creixement indefinit, col·lapse permanent i aleatorietat sense estructura. El model útil és aquell en què la població pot persistir, fluctuar, respondre a pressions i generar diferències entre llinatges.

| Estat observat | Interpretació | Resposta funcional |
| --- | --- | --- |
| Creixement sense límit | Els recursos o els costos són massa permissius; la reproducció és massa fàcil. | Reduir recursos, augmentar costos basals o reproductius, o elevar el llindar de reproducció. |
| Extinció ràpida en gairebé totes les llavors | L'entorn és massa hostil o el metabolisme massa car. | Augmentar recursos inicials o regeneració, reduir cost basal, reduir llindar reproductiu o revisar inicialització. |
| Població estable però sense substitució generacional | Els agents sobreviuen però no hi ha prou naixements o morts per generar evolució. | Ajustar edat màxima, costos i reproducció perquè hi hagi renovació. |
| Molts naixements però cap llinatge persistent | La mutació pot ser massa forta o la selecció massa caòtica. | Reduir magnitud de mutació o estabilitzar parcialment l'entorn. |
| Un sol llinatge domina massa aviat | Pot haver-hi coll d'ampolla, poca diversitat inicial o recursos massa concentrats. | Augmentar població inicial, diversitat genètica inicial o variar distribució de recursos. |
| Resultats molt diferents entre llavors | Alta dependència de l'atzar; cal més repetició abans d'interpretar. | Executar més llavors i separar variabilitat estadística de defecte de model. |

## 6. Protocol d'ajust

El procediment recomanat és seqüencial i conservador. Ajustar molts paràmetres alhora pot donar una configuració aparentment millor, però impedeix saber quin canvi ha produït l'efecte observat.

1. Seleccionar una configuració base registrada i repetible.
2. Executar diverses llavors amb la configuració base.
3. Identificar el problema principal: col·lapse, explosió, manca de reproducció, manca de diversitat o inestabilitat excessiva.
4. Triar una sola família de paràmetres a modificar.
5. Aplicar canvis petits i documentats, evitant salts bruscos.
6. Reexecutar el mateix conjunt de llavors o un conjunt equivalent.
7. Comparar mètriques abans i després del canvi.
8. Acceptar, revertir o refinar l'ajust segons criteris mesurables.
9. Registrar la nova configuració només si millora la zona funcional del model.

## 7. Ordre recomanat de calibratge

No tots els paràmetres tenen el mateix paper. El calibratge ha de començar pels elements que determinen si la població pot existir i només després ajustar la qualitat evolutiva del sistema.

| Ordre | Bloc | Objectiu |
| --- | --- | --- |
| 1 | Recursos i metabolisme basal | Comprovar que els agents poden sobreviure prou temps, però no indefinidament sense alimentar-se. |
| 2 | Alimentació i moviment | Assegurar que moure's té cost i que trobar recursos genera avantatge real. |
| 3 | Reproducció | Ajustar llindars perquè hi hagi naixements sense explosió demogràfica. |
| 4 | Mort i edat màxima | Garantir renovació poblacional i evitar bloquejos per individus persistents. |
| 5 | Mutació | Introduir variació suficient però no destructiva. |
| 6 | Ritme ambiental | Afegir pressió variable només quan el règim bàsic ja sigui estable. |

## 8. Criteris quantitatius de comparació

Els ajustos s'han d'avaluar amb mètriques comparables. No cal una estadística complexa en aquesta fase, però sí una base mínima per evitar interpretacions impressionistes.

- percentatge d'execucions que acaben en extinció;
- temps mitjà fins a l'extinció, si n'hi ha;
- població mitjana i variància poblacional;
- nombre mitjà de naixements i morts per interval;
- edat mitjana i distribució d'edats;
- energia mitjana i percentatge d'agents propers a la fam;
- consum total de recursos i recursos disponibles residuals;
- diversitat genètica mitjana;
- nombre de llinatges persistents;
- descendència acumulada dels llinatges principals;
- resposta de la població abans i després de canvis ambientals.

## 9. Ajust de la mutació

La mutació és especialment delicada. No s'ha d'utilitzar per corregir problemes ecològics. Si la població mor perquè no hi ha recursos suficients, augmentar la mutació no resol el problema; només afegeix soroll.

| Símptoma | Possible causa | Ajust recomanat |
| --- | --- | --- |
| Genomes gairebé idèntics després de moltes generacions | Taxa o magnitud de mutació massa baixa. | Augmentar lleugerament la probabilitat o magnitud de mutació. |
| Descendents sistemàticament pitjors que progenitors | Mutacions massa destructives. | Reduir magnitud de mutació i aplicar límits als pesos. |
| Canvis de comportament erràtics entre generacions | Soroll mutacional excessiu o massa paràmetres heretables muten alhora. | Separar mutació neuronal de mutació de paràmetres heretables i reduir-ne l'abast. |
| Cap millora observable malgrat selecció | Variació insuficient o pressió selectiva mal definida. | Primer revisar pressió ecològica; després ajustar mutació. |

## 10. Ajust de recursos i metabolisme

La relació entre recursos i metabolisme defineix la duresa de l'ecosistema. Abans d'analitzar adaptació, cal comprovar que la supervivència no sigui gratuïta i que la mort per energia sigui funcional.

- Si els agents sobreviuen sense moure's ni alimentar-se: el cost basal és massa baix o l'energia inicial massa alta.
- Si moure's sempre és perjudicial: el cost de moviment pot ser massa alt en relació amb l'energia obtinguda dels recursos.
- Si alimentar-se no canvia el destí dels agents: l'energia dels recursos és massa baixa o el metabolisme massa agressiu.
- Si tots els agents es concentren en una sola zona: la distribució de recursos pot ser massa extrema per a una primera fase.
- Si els recursos mai s'esgoten: la taxa de regeneració o la capacitat màxima són massa altes.

## 11. Ajust de reproducció i mida poblacional

La reproducció ha de ser possible però costosa. Si és massa fàcil, la simulació queda dominada per creixement demogràfic. Si és massa difícil, no hi ha generacions suficients per observar evolució.

| Paràmetre | Quan augmentar-lo | Quan reduir-lo |
| --- | --- | --- |
| Llindar reproductiu | Quan hi ha explosió de naixements o reproducció massa primerenca. | Quan gairebé cap agent arriba a reproduir-se. |
| Cost reproductiu | Quan reproduir-se no té conseqüències energètiques reals. | Quan els progenitors moren sistemàticament després de reproduir-se. |
| Energia inicial del descendent | Quan els descendents moren immediatament sense marge funcional. | Quan els descendents ja neixen amb massa avantatge. |
| Població inicial | Quan la deriva domina per mostres massa petites. | Quan el món comença saturat i col·lapsa abans de qualsevol dinàmica evolutiva. |
| Capacitat màxima operativa | Quan el tall demogràfic interfereix amb resultats normals. | Quan el cost computacional o l'explosió poblacional impedeixen executar. |

## 12. Ajust del ritme ambiental

Els canvis ambientals només s'han d'introduir de manera significativa quan el sistema ja funciona en un entorn relativament estable. Si s'introdueixen massa aviat, serà difícil saber si el col·lapse ve d'una mala economia energètica o d'una pertorbació excessiva.

- començar amb entorn estable o quasi estable;
- introduir canvis suaus i poc freqüents;
- mesurar recuperació poblacional després del canvi;
- comparar llinatges abans i després de la pertorbació;
- evitar canvis tan forts que reiniciïn la selecció a cada cicle.

## 13. Registre obligatori dels ajustos

Cada canvi de paràmetres ha de quedar registrat com una nova configuració, no com una modificació informal. Sense aquest registre, els resultats posteriors no seran comparables.

| Camp | Descripció |
| --- | --- |
| Identificador de configuració | Codi únic per distingir cada conjunt de paràmetres. |
| Configuració base | Configuració de la qual deriva l'ajust. |
| Paràmetres modificats | Nom, valor anterior i valor nou. |
| Motiu del canvi | Problema detectat i hipòtesi que justifica l'ajust. |
| Execucions comparades | Llavors, durada i condicions dels experiments usats per decidir. |
| Resultat de l'ajust | Acceptat, rebutjat o pendent de més proves. |
| Notes d'interpretació | Limitacions, anomalies i riscos detectats. |

## 14. Antipatrons que s'han d'evitar

Els errors de calibratge poden fer que el projecte sembli avançar quan en realitat només s'estan perseguint resultats visuals agradables. Cal evitar explícitament aquests antipatrons:

- canviar diversos paràmetres alhora sense motiu clar;
- ajustar valors mirant només una simulació;
- triar la llavor que dona el resultat més interessant;
- confondre estabilitat poblacional amb adaptació;
- augmentar complexitat per compensar una economia energètica mal calibrada;
- afegir una funció de fitness externa perquè el model no selecciona prou;
- guardar només els experiments reeixits i descartar els col·lapses sense registrar-los;
- interpretar com a evolució el que pot ser deriva per poblacions massa petites.

## 15. Resultat esperat del calibratge

Després del primer procés d'ajust, el model hauria de tenir una configuració base funcional que permeti executar experiments de durada raonable sense col·lapse immediat ni creixement sense límit. Aquesta configuració no serà definitiva, però servirà com a punt de referència per comparar futurs canvis.

El resultat correcte no és una població sempre estable. En un sistema evolutiu, les fluctuacions, extincions parcials i canvis de dominància poden ser informatius. El que cal evitar és que tots els escenaris acabin en el mateix resultat trivial: supervivència garantida, extinció automàtica o soroll sense llinatges persistents.

## 16. Criteris de tancament

Aquest punt es considerarà complet quan es compleixin les condicions següents:

- existeix una configuració base registrada i repetible;
- s'han executat comparacions amb diverses llavors;
- els principals paràmetres ajustables tenen rangs funcionals preliminars;
- els canvis fets tenen justificació i resultat registrat;
- la població no creix indefinidament en condicions normals;
- la població no col·lapsa sistemàticament en condicions normals;
- hi ha naixements, morts i substitució generacional suficient;
- la diversitat genètica no desapareix immediatament ni es converteix en soroll total;
- l'equip pot explicar per què s'ha acceptat cada ajust rellevant.

## 17. Decisió funcional recomanada

Per a la primera fase, és recomanable mantenir una configuració base conservadora: un sol tipus de recurs, costos energètics simples, reproducció asexual amb llindar energètic clar, mutació petita en pesos i biaixos, i canvis ambientals suaus només després de validar l'estabilitat bàsica.

Aquesta decisió pot semblar poc ambiciosa, però és la més robusta. Un model senzill però calibrat donarà resultats més interpretables que un model ric en paràmetres que no es pugui diagnosticar.
