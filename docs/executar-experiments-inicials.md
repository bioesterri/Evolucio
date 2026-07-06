# Projecte Evolució — Executar experiments inicials

**Especificació funcional per definir com executar els primers escenaris controlats per entendre el comportament del model.**

## Metadades

| Camp | Valor |
| --- | --- |
| Nom del document | Executar experiments inicials |
| Tipus de document | Especificació funcional per al prototip inicial |
| Punt del full de ruta | Punt 17 del pla funcional |
| Àmbit | Prototip funcional de vida artificial evolutiva |
| Fase | Validació experimental inicial |
| Propòsit | Definir com executar els primers escenaris controlats per entendre el comportament del model. |
| Nivell de detall | Funcional, sense implementació de codi ni optimització avançada. |
| Estat | Versió funcional per a revisió interna |
| Versió | 1.0 |
| Data | Juliol de 2026 |

## Índex

1. [Objectiu](#1-objectiu)
2. [Principi rector](#2-principi-rector)
3. [Condicions prèvies](#3-condicions-prèvies)
4. [Objectius dels experiments inicials](#4-objectius-dels-experiments-inicials)
5. [Abast dels experiments inicials](#5-abast-dels-experiments-inicials)
6. [Conjunt mínim d'escenaris](#6-conjunt-mínim-descenaris)
7. [Protocol general d'execució](#7-protocol-general-dexecució)
8. [Configuració base recomanada](#8-configuració-base-recomanada)
9. [Variables experimentals principals](#9-variables-experimentals-principals)
10. [Repeticions i llavors aleatòries](#10-repeticions-i-llavors-aleatòries)
11. [Dades que s'han de registrar](#11-dades-que-shan-de-registrar)
12. [Mètriques d'avaluació dels experiments inicials](#12-mètriques-davaluació-dels-experiments-inicials)
13. [Interpretació dels resultats](#13-interpretació-dels-resultats)
14. [Controls necessaris](#14-controls-necessaris)
15. [Errors que els experiments inicials han de detectar](#15-errors-que-els-experiments-inicials-han-de-detectar)
16. [Criteris per considerar una configuració viable](#16-criteris-per-considerar-una-configuració-viable)
17. [Resultats esperats de la fase](#17-resultats-esperats-de-la-fase)
18. [Criteri de tancament](#18-criteri-de-tancament)
19. [Risc principal](#19-risc-principal)
20. [Conclusió funcional](#20-conclusió-funcional)

## 1. Objectiu

L'objectiu d'aquest document és definir com s'han d'executar els experiments inicials del Projecte Evolució. Aquests experiments no busquen encara descobrir comportaments complexos ni demostrar adaptació robusta. La seva funció principal és comprovar que el motor de simulació produeix dinàmiques coherents, repetibles i interpretables sota configuracions simples.

El resultat esperat és un primer conjunt de dades comparables que permeti detectar configuracions inviables, col·lapses poblacionals previsibles, estabilitats artificials, errors de disseny i sensibilitats excessives als paràmetres.

**Decisió funcional:** els experiments inicials han de ser petits, controlats i repetibles. Una execució espectacular però no explicable no és útil en aquesta fase.

## 2. Principi rector

Els primers experiments han de servir per validar el comportament del sistema, no per buscar resultats impressionants. Encara no és el moment de fer simulacions llargues amb molts agents, múltiples recursos o condicions ambientals complexes.

El principi rector és: **canviar pocs paràmetres cada vegada, registrar-ho tot i interpretar els resultats amb prudència**. Si una diferència només apareix en una llavor aleatòria, no s'ha de considerar un patró evolutiu estable.

## 3. Condicions prèvies

Abans d'executar experiments inicials, el sistema ha de complir unes condicions mínimes:

- el bucle principal de simulació s'executa sense intervenció manual;
- les regles d'energia, recursos, reproducció i mort ja han passat proves bàsiques;
- cada execució registra configuració, llavor aleatòria i versió del model;
- les mètriques evolutives mínimes ja es generen de manera automàtica;
- la visualització mínima permet detectar errors grossos;
- els resultats rellevants persisteixen més enllà de la memòria temporal.

Si aquestes condicions no es compleixen, executar experiments llargs seria prematur. El risc seria interpretar com a biologia artificial allò que en realitat és un error de simulació.

## 4. Objectius dels experiments inicials

Els experiments inicials han de respondre preguntes concretes:

- la població pot sobreviure durant un nombre mínim de passos sense col·lapsar immediatament?
- la població pot col·lapsar quan els recursos són clarament insuficients?
- la població creix indefinidament si els recursos són massa abundants?
- les mutacions introdueixen variació sense destruir completament la continuïtat dels llinatges?
- les diferències entre llavors aleatòries són raonables o dominen completament el resultat?
- els canvis ambientals simples produeixen respostes mesurables?
- les mètriques permeten distingir estabilitat, col·lapse, expansió i adaptació aparent?

## 5. Abast dels experiments inicials

Els experiments inicials inclouen escenaris controlats amb:

- una sola espècie d'agent;
- un únic tipus de recurs energètic;
- un món 2D simple;
- observacions locals limitades;
- accions bàsiques ja definides;
- reproducció asexual amb mutació;
- morts registrades per causa;
- mètriques poblacionals, energètiques, genealògiques i genètiques bàsiques.

Queden fora d'aquesta fase:

- optimització automàtica de paràmetres;
- cerca exhaustiva d'hiperparàmetres;
- entorns amb múltiples recursos;
- reproducció sexual;
- aprenentatge individual;
- interaccions socials complexes;
- visualitzacions avançades;
- conclusions científiques fortes sobre adaptació.

## 6. Conjunt mínim d'escenaris

La primera bateria d'experiments ha de cobrir diferents règims del sistema: inviabilitat, supervivència mínima, estabilitat relativa, expansió excessiva i resposta a canvi ambiental.

| Codi | Escenari | Finalitat |
| --- | --- | --- |
| E00 | Prova fum | Comprovar que el sistema executa pocs passos sense errors ni inconsistències. |
| E01 | Sense reproducció | Validar consum energètic, mort i esgotament poblacional sense naixements. |
| E02 | Recursos escassos | Comprovar que la manca de recursos genera col·lapse o forta selecció. |
| E03 | Recursos moderats | Buscar una zona on la població pugui mantenir-se sense créixer indefinidament. |
| E04 | Recursos abundants | Detectar si hi ha creixement explosiu o falta de mecanismes limitants. |
| E05 | Mutació desactivada | Establir una línia base sense variació genètica acumulativa. |
| E06 | Mutació baixa | Comprovar continuïtat de llinatges amb variació petita. |
| E07 | Mutació alta | Detectar si l'excés de mutació converteix el sistema en soroll. |
| E08 | Canvi ambiental simple | Comprovar si el sistema respon a una variació de recursos o condicions. |
| E09 | Repetició amb llavors | Mesurar variabilitat entre execucions equivalents. |

## 7. Protocol general d'execució

Cada experiment ha de seguir un protocol comú per evitar comparacions contaminades:

1. Definir una configuració base documentada.
2. Assignar una llavor aleatòria explícita.
3. Executar una simulació curta de validació.
4. Verificar que no hi ha errors de consistència.
5. Executar la durada experimental prevista.
6. Registrar mètriques per pas temporal i resum final.
7. Guardar artefactes rellevants: configuració, resultats, millors genomes, genealogies i gràfics bàsics.
8. Repetir l'escenari amb diverses llavors quan el resultat sigui rellevant.
9. Comparar amb escenaris control equivalents.
10. Classificar el resultat com a col·lapse, estabilitat, expansió, deriva o adaptació aparent.

## 8. Configuració base recomanada

La configuració base no ha de ser considerada òptima. Ha de ser només un punt de partida prou simple per poder interpretar els efectes dels canvis.

| Paràmetre | Criteri funcional inicial |
| --- | --- |
| Mida del món | Petita o mitjana, suficient per observar competència local sense cost excessiu. |
| Població inicial | Moderada, evitant tant l'extinció immediata per baixa densitat com la saturació inicial. |
| Recursos inicials | Distribució irregular però controlada. |
| Regeneració de recursos | Baixa o moderada, per evitar supervivència gratuïta. |
| Energia inicial dels agents | Suficient per explorar, insuficient per reproduir-se indefinidament sense alimentar-se. |
| Cost metabòlic basal | Sempre positiu. |
| Cost de moviment | Positiu, però no tan alt que impedeixi exploració. |
| Cost de reproducció | Elevat respecte de l'energia mitjana disponible. |
| Taxa de mutació | Inicialment baixa i configurable. |
| Durada | Curta per validació; mitjana per primers resultats comparables. |

## 9. Variables experimentals principals

Els experiments inicials només han de variar un conjunt reduït de factors:

| Variable | Què permet avaluar | Risc si es configura malament |
| --- | --- | --- |
| Densitat inicial d'agents | Competència local i risc d'extinció inicial. | Una densitat massa alta pot provocar col·lapse artificial; massa baixa pot eliminar interacció. |
| Quantitat de recursos | Pressió selectiva per alimentació. | Recursos excessius eliminen selecció; recursos insuficients maten tota la població. |
| Regeneració de recursos | Capacitat de sostenir població. | Regeneració massa alta crea creixement gratuït. |
| Cost metabòlic | Pressió energètica de fons. | Cost massa baix genera immobilisme viable; cost massa alt impedeix supervivència. |
| Cost de moviment | Valor adaptatiu d'explorar o estalviar energia. | Si és massa barat, moure's sempre pot dominar; si és massa car, immobilitza agents. |
| Llindar reproductiu | Dificultat per deixar descendència. | Llindar massa baix crea explosió; massa alt impedeix evolució. |
| Taxa de mutació | Variació hereditària. | Massa baixa limita innovació; massa alta destrueix continuïtat. |
| Canvi ambiental | Resposta del sistema a pertorbacions. | Canvis massa bruscos poden confondre adaptació amb extinció inevitable. |

## 10. Repeticions i llavors aleatòries

Cap resultat important s'ha d'interpretar a partir d'una sola llavor aleatòria. Les primeres execucions poden ser úniques per depuració, però qualsevol escenari considerat rellevant ha de repetir-se amb diverses llavors.

Com a criteri funcional inicial:

- una prova de fum pot usar una sola llavor;
- un escenari de validació hauria d'usar diverses llavors;
- una comparació entre configuracions ha d'usar el mateix conjunt de llavors per a cada configuració;
- els resultats s'han d'analitzar amb mitjanes, dispersió i casos extrems, no només amb una corba representativa.

## 11. Dades que s'han de registrar

Cada execució inicial ha de guardar, com a mínim:

- identificador únic de l'experiment;
- configuració completa;
- llavor aleatòria;
- versió del model i del codi;
- estat inicial agregat;
- mètriques per pas temporal;
- mètriques finals agregades;
- nombre de naixements i morts;
- causes de mort;
- consum i regeneració de recursos;
- llinatges principals;
- genomes destacats segons criteris definits;
- artefactes visuals bàsics.

Sense aquestes dades, un resultat no es podrà repetir ni comparar de manera fiable. Un experiment no registrat correctament s'ha de considerar invàlid.

## 12. Mètriques d'avaluació dels experiments inicials

| Mètrica | Interpretació funcional |
| --- | --- |
| Població viva al llarg del temps | Indica col·lapse, estabilitat o creixement excessiu. |
| Energia mitjana i distribució | Mostra si els agents sobreviuen amb marge o al límit. |
| Taxa de naixements | Indica capacitat reproductiva del sistema. |
| Taxa de morts i causes | Permet saber quina pressió domina la selecció. |
| Recursos disponibles | Mostra si el món queda esgotat o massa ric. |
| Descendència per llinatge | Permet identificar èxit reproductiu diferencial. |
| Diversitat genètica aproximada | Ajuda a distingir selecció, deriva o col·lapse de variació. |
| Resposta després de canvi ambiental | Permet detectar resiliència o extinció davant pertorbacions. |
| Variabilitat entre llavors | Mesura si els resultats són robustos o accidentalment dependents de l'atzar. |

## 13. Interpretació dels resultats

Els resultats inicials s'han de classificar amb prudència. Es proposen les categories següents:

| Categoria | Descripció | Acció recomanada |
| --- | --- | --- |
| Col·lapse immediat | La població desapareix molt aviat. | Revisar energia inicial, recursos, costos i llindars de reproducció. |
| Col·lapse tardà | La població sembla viable però acaba desapareixent. | Analitzar esgotament de recursos, manca de reproducció o mutació destructiva. |
| Estabilitat relativa | La població oscil·la dins un rang raonable. | Conservar com a configuració candidata per experiments posteriors. |
| Explosió poblacional | La població creix fins a saturar el món. | Incrementar costos, limitar recursos o revisar reproducció. |
| Deriva sense patró | Els llinatges canvien però sense diferències consistents. | Augmentar durada, revisar pressió selectiva o reduir soroll mutacional. |
| Adaptació aparent | Alguns llinatges mostren avantatge repetit. | Repetir amb més llavors i controls abans de considerar-ho adaptació real. |

## 14. Controls necessaris

Els controls són imprescindibles per no confondre evolució amb artefactes del sistema. Com a mínim cal incloure:

- control sense mutació, per veure què passa només amb la dinàmica ecològica inicial;
- control sense reproducció, per validar mortalitat i consum energètic;
- control amb recursos clarament insuficients, per comprovar que el col·lapse és possible;
- control amb recursos abundants, per detectar si el sistema té mecanismes limitants;
- control amb diverses llavors, per estimar la variabilitat aleatòria;
- control de durada curta i mitjana, per separar errors inicials de dinàmiques emergents.

## 15. Errors que els experiments inicials han de detectar

Els experiments inicials han de servir per revelar problemes com:

- agents que sobreviuen sense consumir recursos;
- agents que es reprodueixen sense cost real;
- recursos que es regeneren massa ràpid o no es consumeixen correctament;
- morts no registrades o registrades amb causa incorrecta;
- creixement poblacional sense límit funcional;
- mutacions que no s'apliquen o s'apliquen massa fort;
- llinatges que no mantenen traça genealògica;
- mètriques inconsistents amb l'estat real de la simulació;
- resultats impossibles de repetir amb la mateixa llavor.

## 16. Criteris per considerar una configuració viable

Una configuració inicial es pot considerar viable si:

- la població no s'extingeix sempre de manera immediata;
- la població tampoc creix indefinidament sense restricció;
- els recursos mostren escassetat funcional en algun moment;
- els agents necessiten alimentar-se per mantenir-se vius;
- la reproducció depèn realment de l'energia acumulada;
- els llinatges presenten diferències mesurables de descendència;
- les mutacions generen variació però no destrueixen tota continuïtat;
- els resultats són raonablement repetibles entre llavors.

## 17. Resultats esperats de la fase

Al final d'aquesta fase, l'equip hauria de disposar de:

- un conjunt d'execucions inicials registrades;
- una matriu bàsica de configuracions provades;
- identificació de configuracions inviables;
- una o més configuracions candidates per experiments més llargs;
- gràfics bàsics de població, energia, recursos i naixements/morts;
- un informe breu de patrons observats;
- una llista de paràmetres que cal ajustar abans de la fase següent.

## 18. Criteri de tancament

Aquest punt es considerarà tancat quan:

- s'hagin executat diverses configuracions inicials comparables;
- cada execució tingui llavor, configuració i versió registrades;
- les mètriques principals estiguin disponibles per anàlisi;
- hi hagi almenys un escenari de col·lapse explicable;
- hi hagi almenys un escenari de supervivència temporal o estabilitat relativa;
- s'hagi provat l'efecte de la mutació respecte d'un control sense mutació;
- s'hagi estimat la variabilitat entre llavors;
- l'equip pugui justificar quines configuracions passen a experiments posteriors i quines es descarten.

## 19. Risc principal

El risc principal és interpretar massa aviat qualsevol patró com a adaptació. Una simulació pot produir formes, oscil·lacions i llinatges dominants sense que això impliqui adaptació robusta. Per això, els experiments inicials han de ser tractats com una fase de calibratge i validació, no com una fase de demostració científica.

## 20. Conclusió funcional

Executar experiments inicials és el primer contacte real entre el disseny del model i el seu comportament dinàmic. Aquesta fase ha de permetre saber si el sistema és viable, si els paràmetres són raonables i si les mètriques permeten interpretar el que passa.

El resultat mínim acceptable no és obtenir agents aparentment intel·ligents, sinó disposar d'execucions repetibles, comparables i prou informatives per decidir com continuar desenvolupant el Projecte Evolució.
