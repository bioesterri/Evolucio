# Projecte Evolució — Document funcional inicial

**Full de ruta per construir un prototip de vida artificial amb agents neuronals, reproducció asexual, mutació i selecció ambiental emergent.**

## Metadades

| Camp | Valor |
| --- | --- |
| Destinatari | CTO, direcció tècnica o equip fundador |
| Versió | 0.1 |
| Data | 1 de juliol de 2026 |
| Estat | Document funcional inicial |

## Propòsit executiu

Aquest document defineix les tasques funcionals d’alt nivell que cal completar per arribar a un primer prototip robust, mesurable i repetible.

No és encara una especificació tècnica ni una arquitectura de codi; és el marc de treball que permet alinear abast, prioritats i criteris d’èxit.

## 1. Visió funcional

El Projecte Evolució té com a objectiu construir una simulació de vida artificial on la pressió selectiva neixi de l’entorn i de la interacció entre agents, no de conductes preprogramades.

El sistema ha de permetre observar si determinats llinatges aconsegueixen més supervivència, més descendència o més adaptabilitat davant canvis ambientals.

La primera fase no busca intel·ligència complexa ni comportaments espectaculars. Busca una base funcional sòlida:

- agents simples;
- món clar;
- regles energètiques coherents;
- herència amb mutació;
- mètriques traçables;
- experiments repetibles.

## 2. Principis de decisió

- Construir primer una simulació mínima però completa abans d’afegir complexitat.
- No programar estratègies intel·ligents dins dels agents; les conductes han d’emergir de la selecció.
- Separar clarament món, agents, genoma, xarxa neuronal, reproducció, mètriques, persistència i visualització.
- Fer que cada experiment sigui repetible: configuració, llavor aleatòria, resultats i genealogia han de quedar registrats.
- Mesurar adaptació amb dades, no amb impressions visuals.

## 3. Mapa de treball

| Bloc | Objectiu | Resultat funcional |
| --- | --- | --- |
| Abast i arquitectura | Fixar què entra i què queda fora del primer prototip. | Equip alineat i risc de dispersió reduït. |
| Model ecològic | Definir món, agents, observacions, accions, energia i recursos. | Simulació coherent i tancada funcionalment. |
| Evolució | Implementar reproducció, herència, mutació, mort i selecció ambiental. | Diferències entre llinatges mesurables. |
| Execució i dades | Registrar mètriques, configuracions i resultats. | Experiments comparables i repetibles. |
| Validació | Provar, visualitzar, executar i ajustar. | Base preparada per ampliar sense reescriure-ho tot. |

## 4. Tasques funcionals d’alt nivell

### 4.1 Definir l’abast del prototip inicial

**Definició funcional:** fixar quin és el mínim sistema complet que ha de funcionar: món, agents, recursos, energia, reproducció, mort, mutació, mètriques i registre. També cal deixar explícit què no es farà encara.

**Resultat esperat:** un prototip reduït però executable que permeti estudiar evolució poblacional sense carregar el sistema amb funcionalitats secundàries.

**Criteri de tancament:** hi ha una llista clara d’objectius inclosos, exclusions temporals i criteris mínims perquè la primera versió es consideri útil.

### 4.2 Dissenyar l’arquitectura general del projecte

**Definició funcional:** organitzar el projecte en blocs funcionals independents: simulació, entorn, agent, genoma, xarxa neuronal, reproducció, mètriques, persistència, anàlisi i visualització.

**Resultat esperat:** una estructura que permeti canviar una part del sistema sense trencar-ne la resta i que faciliti proves, manteniment i ampliacions.

**Criteri de tancament:** l’equip pot explicar on viu cada responsabilitat i quines dades circulen entre blocs sense entrar encara en implementació detallada.

### 4.3 Definir el model del món 2D

**Definició funcional:** descriure l’espai on viuen els agents: dimensions, límits, distribució de recursos, canvis ambientals i condicions que afecten la supervivència.

**Resultat esperat:** un entorn simple però ecològicament significatiu, capaç de generar pressió selectiva real sobre els agents.

**Criteri de tancament:** el món pot inicialitzar-se, evolucionar en el temps i afectar els agents de manera clara i mesurable.

### 4.4 Definir el model d’agent

**Definició funcional:** establir què és un agent dins la simulació: posició, energia, edat, estat intern, identificador, progenitor, genoma i estat vital.

**Resultat esperat:** agents prou simples per simular poblacions grans, però prou complets per reproduir-se, morir i deixar traça genealògica.

**Criteri de tancament:** cada agent té un estat funcional complet i pot ser seguit des del naixement fins a la mort.

### 4.5 Definir el genoma i la xarxa neuronal

**Definició funcional:** determinar com es representa la informació heretable i com aquesta informació controla les decisions de l’agent a través d’una xarxa neuronal petita.

**Resultat esperat:** un vincle clar entre herència, comportament i selecció: el genoma condiciona accions, i les accions condicionen supervivència i reproducció.

**Criteri de tancament:** el genoma és mutable, transmissible i suficient per reconstruir el sistema de decisió d’un agent.

### 4.6 Definir les observacions disponibles per als agents

**Definició funcional:** fixar quina informació local percep cada agent: recursos propers, energia pròpia, presència d’altres agents, senyals ambientals i possibles limitacions perceptives.

**Resultat esperat:** agents que decideixen amb informació parcial, cosa necessària perquè emergeixin estratègies adaptatives en lloc de respostes trivials.

**Criteri de tancament:** les observacions són limitades, consistents i no donen avantatges artificials impossibles dins del món simulat.

### 4.7 Definir les accions possibles

**Definició funcional:** establir el conjunt mínim d’accions: moure’s, alimentar-se, reproduir-se i, si és necessari, competir o interactuar amb altres agents.

**Resultat esperat:** un espai d’acció reduït però suficient perquè apareguin diferències de supervivència entre agents i llinatges.

**Criteri de tancament:** cada acció té un efecte clar sobre el món, l’energia de l’agent o altres agents, i cap acció codifica una estratègia intel·ligent explícita.

### 4.8 Definir energia, recursos i metabolisme

**Definició funcional:** formalitzar com els agents guanyen energia, com la gasten i com els recursos apareixen, desapareixen o es regeneren dins l’entorn.

**Resultat esperat:** una economia energètica que faci impossible la supervivència gratuïta i obligui els agents a competir indirectament per recursos limitats.

**Criteri de tancament:** la població no creix indefinidament sense restricció i la manca d’energia genera conseqüències funcionals.

### 4.9 Definir reproducció, herència i mutació

**Definició funcional:** descriure quan un agent pot reproduir-se, quin cost té, què rep el descendent i com s’introdueixen mutacions petites i configurables.

**Resultat esperat:** una dinàmica evolutiva real: els descendents s’assemblen als progenitors però poden variar prou per permetre selecció acumulativa.

**Criteri de tancament:** cada naixement registra progenitor, genoma heretat, mutacions aplicades i condicions mínimes de reproducció.

### 4.10 Definir mort i selecció ambiental

**Definició funcional:** establir les causes de mort: falta d’energia, edat, condicions ambientals, competència o altres pressions definides pel model.

**Resultat esperat:** la selecció emergeix perquè no tots els agents sobreviuen ni es reprodueixen amb la mateixa probabilitat.

**Criteri de tancament:** les morts són explicables per causes del sistema i es poden agregar com a mètrica d’anàlisi.

### 4.11 Implementar el bucle principal de simulació

**Definició funcional:** definir l’ordre funcional de cada pas temporal: actualitzar entorn, calcular observacions, decidir accions, aplicar efectes, resoldre naixements i morts, i registrar estat.

**Resultat esperat:** un cicle complet i repetible que converteixi regles individuals simples en dinàmiques poblacionals observables.

**Criteri de tancament:** la simulació pot executar-se durant molts passos sense intervenció manual i mantenint consistència de dades.

### 4.12 Implementar mètriques evolutives

**Definició funcional:** definir quines dades indiquen si hi ha evolució o només soroll: supervivència, descendència, diversitat genètica, consum energètic, estabilitat poblacional i resposta a canvis.

**Resultat esperat:** un sistema d’avaluació objectiu per comparar experiments, configuracions i llinatges.

**Criteri de tancament:** cada execució produeix mètriques suficients per valorar si hi ha adaptació mesurable.

### 4.13 Implementar registre d’experiments

**Definició funcional:** garantir que cada execució guardi configuració, llavor aleatòria, versió del model, paràmetres principals, mètriques i artefactes rellevants.

**Resultat esperat:** traçabilitat completa: qualsevol resultat important s’ha de poder revisar, comparar i, idealment, repetir.

**Criteri de tancament:** no hi ha experiments “orfes”: tota execució important queda identificada i documentada.

### 4.14 Implementar persistència de dades

**Definició funcional:** decidir quines dades es guarden de manera estructurada: experiments, agents, genealogies, mètriques, millors genomes i sèries temporals massives.

**Resultat esperat:** una base de dades funcional per consultar resultats sense dependre de fitxers dispersos o memòria temporal.

**Criteri de tancament:** els resultats rellevants sobreviuen al final de la simulació i poden ser analitzats posteriorment.

### 4.15 Crear proves bàsiques de validació

**Definició funcional:** definir comprovacions mínimes que evitin errors silenciosos: conservació de regles, costos energètics, naixements, morts, mutacions i coherència de població.

**Resultat esperat:** confiança en el comportament del sistema abans d’interpretar qualsevol resultat evolutiu.

**Criteri de tancament:** els casos bàsics passen proves automàtiques o verificacions sistemàtiques abans d’executar experiments llargs.

### 4.16 Crear visualització mínima

**Definició funcional:** proporcionar una representació simple del món, dels agents, dels recursos i de l’evolució de mètriques principals.

**Resultat esperat:** una eina de diagnòstic ràpida per detectar errors grossos, col·lapses poblacionals o dinàmiques inesperades.

**Criteri de tancament:** es pot veure una execució o revisar-ne un resum visual sense confondre visualització amb prova científica.

### 4.17 Executar experiments inicials

**Definició funcional:** córrer escenaris controlats amb configuracions simples per comprovar si la simulació produeix dinàmiques estables, col·lapses o patrons repetibles.

**Resultat esperat:** primer conjunt de dades real per entendre el comportament del model i detectar paràmetres inviables.

**Criteri de tancament:** hi ha resultats de diverses execucions comparables, amb llavors i configuracions registrades.

### 4.18 Analitzar resultats

**Definició funcional:** examinar mètriques i genealogies per identificar si alguns llinatges tenen avantatge, si hi ha diversitat genètica i si la població respon als canvis ambientals.

**Resultat esperat:** una interpretació funcional dels experiments basada en dades, no en observacions anecdòtiques.

**Criteri de tancament:** l’equip pot distingir entre adaptació, deriva, error de model i simple atzar estadístic.

### 4.19 Ajustar paràmetres del model

**Definició funcional:** modificar de manera controlada taxes de mutació, costos energètics, densitat de recursos, ritme ambiental, mida poblacional i condicions de reproducció.

**Resultat esperat:** un model calibrat prou bé perquè no sigui ni trivial ni impossible: ha de permetre evolució observable sense col·lapse permanent.

**Criteri de tancament:** els ajustos es fan comparant experiments registrats, no canviant valors a cegues.

### 4.20 Preparar la base per a versions més complexes

**Definició funcional:** deixar el sistema llest per incorporar complexitat posterior: entorns més rics, interaccions socials, noves observacions, xarxes més grans o visualització avançada.

**Resultat esperat:** una plataforma extensible que no obligui a reescriure el prototip quan el projecte avanci.

**Criteri de tancament:** la primera versió queda tancada amb arquitectura clara, resultats mesurables i una llista prioritzada de millores futures.

## 5. Resultat esperat de la primera fase

El resultat mínim acceptable és una simulació que permeti observar una població d’agents neuronals en un entorn amb recursos limitats, on alguns llinatges sobrevisquin i es reprodueixin millor que altres per efecte de mutacions heretables i pressió ambiental.

La prova d’èxit no és que els agents semblin intel·ligents. La prova d’èxit és que el sistema generi diferències mesurables entre llinatges, que aquestes diferències siguin traçables i que els experiments es puguin repetir.

## 6. Decisions que cal evitar en aquesta fase

- Afegir aprenentatge individual abans de tenir evolució poblacional validada.
- Programar conductes aparentment intel·ligents dins les regles de l’agent.
- Crear una arquitectura massa complexa abans de demostrar que el model ecològic funciona.
- Interpretar una visualització atractiva com si fos evidència d’adaptació.
- Executar experiments sense llavor, configuració i mètriques registrades.

## 7. Documents relacionats

- [Abast del prototip inicial](abast-prototip-inicial.md): especificació funcional detallada del sistema mínim necessari per validar selecció diferencial observable, incloent components inclosos, exclusions, mètriques, registre experimental i criteris d'acceptació.
- [Bucle principal de simulació](bucle-principal-simulacio.md): especificació funcional del pas temporal discret que coordina entorn, agents, accions, reproducció, mort, mètriques, validacions i repetibilitat.
- [Mètriques evolutives](metriques-evolutives.md): especificació funcional del sistema d'indicadors per distingir evolució mesurable de soroll aleatori.
- [Registre d'experiments](registre-experiments.md): especificació funcional del sistema de traçabilitat que vincula cada execució amb configuració, llavor, versió, mètriques i artefactes.
- [Valoració del registre d'experiments](valoracio-registre-experiments.md): revisió d'adequació del registre amb la resta de documents funcionals i recomanacions d'implementació.
