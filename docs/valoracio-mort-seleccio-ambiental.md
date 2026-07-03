# Valoració d'adequació — Mort i selecció ambiental

## Objectiu de la valoració

Aquest document comprova si l'especificació **Mort i selecció ambiental** s'adequa a la resta de la documentació funcional del Projecte Evolució i identifica punts forts, encaixos, riscos i recomanacions d'implementació.

## Veredicte resumit

La valoració és **positiva**: el document és coherent amb l'arquitectura funcional existent i reforça el nucli evolutiu definit pels documents d'energia, reproducció, accions, model d'agent i món 2D. La proposta manté el criteri central del projecte: no introduir una funció de fitness externa, sinó deixar que la selecció emergeixi de recursos limitats, costos energètics, mortalitat i reproducció diferencial.

## Encaix amb la documentació existent

| Document relacionat | Encaix observat | Valoració |
| --- | --- | --- |
| `energia-recursos-metabolisme.md` | La mort per manca d'energia és una conseqüència directa del model energètic i dels recursos finits. | Molt coherent: converteix l'energia en mecanisme terminal i no només en estat intern. |
| `reproduccio-herencia-mutacio.md` | L'ordre temporal proposat evita que agents morts o inviables es reprodueixin, i relaciona mortalitat amb descendència i llinatge. | Molt coherent: reforça la reproducció com a resultat de supervivència, no com a premi extern. |
| `accions-agents.md` | Els costos d'acció i moviment poden generar mortalitat indirecta si l'agent actua de manera energèticament ineficient. | Coherent: les accions tenen conseqüències acumulades i seleccionables. |
| `model-agent.md` | La distinció entre agent viu, agent mort i registre històric encaixa amb la necessitat de mantenir identitat, estat i genealogia. | Coherent: caldrà assegurar que l'estat mort queda separat del conjunt actiu. |
| `model-mon-2d.md` | La competència indirecta per recursos i espai viable encaixa amb un món discret amb recursos locals. | Coherent: no requereix depredació ni combat per generar pressió ecològica. |
| `genoma-xarxa-neuronal.md` | La selecció opera sobre diferències de comportament produïdes pel genoma i expressades per la xarxa, sense puntuar genomes explícitament. | Coherent: preserva el principi d'emergència evolutiva. |
| `observacions-agents.md` | La mortalitat dependrà indirectament de la qualitat de les observacions disponibles per trobar recursos i evitar zones costoses. | Coherent: les observacions continuen sent informació, no fitness. |
| `arquitectura-general.md` | La proposta demana esdeveniments de mort, mètriques agregades i traçabilitat, cosa alineada amb una arquitectura modular i auditable. | Coherent: introdueix un contracte clar per al motor de simulació i el subsistema de mètriques. |
| `abast-prototip-inicial.md` | Les exclusions temporals mantenen el prototip simple i eviten afegir soroll abans de validar energia, reproducció i llinatges. | Molt coherent: evita depredació, malalties i catàstrofes prematures. |

## Punts forts

- **Coherència ecològica:** la mort s'explica per regles internes del món, especialment energia, edat, ambient i competència indirecta.
- **Absència de fitness extern:** el document manté explícitament que la selecció no és una puntuació separada.
- **Traçabilitat:** cada mort queda definida com a esdeveniment amb causa, agent, pas temporal, energia, edat i llinatge.
- **Ordre temporal clar:** situar l'avaluació de mort abans de la reproducció evita estratègies suïcides artificials.
- **Calibratge analitzable:** els símptomes de població massa fàcil, massa dura o massa aleatòria tenen ajustos funcionals concrets.
- **Compatibilitat amb el prototip inicial:** no introdueix mecanismes complexos com combat, malalties o cadenes tròfiques.

## Riscos o punts a vigilar

1. **Conflicte potencial entre mort i reproducció dins el pas temporal.**  
   Cal implementar de manera estricta l'ordre proposat: accions, alimentació, metabolisme, mort, retirada d'agents morts i només després reproducció vàlida.

2. **Excés de morts per edat màxima.**  
   Si domina `MAX_AGE`, el sistema indicaria que l'economia energètica no exerceix prou pressió selectiva.

3. **Ambient massa terminal.**  
   `ENV_STRESS` hauria de començar com a modificador gradual de costos o recursos, no com a eliminació massiva.

4. **Ambigüitat de `COMPETITIVE_EXCL`.**  
   En el prototip inicial és millor registrar-la com a causa secundària o factor contextual, perquè la mort funcional final serà sovint `ENERGY_DEPLETION`.

5. **Dades postmortem insuficients.**  
   Sense `lineage_id`, `offspring_count`, `genome_id` i `environment_state`, la mortalitat seria difícil d'interpretar evolutivament.

## Recomanacions d'implementació

- Crear una enumeració estable de causes de mort amb els codis `ENERGY_DEPLETION`, `MAX_AGE`, `ENV_STRESS`, `COMPETITIVE_EXCL` i `INVALID_STATE`.
- Implementar una funció única de resolució de mort que apliqui la prioritat definida i retorni causa principal i causes secundàries.
- Separar clarament el conjunt d'agents actius del registre històric d'agents morts.
- Registrar cada mort com a esdeveniment immutable per permetre reproducció d'experiments i anàlisi posterior.
- Afegir mètriques agregades per pas temporal i per llinatge abans d'interpretar resultats evolutius.
- Tractar l'ambient com a modificador de cost, regeneració o risc fins que el metabolisme i la reproducció estiguin calibrats.

## Conclusió

El document **Mort i selecció ambiental** és adequat per incorporar-se a la documentació funcional del Projecte Evolució. Completa una peça imprescindible del model: defineix per què els agents deixen d'actuar, com es registra aquesta transició i com aquesta mortalitat es converteix en pressió selectiva sense introduir una funció de fitness externa.

La seva incorporació és especialment valuosa perquè connecta els documents d'energia i reproducció amb les mètriques evolutives. La recomanació és acceptar-lo com a especificació funcional inicial i utilitzar-lo com a contracte per implementar el subsistema de mortalitat, registre postmortem i anàlisi de selecció ambiental.
