# Valoració d'adequació — Proves bàsiques de validació

## Objectiu de la valoració

Aquest document comprova si l'especificació **Proves bàsiques de validació** s'adequa a la resta de la documentació funcional del Projecte Evolució i identifica punts forts, encaixos, riscos i recomanacions d'implementació.

## Veredicte resumit

La valoració és **molt positiva**: el document encaixa de manera transversal amb tota la documentació existent perquè converteix les regles funcionals del motor en criteris verificables abans d'interpretar resultats evolutius. És especialment adequat com a punt 15 del pla funcional, després del registre d'experiments i la persistència de dades, perquè estableix una barrera de qualitat prèvia als experiments llargs.

El document és coherent amb el principi del projecte: no n'hi ha prou amb observar patrons visuals o mètriques aparentment interessants; cal demostrar que el motor conserva invariants bàsics d'energia, població, recursos, reproducció, mutació, mort, genealogia i repetibilitat. Aquesta orientació redueix el risc que errors silenciosos es confonguin amb adaptació evolutiva.

## Encaix amb la documentació existent

| Document relacionat | Encaix observat | Valoració |
| --- | --- | --- |
| `document-funcional-inicial.md` | Dona resposta directa a la necessitat de validar el prototip abans d'interpretar evolució real. | Molt coherent: transforma el full de ruta en criteris de sanitat verificables. |
| `arquitectura-general.md` | Respecta la separació entre motor, proves, registre, persistència, anàlisi i visualització. | Molt coherent: valida el motor sense barrejar proves amb lògica evolutiva. |
| `abast-prototip-inicial.md` | Manté l'abast del prototip centrat en proves petites, ràpides i deterministes. | Coherent: evita validar funcionalitats futures abans d'hora. |
| `model-mon-2d.md` | Verifica límits espacials, recursos, regeneració i escassetat ambiental. | Molt coherent: cobreix els riscos principals d'entorn incoherent. |
| `model-agent.md` | Exigeix identitat, posició, energia, edat, genoma, progenitor i estat vital vàlids. | Molt coherent: converteix l'estat de l'agent en invariant testable. |
| `energia-recursos-metabolisme.md` | Prioritza costos basals, alimentació, energia màxima i mort energètica. | Molt coherent: protegeix la pressió selectiva energètica. |
| `accions-agents.md` | Defineix proves per a inacció, moviment, alimentació, reproducció i competència indirecta. | Coherent: valida efectes simples i evita estratègies ocultes. |
| `reproduccio-herencia-mutacio.md` | Cobreix condicions de naixement, cost, genealogia, herència i mutació acotada. | Molt coherent: reforça el pont entre comportament i evolució. |
| `mort-seleccio-ambiental.md` | Demana causes de mort detectables, agents morts inactius i mortalitat agregable. | Molt coherent: evita desaparicions sense explicació analítica. |
| `metriques-evolutives.md` | Contrasta mètriques contra recomptes manuals en mons petits. | Molt coherent: evita estadístiques desconnectades de l'estat real. |
| `bucle-principal-simulacio.md` | Reforça l'ordre estable del pas temporal i permet controladors de prova. | Molt coherent: detecta incoherències temporals entre observació, acció i efecte. |
| `registre-experiments.md` | Requereix registrar llavor, configuració, versió, mètriques i artefactes mínims. | Coherent: converteix la validació en prerequisit d'un run interpretable. |
| `persistencia-dades.md` | Inclou persistència mínima com a part del criteri de tancament. | Coherent: assegura que els resultats validats sobreviuen al final de l'execució. |

## Punts forts

- **Cobertura transversal:** valida configuració, món, agents, energia, accions, reproducció, mutació, mort, mètriques, persistència i repetibilitat.
- **Orientació a invariants:** prioritza identitat única, posició vàlida, energia finita, mort irreversible, genealogia i balanç poblacional.
- **Protecció contra errors silenciosos:** identifica fallades que poden semblar resultats biològicament interessants si no es detecten aviat.
- **Escenaris petits i deterministes:** proposa mons controlats com món buit, recurs únic, moviment bloquejat, reproducció possible/impossible i llavor repetida.
- **Separació entre validació i experiment:** deixa clar que aquests escenaris no són experiments evolutius, sinó proves de sanitat del motor.
- **Integració amb registre i persistència:** exigeix que la configuració, la llavor, les mètriques i els artefactes quedin disponibles per auditoria.
- **Criteri de bloqueig d'experiments llargs:** estableix que no s'haurien d'interpretar resultats si fallen invariants bàsics.

## Riscos o punts a vigilar

1. **Traduir l'especificació a proves automàtiques concretes.**  
   El document és funcionalment complet, però la implementació haurà de convertir cada invariant en tests executables amb dades petites i resultats esperats explícits.

2. **Dependència de determinisme real.**  
   La repetibilitat amb la mateixa llavor requereix controlar totes les fonts de no determinisme, incloent l'ordre d'iteració, el generador pseudoaleatori i possibles optimitzacions numèriques.

3. **Ambigüitats de regla que cal tancar abans de provar.**  
   Alguns casos permeten diverses polítiques vàlides, com moviment invàlid amb cost o sense cost, causa de mort única o múltiple, o clamping d'energia. Caldrà fixar una decisió funcional abans d'escriure tests estrictes.

4. **Risc de proves massa acoblades a la implementació.**  
   Les proves han de validar comportament i invariants, no detalls interns que podrien canviar sense afectar la funció evolutiva.

5. **Cobertura de persistència mínima.**  
   La persistència s'inclou correctament, però caldrà definir un conjunt mínim d'artefactes obligatoris perquè la validació no quedi només en memòria.

6. **Manteniment de regressions amb canvis de model.**  
   Si canvien regles del motor, els resultats esperats amb llavor fixa s'hauran d'actualitzar de manera controlada i documentada, no simplement sobrescriure.

## Recomanacions d'implementació

- Crear una suite de tests separada per capes: configuració, món, agent, energia, accions, reproducció/mutació, mort, mètriques i integració.
- Afegir fixtures de món mínim amb llavor fixa, pocs agents i recursos comptables manualment.
- Implementar asserts d'invariants després de cada pas del bucle principal, almenys en mode test o debug.
- Preparar controladors d'acció deterministes per substituir temporalment la xarxa neuronal en proves de motor.
- Definir una prova de regressió curta que executi dues vegades la mateixa configuració i compari mètriques, població, morts, naixements i hashes de genoma.
- Establir un manifest mínim de validació que indiqui quines proves han passat abans d'un experiment llarg.
- Fer que les fallades d'invariant siguin explícites i diagnòstiques: agent afectat, pas temporal, configuració, llavor i estat rellevant.
- Documentar les decisions funcionals pendents quan una prova admeti diverses polítiques vàlides.

## Conclusió

El document **Proves bàsiques de validació** és adequat per incorporar-se a la documentació funcional del Projecte Evolució com a punt 15. Complementa la resta del pla perquè converteix les especificacions prèvies en una bateria mínima de comprovacions repetibles i impedeix que experiments llargs es considerin fiables sense haver validat abans el motor.

La recomanació és acceptar-lo com a especificació funcional inicial i utilitzar-lo com a base per dissenyar la primera suite automàtica de proves del prototip. La prioritat hauria de ser implementar primer els tests d'invariants crítics: configuració, límits del món, energia, naixements, morts, mutació zero, balanç poblacional, mètriques i repetibilitat amb llavor fixa.
