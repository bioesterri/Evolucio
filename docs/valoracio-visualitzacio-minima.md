# Valoració d'adequació — Visualització mínima

## Objectiu de la valoració

Aquest document comprova si l'especificació **Visualització mínima** s'adequa a la resta de la documentació funcional del Projecte Evolució i identifica punts forts, encaixos, riscos i recomanacions d'implementació.

## Veredicte resumit

La valoració és **molt positiva**: el document encaixa amb el full de ruta del projecte perquè defineix la visualització com una capa de diagnòstic desacoblada del motor, no com una font d'evidència científica. Aquesta distinció és coherent amb la resta de documents, que prioritzen mètriques, proves, persistència i traçabilitat abans d'interpretar adaptació.

El document és especialment adequat com a punt 16 del pla funcional perquè concreta una necessitat ja prevista: poder veure el món, els agents, els recursos i les mètriques principals sense convertir la interfície en una dependència del nucli de simulació. També manté bé l'abast del prototip: imatges, animacions simples i dashboards estàtics abans d'una interfície interactiva completa.

## Encaix amb la documentació existent

| Document relacionat | Encaix observat | Valoració |
| --- | --- | --- |
| `document-funcional-inicial.md` | Desenvolupa directament el punt 4.16, que demana una representació simple del món, els agents, els recursos i l'evolució de mètriques principals. | Molt coherent: converteix el punt del full de ruta en criteris funcionals verificables. |
| `arquitectura-general.md` | Respecta la separació entre nucli de simulació, persistència, anàlisi i visualització. | Molt coherent: la visualització consumeix estats i mètriques, però no decideix regles del món ni accions dels agents. |
| `model-mon-2d.md` | La vista espacial basada en graella, recursos i límits visibles encaixa amb el món 2D discret recomanat. | Molt coherent: facilita depuració de fronteres, regeneració de recursos i patrons geomètrics artificials. |
| `model-agent.md` | La representació d'agents vius amb posició, energia o edat opcional és compatible amb l'estat funcional mínim de l'agent. | Coherent: evita mostrar massa atributs en la vista principal. |
| `observacions-agents.md` | Diferencia clarament la vista global del desenvolupador de l'observació local disponible per a l'agent. | Molt coherent: evita confondre informació diagnòstica amb percepció de l'agent. |
| `accions-agents.md` | La visualització ajuda a detectar agents immòbils, acumulació a vores o explosions locals derivades de moviment i reproducció. | Coherent: aporta diagnòstic visual sense alterar la resolució d'accions. |
| `energia-recursos-metabolisme.md` | Els gràfics de recursos totals, energia mitjana i mapes d'abundància reforcen el control del balanç energètic. | Molt coherent: permet detectar consum excessiu, regeneració massa forta o pressió selectiva feble. |
| `mort-seleccio-ambiental.md` | Les morts es representen com a mètriques o capes temporals opcionals, no com a soroll permanent en la vista principal. | Coherent: manté la causa de mort com a dada analítica, no com a decoració visual. |
| `reproduccio-herencia-mutacio.md` | Els llinatges destacats i els rànquings de descendència encaixen amb genealogia i herència sense exigir arbres complets. | Molt coherent: limita la visualització de llinatges per evitar il·legibilitat. |
| `metriques-evolutives.md` | Converteix les mètriques principals en sèries temporals, resums finals i dashboards diagnòstics. | Molt coherent: reforça que la visualització no substitueix l'anàlisi quantitativa. |
| `registre-experiments.md` | Exigeix vincular artefactes visuals amb identificadors d'experiment. | Molt coherent: evita fitxers desordenats i facilita comparar execucions. |
| `persistencia-dades.md` | Utilitza snapshots, mètriques agregades i esdeveniments persistits com a entrada de visualització posterior. | Molt coherent: permet generar visuals sobre dades guardades sense executar el motor. |
| `proves-basiques-validacio.md` | La visualització es presenta com a ajuda per detectar errors grossos, però no com a substitut de proves automàtiques. | Molt coherent: redueix el risc d'acceptar simulacions perquè “semblen correctes”. |

## Punts forts

- **Manté el desacoblament del motor:** el motor pot executar-se sense gràfics i la visualització consumeix dades sense alterar-les.
- **Diferencia diagnòstic i evidència:** repeteix explícitament que una imatge atractiva no demostra adaptació.
- **Cobreix els dos nivells essencials:** separa vista espacial del món i visualització temporal de mètriques per evitar pantalles sobrecarregades.
- **Prioritza sortides reproduïbles:** recomana fotogrames, animacions simples i dashboards generats a partir de dades guardades.
- **Inclou llinatges amb prudència:** permet veure llinatges destacats però evita arbres genealògics complets o massa colors simultanis.
- **És útil per depurar errors grossos:** identifica símptomes visuals concrets, com agents atrapats, recursos sempre plens, extincions immediates o patrons geomètrics artificials.
- **Encaixa amb persistència i registre:** demana artefactes associats a l'identificador d'experiment i generables posteriorment.

## Riscos o punts a vigilar

1. **Confondre una animació convincent amb validació científica.**  
   Cal mantenir dashboards i imatges sempre vinculats a mètriques, proves i configuracions d'experiment.

2. **Incrementar massa aviat l'abast interactiu.**  
   Una interfície web completa, filtres avançats o reproducció pas a pas podrien retardar el motor i la validació funcional.

3. **Guardar massa snapshots.**  
   Les animacions denses poden créixer ràpidament en volum. Cal definir una freqüència de mostreig explícita per experiment.

4. **Acoblar formats visuals al nucli.**  
   El motor no ha de saber si la sortida final és PNG, SVG, GIF, HTML o un dashboard; només ha de produir estats, mètriques i esdeveniments.

5. **Saturar la vista de llinatges.**  
   Mostrar tots els llinatges alhora pot fer que la visualització perdi valor diagnòstic. S'han de mostrar només llinatges seleccionats per criteris objectius.

6. **Consumir aleatorietat funcional durant la visualització.**  
   Si la generació visual fa servir aleatorietat per colors o mostreig, aquesta aleatorietat ha d'estar separada de la llavor de simulació.

## Recomanacions d'implementació

- Crear un mòdul o paquet de visualització separat del nucli de simulació.
- Definir un format estable de snapshot visual amb dimensions del món, mapa de recursos, agents vius i metadades d'experiment.
- Generar primer una imatge estàtica del món per a un pas concret i un dashboard final amb població, energia, naixements, morts i recursos.
- Fer que la visualització pugui treballar sobre dades persistides, no només sobre objectes vius del motor.
- Incloure sempre l'identificador d'experiment, seed, configuració resumida i pas temporal en els artefactes visuals.
- Separar la generació de fotogrames de la composició d'animacions per poder mostrejar cada N passos.
- Utilitzar paletes i escales consistents entre experiments comparables per reduir biaix visual.
- Afegir una nota visible als dashboards indicant que la visualització és diagnòstica i que les conclusions evolutives requereixen mètriques i proves.
- Limitar els llinatges mostrats als principals segons descendència acumulada, persistència o resposta a canvi ambiental.

## Conclusió

El document **Visualització mínima** és adequat per incorporar-se a la documentació funcional del Projecte Evolució. Completa el punt 16 del pla inicial amb un abast realista i coherent amb l'arquitectura modular del projecte.

La recomanació és acceptar-lo com a especificació funcional inicial de la capa visual. La seva implementació hauria de començar per artefactes estàtics i reproduïbles vinculats a experiments registrats: una vista espacial del món, un dashboard temporal de mètriques i una visualització limitada de llinatges destacats. La interfície interactiva completa ha de continuar fora del prototip inicial fins que el motor, les mètriques, la persistència i les proves bàsiques siguin estables.
