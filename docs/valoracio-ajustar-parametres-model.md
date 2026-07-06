# Valoració d'adequació — Ajustar paràmetres del model

## Objectiu de la valoració

Aquest document comprova si l'especificació **Ajustar paràmetres del model** s'adequa a la resta de la documentació funcional del Projecte Evolució i identifica punts forts, encaixos, riscos i recomanacions d'implementació.

## Veredicte resumit

La valoració és **molt positiva**: el document encaixa de manera natural després de l'execució dels experiments inicials i concreta una disciplina de calibratge prudent, traçable i compatible amb el criteri general del projecte. El seu punt fort principal és que prohibeix l'ajust intuïtiu o reactiu de valors i exigeix comparacions entre execucions registrades, preferiblement amb diverses llavors i canvis petits.

És adequat com a document funcional del punt 19 perquè converteix els resultats dels primers experiments en decisions de configuració justificades. També evita dos riscos recurrents en models evolutius artificials: confondre supervivència trivial amb adaptació i confondre extinció sistemàtica amb pressió selectiva útil.

## Encaix amb la documentació existent

| Document relacionat | Encaix observat | Valoració |
| --- | --- | --- |
| `document-funcional-inicial.md` | Manté l'objectiu global de construir una simulació evolutiva observable, però insisteix que l'evolució només és interpretable si el sistema no és ni trivial ni impossible. | Molt coherent: reforça el full de ruta sense ampliar prematurament l'abast. |
| `abast-prototip-inicial.md` | Respecta l'abast del prototip: un recurs, costos simples, reproducció asexual i mutació petita abans d'introduir complexitat ambiental. | Molt coherent: evita que el calibratge es converteixi en redisseny del model. |
| `energia-recursos-metabolisme.md` | Situa recursos i metabolisme basal com el primer bloc de calibratge. | Molt coherent: reconeix que l'economia energètica és la base de la pressió selectiva. |
| `reproduccio-herencia-mutacio.md` | Tracta reproducció i mutació com mecanismes ajustables però separats, evitant usar la mutació per corregir problemes ecològics. | Molt coherent: protegeix la continuïtat de llinatges i la interpretabilitat genètica. |
| `mort-seleccio-ambiental.md` | Requereix morts, edat màxima i renovació generacional per evitar poblacions persistents sense substitució evolutiva. | Coherent: vincula mortalitat amb temps evolutiu útil. |
| `metriques-evolutives.md` | Reutilitza mètriques de població, energia, naixements, morts, diversitat, llinatges i resposta ambiental com base de comparació. | Molt coherent: converteix mètriques descriptives en criteris de decisió. |
| `registre-experiments.md` | Exigeix que cada canvi de paràmetres quedi registrat com una configuració derivada amb motiu, execucions comparades i resultat de l'ajust. | Molt coherent: impedeix experiments orfes i ajustos no auditables. |
| `persistencia-dades.md` | Necessita configuracions persistides, sèries temporals i resums per comparar abans i després dels ajustos. | Coherent: dona una finalitat clara a la persistència experimental. |
| `visualitzacio-minima.md` | La visualització pot ajudar a detectar col·lapses o explosions, però el document evita usar-la com a única evidència. | Coherent: manté la visualització com a suport diagnòstic. |
| `proves-basiques-validacio.md` | Assumeix que les regles bàsiques ja han estat validades abans d'interpretar calibratges. | Coherent: no substitueix les proves de sanitat del motor. |
| `executar-experiments-inicials.md` | Depèn directament dels experiments inicials i els converteix en comparacions de configuracions. | Molt coherent: és la continuació funcional lògica del punt 17. |
| `analitzar-resultats.md` | Complementa l'anàlisi amb decisions operatives sobre acceptar, revertir o refinar ajustos. | Coherent: transforma interpretació en governança de paràmetres. |

## Punts forts

- **Governança clara dels paràmetres:** estableix que cap canvi important s'ha de fer sense configuració base, llavors comparables, mètriques i justificació.
- **Separació funcional de famílies:** distingeix recursos, energia, reproducció, mutació, demografia i entorn, cosa que facilita diagnosticar causes.
- **Ordre de calibratge robust:** prioritza supervivència ecològica i metabolisme abans de reproducció, mutació i canvis ambientals.
- **Criteris de resposta accionables:** ofereix respostes funcionals per a creixement sense límit, extinció ràpida, manca de substitució, soroll mutacional i dominància prematura.
- **Bon tractament de la mutació:** adverteix explícitament que la mutació no ha de corregir una economia energètica mal calibrada.
- **Traçabilitat dels ajustos:** defineix camps obligatoris per registrar cada configuració derivada i el motiu de l'acceptació o rebuig.
- **Antipatrons ben identificats:** evita perseguir resultats visuals atractius, triar llavors convenients o guardar només experiments reeixits.

## Riscos o punts a vigilar

1. **Quantificar els rangs funcionals preliminars.**  
   El document defineix criteris funcionals, però la implementació haurà de convertir expressions com “petit”, “moderat” o “massa alt” en rangs versionats i comparables.

2. **Evitar calibratge excessiu sobre una bateria curta.**  
   Una configuració pot semblar equilibrada en execucions curtes i fallar en simulacions llargues. Les configuracions acceptades inicialment s'han d'etiquetar com a base provisional.

3. **Mantenir canvis d'una sola família sempre que sigui possible.**  
   En alguns casos, recursos i metabolisme o reproducció i energia estan acoblats. Quan calgui tocar més d'una família, s'haurà de justificar explícitament per no perdre interpretabilitat.

4. **Definir criteris mínims d'acceptació.**  
   El document enumera mètriques, però caldrà establir llindars operatius inicials per decidir quan una configuració queda acceptada, rebutjada o pendent.

5. **Controlar la deriva per poblacions petites.**  
   El document detecta el risc, però els primers experiments hauran de diferenciar bé entre selecció real, deriva i efectes de coll d'ampolla.

## Recomanacions d'implementació

- Crear un catàleg de configuracions versionades amb camps `config_id`, `base_config_id`, paràmetres modificats, motiu, conjunt de llavors i estat de decisió.
- Definir una bateria estàndard de llavors per calibratge, reutilitzable en comparacions entre configuracions.
- Generar automàticament un resum comparatiu abans/després amb extincions, temps fins a extinció, població mitjana, naixements, morts, energia, recursos i diversitat.
- Separar formalment els perfils de calibratge ecològic, reproductiu, mutacional i ambiental.
- Marcar com a invàlids els ajustos que no tinguin configuració base o execucions comparades.
- Documentar explícitament els ajustos rebutjats, no només els acceptats.
- Retardar canvis ambientals significatius fins que hi hagi una configuració estable en entorn simple.

## Conclusió

El document **Ajustar paràmetres del model** és adequat per incorporar-se a la documentació funcional del Projecte Evolució. Té una relació forta amb el registre d'experiments, les mètriques evolutives, l'anàlisi de resultats i els experiments inicials, i aporta el procediment que faltava per transformar observacions en canvis de configuració justificats.

La recomanació és acceptar-lo com a especificació funcional inicial del punt 19 i utilitzar-lo com a base per definir perfils de configuració versionats, rangs preliminars de paràmetres i criteris d'acceptació/rebuig dels ajustos.
