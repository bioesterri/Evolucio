# Valoració d'adequació — Executar experiments inicials

## Objectiu de la valoració

Aquest document comprova si l'especificació **Executar experiments inicials** s'adequa a la resta de la documentació funcional del Projecte Evolució i identifica punts forts, encaixos, riscos i recomanacions d'implementació.

## Veredicte resumit

La valoració és **molt positiva**: el document encaixa amb la fase de validació experimental inicial i actua com a pont entre les proves bàsiques del motor i els experiments evolutius més llargs. El seu enfocament és prudent, traçable i coherent amb el principi del projecte: no interpretar patrons visuals o llinatges dominants com a adaptació fins que hi hagi controls, repeticions, mètriques i registre suficient.

És especialment adequat com a punt 17 del pla funcional perquè només proposa experiments petits, controlats i repetibles després que el bucle principal, les regles ecològiques, les mètriques, el registre, la persistència i la visualització mínima ja existeixin.

## Encaix amb la documentació existent

| Document relacionat | Encaix observat | Valoració |
| --- | --- | --- |
| `document-funcional-inicial.md` | Desenvolupa la fase final de validació del prototip i manté l'objectiu de fer experiments repetibles amb configuració, llavor, resultats i genealogia registrats. | Molt coherent: converteix el full de ruta en una bateria inicial d'escenaris comparables. |
| `abast-prototip-inicial.md` | Respecta l'abast reduït: una espècie, un recurs, món 2D simple, accions bàsiques i reproducció asexual. | Molt coherent: evita ampliar el prototip abans de validar-ne la base. |
| `arquitectura-general.md` | Assumeix separació entre simulació, registre, mètriques, persistència, anàlisi i visualització. | Coherent: no introdueix dependències funcionals que barregin responsabilitats. |
| `bucle-principal-simulacio.md` | Exigeix que el bucle s'executi sense intervenció manual i que produeixi mètriques per pas temporal. | Molt coherent: els experiments són consumidors naturals del bucle temporal validat. |
| `energia-recursos-metabolisme.md` | Fa dels recursos, costos i energia inicial variables principals dels escenaris E01-E04. | Molt coherent: valida la pressió energètica com a motor ecològic. |
| `reproduccio-herencia-mutacio.md` | Inclou controls de reproducció, mutació desactivada, mutació baixa i mutació alta. | Molt coherent: separa continuïtat de llinatges, variació i soroll mutacional. |
| `mort-seleccio-ambiental.md` | Demana morts registrades per causa i col·lapses explicables. | Coherent: evita extincions opaques o interpretacions sense causa funcional. |
| `metriques-evolutives.md` | Utilitza població, energia, naixements, morts, recursos, descendència, diversitat i variabilitat entre llavors. | Molt coherent: les mètriques definides passen a criteris d'interpretació experimental. |
| `registre-experiments.md` | Requereix configuració completa, llavor, versió, mètriques, artefactes i resum final per cada run. | Molt coherent: cap experiment inicial és vàlid si queda orfe de context. |
| `persistencia-dades.md` | Dona sentit a la persistència de configuracions, sèries temporals, genomes, genealogies i artefactes visuals. | Coherent: els resultats dels escenaris necessiten sobreviure a l'execució. |
| `visualitzacio-minima.md` | Requereix gràfics bàsics suficients per detectar errors grossos, no visualitzacions avançades. | Coherent: la visualització és suport diagnòstic, no evidència única. |
| `proves-basiques-validacio.md` | Estableix les condicions prèvies abans d'executar experiments amb interpretació funcional. | Molt coherent: separa sanitat del motor d'experimentació inicial. |

## Punts forts

- **Prudència metodològica:** insisteix que la fase és de calibratge i validació, no de demostració científica.
- **Escenaris mínims clars:** defineix E00-E09 per cobrir fum, controls, recursos escassos/moderats/abundants, mutació i variabilitat entre llavors.
- **Comparabilitat:** recomana canviar pocs paràmetres cada vegada i reutilitzar el mateix conjunt de llavors per comparacions entre configuracions.
- **Controls explícits:** inclou controls sense mutació, sense reproducció, amb recursos insuficients, amb recursos abundants i amb diverses llavors.
- **Criteris d'interpretació útils:** diferencia col·lapse immediat, col·lapse tardà, estabilitat relativa, explosió poblacional, deriva i adaptació aparent.
- **Traçabilitat exigida:** considera invàlid qualsevol experiment que no registri configuració, llavor, versió, mètriques i artefactes mínims.
- **Bona alineació amb riscos del projecte:** detecta errors com supervivència sense consum, reproducció sense cost, regeneració excessiva, morts mal registrades o resultats no repetibles.

## Riscos o punts a vigilar

1. **Definir valors concrets de configuració.**  
   El document és funcional i evita fixar números, però la implementació haurà de convertir criteris com “moderada”, “baixa” o “elevat” en perfils versionats i comparables.

2. **Evitar confondre estabilitat curta amb viabilitat robusta.**  
   Una població que sobreviu en execucions mitjanes pot col·lapsar en durades més llargues. Cal etiquetar aquestes configuracions com a candidates, no com a validades definitivament.

3. **Controlar el volum d'artefactes.**  
   Guardar configuració, resultats, millors genomes, genealogies i gràfics és necessari, però caldran polítiques de freqüència, retenció i resum per no generar dades excessives.

4. **Assegurar repetibilitat real.**  
   Repetir llavors només serà significatiu si totes les fonts d'atzar i l'ordre d'execució són controlables i si la versió del model queda identificada correctament.

5. **Formalitzar la classificació final.**  
   Les categories proposades són bones, però caldrà una regla operativa mínima perquè diferents persones classifiquin igual una execució com a col·lapse, estabilitat, expansió, deriva o adaptació aparent.

## Recomanacions d'implementació

- Crear perfils de configuració versionats per a E00-E09, derivats d'una configuració base comuna.
- Fer que cada escenari tingui una durada curta de validació i una durada mitjana experimental separades.
- Definir un conjunt inicial de llavors estàndard per comparacions, per exemple una llista curta reutilitzable en tots els escenaris rellevants.
- Bloquejar o marcar com a invàlids els runs que no tinguin configuració resolta, llavor, versió de model i mètriques mínimes.
- Generar un resum tabular per escenari amb estat final, població final, població mínima/màxima, naixements, morts per causa, recursos finals i classificació.
- Guardar gràfics bàsics de població, energia, recursos i naixements/morts com a artefactes derivats de les dades primàries.
- Afegir una nota d'anàlisi curta per cada bateria experimental indicant quines configuracions es descarten i quines passen a experiments posteriors.

## Conclusió

El document **Executar experiments inicials** és adequat per incorporar-se a la documentació funcional del Projecte Evolució. Manté la coherència amb l'abast del prototip, depèn correctament de les proves bàsiques, mètriques, registre, persistència i visualització mínima, i evita el risc principal de sobreinterpretar resultats inicials.

La recomanació és acceptar-lo com a especificació funcional inicial del punt 17 i utilitzar-lo com a base per preparar la primera matriu d'experiments controlats E00-E09, amb configuracions versionades, llavors explícites i criteris de classificació verificables.
