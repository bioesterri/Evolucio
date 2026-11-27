# Comparació experimental de trajectòries

Un disseny v1 identifica l'experiment i la revisió, el baseline, els controls i
les variants. Declara la pregunta, el factor, el criteri de comparació,
l'horitzó i les mètriques principals, secundàries i exploratòries. La direcció
esperada és opcional i es declara per mètrica. Les decisions analítiques del
disseny són independents del motor.

## Disseny, controls i execució

Cada cas executa el mateix `SeedSet` amb el `BatchPlan` del PR-34. La unitat
de rèplica és la trajectòria independent d'una configuració i una llavor.
Agents, llinatges, passos i checkpoints comparteixen la història del run.
Una repetició determinista d'una cel·la mesura repetibilitat. Diverses llavors
mesuren variabilitat estocàstica. Compartir la llavor estableix un bloc
aparellat. Després de divergir, dues trajectòries poden consumir números
aleatoris diferents.

L'execució congela el document i els hashes base i per llavor abans
d'inicialitzar el batch. Desa el pla a `experiment_design.json` i els intents
a `attempts.jsonl`. El directori d'execució és exclusiu de l'experiment i
revisió. Per canviar una hipòtesi o un protocol, incrementa `design_version`
i executa en un directori nou. El pla exportat remet als fitxers de
configuració originals. Conserva aquests fitxers i el directori d'execució
per reproduir l'informe.

Els exemples complets són a `examples/experimental_comparison`.

| Control | Camp autoritzat | Condició de l'exemple |
|---|---|---|
| Sense mutació | `evolution.weight_mutation_rate`, `evolution.bias_mutation_rate` | Ambdues taxes a 0. L'herència preserva els pesos i biaixos del progenitor. La diversitat dels founders i la selecció sobre aquesta variació poden persistir. |
| Entorn constant | `world.environment_schedule` | Calendari buit amb el mateix ambient inicial. El consum i la regeneració continuen actius. |
| Recursos abundants | `world.initial_resource_mean`, `world.resource_capacity` | 15 i 20 unitats per cel·la respecte de 5 i 10 al baseline. Examina consum, existències i mortalitat per valorar si hi ha limitació efectiva. |

La política d'agents aleatoris fixos queda pendent. Requereix un contracte de
política separat que especifiqui la distribució d'accions i els streams RNG
que la governen. Amb mutació a zero cada agent continua actuant segons la
seva xarxa heretada.

## Comparabilitat i selecció

Cada cel·la registra el seu estat, els candidats i els motius d'exclusió.
Un resultat absent sense evidència d'execució queda com a desconegut. Es
requereixen llavor i configuració previstes, estat completat, cobertura
confirmada de `[0, total_steps)`, i validació científica del PR-42. Una
recuperació elegible aporta la trajectòria completa verificada pel PR-41.
L'intent font, els intents fallits i les repeticions consten a l'auditoria.
Amb diversos candidats elegibles cal indicar un `run_id` de manera explícita.
Cap criteri de selecció depèn de la població, la descendència o l'extinció.

Les diferències es classifiquen segons el schema normalitzat. Els camps
científics de cada contrast han d'estar autoritzats expressament. La
projecció `configuration_group_hash` exclou només `seed`,
`runtime.chunk_size` i `persistence.output_dir`. Els `config_hash` originals
continuen presents a l'auditoria i a la taula per run. Les opcions de captura
es documenten i es comproven a part. Les diferències en mostreig verificat
poden limitar mètriques concretes sense invalidar les mètriques exactes
reconstruïdes a partir dels esdeveniments complets.

La comparabilitat comprova els schemas persistits, el commit i l'entorn
d'execució, i verifica la signatura de compilació de cada configuració. La
signatura inclou els contractes d'observació, política, mutació i RNG. Si es
coneix el commit però hi ha canvis locals sense verificar, `STRICT`
bloqueja el contrast i `AVAILABLE_PAIRS` adverteix de la limitació.
Executa amb un commit identificat i l'arbre net per verificar la procedència.

`STRICT` requereix totes les parelles i els valors de les mètriques
principals. Els contrasts bloquejats conserven el material d'auditoria.
`AVAILABLE_PAIRS` permet l'anàlisi exploratòria de parelles verificades i
publica les exclusions i el denominador real. Les fallades tècniques poden
dependre de la configuració, de manera que la selecció per disponibilitat
pot introduir biaix. Les execucions locals independents o consultades a
PostgreSQL es presenten com a exploratòries si no tenen un pla congelat
verificat. Una selecció manual, una tolerància canviada o un bootstrap
afegit posteriorment també identifiquen l'anàlisi com a exploratòria.

## Lectura de les mètriques

El registre `metric_registry.json` descriu la unitat, l'agregació, la font
PR-43/44 i els requisits de disponibilitat de cada mètrica.

| Família | Interpretació |
|---|---|
| Població | Cens, esdeveniments o exposició al llarg de l'horitzó. La durada dels colls d'ampolla és la durada observada sota el llindar declarat. |
| Energia i recursos | Els fluxos comparats cobreixen el mateix interval complet. El balanç residual descriu tancament comptable, i els quocients depenen del seu denominador. Les existències d'inici i final descriuen disponibilitat als extrems. |
| Mortalitat i supervivència | Causes observades, edat de les morts i corba de Kaplan-Meier. Les fraccions sense morts i les medianes no identificables queden nul·les. L'edat fixa s'ha de declarar prèviament. |
| Reproducció | Descendència observada al llarg del seguiment. Les cohorts recents han tingut menys oportunitats. Per comparar cohorts cal igualar el seguiment amb les APIs del PR-44. |
| Llinatges | Diversitat i dominància a l'estat final, extincions de llinatge durant l'horitzó. Els identificadors són locals al run. |
| Genètica | Variància i distància de la població viva al pas terminal si hi ha mostra exacta. Les components per llinatge requereixen un checkpoint terminal. La divergència del fundador resumeix la distància al representant retingut de cada llinatge i descriu aquest subconjunt. |

Les files nul·les conserven la qualitat disponible i el motiu. Una diferència
entre configuracions és una observació del contrast i requereix evidència
addicional d'herència, selecció i resposta funcional per inferir adaptació.

## Distribucions i diferències

Es conserven tots els valors per llavor, inclosos els extrems i les
extincions. Els quartils de Polars fan servir interpolació lineal. La
desviació estàndard mostral usa `ddof=1` i queda nul·la amb menys de dos
valors. El contrast calcula `variant - baseline`. La diferència relativa
requereix una mètrica d'escala de raó i un baseline estrictament positiu.
La tolerància absoluta de signe per defecte és `1e-6` en la unitat de la
mètrica. Les proporcions de signe descriuen la consistència entre llavors.

El bootstrap opcional NumPy remostreja els deltes aparellats amb llavor
analítica explícita, 10.000 remostrejos i nivell de confiança 0,95 per
defecte. Els intervals percentils són exploratoris i amb una sola parella
queden nuls. La dispersió entre llavors, la incertesa de la mitjana i els
límits del model són fonts diferents d'incertesa. El nombre útil de llavors
depèn de la variabilitat i de l'efecte que es vol estudiar.

L'extinció s'expressa com a fracció observada de runs elegibles amb el mateix
horitzó. La taula aparellada distingeix quatre resultats conjunts.
Les trajectòries que arriben vives al final tenen seguiment censurat per la
dreta. `extinction_observations.parquet` conserva la durada observada i
l'indicador d'esdeveniment per aplicar el helper Kaplan-Meier del PR-43.

## Comparacions temporals

Els passos són absoluts. `EXACT` exigeix els mateixos passos observats.
`COMMON_STEPS` pren la intersecció verificada sense interpolació. Cada run
aporta un valor per pas. La mitjana poblacional preserva els zeros confirmats
després de l'extinció. Les mètriques genètiques indefinides amb població zero
queden nul·les. La sortida publica el nombre de runs i de parelles amb valor
per pas.

L'API temporal també accepta prefixos terminals validats quan totes les
trajectòries cobreixen la finestra indicada. Aquest ús és exploratori i el
resum final continua exigint l'horitzó complet. Per a una pertorbació,
`environment_change_step` identifica una transició real del calendari.
Els passos anteriors, durant la fase i posteriors corresponen a les mateixes
finestres en ambdós casos. El contrast descriptiu és
`(després − abans)_baseline − (després − abans)_control` per llavor. Pot
reflectir plasticitat, mortalitat diferencial o altres mecanismes.

## CLI i reproducció

```console
evolucio experiment validate-design examples/experimental_comparison/design.json
evolucio experiment execute-design examples/experimental_comparison/design.json --dry-run
evolucio experiment execute-design examples/experimental_comparison/design.json --execution-dir /tmp/experiment-45
evolucio analysis compare examples/experimental_comparison/design.json --execution-dir /tmp/experiment-45 --output-dir /tmp/comparison-45 --temporal-metric population_after --environment-control constant_environment
```

`--runs-root` analitza runs existents. `--registered` localitza els runs per
identitat experimental al registre PostgreSQL i llegeix els artefactes amb
el mateix loader local. `--allow-available-pairs` activa la política
exploratòria i `--select-run CASE:SEED=RUN_ID` resol una ambigüitat.
`--bootstrap-seed` activa els intervals exploratoris.
`--common-steps` autoritza l'alineació sobre passos coincidents.

Comprova l'ID, l'abast i el manifest de la comparació. El manifest registra
els hashes originals dels manifests i artefactes font, els schemas,
l'entorn analític, la política, les opcions temporals i la configuració de
bootstrap. L'exportació crea un directori derivat independent i llegeix els
originals en mode de consulta. PostgreSQL i MLflow continuen sent punts de
localització i traçabilitat. La font científica és el run verificat.
