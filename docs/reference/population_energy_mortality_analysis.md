# Anàlisi poblacional, energètica i de mortalitat

L'anàlisi local llegeix el manifest, la configuració congelada, el resum i els
artefactes Parquet i Zarr amb els loaders versionats del PR-42. La validació
científica comprova la integritat de les dades utilitzades, els compromisos de
chunks, el mostreig, la genealogia i el balanç poblacional observat. La
validació examina les metadades dels genomes destacats que relaciona amb els
orígens. Les consultes analítiques treballen amb taules Polars i es poden
executar sense serveis externs ni reexecució de la simulació. Les ordenacions,
els joins i la validació científica poden materialitzar dades intermèdies.

Els runs recuperats incorporen la seva història fins al checkpoint a la
mateixa trajectòria. Els resultats registren l'arrel, el pare de recuperació i
la profunditat. Els runs fallits o aturats s'analitzen fins a l'últim pas
confirmat. Els índexs antics sense compromisos de chunks no acrediten la
cobertura necessària per a aquesta anàlisi. Una continuació sense prefix
materialitzat pot proporcionar fluxos i mortalitat dins el tram confirmat.
La reconstrucció completa i l'exposició requereixen un origen verificable.

## Temps, població i finestres

Els passos són absoluts. Una anàlisi cobreix `[start_step, end_step_exclusive)`.
El valor produït al final del pas `t` correspon a l'estat `state_step = t + 1`.
La població inicial d'una finestra és l'estat abans del seu primer pas. La
població final i les estadístiques temporals es calculen sobre els estats
posteriors als passos. Una finestra sense naixements ni morts conserva la
població. Els compromisos dels chunks acrediten aquests zeros.

Les finestres s'alineen a múltiples absoluts de `window_steps`. `window_start`
és l'ancoratge alineat i `covered_start` el primer pas inclòs. Els extrems
retallats s'indiquen amb `window_incomplete`. L'exposició és la suma de
poblacions inicials de cada pas i aproxima el temps que els agents han estat
exposats. Les fraccions demogràfiques són descriptives.

L'extinció es registra al primer estat amb població zero, inclòs l'estat
inicial quan correspongui. El temps fins a l'extinció comença a l'origen de
la trajectòria. El llindar dels colls d'ampolla és la població fundadora
multiplicada per `bottleneck_fraction`. L'episodi comprèn els estats observats
amb població estrictament positiva i inferior al llindar. Si s'observa
encara sota el llindar a l'últim estat de la finestra, es marca amb censura
a la dreta. És un criteri demogràfic operatiu. La durada observada pot
quedar retallada als extrems. L'interval d'estats acaba en el primer estat
exclòs, de manera que pot acabar a `end_step_exclusive + 1`. El rang,
la desviació estàndard amb `ddof=0` i el coeficient de variació descriuen la
variabilitat. La periodicitat, l'estabilització i els efectes genètics
requereixen definicions i evidència addicionals.

## Stocks, fluxos i mostreig

Població, recursos disponibles i energia viva són stocks d'un estat. Les
seves observacions temporals es poden resumir amb mínims, màxims i mitjanes.
Els naixements, morts, consums, guanys i costos són fluxos. Un interval de
mostreig de 10 conserva els passos 0, 10, 20 i següents. Els esdeveniments
demogràfics complets donen totals exactes amb qualsevol interval de mostreig.

Un total de flux només és disponible si hi ha una observació finita a cada pas
o un acumulador de checkpoint íntegre, finit, sense saturació i amb
compromís de telemetria per a l'interval exacte. Els acumuladors són prefixos
des de l'origen de la trajectòria. Els mínims, màxims i mitjanes sobre
observacions espaiades es marquen com a `SAMPLED`. `EXACT` es refereix al
contracte i a les dades persistides. `UNAVAILABLE` es representa amb
`null` i un motiu al resum. Un denominador nul o zero produeix `null`.

El metabolisme basal precedeix la primera comprovació de mortalitat.
El guany alimentari registrat ja incorpora el límit d'energia. El moviment,
l'alimentació i els seus costos precedeixen la mortalitat postacció. La
reproducció té lloc després. El dèbit brut suma quatre costos. En el balanç
viu també s'afegeix l'energia dels fills i es resta l'energia signada dels
agents retirats. Una mort amb energia negativa augmenta el resultat de la
resta algebraica. La suma positiva retirada és una altra mètrica.
El residual és el canvi observat menys el canvi esperat. El càlcul requereix
els dos stocks d'energia i tots els fluxos finits, sense saturació del
checkpoint de tancament. Pot mostrar diferències d'arrodoniment de float32.

La regeneració efectiva es deriva amb el canvi del stock de recursos més el
consum complet. El contracte actual acumula la regeneració ambiental i el
dèbit d'alimentació com a únics canvis de recursos, amb diferències petites
possibles per arrodoniment. El stock inicial es pot obtenir d'una captura o
d'una inicialització uniforme coneguda. Una capacitat o taxa configurada no
determina la regeneració real. Les ràtios d'eficiència tenen interpretació
descriptiva i la conversió alimentària pot estar determinada directament per
la configuració.

## Mortalitat i supervivència

Les causes i fases provenen dels codis versionats del model. La manca
d'energia és una causa ecològica, l'edat màxima és un límit del model i
`INVALID_STATE` és un diagnòstic tècnic inclòs en els recomptes. Les fases
`PRE_ACTION` i `POST_ACTION` identifiquen el punt de registre. Les
estadístiques d'edat descriuen les morts observades. Els quartils fan servir
interpolació lineal i la desviació estàndard té `ddof=0`. Les estadístiques
d'energia inclouen els valors negatius finits i indiquen els descartats per
valor no finit. Els bins són `[inferior, superior)`, amb un bin inferior
sense límit esquerre i un de superior que inclou l'última vora. La columna
`invalid_count` compta els valors exclosos de tot l'histograma.

L'edat de mort prové del registre de mort. El metabolisme incrementa l'edat
abans de la viabilitat i un descendent entra amb edat zero a l'estat posterior
al seu pas de naixement. Per als supervivents, el PR-42 comprova l'edat final
que prové d'`AgentSummary`. En un prefix parcial amb orígens i vides
verificats, l'edat al tall és `state_step - birth_step` per als fundadors i
`state_step - birth_step - 1` per als descendents. Un supervivent contribueix
fins a l'edat observada i queda censurat a la dreta, amb causa nul·la.

Les cohorts es formen per amplada fixa sobre els passos de naixement
absoluts. Els fundadors tenen una categoria pròpia. Kaplan-Meier multiplica
els factors `1 - morts / risc` per edat i inclou totes les morts i censures
empatades en el conjunt a risc abans de retirar-les. La mediana és la
primera edat amb supervivència estimada inferior o igual a 0,5. Quan
aquest nivell queda fora del seguiment observat, la mediana és nul·la.
Els agents poden compartir llinatges, recursos i ambient. Les corbes són
descriptives i la v1 no proporciona intervals de confiança, proves
d'hipòtesis ni estimació amb entrada retardada.

## Diccionari de mètriques

| Nom | Definició | Unitat o denominador | Font | Disponibilitat |
| --- | --- | --- | --- | --- |
| `population_before`, `population_after` | Cens abans i després del pas | agents | Fundadors, naixements i morts | Origen i cobertura verificats |
| `birth_count`, `death_count`, `net_change` | Esdeveniments i saldo dins del pas o finestra | agents | Births i deaths Parquet | Esdeveniments complets |
| Fraccions demogràfiques per pas | Naixements, morts o saldo dividits pel cens inicial del pas | per agent inicial | Sèrie poblacional | Cens inicial positiu |
| `agent_steps_at_start` i taxes de finestra | Suma dels censos inicials i freqüència d'esdeveniments | agent·pas, esdeveniments per agent·pas | Sèrie poblacional | Taxes amb exposició positiva |
| `population_min`, `population_max`, `population_mean`, mediana, desviació, rang i CV | Descriptius dels censos de final de pas | agents, CV dividit per la mitjana | Sèrie poblacional | Passos executats i mitjana positiva per al CV |
| Extinció i temps fins a l'extinció | Primer estat buit i distància a l'origen | passos | Sèrie poblacional | Trajectòria verificable |
| Colls d'ampolla | Episodis sota el llindar demogràfic i sobre zero | passos, agents | Sèrie poblacional i política | Població fundadora positiva |
| `total_energy_alive`, `mean_energy_alive`, `total_resources_available` | Stocks observats per pas | energia, energia per agent viu, recursos | Metrics i captures | Segons mostreig i captures |
| `feeding_energy_gained_total`, costos per fase, `resource_consumed_total` | Fluxos totals | energia o recursos | Mètriques completes o acumulador | Cobertura exacta i termes finits |
| `gross_energy_cost_total` | Suma dels quatre dèbits | energia | Fluxos energètics | Quatre costos disponibles |
| `offspring_energy_total`, `net_reproductive_energy_cost_total` | Energia inicial dels fills i dèbit de reproducció menys l'energia transferida | energia | Naixements i flux reproductiu | Esdeveniments complets i cost per al net |
| `energy_at_death_total`, `positive_energy_removed_on_death` | Suma signada o suma de valors estrictament positius | energia | Morts | Valors finits a tots els events |
| `expected_delta_energy`, `observed_delta_energy`, `energy_balance_residual` | Canvi comptable, canvi mesurat i diferència | energia | Stocks, mètriques, naixements i morts | Termes complets segons la mètrica |
| `observed_min_resources`, `observed_max_resources`, `observed_mean_resources` | Estadístics dels stocks registrats | recursos | Mètriques | Com a mínim una observació |
| `resource_regenerated_total` | Canvi del stock més consum | recursos | Stocks d'extrem i consum | Tots tres termes disponibles |
| `births_per_resource_consumed`, `feeding_energy_per_resource` | Naixements o guany alimentari per consum | naixements o energia per recurs consumit | Events i fluxos | Finestra comuna amb consum positiu |
| `reproduction_cost_per_birth`, `agent_steps_per_energy_cost` | Dèbit reproductiu per naixement i exposició per dèbit brut | energia per naixement, agent·pas per energia | Exposició i fluxos | Denominador positiu i finestra comuna |
| `death_count`, `death_fraction`, `deaths_per_agent_step` | Morts per causa o fase, fracció del total i taxa temporal | morts, morts totals o agent·pas | Morts i exposició | Fracció i taxa amb denominador positiu |
| Distribucions i histogrames de mort | Descriptius i recomptes per intervals d'edat o energia | passos, energia, morts per bin | Morts | Energia finita per als descriptius |
| `observed_age`, `event_observed`, `right_censored_count` | Seguiment, indicador de mort i censures per cohort | passos i agents | Resum d'agents o prefix de vides verificat | Origen i tall coneguts |
| `survival_probability`, `median_survival_age` | Corba Kaplan-Meier i primera edat amb probabilitat ≤ 0,5 | proporció i passos | Observacions d'edat i censura | Dins del seguiment observat |

## CLI i exportació

```console
evolucio analysis summary runs/<run-id> --format json
evolucio analysis population runs/<run-id> --window-steps 100
evolucio analysis energy runs/<run-id> --window-steps 100
evolucio analysis mortality runs/<run-id> --start-step 10 --end-step-exclusive 20
evolucio analysis survival runs/<run-id> --cohort-width-steps 25
evolucio analysis export runs/<run-id> --output-dir derived/<run-id>
```

L'API `analyze_run` retorna un resum compacte amb la política i la cobertura.
Les consultes tabulars retornen `LazyFrame`, incloent `bottlenecks`,
`death_age_histogram`, `death_energy_histogram` i `survival_curve`.
L'exportació crea un directori separat i registra els hashes, la política i
les versions dels esquemes. L'exportació inclou taules de supervivència quan
el seguiment complet és disponible per al tram analitzat.

Els esquemes `population_analysis`, `energy_analysis`, `mortality_analysis`
i `survival_analysis` comencen a la versió 1. Les seves versions cobreixen
denominadors, finestres, llindars, mostreig, semàntica temporal, valors
absents i el mètode de supervivència.
