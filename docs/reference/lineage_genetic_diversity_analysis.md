# Anàlisi de llinatges, descendència i diversitat genètica

El PR-44 analitza un run terminal local amb els loaders i la validació científica
del PR-42. Les finestres usen passos absoluts i l'interval semitancat
`[start_step, end_step_exclusive)` del PR-43. Un run aturat o fallit pot aportar
el prefix íntegre verificat. Els vius al tall tenen seguiment censurat i el seu
destí posterior és desconegut. Un run recuperat aporta la pròpia trajectòria,
amb `root_run_id` i `source_run_id` per rastrejar l'origen. Els intents derivats
del mateix run comparteixen història i no són rèpliques independents.

## Identitats, temps i cobertura

Dins d'un `run_id`, `agent_id` identifica l'individu, `genome_id` identifica el genoma
assignat i `lineage_id` identifica l'arbre del fundador. Cada descendent asexual
té un progenitor, una generació superior a la del progenitor i el mateix llinatge.
La generació és la profunditat de l'arbre. Una cohort agrupa agents segons el
moment de naixement i un mateix pas pot contenir agents de diverses generacions.

`genealogy_nodes` conserva els morts. `get_ancestors` comença pel progenitor,
exclou l'agent consultat i retorna una cadena buida per a un fundador.
`get_descendants` retorna les distàncies en arestes, el recompte exacte de
fills directes i un recompte total quan el recorregut és complet. Els límits de
nodes o profunditat marquen el resultat com a parcial si ometen descendents.

Els fundadors entren a l'estat inicial. Un fill nascut al pas `b` entra amb edat
zero a l'estat `b + 1`. La primera reproducció és el primer event de naixement
associat al progenitor i el temps transcorregut es compta des de l'estat d'entrada.
L'edat observada fins a una mort s'obté de l'event corresponent. Els agents vius
sense fills observats estan censurats, i la mort abans de la primera reproducció
és un esdeveniment competidor. El PR-44 calcula estadístiques descriptives.

`generation_summary(..., follow_up_steps=k)` restringeix el recompte de fills
als primers `k` passos des de l'entrada. Inclou les cohorts de naixement amb
`k` passos disponibles fins a la frontera del prefix, i conserva les morts
primerenques. Les columnes de mortalitat i edat continuen descrivint el seguiment
observat al prefix dels agents inclosos. Comparar les mitjanes sense tenir en
compte l'exposició de les generacions recents pot induir conclusions errònies.

## Diccionari de mètriques

| Mètrica | Definició i fórmula | Font | Unitat | Disponibilitat i interpretació |
|---|---|---|---|---|
| `direct_offspring_count` | Nombre de naixements amb l'agent com a progenitor | Births i AgentSummary validat quan hi és | agents | Inclou zeros. Censura al tall per als vius |
| `total_descendant_count` | Nombre de nodes accessibles seguint les arestes cap als fills | Founders, births i genealogia validada | agents | Requereix un recorregut complet en la consulta |
| `agents_without_offspring_observed`, `agent_fraction` | Recompte d'agents amb zero fills i proporció de cada valor de fills sobre tots els agents | Founders i births | agents, proporció | Proporcions nul·les amb denominador zero |
| `mean_direct_offspring`, `median_direct_offspring`, `offspring_std` | Mitjana, mediana i desviació típica poblacional dels fills directes | Births i orígens | fills per agent | Inclou agents amb zero fills |
| `first_reproduction_step`, `time_to_first_reproduction` | Primer naixement per progenitor i pas del naixement menys estat d'entrada del progenitor | Births i orígens | pas, passos | Dos valors nuls si no s'observa reproducció |
| `observed_age`, `follow_up_steps` | Edat a la mort o al tall, i temps entre l'estat d'entrada i el tall | AgentSummary, founders, births i deaths | passos | El seguiment potencial fins al tall pot excedir l'edat observada en agents morts |
| `population_before`, `population_after`, `net_change` | Cens del llinatge abans i després del pas. `net_change = births - deaths` | Cens fundador i events complets | agents | Els intervals mantenen la darrera població posterior |
| `extinction_event_step`, `extinction_state_step` | Pas de mort del darrer agent i estat següent `D + 1` | LineageSummary o events | pas | Nuls per a llinatges vius al tall. Una interrupció deixa el destí posterior desconegut |
| `duration_observed_steps` | Estat d'extinció o estat de tall menys pas del fundador | Cens, morts i frontera | passos | Censurada si el llinatge encara és viu |
| `population_mean`, `agent_steps_at_start` | Mitjana de poblacions posteriors i suma de poblacions anteriors sobre els passos de la finestra | Intervals i balanços | agents, agent·pas | L'expansió i el declivi es determinen pel signe de `net_change`. Pot haver-hi renovació amb saldo zero |
| `births_per_agent_step`, `deaths_per_agent_step` | Esdeveniments de la finestra dividits per l'exposició `agent_steps_at_start` | Intervals i events | esdeveniments per agent·pas | Nuls amb exposició zero |
| `population_fraction` | `p_L = N_L / N` per llinatge viu | Censos reconstruïts | proporció | Sense files si no hi ha població, i suma 1 quan hi ha vius |
| `active_lineages` | Riquesa de llinatges amb població viva positiva | Censos reconstruïts, contrastats amb metrics | llinatges | Zero quan la població és zero |
| `dominant_lineage_id`, `dominant_lineage_fraction` | Llinatge amb proporció màxima i la seva proporció | Censos reconstruïts | identificador, proporció | En empat es mostra el menor ID i es conserva el nombre de llinatges empatats. Nuls en extinció total |
| `shannon_entropy` | `H = -Σ p_L ln(p_L)`, amb logaritme natural | Proporcions positives | nats | Nul sense població |
| `simpson_concentration`, `simpson_diversity` | `C = Σ p_L²` i `1 - C` | Proporcions positives | proporció | Nuls sense població |
| `effective_lineages_shannon`, `effective_lineages_simpson` | `exp(H)` i `1/C`, equivalents de llinatges igualment representats | Shannon, Simpson | llinatges equivalents | Nuls sense població. Interpretació diferent de `Ne`, riquesa genòmica i espècies |
| `lineage_evenness` | `H / ln(riquesa)` | Shannon i riquesa | proporció | Nul si la riquesa és zero o un |
| `genetic_parameter_variance` | Mitjana sobre els paràmetres de la variància poblacional amb `ddof=0` entre agents vius | Metrics Parquet, semàntica PR-25 | paràmetre² | Zero per a zero o un agent. Es conserva el mostreig temporal original |
| `mean_pairwise_squared_distance_per_parameter`, `pairwise_rms_distance_per_parameter` | `2nV/(n-1)` i la seva arrel per a parelles diferents d'agents | Variància poblacional `V` i cens `n` | paràmetre², paràmetre | Nuls per a `n < 2`. L'arrel és RMS, que en general difereix de la mitjana de distàncies |
| `total_parameter_variance`, `within_lineage_variance`, `between_lineage_variance` | Mitjanes sobre paràmetres de `Var(X)`, `Σ p_L Var(X ∣ L)` i `Σ p_L (mean_L - mean_global)²` | Genomes vius d'un checkpoint | paràmetre² | Es verifica que els dos components sumin la variància total |
| `mean_parameter_variance` per llinatge | Mitjana de la variància amb `ddof=0` dins del llinatge | Checkpoint complet | paràmetre² | Zero quan hi ha un individu viu al grup |
| `distinct_genome_contents` | Nombre de `genome_content_hash` diferents | Fitxers verificats o checkpoint | continguts | Només per a la selecció explícita o els genomes vius del llinatge en aquella captura |
| Distàncies L1, L2, MAE, RMS i màxim | Sobre `d = a - b`, `Σ abs(d)`, `sqrt(Σ d²)`, `mean(abs(d))`, `sqrt(mean(d²))` i `max(abs(d))` | Parell verificat de genomes compatibles | paràmetre | Comparacions exactes en l'espai de pesos, amb identificadors i schema validats |
| `mutation_selected_total`, `mutation_effective_total` | Sumes de paràmetres seleccionats i de paràmetres amb canvi real | Births Parquet | paràmetres | Soroll zero i clipping poden fer divergir els recomptes. Els diagnòstics de fallada del core no es compten com a naixements |
| `effective_mutation_fraction` | Canvis efectius dividits pels paràmetres visitats pel mecanisme en tots els naixements observats | Births i schema de policy | proporció | Nul sense naixements. Mesura observada diferent de la probabilitat configurada |
| `path_effective_mutations`, `path_sum_abs_delta` | Sumes dels recomptes i magnituds absolutes al camí entre el fundador i l'agent | Genealogia Parquet | paràmetres, paràmetre | Canvis compensatoris poden reduir la divergència final a zero |
| `survival_probability` per llinatge | Kaplan–Meier d'edat individual amb els vius censurats | Vida observada i PR-43 | proporció | Supervivència individual, distinta de la persistència de l'arbre |

## Genomes conservats

El registre conserva genomes seleccionats, com ara els fundadors i els representants
finals. Les seves distàncies i els seus hashes descriuen aquesta selecció deliberada.
Un checkpoint pot aportar un genoma absent del registre quan l'agent era viu i se'n
verifiquen els identificadors. Un snapshot ordinari només aporta els camps que
realment conté. L'absència d'un genoma deixa la comparació concreta com a
`UNAVAILABLE` i permet continuar consultant la genealogia i la descendència.

Els genomes s'aplanen en ordre canònic `layer1.weight`, `layer1.bias`,
`layer2.weight`, `layer2.bias`. Les fulles han de tenir els tipus i formes de
l'esquema i valors finits. Els càlculs host usen Float64. Els identificadors de
genoma i llinatge poden ser diferents amb continguts idèntics. Les distàncies de
paràmetres admeten funcions neuronals equivalents, i no permeten inferir
conducta, intel·ligència, adaptació, selecció causal o especiació.

`EXACT` indica un resultat dins la cobertura que descriu. Els events complets
permeten reconstruir demografia i diversitat genealògica en passos que no tenen
mètriques persistides. La cobertura genètica temporal és `SAMPLED` si falten
passos i la dada de cada pas registrat descriu tots els agents vius en aquell
instant. Un checkpoint és una captura concreta i s'alinea amb la mètrica del
pas executat immediatament anterior. Les sèries genètiques no s'interpolen.

## Consultes i exportació

```console
evolucio analysis lineages runs/RUN_ID
evolucio analysis offspring runs/RUN_ID
evolucio analysis generations runs/RUN_ID --follow-up-steps 20
evolucio analysis diversity runs/RUN_ID --start-step 0 --end-step-exclusive 200
evolucio analysis diversity runs/RUN_ID --state-step 200
evolucio analysis mutations runs/RUN_ID
evolucio analysis ancestors runs/RUN_ID --agent-id 12
evolucio analysis descendants runs/RUN_ID --agent-id 0 --max-depth 4 --max-nodes 100
evolucio analysis genome-distance runs/RUN_ID --genome-a 0 --genome-b 9
evolucio analysis export runs/RUN_ID --lineages --output-dir derived/RUN_ID
```

`--lineages` afegeix l'exportació nova al mecanisme del PR-43. El resum JSON
registra schemas, cobertura, política, limitacions i hashes de les fonts. Els
Parquet es creen en un directori derivat nou. Les consultes de finestra exporten
les sèries temporals i ometen els resums que necessiten el seguiment complet.
El càlcul funciona sense PostgreSQL, MLflow ni execució del simulador.

## Exemple validat

La prova `test_recorded_multigeneration_analysis` executa sis passos amb cinc
fundadors, deu naixements, una mort i generacions de 0 a 3. Al darrer pas hi ha
14 agents i quatre llinatges vius, amb poblacions d'11, 1, 1 i 1 agents.
Shannon és 0,754997 i la diversitat de Simpson és 0,367347. El checkpoint
de l'estat 6 dona una variància total de 0,02370517, amb 0,00002584 dins dels
llinatges i 0,02367933 entre llinatges. Els deu naixements registren en total
3.750 canvis efectius de paràmetres.

## Cost de les consultes

Les consultes d'arbre indexen els agents en memòria i recorren una sola branca o
un subarbre acotat. No es materialitza la clausura transitiva. La població per
llinatge s'agrega i ordena per esdeveniments i llinatges. El balanç inicial i els
fluxos en cada pas es contrasten amb PR-43, de manera que també es verifica la
població als intervals constants. La diversitat es calcula en un recorregut
ordenat i actualitza la suma de quadrats, el terme d'entropia i un heap per a
la dominància. El cost és `O(E log E)` en temps i `O(E)` en memòria per a `E`
files de canvi i cens. La sèrie densa s'activa explícitament i té un límit de
files. Les estadístiques per finestres materialitzen les interseccions entre
intervals i finestres demanades.

En una captura, la variància usa `O(NP)` càlculs per a `N` agents i `P`
paràmetres més l'ordenació dels llinatges. Es carrega una captura cada vegada.
Les distàncies agregades L1 i L2 entre destacats recorren els parells amb una
selecció explícita acotada, en temps `O(K²P)` i memòria `O(KP)` per a `K`
genomes. El càlcul és determinista i no construeix cap matriu de parelles.
