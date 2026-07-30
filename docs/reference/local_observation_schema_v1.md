# Esquema d'observacions locals v1

## Objectiu i identitat

`local_cardinal_v1` transforma un únic `SimulationState` coherent en un array `float32 [C,15]` local, determinista i compatible amb JAX/JIT. La versió és **1**, la mida **15** i el digest SHA-256 canònic és `14bd8098500e8537b6144fa92d0e6b08f1a7d30340283411890586daa4781515`.

El descriptor ordenat, les fórmules, els raigs, l'exclusió focal, l'orientació i el bitmask formen el payload canònic. Canviar-ne semàntica, ordre o mida requereix una versió i digest nous; els genomes d'esquemes diferents són incompatibles.

## Camps congelats

| Índex | Nom | Rang | Font | Normalització |
|---:|---|---:|---|---|
| 0 | `ENERGY_RELATIVE` | `[0,1]` | energia pròpia | `clip(energy / maximum_energy, 0, 1)` |
| 1 | `AGE_RELATIVE` | `[0,1]` | edat pròpia | `clip(age / maximum_age, 0, 1)` |
| 2 | `REPRODUCTION_MARGIN` | `[-1,1]` | energia pròpia | `clip((energy-reproduction_threshold)/maximum_energy,-1,1)` |
| 3 | `CURRENT_RESOURCE` | `[0,1]` | `resources[y,x]` | `clip(value/resource_capacity,0,1)` |
| 4 | `RESOURCE_NORTH` | `[0,1]` | raig nord | `clip(sum/(resource_capacity*r),0,1)` |
| 5 | `RESOURCE_SOUTH` | `[0,1]` | raig sud | igual |
| 6 | `RESOURCE_EAST` | `[0,1]` | raig est | igual |
| 7 | `RESOURCE_WEST` | `[0,1]` | raig oest | igual |
| 8 | `AGENTS_NORTH` | `[0,1]` | ocupació del raig nord | `clip(sum/max_agents,0,1)` |
| 9 | `AGENTS_SOUTH` | `[0,1]` | ocupació del raig sud | igual |
| 10 | `AGENTS_EAST` | `[0,1]` | ocupació del raig est | igual |
| 11 | `AGENTS_WEST` | `[0,1]` | ocupació del raig oest | igual |
| 12 | `LOCAL_DENSITY` | `[0,1]` | recompte quadrat local | `clip(max(local_count-1,0)/max_agents,0,1)` |
| 13 | `ENVIRONMENTAL_STRESS` | `[0,1]` | `environment[y,x]` actual | `clip(value,0,1)` |
| 14 | `MOVEMENT_BLOCKED` | `[0,1]` | límits tancats | `(north*1+south*2+east*4+west*8)/15` |

Una escala no positiva retorna zero als helpers segurs; això permet `resource_capacity=0` sense NaN. Les escales funcionals d'energia i edat continuen validades positivament al host.

## Agregació, vores i focal

Les posicions són `[x,y]`, els mapes `[y,x]`. Nord és `(0,-1)`, sud `(0,1)`, est `(1,0)` i oest `(-1,0)`. Cada raig conté distàncies `1..r`, no la cel·la actual ni diagonals. Fora del rectangle aporta zero, el denominador conserva `r`, i no hi ha wrapping ni clipping. L'extracció substitueix només l'índex físic invàlid per una cel·la segura i després l'emmascara.

La densitat usa el veïnat quadrat del PR-11. Per un focal viu i vàlid resta exactament una unitat: conserva altres agents que comparteixin cel·la. Les entrades direccionals no inclouen la cel·la focal. El bitmask immediat diferencia direccions; l'ocupació no bloqueja moviment.

## Localitat, temps i emmascarament

Un slot inactiu o un agent viu fora de límits rep quinze zeros, independentment de residus. No s'observen coordenades absolutes, totals globals, identitats o genealogia, energia/edat alienes, RNG, fitness, mètriques, accions, regeneració futura ni calendari ambiental futur. L'ambient i els recursos són els actuals; ocupació i població han de ser coherents. La funció no actualitza estat, RNG ni configuració.

## Configuració i compilació

| Camp | Classificació | Identitat |
|---|---|---|
| `observations.schema_version` | estàtic, literal `1` | `config_hash` i `CompileSignature` |
| `observations.perception_radius` | estàtic, enter estricte `1..3` | `config_hash` i `CompileSignature` |
| mida i digest | contracte congelat derivat | `CompileSignature` |
| `maximum_energy`, `maximum_age`, `reproduction_threshold`, `resource_capacity` | scalars dinàmics | només `config_hash`; modifiquen valors, no forma |

`CompileSignature` v4 persisteix versió, mida, digest i radi. El radi és estàtic perquè determina la forma intermèdia `[C,4,r,2]`; les escales numèriques no determinen formes. PR-13 haurà d'usar la mida 15 com entrada de `PolicyMLP`. La persistència posterior haurà de vincular digest i genomes.

## Responsabilitats posteriors

| Funcionalitat | PR responsable |
|---|---|
| Topologia `PolicyMLP` | PR-13 |
| Genomes batched | PR-14 |
| Inferència i selecció | PR-15 |
| Validació d'accions | PR-16 |
| Moviment | PR-17 |
| Alimentació | PR-18 |
| Viabilitat | PR-20 i PR-21 |
| Integració temporal | PR-27 |
| Persistència del digest amb genomes | PR-35, PR-39 o adaptador corresponent |
