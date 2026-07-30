# Ocupació i densitat espacials

## Objectiu i estat persistent

El PR-11 converteix una `PopulationState` de capacitat fixa en mapes espacials generals,
deterministes i compatibles amb JAX. `WorldState.occupancy` és l'únic mapa persistent: conté
recomptes enters primaris. La densitat és una vista derivada i no es desa, evitant duplicació,
desincronització i un invariant addicional després de naixements, morts o moviments.

Les posicions tenen ordre `[x, y]`, mentre que els mapes tenen ordre `[y, x]` i forma `[H, W]`.
Per a cada slot viu i vàlid, l'índex lineal és `y * width + x`. Els slots inactius i els agents
vius invàlids reben temporalment la coordenada segura zero, però només els vàlids tenen pes u.
Un `scatter-add` suma els pesos, inclosos els duplicats, sense limitar els agents per cel·la.

Els slots inactius no contribueixen ni es consideren invàlids. Un agent viu fora dels límits no
es plega, retalla ni compta: incrementa `invalid_alive_count`. Es manté l'invariant:

```text
sum(occupancy) + invalid_alive_count == sum(population.alive)
```

## Densitat i veïnat local

La densitat per cel·la és `occupancy.astype(REAL_DTYPE) / max_agents`. La capacitat fixa és
l'únic límit superior global estable; no es normalitza per població viva, ocupació observada,
nombre de cel·les ni àrea de finestra. No s'aplica clipping.

Per un radi Python estàtic `r >= 0`, el veïnat és `[x-r, x+r] × [y-r, y+r]`, un quadrat de
costat `2*r+1` i distància de Chebyshev. El recompte usa `lax.reduce_window` amb stride u,
sortida `SAME` i padding zero: fora del món no contribueix i no hi ha wrapping, reflexió,
replicació ni renormalització de vores. Inclou la cel·la central, els agents solapats i l'agent
focal. La densitat local és `local_count / max_agents`, no una mitjana per cel·la. Amb radi zero,
recompte i densitat locals coincideixen amb els mapes de cel·la.

## Formes, dtypes i execució

| Valor | Forma | Dtype | Semàntica |
|---|---:|---|---|
| `PopulationState.position` | `[C, 2]` | `INDEX_DTYPE` | Coordenades `[x, y]` |
| `PopulationState.alive` | `[C]` | `MASK_DTYPE` | Slots actius |
| `occupancy` | `[H, W]` | `COUNT_DTYPE` | Recompte enter `[y, x]` |
| `invalid_alive_count` | `()` | `COUNT_DTYPE` | Vius fora dels límits |
| densitat de cel·la | `[H, W]` | `REAL_DTYPE` | `occupancy / max_agents` |
| recompte local | `[H, W]` | `COUNT_DTYPE` | Suma amb padding zero |
| densitat local | `[H, W]` | `REAL_DTYPE` | `local_count / max_agents` |

Les formes depenen de la capacitat i les dimensions estàtiques, mai de `sum(alive)`. Les
operacions no fan I/O ni usen RNG i són compatibles amb eager, `eqx.filter_jit` i `lax.scan`.

## Límit entre PR

| Funcionalitat | PR responsable |
|---|---|
| Ocupació global | PR-11 |
| Densitat global | PR-11 |
| Recompte local | PR-11 |
| Densitat local | PR-11 |
| Extracció per agent | PR-12 |
| Exclusió de l'agent focal | PR-12 |
| Densitats direccionals | PR-12 |
| Pressió competitiva observada | PR-12 |
| Recàlcul després de moviment | PR-17 / PR-27 |
| Recàlcul després de naixements i morts | PR-22 / PR-24 / PR-27 |
| Diagnòstics d'invariants | PR-26 |

El PR-12 extraurà els mapes per agent, construirà direccions i restarà l'agent focal quan la
variable signifiqui «altres agents». Els mapes generals del PR-11 no anticipen aquesta semàntica.

## Consum perceptiu

L'extracció per slot, els raigs cardinals i l'exclusió exacta del focal es documenten a [Esquema d'observacions locals v1](local_observation_schema_v1.md).
