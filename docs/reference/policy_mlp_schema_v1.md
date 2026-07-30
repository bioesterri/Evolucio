# Esquema PolicyMLP v1

## Contracte

`PolicyMLP` transforma una observació local `float32[15]` en set **scores lineals**. L'esquema és `policy_mlp_tanh_15_16_7_v1`, versió 1, digest `b1c8336c8c0be45f3bacbbf384ccb5d8cbf352651c084da3ac98fcab6bbc7e90`. La topologia fixa és 15 → 16 → 7, amb una única capa oculta:

```text
hidden = tanh(layer1.weight @ observation + layer1.bias)
scores = layer2.weight @ hidden + layer2.bias
```

S'utilitza `jax.nn.tanh` només a la capa oculta. No hi ha softmax ni cap altra activació, normalització o selecció a la sortida: els valors no són probabilitats.

| Capa | Entrada | Sortida | Activació | Biaix | Paràmetres |
|---|---:|---:|---|---|---:|
| `layer1` | 15 | 16 | `tanh` | sí | 256 |
| `layer2` | 16 | 7 | lineal | sí | 119 |

El total és de **375 paràmetres**.

## Fulles PyTree i genomes futurs

L'`eqx.Module` és immutable i no conté estat recurrent, RNG, inicialització ni aprenentatge individual. L'ordre canònic de les quatre fulles també és l'ordre de persistència i reconstrucció futura:

| Fulla | Forma individual | Forma futura batched | Dtype |
|---|---|---|---|
| `layer1.weight` | `[16,15]` | `[C,16,15]` | `float32` |
| `layer1.bias` | `[16]` | `[C,16]` | `float32` |
| `layer2.weight` | `[7,16]` | `[C,7,16]` | `float32` |
| `layer2.bias` | `[7]` | `[C,7]` | `float32` |

El model se separa deliberadament de la inicialització: PR-13 només construeix des de quatre arrays explícits, finits i `float32`. Aquesta estructura prepara l'eix de capacitat fixa `C` sense implementar-lo.

## Observacions, accions i versionat

El payload canònic incorpora nom, versió, topologia, activacions, biaixos, dtype, descriptors ordenats, total de paràmetres, semàntica lineal, i el nom, versió, mida i digest de l'[esquema d'observacions](local_observation_schema_v1.md). També incorpora l'ordre complet d'`ActionCode`: `STAY`, `MOVE_NORTH`, `MOVE_SOUTH`, `MOVE_EAST`, `MOVE_WEST`, `EAT`, `REPRODUCE` (índexs 0–6). Així, un canvi perceptiu o una renumeració d'accions canvia la identitat de la política.

Els genomes són incompatibles entre versions de l'esquema. Persistència i reconstrucció hauran de validar versió, digest, ordre, formes i dtype; no hi ha migració automàtica. `PolicyMLP` només expressa preferències: la legalitat i la resolució d'accions no formen part del model.

| Funcionalitat | PR responsable |
|---|---|
| Esquema d'observacions | PR-12 |
| Model neuronal individual | PR-13 |
| Inicialització de genomes | PR-14 |
| Representació batched | PR-14 |
| Inferència poblacional | PR-15 |
| Selecció d'accions | PR-15 |
| Validació d'accions | PR-16 |
| Mutació | PR-23 |
| Persistència de genomes | PR-39 |
