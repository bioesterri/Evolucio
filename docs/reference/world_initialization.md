# Inicialització del món 2D

## Contracte espacial

El món inicial és una graella discreta de forma `[height, width]`. Les coordenades dels agents
s'expressen com `[x, y]`, però els mapes s'indexen com `[y, x]`. Els límits són tancats i no
toroidals: una coordenada és vàlida exactament quan `0 <= x < width` i
`0 <= y < height`; no hi ha *wrapping*, clipping ni substitució de posicions.

`WorldState` conté `resources` i `environment` en `float32`, i `occupancy` en `int32`, sempre amb
forma `[height, width]`. `initial_resource_mean` és la **mitjana inicial per cel·la**, no un total,
una probabilitat ni una fracció.

## Recursos

El control uniforme usa `resources[y, x] = initial_resource_mean` i no consumeix aleatorietat.
El patró de taques deriva claus separades per als centres `x` i `y` i mostra `K` centres uniformes
dins del rectangle. Per a cada centre `k`:

```text
d²_k = (x - center_x_k)² + (y - center_y_k)²
raw[y,x] = sum_k exp(-d²_k / (2 * resource_patch_radius²))
centered = raw - mean(raw)
pattern = centered / max(max(abs(centered)), epsilon)
margin = min(initial_resource_mean, resource_capacity - initial_resource_mean)
resources = initial_resource_mean + resource_patch_contrast * margin * pattern
```

El patró té mitjana aproximadament zero i interval `[-1, 1]`; per això ambdós modes conserven la
mateixa mitjana global dins la tolerància de `float32`. Un `clip` final a
`[0, resource_capacity]` protegeix només de l'arrodoniment: no substitueix la validació host.
Les distàncies són euclidianes ordinàries i les taques de vora queden truncades.

## Configuració i compilació

| Camp | Validació | Categoria | Efecte |
|---|---|---|---|
| `width`, `height` | enters estrictes `> 0` | estàtic | forma |
| `boundary_mode` | `closed` | estàtic | semàntica de límits |
| `resource_distribution` | `uniform` o `patches` | estàtic | algoritme |
| `resource_patch_count` | enter estricte `>= 1` | estàtic | nombre de centres |
| `resource_capacity` | finit, `>= 0` | dinàmic `float32` | límit local |
| `initial_resource_mean` | finit, entre zero i capacitat | dinàmic `float32` | mitjana inicial |
| `resource_patch_radius` | finit, `> 0` | dinàmic `float32` | escala espacial |
| `resource_patch_contrast` | finit, `[0, 1]` | dinàmic `float32` | amplitud |
| `environment_initial_value` | finit, `[0, 1]` | dinàmic `float32` | valor basal |

Els camps estàtics formen part de `CompileSignature`; els dinàmics canvien `config_hash`, però no
la signatura ni les formes. L'esquema de configuració és 1.1 i la signatura de compilació és 3.

## RNG, ambient i ocupació

`initialize_world` deriva primer `WORLD_INITIALIZATION` de la root key i després
`RESOURCE_INITIALIZATION`. No avança `RngState`: la derivació és pura, reproduïble i independent
de l'ordre d'altres streams. Els centres `x` i `y` reben claus filles diferents.

L'ambient inicial és uniforme amb `environment_initial_value`, sense soroll, gradient ni canvi
temporal. L'ocupació inicial és zero perquè aquest PR no crea població.

## Garanties i treball posterior

La generació vectoritzada costa `O(height × width × resource_patch_count)` en temps i memòria
intermèdia. Les formes són fixes, els dtypes explícits i no hi ha NumPy, I/O ni estat aleatori
global.

| Funcionalitat | PR responsable |
|---|---|
| Regeneració | PR-09 |
| Calendari ambiental | PR-09 |
| Població inicial | PR-10 |
| Posicions inicials | PR-10 |
| Ocupació real | PR-11 |
| Observacions locals | PR-12 |
| Moviment | PR-17 |
| Alimentació | PR-18 |
