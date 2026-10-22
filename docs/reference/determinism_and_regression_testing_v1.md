# Determinisme i proves de regressió v1

## Contracte de determinisme

El projecte considera determinista una execució quan el mateix codi, les mateixes versions de
dependències rellevants, el mateix backend i entorn, la mateixa configuració, el mateix estat
inicial i la mateixa llavor produeixen el mateix resultat semàntic.

Aquest contracte no promet identitat bit a bit entre CPU i GPU, arquitectures diferents ni
versions diferents de JAX, XLA o les dependències numèriques.

## Eager, JIT i particionament

Les proves comparen `simulation_step` eager amb `eqx.filter_jit`, i `run_chunk_impl` amb
`run_chunk`. També comparen una execució monolítica amb particions diferents de chunks. La mida
del chunk és una decisió tècnica i no una variable ecològica.

## Política numèrica

Booleans, enters, identificadors, codis, màscares, posicions, genealogia i passos exigeixen
igualtat exacta. Els `float32` exigeixen forma i dtype idèntics, valors finits i toleràncies
globals `atol=1e-6` i `rtol=1e-6`.

## Fixture de llavor fixa

`tests/regression/fixtures/fixed_seed_v1.json` és un resum auditable, no un dump de l'estat.
Conté la llavor, els passos, la identitat de configuració i contractes, totals finals,
checkpoints temporals i una mostra genealògica amb resum de mutació.

La baseline només es modifica davant un canvi semàntic intencionat i revisat. Quan pertoqui,
primer s'ha d'actualitzar la versió del contracte afectat. `pytest` mai no regenera ni accepta
automàticament la baseline.
