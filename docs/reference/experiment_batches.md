# Lots d’experiments i conjunts de llavors

Un **case** és una variant identificada explícitament per `case_id`, amb un fitxer de
configuració completa. Un **run** és una simulació independent d’un case amb una seed.
Un **batch** combina tots els cases amb el mateix `SeedSet` explícit, immutable i ordenat.

## Disseny experimental

Les configuracions comparteixen les mateixes llavors mestres per permetre comparar
`A(seed_i)` amb `B(seed_i)` sense confondre el canvi de configuració amb un canvi de
llavors. No es força la sincronització de subkeys després que les trajectòries divergeixin:
es conserva el contracte RNG existent.

L’ordre és case-major, preservant l’ordre indicat dels cases i de les seeds:

```text
2 configs × 3 seeds = 6 runs
A/101
A/202
A/303
B/101
B/202
B/303
```

No s’ordenen ni es generen seeds automàticament. S’admeten enters Python de 0 a
`2**32 - 1`, segons `create_rng_state`; es rebutgen bools, duplicats i conjunts buits.
Els `case_id` són únics dins del lot. Dos cases amb identitats diferents poden apuntar
al mateix fitxer i el preflight en mostrarà els mateixos hashes.

Per dissenyar una comparació d’un factor cada vegada, prepara configuracions que només
difereixin en el factor controlat. El batch no dedueix ni valida aquest criteri científic.
**Una sola seed no és base suficient per concloure adaptació.** Calen controls, diverses
seeds, distribucions i anàlisi posterior.

## API host i identitat

```python
from pathlib import Path
from evolucio.experiments import (
    BatchPlan, ExperimentCase, SeedSet, expand_batch_plan,
    preflight_batch, run_batch, seeds_for_case,
)

plan = BatchPlan(
    cases=(
        ExperimentCase("baseline", Path("configs/baseline.json")),
        ExperimentCase("variant", Path("configs/variant.json")),
    ),
    seed_set=SeedSet((11, 22, 33)),
    total_steps=10000,
    chunk_steps=256,
)
view = preflight_batch(plan)
specs = expand_batch_plan(plan)
result = run_batch(plan=plan)
assert seeds_for_case(result, "baseline") == seeds_for_case(result, "variant")
```

`BatchRunSpec.identity`, per exemple `baseline/seed-11`, és una identitat transitòria;
no és un identificador persistent. Els ordinals comencen a 1. `expand_batch_plan` és pur,
determinista, sense I/O ni JAX. `BatchPreflight` conté només rutes, hashes i escalars/tuples.
La versió host `BATCH_INTERFACE_VERSION = 1` no entra a `config_hash` ni a `CompileSignature`.

La seed actualment forma part d’`ExperimentConfig` i del hash canònic complet. Cada fitxer
es carrega una vegada per case; per cada seed es crea una còpia normalitzada amb
`ExperimentConfig.model_validate` i es calcula `freeze_config`. No es modifica el document
base ni es redefineix el hash: diferents seeds poden tenir hashes diferents. El preflight
mostra el hash del document base i els hashes de les còpies per run. Els overrides de passos,
chunks i mode es passen al runtime; no modifiquen el document ni el hash.

Cada resultat conserva `case_id`, seed, mode, hash de la còpia validada i digest canònic de
`CompileSignature`. Signatures diferents són vàlides i poden causar recompilacions normals
de JAX. No hi ha cache manual de compilacions.

## Prevalidació i independència

Abans del primer run, es comproven tots els paths, es carreguen i validen tots els documents
i totes les còpies amb seeds, i es construeix la matriu completa. En execució real també es
converteixen totes les configuracions amb `compile_config` per detectar errors de
representabilitat abans d’inicialitzar cap simulació. El dry-run fa només la validació host:
no importa JAX/Equinox, no crea arrays, no inicialitza simulacions ni compila kernels.
Els errors que només pot detectar la conversió a `CoreConfig` requereixen execució real.

Cada run crida `initialize_simulation` i `create_empty_metrics_accumulator`, crea el seu
`RunPlan` i executa `run_with_mode`. Es reinicialitzen state, RNG, IDs, genealogia, events i
acumulador. Cap carry final es conserva a `BatchRunResult`; només hi ha observacions escalars.
El resum materialitza les màscares/llinatges finals i els totals de l’acumulador, també en
benchmark, sense activar telemetria en aquest mode.

`consumer_factory(spec)` crea un consumer independent per run segons el contracte PR-30
`ChunkConsumer(HostChunkBatch)`. Pot retornar `None`; ometre la factory és vàlid. Per obtenir
DTOs PR-31, la factory pot retornar `DTOChunkConsumer(el_seu_consumer)`. Benchmark conserva
la seva semàntica PR-33: no lliura chunks al consumer.

`experiment` és el mode recomanat per lots científics normals. `smoke` i `debug` serveixen
principalment per diagnòstic; `benchmark` mesura rendiment. Tots quatre són vàlids.

## Errors

- Errors de definició o configuració: el batch no comença; `continue_on_error` no hi aplica.
- `continue_on_error=True` (defecte): una fallada tècnica d’inicialització, runtime, factory o
  consumer produeix `FAILED` amb case, seed, tipus i missatge; continua el run següent.
- `continue_on_error=False`: es propaga `BatchExecutionError`, encadenada amb l’error original.
  Conté `runs` (prefix executat, inclosa la fallada) i `requested_run_count`. No retorna un
  `BatchResult` parcial ni executa specs posteriors.

Quan un run falla, el runtime actual no retorna el carry parcial. Per tant `steps_executed`,
`final_step` i les mètriques són `None`; no s’inventen recomptes ni es classifica com completat.
Un `BatchResult` retornat sempre satisfà `completed + failed == requested == len(runs)`.
Els callbacks host opcionals `on_preflight(view, specs)` i `on_result(result)` permeten
presentació; els seus errors es propaguen com a errors d’orquestració.

## CLI

```console
uv run evolucio batch run \
  --config baseline=configs/baseline.json \
  --config variant=configs/variant.json \
  --seeds 11,22,33 --steps 10000 --chunk-steps 256 \
  --mode experiment --dry-run
```

La sortida mostra nombre de cases, seeds i runs, mode, passos, chunk, hashes i matriu completa.
Per executar el mateix pla, elimina `--dry-run`:

```console
uv run evolucio batch run \
  --config baseline=configs/baseline.json \
  --config variant=configs/variant.json \
  --seeds 11,22,33 --steps 10000 --chunk-steps 256 --mode experiment
```

La CLI mostra tota la matriu abans del primer run i una línia per run acabat, sense línies
per step/chunk. Al final mostra sol·licitats, executats, completats i fallits; els errors amb
case i seed van a stderr. `--fail-fast` atura després de la primera fallada.

Codis de sortida: **0** tots completen; **1** fallada tècnica, encara que el lot continuï;
**2** CLI, pla o configuració invàlids.

## Límits

`BatchResult` són observacions de múltiples runs, no una prova d’adaptació. Es preserven
les observacions per seed; no es calculen mitjanes, diferències A-B, rankings ni fitness.
L’execució és seqüencial. Sense estadística inferencial, sweeps automàtics, generació
aleatòria de seeds, paral·lelisme, persistència, IDs persistents, checkpoint/resume ni
aturada anticipada. L’anàlisi posterior amb Polars queda fora d’aquest PR.

## Registre local opcional (PR-35)

L’API accepta `run_batch(..., output_root=Path("runs"))`: cada case/seed rep un UUID i
manifest propis. La CLI batch l’activa per defecte i admet `--output-root`. No hi ha un
manifest de run compartit per tot el batch. Vegeu [registres locals](local_run_records.md).
Aquesta extensió substitueix la limitació de manca de persistència local del contracte inicial;
no afegeix persistència massiva ni modifica el pla, les seeds o els hashes científics.
