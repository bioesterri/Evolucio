# PR-34 — Execució de lots d’experiments i conjunts de llavors

Branca: `pr-34/experiment-batches-seed-sets`.

## Descripció del PR

Faltava una unitat host per comparar configuracions amb el mateix conjunt explícit de
llavors. Afegeix `evolucio.experiments`: `SeedSet`, `ExperimentCase`, `BatchPlan`, expansió
pura a `BatchRunSpec`, preflight i resultats escalars individuals (`BatchRunResult`) i del
lot (`BatchResult`). L’ordre és case-major: 2 configuracions × 3 seeds produeixen 6 runs,
`A/101, A/202, A/303, B/101, B/202, B/303`. No s’eliminen les observacions per seed.

Tots els documents i les còpies normalitzades amb seeds es validen abans del primer run.
L’execució real també comprova la conversió canònica de tots els documents a `CoreConfig`
abans d’inicialitzar. Cada run inicialitza state, RNG, IDs i acumulador nous i reutilitza
`RunPlan` i `run_with_mode` dels PR-30/33. La factory de consumers rep el spec del run i
pot retornar `None`. No es comparteix cap consumer implícit ni s’emmagatzemen carries finals.

La seed forma part del document host actual i del hash canònic. Es valida una còpia nova
per seed i es calcula el hash amb `freeze_config`, preservant exactament el contracte
existent. El preflight mostra hashes base i per run; els resultats registren també el
digest de `CompileSignature`. El pla de batch no entra a cap d’aquestes identitats.

Les fallades tècniques s’associen a case i seed i el lot continua per defecte. Amb fail-fast
es propaga `BatchExecutionError`, amb el prefix executat i el recompte sol·licitat; no es
retorna un resultat parcial ambigu. Els errors de definició impedeixen començar el lot.
En un run fallit, el runtime no ofereix carry parcial i els passos/mètriques són `None`.

La CLI afegeix `evolucio batch run --config CASE=PATH` repetible, `--seeds 11,22,33`,
`--steps`, `--chunk-steps`, `--mode`, `--dry-run` i `--fail-fast`. Mostra la matriu completa
abans d’executar, una línia per run i els totals finals; les fallades van a stderr amb
case i seed. Codis: 0 èxit complet, 1 fallada tècnica, 2 definició/CLI/config invàlides.
El dry-run és host-only: sense imports JAX/Equinox, inicialització ni compilació.

## Disseny experimental i límits

Paired seeds permet comparar `A(seed_i)` amb `B(seed_i)` sota el mateix contracte RNG,
sense confondre canvis de configuració amb conjunts de llavors diferents. No sincronitza
artificialment trajectòries divergents. `experiment` és el mode recomanat; smoke/debug
són diagnòstics i benchmark conserva el seu ús de rendiment sense telemetria.

`BatchResult` són observacions, no una prova d’adaptació. Una sola seed no és base suficient
per concloure adaptació. Calen controls, distribucions i anàlisi posterior. Sense estadística,
rankings, fitness, sweeps automàtics, seeds aleatòries, paral·lelisme, persistència,
identificadors persistents, checkpoint/resume ni early stopping. Cap canvi al core ni
cap dependència nova; `uv.lock` es manté intacte. No es fa merge automàtic.

## Fitxers

- `src/evolucio/experiments/{__init__,seeds,types,preflight,batch}.py`: API i orquestració.
- `src/evolucio/cli/batch_command.py`, `src/evolucio/cli/main.py`: CLI de lots.
- `pyproject.toml`: contracte d’importació `core` no pot dependre d’`experiments`.
- `tests/unit/experiments/{__init__,conftest,test_seeds,test_batch_plan,test_preflight,test_batch_runner}.py`.
- `tests/unit/cli/test_batch_command.py`: sortides, errors, exit codes i dry-run.
- `tests/experiment_helpers.py`, `tests/integration/test_experiment_batch.py`,
  `tests/integration/test_batch_seed_pairing.py`: casos petits i execucions reals.
- `docs/reference/experiment_batches.md`, guia CLI, índex documental i `README.md`.
- Aquest document de descripció del PR.

## Proves

La prova principal verifica exactament `A/101, A/202, A/303, B/101, B/202, B/303` i
`seeds(A) == seeds(B) == (101, 202, 303)`. Dues configuracions difereixen només en el cost
basal i conserven la mateixa signatura de compilació; no es declara cap guanyador.

La prova crítica d’independència executa `A/10, A/20`, després `A/20` sol i repeteix el
lot. Compara el resum final, població, món, genomes, RNG, IDs i acumulador amb el contracte
numèric del PR-29. Comprova també inicialitzacions separades, IDs fundadors reiniciats,
acumuladors buits, naixements i morts reals. Reordenar A/B només altera l’ordinal.

Es cobreixen els quatre modes, factories i consumers separats, absència de consumer,
fallades d’inicialització/runtime/factory/consumer, continuació, fail-fast i prevalidació
amb zero inicialitzacions, inclosos errors de representabilitat float32.

## Verificacions

- Suite prèvia: `uv run pytest`, **684 proves passades** abans de modificar codi.
- Suite final: `uv run pytest`, **753 proves passades** (69 noves), sense skips ni xfails.
- `uv run coverage erase`, `uv run coverage run -m pytest`, `uv run coverage report`:
  **753 proves passades**, **97 % global**, **100 % a experiments i a batch_command.py**.
- `uv run pytest tests/unit/experiments`: **45 proves passades**.
- `uv run pytest tests/integration/test_experiment_batch.py`: **6 proves passades**.
- `uv run pytest tests/integration/test_batch_seed_pairing.py`: **1 prova passada**.
- `uv sync --group dev`, `uv sync --locked --group dev`, `uv lock --check`: correctes.
- `uv run ruff check .`, `uv run ruff format --check .`: correctes.
- `uv run pyright`: 0 errors i 0 warnings.
- `uv run lint-imports`: 6 contractes conservats, 0 trencats.
- `uv pip check`: 36 paquets compatibles.
- `uv build`: sdist i wheel construïts correctament.
- `git status --short` revisat i `git diff --check` correcte.
- Smoke CLI dry-run: 2 cases × 2 seeds, matriu completa, codi 0.
- Smoke CLI real: 4 runs completats, 0 fallits, codi 0.

Les configuracions petites dels smoke tests difereixen només en el cost basal. Ordres
executades des del repositori:

```console
uv run evolucio batch run \
  --config baseline=/workspace/work/pr34/A.json \
  --config variant=/workspace/work/pr34/B.json \
  --seeds 11,22 --steps 4 --chunk-steps 2 --mode experiment --dry-run

uv run evolucio batch run \
  --config baseline=/workspace/work/pr34/A.json \
  --config variant=/workspace/work/pr34/B.json \
  --seeds 11,22 --steps 4 --chunk-steps 2 --mode experiment
```

Cap verificació pendent. Implementació a la branca indicada, sense merge automàtic.
