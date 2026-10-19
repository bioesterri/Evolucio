# PR-33 — Modes smoke, debug, experiment i benchmark

Branca: `pr-33/run-modes`.

## Descripció del PR

Afegeix quatre perfils tècnics del host amb polítiques immutables i un dispatch
únic `run_with_mode`. La CLI accepta `--mode`, amb `experiment` per defecte;
el dry-run mostra el mode, el pla, els diagnòstics i si hi haurà warm-up i timing,
sense importar JAX ni Equinox.

| Mode | Diagnòstics | Camí | Telemetria al consumer | Timing |
| --- | --- | --- | --- | --- |
| smoke | Inicial/final | Runtime per chunks PR-30 | Sí | No |
| debug | Cada step i transició | Checked step PR-27 compilat | Sí, batches d'1 step | No |
| experiment | Cap check addicional | Runtime PR-30 sense canvis | Sí | No |
| benchmark | Cap check addicional | Mateix runtime PR-30 | No | Sí, amb warm-up |

Smoke comprova l'estructura inicial i l'estat final amb els invariants canònics.
Debug comprova l'estat inicial i inspecciona al host els reports de cada
`simulation_step_checked`. Una violació produeix `RunDiagnosticError` amb step,
fase, codi i recompte; atura abans de dispatchar el pas següent. La CLI la mostra
a stderr sense traceback i amb exit code 1, també amb `--quiet`. No repara estats
ni conserva historial de diagnòstics.

Les mètriques i els events de debug reutilitzen `HostChunkBatch` i l'adaptador
DTO del PR-31. Experiment conserva el camí i el resum del PR-32. Els runs buits
o extingits continuen fins al final de `RunPlan` si no hi ha fallades tècniques.

Benchmark escalfa cada longitud necessària, inclòs el remainder, sincronitza i
descarta els resultats. Torna a començar des de l'estat i l'acumulador originals:
el warm-up no consumeix passos, IDs ni RNG del run mesurat. `perf_counter` mesura
el runtime host + kernel normal + frontera device-host requerida, amb
`consumer=None` i sense conversió de DTOs. Sincronitza l'estat i l'acumulador
finals abans d'aturar el timer. Mostra temps, passos/s, plataforma i dispositiu;
`--quiet` conserva aquest report. No és temps pur de kernel ni una mètrica evolutiva.

## Neutralitat i proves

La prova central executa els quatre modes sobre un escenari petit amb moviment,
alimentació, reproducció, mutació i mort. Compara població, món, genomes, RNG,
IDs i acumulador finals, i la telemetria de smoke/debug/experiment, incloses
identitats genealògiques i resums de mutació. Reutilitza les toleràncies del PR-29:
exactitud discreta i `atol=rtol=1e-6` per floats.

També comprova hash i `CompileSignature` immutables, població buida, pas inicial
no zero, durada exacta després del warm-up, corrupció final de smoke, corrupció
d'una transició debug amb pas exacte i absència de passos posteriors, ordre de
warm-up/sincronització/timer, longituds amb i sense remainder i reports CLI.

No hi ha assertions de rendiment mínim ni speedup dependent de hardware. La
prova del dry-run en un procés nou comprova l'absència de JAX/Equinox per a cada mode.

## Fitxers

- `runtime/modes.py`: enums, polítiques, reports i dispatch.
- `runtime/diagnostics.py`: smoke i debug sobre checks canònics.
- `runtime/benchmark.py`: warm-up, sincronització i timing.
- `runtime/__init__.py`: API pública, sense imports de dispositiu per als plans.
- `cli/main.py`, `cli/run_command.py`: selecció, dry-run, errors i reports.
- Proves unitàries de modes/debug/benchmark i CLI; prova d'integració de neutralitat.
- Marcadors de paquet de tests per executar també els subconjunts independentment.
- `docs/reference/run_modes.md`, guia CLI i índex documental.

## Abast i limitacions

`run mode != model configuration`: no modifica `CoreConfig`, `config_hash`,
`CompileSignature`, durada, RNG ni regles o mètriques científiques. No hi ha
canvis al core ni dependències noves. Smoke/debug es recomanen per runs petits,
sense límits arbitraris. El timing depèn de hardware, backend, versions i càrrega;
exclou inicialització, warm-up i format del report.

Sense persistència, registry, experiment_id, run_id, snapshots, checkpoints,
resume, memory/GPU/XLA profiling, sweeps, early stopping ecològic ni canvis
científics per mode. No es fa merge automàtic.

## Verificacions executades

- Suite inicial sense canvis: `uv run pytest`, 652 proves passades.
- Suite final: `uv run coverage run -m pytest`, 684 proves passades;
  cobertura global del 97 % amb llindar del 90 %.
- `uv run pytest tests/unit/runtime`: 109 proves passades.
- Neutralitat + entry point: 8 proves passades, inclòs el dry-run dels quatre modes.
- `uv sync --group dev`, `uv sync --locked --group dev`, `uv lock --check`,
  Ruff lint i format, Pyright, els cinc contractes d'importació, `uv pip check`
  i `git diff --check`: correctes.
- `uv build`: wheel i distribució font construïts correctament.
- Les quatre ordres CLI documentades s'han executat sobre la configuració del
  repositori: smoke 4 passos, debug 2, experiment 4 i benchmark 4; totes amb
  `completed`. Smoke informa dels checks inicial/final correctes. Benchmark
  informa de warm-up completat, pas final 4 i temps/taxa positius sobre CPU.
- Cap dependència nova, cap canvi a `uv.lock` ni als schemas del core.
