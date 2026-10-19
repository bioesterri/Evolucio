# Modes d'execució

`RunMode` és un perfil tècnic del host: **run mode != model configuration**.
No entra a `CoreConfig`, `config_hash` ni `CompileSignature` i no canvia cap
paràmetre ecològic, mètrica científica, RNG, ID o durada. L'horitzó sempre és
`RunPlan.total_steps`; smoke i debug es recomanen per runs petits, sense límits arbitraris.

| Mode | Diagnòstics | Camí | Telemetria al consumer | Timing |
| --- | --- | --- | --- | --- |
| smoke | Estructurals inicials i estat final | Fast path per chunks | Sí | No |
| debug | Inicial, final i transició de cada step | Checked step compilat | Sí, batches d'1 step | No |
| experiment | Sense checks addicionals | Fast path PR-30 | Sí | No |
| benchmark | Sense checks addicionals | Mateix fast path PR-30 | Sense consumer ni DTOs | Sí, després de warm-up |

`get_mode_policy` retorna una política immutable centralitzada. `run_with_mode`
és l'únic dispatch dels modes. El mode per defecte és `experiment`, que conserva
el resum i el comportament normal del PR-32. La CLI valida i inicialitza amb les
mateixes API canòniques abans de cridar aquest orquestrador.

## Smoke i debug

Smoke comprova els invariants estructurals abans del runtime i els invariants
finals després del runtime. És una comprovació ràpida dels extrems, no de totes
les transicions. Una corrupció informa de l'invariant, el nombre de violacions
i la fase `initial` o `final`; no repara l'estat.

Debug comprova l'estat inicial i executa `simulation_step_checked` del PR-27
compilat amb `eqx.filter_jit`. Després de cada pas materialitza els reports al
host. Una fallada atura el bucle abans del pas següent i informa del step exacte,
la fase i els codis i recomptes no nuls. El step del diagnòstic és el pas que
s'intentava executar; el step del check final de smoke és el de l'estat final.
`RunDiagnosticError.failure` conté aquesta informació sense cap dump d'estat.

Cada pas correcte lliura mètriques i events com un `HostChunkBatch` d'un step,
compatible amb `DTOChunkConsumer` del PR-31. No hi ha prints, callbacks o logging
dins JIT. Només es reté l'últim `DebugStepDiagnostic`, sense historial de reports.
Debug és deliberadament més lent que smoke perquè comprova totes les transicions
i creua la frontera amb el host a cada step.

L'extinció, la manca de reproducció o una població baixa no són errors tècnics:
els quatre modes completen tot l'horitzó si no falla cap invariant ni el runtime.

## Experiment i benchmark

Experiment delega al runtime PR-30, adequat per obtenir telemetria científica.
Benchmark mesura el cost tècnic d'aquest mateix runtime. `steps_per_second` no
és una mètrica evolutiva. `RunMode.EXPERIMENT` tampoc és un `experiment_type`
funcional ni crea un `experiment_id` o un registre d'experiments.

El warm-up executa `run_chunk` per a cada longitud realment necessària i
sincronitza abans de descartar-ne els resultats. Per 10 passos amb chunks de 4
escalfa 4 i 2; per 2 passos amb chunks de 4 només escalfa 2. Totes les crides
reutilitzen l'estat i l'acumulador inicials, purs i sense donació de buffers.
El run mesurat torna a començar exactament des d'aquests inputs: el warm-up
no consumeix passos, RNG, IDs ni mètriques del run real.

El cronòmetre `time.perf_counter` comença després del warm-up i la sincronització
inicial. Mesura **host runtime + run_chunk + frontera device-host necessària**,
amb `consumer=None`. El runtime conserva els seus `device_get` normals; no es
converteixen els chunks a DTOs. Abans d'aturar el cronòmetre es força
`jax.block_until_ready` sobre l'estat i l'acumulador finals.

Això no és temps pur de kernel. Exclou validació, inicialització, warm-up i
format del report final. Depèn del backend, del dispositiu, de versions, càrrega
del sistema, formes i llargada del run; per això les proves només exigeixen
temps i taxa positius, recompte correcte i resultat ecològic equivalent. No
s'ofereixen mètriques de memòria ni profiling de GPU/XLA: no hi ha una API
portable fiable incorporada. El report identifica `platform` i `device_kind`.

## CLI

```console
uv run evolucio run tests/fixtures/config/valid_v1.json --mode smoke --steps 4 --chunk-steps 2
uv run evolucio run tests/fixtures/config/valid_v1.json --mode debug --steps 2
uv run evolucio run tests/fixtures/config/valid_v1.json --mode experiment --steps 4 --chunk-steps 2
uv run evolucio run tests/fixtures/config/valid_v1.json --mode benchmark --steps 4 --chunk-steps 2
uv run evolucio run tests/fixtures/config/valid_v1.json --mode debug --steps 10 --dry-run
```

El dry-run mostra el mode, el pla, la política de diagnòstics i si el timing i
el warm-up estan habilitats. No crida `run_with_mode`, no inicialitza l'estat i
continua sense importar JAX ni Equinox. `--quiet` suprimeix el resum normal de
smoke, debug i experiment, però conserva els errors; en benchmark conserva el
report temporal mínim. No s'imprimeix progrés per step.

Els resultats van a stdout. Les violacions diagnosticades van a stderr, sense
traceback i amb codi 1; la configuració o les opcions invàlides continuen donant
codi 2. No s'afegeixen persistència, snapshots, checkpoints, resume, sweeps,
profilers ni canvis ecològics.
