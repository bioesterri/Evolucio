# PR-32 — CLI de validació, execució i inspecció de configuracions

Branca: `pr-32/cli`.

## Descripció del PR

Exposa l'entry point `evolucio` amb Typer, ja present al projecte. Afegeix
`config validate`, `config inspect` en text i JSON, i `run` amb `--steps`
obligatori, `--chunk-steps`, `--quiet` i `--dry-run`. La configuració invàlida
impedeix inicialitzar i executar; el dry-run mostra el pla sense compilar arrays,
inicialitzar l'estat ni executar passos. Una prova en un procés nou comprova
que tampoc importa JAX ni Equinox.

Reutilitza `load_config`, la validació Pydantic, `freeze_config`, `config_hash`,
`build_compile_signature` i `compile_signature_digest`. La seed obligatòria és
la del fitxer; no hi ha overrides ecològics. `--chunk-steps` pren per defecte
`runtime.chunk_size` i configura `RunPlan`, sense modificar `CoreConfig` ni el hash.

El flux d'execució és configuració → `compile_config` → inicialització →
`RunPlan` → `run_simulation` del PR-30 → `DTOChunkConsumer` del PR-31 → resum.
Com que no existia un inicialitzador integral de `SimulationState`, s'afegeix
`runtime.initialization.initialize_simulation`, que compon les API canòniques
de RNG, món, població, IDs i genomes i comprova els seus diagnòstics.

Els exports del runtime es carreguen quan es necessiten i les anotacions dels
seus contractes no importen el core en execució; `RunPlan` és accessible sense
inicialitzar dispositius. La CLI és host-side. Un contracte d'importació impedeix que core, config,
runtime i host en depenguin. El consumer només conserva l'última mètrica i els
totals de naixements i morts. Una extinció no atura l'horitzó sol·licitat.

Resultats a stdout; errors a stderr. Codis: 0 èxit, 1 error inesperat d'execució,
2 ús/configuració/inicialització invàlids. Els errors esperables no mostren traceback.
La versió prové de la metadata real del paquet.

## Validació

Proves directes de `main(argv)` per a fitxers i configuracions invàlids,
inspecció determinista amb claus reordenades, JSON serialitzable, hash, seed,
ajuda, versió, runs petits, múltiples chunks, remainder, quiet, errors runtime,
població buida i extinció. Spies obligatoris per demostrar que la configuració
invàlida i el dry-run no executen. Neutralitat entre chunks de 4 i 7 amb 20 passos,
repetibilitat, consumer de memòria limitada i entry point instal·lat.

Les ordres d'ús i el contracte complet es documenten a
[la guia CLI](reference/cli.md).

## Abast

Sense dependències noves: `uv lock` no modifica `uv.lock`.
Sense persistència, manifests, checkpoints, resume, snapshots automàtics,
visualització, sweeps, early stopping ni overrides ecològics.
No es fa merge automàtic.

## Resultats de verificació

- Base inicial: `uv run pytest`, 602 proves passades.
- Versió definitiva: `uv run coverage run -m pytest`, 652 proves passades;
  `uv run coverage report`, 97 % de cobertura global (llindar: 90 %).
- Proves específiques: 46 unitàries CLI i 3 d'integració de l'entry point;
  conjunt CLI + runtime + entry point, 143 proves passades.
- `uv sync --group dev`, `uv sync --locked --group dev`, `uv lock --check`,
  `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright`,
  `uv run lint-imports`, `uv pip check` i `git diff --check`: correctes.
- Cinc contractes d'importació respectats; cap error de tipus.
- `uv build`: distribució font i wheel construïts correctament.
- Smoke tests reals d'ajuda, versió, validate, inspect text/JSON, dry-run i run
  amb 2 passos i chunks d'1: correctes. Run final: `completed`, `final_step: 2`,
  `alive_count: 128`. Proves executades amb CPU.
