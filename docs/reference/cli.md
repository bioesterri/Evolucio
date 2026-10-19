# CLI de validació i execució

La CLI `evolucio` reutilitza Typer i les API canòniques del projecte. Accepta
configuracions JSON, YAML i YML. Exemples amb la configuració del repositori:

```console
evolucio --help
evolucio --version
evolucio config validate tests/fixtures/config/valid_v1.json
evolucio config inspect tests/fixtures/config/valid_v1.json
evolucio config inspect tests/fixtures/config/valid_v1.json --format json
evolucio run tests/fixtures/config/valid_v1.json --steps 2 --chunk-steps 1
evolucio run tests/fixtures/config/valid_v1.json --steps 10 --chunk-steps 4 --dry-run
```

`config validate` carrega i valida mitjançant `load_config`, congela mitjançant
`freeze_config` i mostra `schema_version`, `config_hash` i el digest canònic de
`CompileSignature`. No inicialitza l'estat ni executa passos.

`config inspect` mostra tots els camps normalitzats, inclosos els valors per defecte,
i `static_compile_contract`, obtingut directament de `build_compile_signature`.
El contracte estàtic identifica els camps que determinen compilació. La configuració
completa conserva els blocs canònics: inclou paràmetres ecològics dinàmics i camps
estàtics; no els reclassifica. El bloc `runtime` conté els paràmetres tècnics del fitxer.
L'ordre és determinista i `--format json` permet redirecció sense text addicional.

`run` requereix `--steps` explícit. `--chunk-steps` reutilitza `runtime.chunk_size`
quan s'omet. Les dues opcions construeixen `RunPlan`; no modifiquen `CoreConfig`,
el fitxer normalitzat ni el seu hash. Canviar la partició en chunks no canvia la
semàntica ecològica. La signatura mostrada és la del fitxer, inclòs el seu
`runtime.chunk_size`; la longitud efectiva dels chunks és la de `RunPlan`.
La seed és sempre la del fitxer i continua sent obligatòria. No hi ha overrides
de paràmetres ecològics.

El flux és configuració validada → `compile_config` → `initialize_simulation`
(composició dels inicialitzadors canònics de món, població i genomes) → `RunPlan`
→ `run_simulation` → `DTOChunkConsumer` → resum. No es conserva historial:
només l'última mètrica i els totals de naixements i morts. No s'atura per extinció.
Per defecte només s'imprimeix el resum final; `--quiet` elimina aquesta sortida.

`--dry-run` valida configuració i opcions, i mostra ruta, hash, seed, passos,
chunks complets, nombre total de chunks i remainder. No crida `compile_config`,
no inicialitza l'estat ni compila o executa `run_chunk`. Els imports del runtime
es carreguen quan cal: el dry-run no importa JAX ni Equinox ni inicialitza cap backend.

| Codi | Significat | Canal |
|---|---|---|
| 0 | Èxit, ajuda o versió | Resultats a stdout |
| 1 | Error inesperat d'execució, amb tipus i causa breu | stderr |
| 2 | Ús, fitxer, configuració o inicialització invàlids | stderr |

Els errors esperables no mostren traceback. No s'escriu cap resultat persistent,
manifest, snapshot ni checkpoint, encara que el fitxer contingui intencions de
persistència: aquest adaptador només valida, inspecciona i executa.

## Modes tècnics

`run --mode smoke|debug|experiment|benchmark` selecciona instrumentació al host.
El valor per defecte és `experiment`. Consulta [els modes d’execució](run_modes.md)
per als checks, el warm-up, el timing i la semàntica de `--quiet`. El dry-run
mostra també el mode i la política sense inicialitzar el backend.
