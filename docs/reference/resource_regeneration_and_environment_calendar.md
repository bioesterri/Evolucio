# Regeneració de recursos i calendari ambiental

## Objectiu i ordre temporal

El PR-09 prepara el món per a les decisions del pas actual. Primer resol la fase global, després actualitza tota la capa ambiental i regenera recursos abans que es calculin les observacions del pas. Al pas zero la fase ja és observable i el seu multiplicador també modula la regeneració del dèficit inicial. El PR-27 cridarà `update_world_for_step(state.world, state.step, config.world)` abans del metabolisme, la viabilitat preacció, les observacions, la política i les accions.

## Regeneració

Per cel·la, amb recurs `R`, capacitat `K`, taxa basal `r` i multiplicador actiu `m`:

```text
f = r * m
gap = K - R
R_next = R + gap * f
```

La configuració garanteix `0 <= r < 1` i `0 <= m <= 1`. Amb `0 <= R <= K`, la fórmula garanteix `R <= R_next <= K` sense clipping. Una cel·la plena no canvia; una cel·la buida recupera una fracció del dèficit; taxa o multiplicador zero anul·len la regeneració. La recuperació s'alenteix en apropar-se a la capacitat i una taxa pròxima a u és molt ràpida, però no instantània.

## Calendari i ambient

Es reutilitza el contracte host existent: `regeneration_rate` és la taxa, `environment_schedule` el calendari, `end_step` expressa el final en lloc d'una durada, i `stress_level` és el valor aplicat a la capa ambiental. Cada fase ocupa `[start_step, end_step)` i ha d'acabar dins de `runtime.steps` per evitar fases parcialment o totalment inexecutables. Les fases estan estrictament ordenades, no se solapen, poden ser consecutives o deixar buits. Als buits i amb calendari buit s'apliquen el multiplicador basal `1.0` i `environment_initial_value`.

La resolució compara vectorialment el pas amb tots els intervals. Si una fase és activa selecciona els seus valors; altrament retorna `NO_ACTIVE_PHASE`. La capa `[H,W]` queda global i uniforme. El calendari és piecewise constant: no hi ha interpolació, variació regional, soroll ni RNG.

## Configuració compilada i signatura

`EnvironmentCalendarCoreConfig.phase_count` és estàtic perquè determina les formes `[L]`. Inicis i finals (`STEP_DTYPE`), multiplicadors i valors ambientals (`REAL_DTYPE`), taxa, capacitat i valor basal són dinàmics. Tots afecten `config_hash`; només canviar `L` altera `CompileSignature.environment_schedule_length`. Aquest camp ja existia i no s'ha incrementat la versió 3 de la signatura. Calendaris llargs tenen cost de resolució lineal i una longitud diferent implica una classe d'executable diferent.

## Garanties i limitacions

Es preserven formes i dtypes de recursos, ambient i ocupació; l'ocupació queda intacta. Amb entrades vàlides els resultats són finits i acotats. Calendaris amb multiplicadors baixos durant períodes llargs poden reduir fortament la productivitat. El model només admet fases globals constants i no diagnostica estats invàlids (PR-26).

| Funcionalitat | PR responsable |
|---|---|
| Població inicial | PR-10 |
| Ocupació real | PR-11 |
| Observacions ambientals | PR-12 |
| Alimentació i consum | PR-18 |
| Metabolisme afectat per ambient | PR-19 o PR posterior explícit |
| Mortalitat i viabilitat | PR-20 i PR-21 |
| Mètriques ambientals | PR-25 |
| Integració del step | PR-27 |
| Execució amb `lax.scan` | PR-28 |
