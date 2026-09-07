# Metabolisme, costos d'acció i balanç energètic v1

## Ordre temporal

El contracte separa dues funcions pures que el futur `step` haurà d'invocar en punts diferents:

1. **Preacció:** actualització ambiental → metabolisme basal i increment d'edat → viabilitat
   preacció → observacions i política.
2. **Postacció:** moviment → alimentació → costos de les accions executades → viabilitat
   postacció → reproducció.

`apply_basal_metabolism_and_age` resta `basal_cost` a cada agent viu amb energia finita i
incrementa una vegada l'edat `int32` vàlida. Un slot inactiu conserva energia i edat. Una energia
negativa és legítima: no hi ha clipping ni mort en aquest contracte. Energia activa no finita,
edat negativa o edat que desbordaria `int32` es conserven i eleven els diagnòstics escalars.

`apply_action_energy_costs` consulta els resultats, no les propostes: només `MOVED` paga moviment,
i només `FED_FULL` o `FED_PARTIAL` paga alimentació. Intents fallits, `STAY`, `REPRODUCE` i slots
inactius paguen zero. Una incoherència anul·la tots els costos del slot i incrementa
`invalid_action_cost_input_count`. El cost reproductiu queda diferit al naixement atòmic.

## Contracte vectoritzat

| Resultat | Forma | Dtype |
|---|---:|---|
| `basal_cost_applied`, costos de moviment, alimentació i acció | `[C]` | `REAL_DTYPE` (`float32`) |
| `age_incremented` | `[C]` | `MASK_DTYPE` (`bool`) |
| recomptes d'inputs invàlids | `()` | `COUNT_DTYPE` (`int32`) |
| energia esperada i residu | `[C]` | `REAL_DTYPE` (`float32`) |
| totals i residu absolut màxim | `()` | `REAL_DTYPE` (`float32`) |

Les formes depenen exclusivament de la capacitat fixa. Les funcions no empren bucles per agent,
callbacks, efectes laterals ni RNG, i són compatibles amb eager, JIT i `lax.scan`.

## Balanç

Per als slots de `tracked_mask`:

```text
expected_energy_after = energy_before - basal_cost_applied
                        + feeding_energy_gained
                        - movement_cost_applied - feeding_cost_applied
residual = energy_after - expected_energy_after
```

La funció no incorpora toleràncies. Els vectors no seguits es representen amb zero al report i
no entren als totals ni a `max_abs_residual`.

## Esquema i configuració

L'esquema és `preaction_basal_postaction_success_costs_v1`, versió 1. El digest SHA-256 es calcula
sobre JSON canònic i, amb la versió, forma part de `CompileSignature` v11. `basal_cost` és finit i
estrictament positiu; `movement_cost` i `feeding_cost` són finits i no negatius. Els tres són
escalars dinàmics `REAL_DTYPE`: alteren `config_hash`, però no les formes ni la signatura.

## Límits posteriors

- PR-20 decidirà viabilitat i mort abans i després de les accions.
- PR-21 i PR-22 implementaran reproducció, cost reproductiu i naixement atòmic.
- PR-25 compondrà el pas complet, inclosa l'energia retirada per morts.
- PR-27 integrarà execució i comptabilització completa de morts i naixements.

Aquest PR no modifica `SimulationState` ni implementa viabilitat, mort, reproducció, naixements,
`step`, persistència o mètriques acumulades.
