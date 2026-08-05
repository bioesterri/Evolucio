# Contracte d'accions i validació local v1

## Objectiu i nivells d'acció

El PR-16 transforma les decisions neuronals `proposed_actions` del PR-15 en
`routed_actions`, intencions localment vàlides que encara han de passar pels resolutors
especialitzats. Una futura `executed_actions` indicarà efectes realment aplicats; no existeix en
aquest PR. Per tant, `ACCEPTED` significa «encaminada», mai «executada amb èxit».

Les accions `EAT` encaminades es resolen segons el contracte d'[alimentació](feeding_resource_competition_and_energy_transfer_v1.md).

La validació és pura, determinista i preliminar. Comprova slot actiu, codi conegut, posició
`[x,y]` dins dels límits tancats, destinació cardinal dins del món i recurs estrictament positiu
per `EAT`. Tot rebuig s'encamina exclusivament a `STAY`, sense cercar una segona acció útil.

## Codis, deltes i encaminament

El primer component és `x`, el segon `y`, i el nord redueix `y`. No hi ha wrapping ni clipping.

| Acció | Codi | Delta `[dx,dy]` | Validació local | Resolutor posterior |
|---|---:|---:|---|---|
| `STAY` | 0 | `[0,0]` | Slot, codi i posició vàlids | PR-19 |
| `MOVE_NORTH` | 1 | `[0,-1]` | Destinació dins del món | PR-17 |
| `MOVE_SOUTH` | 2 | `[0,1]` | Destinació dins del món | PR-17 |
| `MOVE_EAST` | 3 | `[1,0]` | Destinació dins del món | PR-17 |
| `MOVE_WEST` | 4 | `[-1,0]` | Destinació dins del món | PR-17 |
| `EAT` | 5 | `[0,0]` | Recurs local `> 0` | PR-18 |
| `REPRODUCE` | 6 | `[0,0]` | Slot, codi i posició vàlids | PR-21 / PR-22 |

| Acció proposada | Condició local | Acció encaminada | Resolutor posterior |
|---|---|---|---|
| `STAY` | Slot i posició vàlids | `STAY` | PR-19 |
| Moviment | Destinació dins el món | Mateix moviment | PR-17 |
| Moviment | Destinació fora | `STAY` | Cap |
| `EAT` | Recurs local `> 0` | `EAT` | PR-18 |
| `EAT` | Sense recurs | `STAY` | Cap |
| `REPRODUCE` | Slot i posició vàlids | `REPRODUCE` | PR-21 / PR-22 |
| Codi invàlid | — | `STAY` | Cap |
| Slot inactiu | — | `STAY` | Cap |

## Causes i precedència

La precedència és explícita i no deriva del valor numèric: `INACTIVE_SLOT`,
`INVALID_ACTION_CODE`, `INVALID_ACTOR_POSITION`, `MOVE_OUT_OF_BOUNDS`, `EAT_NO_RESOURCE` i,
finalment, `ACCEPTED`.

| `ActionValidationCode` | Valor | Causa |
|---|---:|---|
| `ACCEPTED` | 0 | La proposta supera la validació local. |
| `INACTIVE_SLOT` | 1 | El slot no representa un agent actiu. |
| `INVALID_ACTION_CODE` | 2 | El codi no pertany a `ActionCode`. |
| `INVALID_ACTOR_POSITION` | 3 | La posició actual activa és fora del món. |
| `MOVE_OUT_OF_BOUNDS` | 4 | La destinació cardinal és fora del món. |
| `EAT_NO_RESOURCE` | 5 | El recurs de la cel·la actual no és positiu. |

Els slots inactius conserven proposta i posició, però produeixen `STAY` i `INACTIVE_SLOT`, fins
i tot amb codi o posició invàlids. Una posició activa invàlida mai no s'interpreta com la cel·la
segura usada internament per evitar indexació il·legal. Un moviment rebutjat conserva com a
`move_targets` la posició actual.

## Inacció explícita i fallback

Una proposta explícita `STAY` vàlida conserva `proposed_actions = routed_actions = STAY` i rep
`ACCEPTED`. Un fallback conserva la proposta original, encamina `STAY` i registra una causa
diferent d'`ACCEPTED`. Això permetrà al PR-25 distingir espera voluntària i intent fallit. La
pèrdua del torn és deliberada: corregir cap a una direcció, alimentació, reproducció, segona
preferència o alternativa aleatòria canviaria la semàntica neuronal i la pressió selectiva.

## Formes, dtypes i JAX

`ActionValidationResult` conté `proposed_actions int32[C]`, `routed_actions int32[C]`,
`validation_codes int32[C]` i `move_targets int32[C,2]`. L'entrada addicional és `alive bool[C]`
i `resources float32[H,W]`. Tota la capacitat es processa amb operacions JAX vectoritzades;
les formes no depenen de `sum(alive)`. Els codis desconeguts se sanititzen a `STAY` abans
d'indexar l'única taula de deltes, però es conserven al resultat. El contracte funciona en eager,
JIT i `lax.scan`, és equivariant a la permutació conjunta de slots i no usa RNG, I/O ni estat
global.

## Esquema versionat i `CompileSignature`

- Nom: `discrete_actions_local_validation_stay_fallback_v1`.
- Versió: `1`.
- Scope: `local_preliminary`.
- Digest: `85dbbbb9418746b480b119e956a2d4c4297b9b3739034db42b1bba79871890c3`.

El payload JSON canònic fixa codis i ordre, recompte, deltes, orientació, límits tancats, causes,
precedència, fallback, semàntica de les dues inaccions, condició estricta de recurs, reproducció
diferida, absència d'ocupació, costos i RNG, i els tres nivells d'acció. No conté dimensions,
capacitat, seed, valors del món ni decisions concretes. `CompileSignature` v8 incorpora versió i
digest del contracte, a més del recompte d'accions i els digests d'observacions, política i
selecció. Un canvi funcional futur exigeix revisar versió, digest, signatura i compatibilitat
experimental.

## Límits de responsabilitat

No es consulta ocupació: un ocupant pot marxar i els moviments simultanis necessiten el PR-17.
Tampoc es tria guanyador entre destinacions coincidents. Diversos agents poden encaminar `EAT`
sobre el mateix recurs positiu perquè repartiment, consum i transferència corresponen al PR-18.
`REPRODUCE` no comprova energia, edat, viabilitat o slots perquè la decisió atòmica pertany als
PR-21 i PR-22. Cap validació aplica metabolisme o costos; aquesta economia correspon al PR-19.

| Funcionalitat | PR responsable |
|---|---|
| Proposta neuronal | PR-15 |
| Validació local i fallback | PR-16 |
| Conflictes de moviment | PR-17 |
| Alimentació i recursos | PR-18 |
| Costos | PR-19 |
| Viabilitat preacció | PR-20 |
| Viabilitat postacció | PR-21 |
| Reproducció atòmica | PR-22 |
| Mètriques d'intents i fallades | PR-25 |
| Integració temporal | PR-27 |
# Continuació espacial

Les intencions cardinals acceptades són resoltes pel contracte de
[moviment cardinal i conflictes espacials v1](cardinal_movement_and_spatial_conflicts_v1.md), que
pot convertir un intent fallit a `STAY` sense reinterpretar el resultat de la validació local.
