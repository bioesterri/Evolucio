# Inicialització de la població fundadora

## Contracte de slots

La població té capacitat fixa `C = max_agents`: tots els camps tenen primer eix `C` i
`position` té forma `[C, 2]`. Els primers `N = initial_agents` slots s'activen amb
`alive[i] = i < N`; `alive` és l'única font de veritat. Un slot és una ubicació computacional
reutilitzable, no una identitat ni una prioritat ecològica. L'índex del slot no és `agent_id`.

`create_empty_population(C)` crea la representació canònica: `alive=False`, tots els IDs a
`NULL_ID`, posició `[-1,-1]`, energia zero i generació, edat i pas de naixement zero. El sentinel
de posició mai no decideix la vida del slot.

## Fundadors, energia i genealogia

Cada fundador rep IDs independents i no reciclables d'agent, genoma i llinatge. Té
`parent_id = NULL_ID`, llinatge propi, generació zero, edat zero, pas de naixement zero i
`initial_energy`. El `genome_id` només reserva la identitat del futur genoma: no existeixen encara
pesos, biaixos ni `GenomeBatch`.

Les tres reserves d'IDs són atòmiques. Si qualsevol espai s'esgota, `overflow=True`, cap comptador
avança, tots els IDs retornats són nuls, la població queda buida i el món d'entrada es conserva.
Les col·lisions espacials o la quantitat de cel·les no són overflow.

## Posicions, RNG i ocupació

La col·locació host `random` correspon al codi estable
`InitialPlacementCode.UNIFORM_WITH_REPLACEMENT = 0`. Es deriva el stream
`AGENT_INITIALIZATION` de la root key i, amb subíndexs 0 i 1, claus diferents per `x` i `y`.
Sempre es mostregen `C` coordenades uniformes i després s'emmascaren els slots inactius; canviar
`initial_agents` només revela mostres ja determinades. La root key no avança i recursos, ambient i
ordre de derivació d'altres streams no influeixen en les posicions.

El mostreig és amb reemplaçament: permet molts agents en una cel·la i, per tant, no exigeix
`initial_agents <= width * height`. Això representa solapament ecològic, no més slots que la
capacitat computacional. Per cada slot viu en `[x,y]` es calcula
`flat_index = y * width + x`; un recompte vectoritzat ponderat per `alive` construeix
`occupancy[y,x]`. Els slots inactius apunten temporalment a zero amb pes exactament zero.
Recursos i ambient es conserven exactament.

## Configuració i compilació

| Camp | Validació host | Categoria | `config_hash` | `CompileSignature` |
|---|---|---|---|---|
| `max_agents` | enter estricte `> 0` | estàtic | sí | sí; fixa formes |
| `initial_agents` | enter estricte `0..max_agents` | dinàmic `int32` | sí | no |
| `placement` | `random` (uniforme amb reemplaçament) | estàtic | sí | sí |
| `initial_energy` | finit, `death_threshold < value <= max_energy` | dinàmic `float32` | sí | no |
| `max_energy`, `death_threshold` | finits i coherents | dinàmics `float32` | sí | no |

`width * height` ha de ser representable com `int32`. `placement` i `max_agents` ja eren a la
signatura v3; no s'ha canviat la versió. Ni claus, IDs, posicions ni ocupació són propietats d'un
executable compilat.

## Límit de responsabilitats

El helper d'ocupació d'aquest PR només construeix l'estat inicial; el PR-11 l'extraurà o
generalitzarà i afegirà densitat. No s'implementen regles vitals ni un inicialitzador integral de
`SimulationState`.

| Funcionalitat | PR responsable |
|---|---|
| Inicialització dels genomes | PR-14 |
| Ocupació general i densitat | PR-11 |
| Observacions locals | PR-12 |
| Política neuronal | PR-13 a PR-15 |
| Moviment | PR-17 |
| Alimentació | PR-18 |
| Metabolisme | PR-19 |
| Mortalitat | PR-20 i PR-21 |
| Nous slots per descendència | PR-22 |
| Genealogia i esdeveniments | PR-24 |
| Inicialització integral del run | PR posterior de runtime |
