# Moviment cardinal i conflictes espacials v1

## Objectiu i límit de fase

El PR-17 resol simultàniament les intencions cardinals que la validació local del PR-16 ha
encaminat. La validació comprova l'actor, el codi, els límits i els recursos locals; la resolució
espacial decideix si una intenció de moviment s'aplica. No executa alimentació o reproducció ni
aplica costos energètics.

La fase posterior resol l'[alimentació i competència pels recursos](feeding_resource_competition_and_energy_transfer_v1.md).

## Instantània i regla espacial

La resolució consulta exclusivament `WorldState.occupancy` anterior al moviment. Una destinació
només és admissible si la seva ocupació inicial és exactament zero i rep com a màxim una entrada.
Una cel·la ocupada bloqueja entrades encara que els ocupants marxin. Això bloqueja intercanvis,
cicles i la propagació de vacants en cadenes. Les co-localitzacions preexistents es toleren i no
provoquen expulsions; els seus membres poden dispersar-se o competir per una cel·la buida.

| Situació | Resultat espacial | Codi |
|---|---|---|
| No és moviment | No canvia | `NOT_MOVEMENT` |
| Moviment únic cap a buit | Es mou | `MOVED` |
| Destinació ocupada | No es mou | `DESTINATION_OCCUPIED` |
| Conflicte resolt, guanyador | Es mou | `MOVED` |
| Conflicte resolt, perdedor | No es mou | `CONFLICT_LOST` |
| Col·lisió final de prioritats | Ningú es mou | `PRIORITY_COLLISION` |
| Contracte inconsistent | No es mou | `INVALID_MOVEMENT_INPUT` |

## Resolució neutral i RNG

L'API rep una `movement_conflict_key` ja derivada per al domini
`RngStreamCode.MOVEMENT_CONFLICT`; no rep ni avança l'arrel persistent. Cada clau d'entitat es
deriva de l'`agent_id`, mai del slot. Tres substreams independents produeixen una tupla de tres
`uint32`. Reduccions segmentades estàtiques calculen el mínim lexicogràfic per destinació. Si dos
finalistes conserven la mateixa tupla completa, tots els reclamants de la destinació fallen: no hi
ha desempat per ID, slot ni una quarta prioritat. La mateixa clau és determinista dins la versió
bloquejada de JAX; una clau de pas nova pot canviar el guanyador.

## Actualització, formes i diagnòstics

Els guanyadors reben `move_target` amb un únic `where` poblacional; els altres conserven la
posició. Un moviment fallit passa a `STAY`, mentre `STAY`, `EAT` i `REPRODUCE` es conserven a
`actions_after_movement`. Després es crida `rebuild_world_occupancy`, única font de veritat del
PR-11. Recursos, ambient, màscara viva, IDs, energia, edat i genomes no canvien.

Els vectors `actions_after_movement` i `movement_codes` tenen forma `[C]` i `CODE_DTYPE`. Els
recomptes de destinacions disputades, col·lisions no resoltes, inputs invàlids i posicions vives
invàlides posteriors són escalars `COUNT_DTYPE`. El recompte de reclamants té longitud estàtica
`height * width`; scatter-add, reduccions segmentades i màscares mantenen compatibilitat amb JIT i
`lax.scan`.

## Esquema versionat i compilació

L'esquema és `simultaneous_empty_snapshot_random_priority_v1`, versió 1, digest
`9209b617b2ed80ae1f1fa90206f13d05a1c69c763ece24ef92be6f64959a2e03`. El payload canònic
inclou el digest del contracte d'accions, codis i precedència, instantània, admissió, co-localització,
bloqueig d'intercanvis/cicles/cadenes, domini i derivació RNG, prioritats, col·lisió final, fallback,
absència de costos i reconstrucció d'ocupació. `CompileSignature` incorpora versió i digest; un
canvi funcional futur requereix una nova versió d'esquema i revisar la signatura. No hi ha una
configuració alternativa de capacitat o política espacial.

## Responsabilitats

| Funcionalitat | PR responsable |
|---|---|
| Proposta neuronal | PR-15 |
| Validació local | PR-16 |
| Moviment i conflictes espacials | PR-17 |
| Alimentació i conflictes de recurs | PR-18 |
| Cost del moviment | PR-19 |
| Viabilitat | PR-20 i PR-21 |
| Mètriques de conflicte | PR-25 |
| Invariants | PR-26 |
| Integració del pas | PR-27 |
| Execució amb `lax.scan` | PR-28 |

## Riscos i limitacions

La instantània conservadora pot reduir la mobilitat en mons densos i no és una propietat biològica
universal. L'ocupació d'entrada ha d'estar sincronitzada amb la població, i cada pas ha d'emprar la
seva clau de conflicte. La reproduïbilitat bit a bit entre versions diferents de JAX no es promet.
Els costos s'han d'aplicar en una fase posterior, quan ja es coneix `MOVED`. La capacitat local no
és configurable en aquesta versió.
