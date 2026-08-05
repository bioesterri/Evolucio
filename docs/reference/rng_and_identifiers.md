# RNG determinista i identificadors interns

## Objectiu i model de claus

El nucli conserva una única clau arrel tipada a `RngState`. Les claus tipades separen el tipus
PRNG dels arrays numèrics ordinaris i eviten tractar accidentalment el seu buffer com
`uint32[2]`. La implementació del prototip queda fixada a `threefry2x32` i la clau inicial es crea
amb `jax.random.key(seed, impl="threefry2x32")`; no depèn de configuració global de JAX.

```text
seed host -> typed root key
                 |
                 +-- split (una vegada per pas)
                       |-- nova root key (persistent)
                       `-- step key (transitòria)
                              |
                              `-- fold_in(stream code) -> stream key
                                      |
                                      `-- fold_in(entity ID/index) -> entity key
```

`split` és exclusiu de l'avanç de l'arrel: el primer fill és la nova arrel i el segon és la clau
del pas. `fold_in` deriva dominis i identitats estables sense fer que el resultat depengui de
l'ordre de derivació ni del nombre de consumidors d'un altre domini.

## Streams estables

| Nom | Codi | Finalitat futura | Estat al PR-07 |
|---|---:|---|---|
| `WORLD_INITIALIZATION` | 0 | Inicialització del món | Només domini reservat |
| `RESOURCE_INITIALIZATION` | 1 | Recursos inicials | Només domini reservat |
| `AGENT_INITIALIZATION` | 2 | Agents fundadors | Només domini reservat |
| `GENOME_INITIALIZATION` | 3 | Genomes fundadors | Només domini reservat |
| `ENVIRONMENT_UPDATE` | 4 | Actualització ambiental | Només domini reservat |
| `ACTION_TIE_BREAK` | 5 | Desempats d'accions | Només domini reservat |
| `MOVEMENT_CONFLICT` | 6 | Conflictes de moviment | Només domini reservat |
| `FEEDING_CONFLICT` | 7 | Conflictes d’alimentació | Només domini reservat |
| `REPRODUCTION_CONFLICT` | 8 | Conflictes reproductius | Només domini reservat |
| `BIRTH_PLACEMENT` | 9 | Col·locació de naixements | Només domini reservat |
| `GENOME_MUTATION` | 10 | Mutació del genoma | Només domini reservat |

Els codis identifiquen dominis, no prioritats ni regles funcionals. No es poden reordenar.

## Consum únic i determinisme

Una clau usada per mostrejar no es reutilitza; una clau pare només deriva claus filles. El codi de
producció d'aquest PR no mostreja cap valor. Les claus d'entitat deriven d'un `agent_id`,
`genome_id`, `lineage_id` o índex amb semàntica explícita, mai de la posició accidental del slot.
En codi JIT amb capacitat fixa, els consumidors futurs derivaran claus sobre arrays de forma fixa
`[C]` i aplicaran la màscara `alive` després per decidir quines claus es poden usar. No s'ha de
fer indexació booleana prèvia que canviï la longitud segons el nombre d'agents vius, i les claus
corresponents a slots inactius o `NULL_ID` no es consumiran per mostrejar.

Amb la mateixa seed, versió de codi i seqüència funcional s'obtenen les mateixes arrels, claus de
pas, streams, claus d'entitat i assignacions d'ID. Seeds, streams o identitats diferents separen
les claus. Reordenar identitats només reordena les claus associades. No es promet identitat bit a
bit entre versions diferents de JAX, jaxlib, XLA, codi o maquinari; el registre experimental haurà
de conservar aquestes versions.

## Comptadors i assignació

`IdCounters` conserva `next_agent_id`, `next_genome_id` i `next_lineage_id` com escalars `int32`.
`NULL_ID = -1` representa absència; els IDs reals ocupen `[0, 2_147_483_646]` i
`MAX_NEXT_ID = 2_147_483_647` representa un comptador esgotat. Un slot és reutilitzable, però un
ID no es recicla mai. `genome_id` és identitat registral o genealògica, no un hash de contingut.

`allocate_ids` usa màscara, suma acumulada i `where` sobre una forma fixa. L'ordre de la màscara és
l'ordre registral. Si les peticions superen l'espai restant, l'operació és atòmica: retorna
`overflow=True`, tots els valors són `NULL_ID` i el comptador no canvia. No hi ha assignació
parcial ni wraparound. La futura reproducció haurà de coordinar de manera atòmica dominis
múltiples; els descendents no rebran automàticament un llinatge nou.

## Continuació futura i exclusions

Un checkpoint futur haurà de conservar com a mínim `step`, la clau arrel tipada, els tres
comptadors, `WorldState` i `PopulationState`, juntament amb versions i configuració experimentals.
Aquest PR no serialitza claus ni escriu checkpoints. Tampoc inicialitza món, recursos, agents o
genomes; no resol accions o conflictes; i no implementa reproducció, mutació, mortalitat,
mètriques ni persistència.
