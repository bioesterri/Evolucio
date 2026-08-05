# Alimentació, competència pels recursos i transferència energètica v1

La fase s'executa després del moviment i consumeix exclusivament `EAT` de
`actions_after_movement`. Per cada intent vàlid, el marge és
`max(maximum_energy - energy, 0)` i la demanda és el mínim entre
`feeding_max_resource_intake` i el marge dividit per `energy_gain_per_resource`.

Les demandes s'agreguen per cel·la en arrays fixos. Si són superiors al recurs, cada consumidor
rep la mateixa fracció `available / total_demand`, sense guanyador, prioritat o RNG. El recurs no
aprofitat roman al món. El guany és `resource_consumed * energy_gain_per_resource` i l'energia se
satura a `maximum_energy`.

Els codis són `NOT_FEEDING`, `FED_FULL`, `FED_PARTIAL`, `NO_RESOURCE`,
`NO_ENERGY_CAPACITY` i `INVALID_FEEDING_INPUT`. Els resultats alimentats conserven `EAT`; els
intents sense efecte passen a `STAY`; les altres accions es conserven.

La transferència és només **recurs ambiental → energia interna**. No hi ha costos, transferència
agent-agent, regeneració ni aleatorietat. L'esquema
`local_proportional_feasible_demand_resource_transfer_v1`, versió 1, té un digest SHA-256
canònic integrat a `CompileSignature`; els valors dinàmics no hi entren.

El PR-19 serà responsable dels costos i el metabolisme. Aquesta fase no decideix viabilitat,
mort, reproducció, herència ni mutació i no avança el pas complet.
