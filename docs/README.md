# Documentació del Projecte Evolució

La carpeta `docs/` agrupa la documentació de referència del Projecte Evolució i evita duplicar decisions en fitxers dispersos.

## Fonts principals

- [Arquitectura tècnica](architecture/projecte_evolucio_arquitectura_tecnica_sistema_v1_0.md): governa les decisions tècniques, els límits entre capes, els invariants del nucli, l'ús de JAX/Equinox, la persistència, l'anàlisi i la visualització.
- [Roadmap de PR](roadmap/projecte_evolucio_llista_pr_prototip_v1_0.md): governa la seqüència general de desenvolupament del prototip inicial.

## Jerarquia documental

1. L'arquitectura tècnica és la font de veritat per a decisions tècniques i límits entre capes.
2. El roadmap de PR defineix l'ordre general de desenvolupament.
3. Les instruccions específiques de cada PR defineixen l'abast concret del canvi.
4. La guia de contribució i `AGENTS.md` defineixen les convencions de treball.

Aquesta pàgina és només un índex. No resumeix ni substitueix els documents de referència.

## Configuració

- [Guia de configuració](configuration.md)
- [JSON Schema de configuració 1.0](schemas/experiment-config-v1.0.json)

## Referència del nucli

- [Tipus, dtypes i codis del nucli](reference/core_types_and_codes.md)
- [Estat del nucli i model poblacional de capacitat fixa](reference/core_state.md)
- [RNG determinista i identificadors interns](reference/rng_and_identifiers.md)
- [Inicialització del món 2D](reference/world_initialization.md)
- [Regeneració de recursos i calendari ambiental](reference/resource_regeneration_and_environment_calendar.md)
- [Inicialització de la població fundadora](reference/population_initialization.md)
- [Ocupació i densitat espacials](reference/spatial_occupancy_and_density.md)
- [Esquema d'observacions locals v1](reference/local_observation_schema_v1.md)
- [Esquema PolicyMLP v1](reference/policy_mlp_schema_v1.md): topologia neuronal fixa, fulles del genoma i ordre dels scores.
