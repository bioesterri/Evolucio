# Viabilitat postacció i porta reproductiva v1

El PR-21 introdueix dues fases pures i separades després dels guanys i costos d'acció. Primer,
`resolve_post_action_viability` reutilitza el mateix avaluador de mortalitat del tall preacció, amb
prioritat `INVALID_STATE > ENERGY_DEPLETION > MAX_AGE`. Les morts es registren abans de
canonicalitzar població i genomes, i després es reconstrueix l'ocupació.

Segon, `evaluate_reproduction_gate` conserva per diagnòstic els intents previs a la mort i només
marca candidats vius que compleixen acció, energia i edat. La supervivència del progenitor es
projecta sense modificar l'estat:

```text
projected_parent_energy = energy - reproduction_energy_cost
projected_parent_energy > death_energy_threshold
```

La desigualtat és estricta perquè una energia igual o inferior al llindar és mortal. El cost no
s'aplica en aquesta fase: el PR-22 només el cobrarà si el naixement té èxit. L'energia inicial del
descendent no és un dèbit addicional del progenitor en l'esquema de configuració 2.1.

La porta no comprova slots lliures, disponibilitat espacial, posicions o conflictes; tampoc crea
identificadors, copia genomes, muta ni consumeix RNG. Tots els arrays conserven la capacitat fixa.

El contracte canònic és
`postaction_death_and_projected_reproduction_survival_v1`; la versió i el digest SHA-256 formen
part de `CompileSignature` v13. Els llindars, l'edat mínima i el cost continuen sent valors
dinàmics: alteren `config_hash`, però no la signatura de compilació.
