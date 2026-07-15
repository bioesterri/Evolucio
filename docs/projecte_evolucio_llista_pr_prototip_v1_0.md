# Checklist de PR del prototip inicial

**Projecte:** Evolució  
**Versió:** 1.0  
**Data:** 15 de juliol de 2026  
**Estat:** Llista pendent de definició detallada

> Aquest document enumera els PR previstos per al prototip inicial i serveix com a checklist de seguiment. No en defineix encara l'abast, els criteris d'acceptació ni les dependències.

## Ús del checklist

- Marcar cada PR com a completat quan el canvi corresponent estigui fusionat o acceptat al repositori.
- Mantenir el títol estable mentre no es defineixi el detall del PR.
- Afegir enllaços, notes d'abast, criteris d'acceptació o dependències en documents específics quan es detallin les fases.

## Llista completa de PR del prototip inicial

### Fase 0 — Base del repositori

- [ ] **PR-00 — Base documental, AGENTS.md i convencions de treball amb Codex**
- [ ] **PR-01 — Inicialització del repositori, paquet Python, dependències i eines de qualitat**
- [ ] **PR-02 — Integració contínua, proves mínimes i portes de qualitat del repositori**

### Fase 1 — Configuració i contractes del nucli

- [ ] **PR-03 — Esquema de configuració, validació, congelació i versionat**
- [ ] **PR-04 — Compilació de la configuració host a CoreConfig i compile_signature**
- [ ] **PR-05 — Tipus comuns, dtypes, enums, codis d'acció i causes de mort**
- [ ] **PR-06 — SimulationState, subestats PyTree i model poblacional de capacitat fixa**
- [ ] **PR-07 — Gestió determinista de RNG, subclaus i identificadors interns**

### Fase 2 — Món i població inicial

- [ ] **PR-08 — Inicialització del món 2D, límits i distribució espacial de recursos**
- [ ] **PR-09 — Regeneració de recursos i calendari de variació ambiental**
- [ ] **PR-10 — Inicialització de la població, slots, posicions i ocupació espacial**
- [ ] **PR-11 — Càlcul vectoritzat de densitat i ocupació del món**

### Fase 3 — Percepció, genoma i política neuronal

- [ ] **PR-12 — Vector d'observacions local, normalització i esquema versionat**
- [ ] **PR-13 — PolicyMLP d'Equinox amb topologia fixa 15-16-7**
- [ ] **PR-14 — Inicialització i representació batched dels genomes**
- [ ] **PR-15 — Inferència neuronal vectoritzada i selecció determinista d'accions**

### Fase 4 — Accions i economia energètica

- [ ] **PR-16 — Contracte d'accions, validació i inacció**
- [ ] **PR-17 — Moviment cardinal i resolució neutral de conflictes espacials**
- [ ] **PR-18 — Alimentació, competència pels recursos i transferència energètica**
- [ ] **PR-19 — Metabolisme basal, costos d'acció i balanç energètic del pas**

### Fase 5 — Viabilitat i evolució

- [ ] **PR-20 — Viabilitat preacció, mort i registre de causes**
- [ ] **PR-21 — Viabilitat postacció i bloqueig de reproducció suïcida**
- [ ] **PR-22 — Reproducció asexual atòmica i assignació de slots de descendència**
- [ ] **PR-23 — Mutació de pesos i biaixos amb límits configurables**
- [ ] **PR-24 — Genealogia, identificadors de llinatge i esdeveniments de naixement**

### Fase 6 — Mètriques i motor compilable

- [ ] **PR-25 — Mètriques de pas, acumuladors i buffers d'esdeveniments**
- [ ] **PR-26 — Invariants del nucli i diagnòstics de consistència**
- [ ] **PR-27 — Integració del step complet de simulació**
- [ ] **PR-28 — run_chunk compilat amb Equinox/JAX i lax.scan**
- [ ] **PR-29 — Equivalència eager/JIT, determinisme i regressió amb llavor fixa**

### Fase 7 — Runtime i operació

- [ ] **PR-30 — Runtime de runs, chunking i frontera dispositiu-host**
- [ ] **PR-31 — DTOs host per mètriques, esdeveniments i snapshots**
- [ ] **PR-32 — CLI de validació, execució i inspecció de configuracions**
- [ ] **PR-33 — Modes smoke, debug, experiment i benchmark**
- [ ] **PR-34 — Execució de lots d'experiments i conjunts de llavors**

### Fase 8 — Persistència i traçabilitat

- [ ] **PR-35 — Manifest de run, configuració congelada i artefactes locals**
- [ ] **PR-36 — Persistència Parquet de mètriques, naixements, morts i genealogia**
- [ ] **PR-37 — Checkpoints i snapshots amb Zarr**
- [ ] **PR-38 — Persistència PostgreSQL de experiments, runs i índexs relacionals**
- [ ] **PR-39 — Registre de genomes destacats i resums d'agents i llinatges**
- [ ] **PR-40 — Integració de seguiment experimental amb MLflow**
- [ ] **PR-41 — Recuperació de checkpoints i gestió de runs fallits o interromputs**

### Fase 9 — Anàlisi i visualització

- [ ] **PR-42 — Loaders versionats i validació de dades persistides**
- [ ] **PR-43 — Anàlisi poblacional, energètica i de mortalitat amb Polars**
- [ ] **PR-44 — Anàlisi de llinatges, descendència i diversitat genètica**
- [ ] **PR-45 — Comparació de runs, controls i múltiples llavors**
- [ ] **PR-46 — Visualització mínima del món i dels recursos**
- [ ] **PR-47 — Gràfics temporals, resums de llinatges i animacions mostrejades**

### Fase 10 — Validació final i lliurament del prototip

- [ ] **PR-48 — Proves d'arquitectura i prohibició d'importacions entre capes**
- [ ] **PR-49 — Bateria end-to-end d'escenaris funcionals obligatoris**
- [ ] **PR-50 — Benchmark base, perfil de rendiment i pressupost de memòria**
- [ ] **PR-51 — Configuracions de referència, documentació d'ús i prototip v0.1**

**Total: 52 PR.**
