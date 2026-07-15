# AGENTS.md — Projecte Evolució

## Projecte

- El Projecte Evolució construeix un prototip de vida artificial evolutiva, reproduïble i científicament auditable.
- El llenguatge principal és Python 3.12.
- JAX i Equinox són la base del nucli computacional.
- L'evolució poblacional sorgeix de reproducció asexual i mutació heretable.
- El prototip inicial no inclou aprenentatge individual durant la vida dels agents.

## Fonts de veritat

- [Arquitectura tècnica](docs/architecture/projecte_evolucio_arquitectura_tecnica_sistema_v1_0.md): font de veritat per a decisions tècniques, capes i invariants.
- [Roadmap de PR](docs/roadmap/projecte_evolucio_llista_pr_prototip_v1_0.md): seqüència general de desenvolupament.
- [Guia de contribució](CONTRIBUTING.md): flux de treball i convencions bàsiques.

Abans de començar una implementació, Codex ha de llegir l'arquitectura tècnica i el document o instrucció específica del PR corresponent.

## Disciplina d'abast

- Implementa un únic PR cada vegada.
- No avancis funcionalitats de PR futurs.
- Fes canvis petits i directament relacionats amb la tasca.
- No modifiquis decisions arquitectòniques sense instrucció explícita.
- No afegeixis abstraccions especulatives.
- No ocultis decisions importants en constants màgiques.
- No afirmis que una prova ha passat si no s'ha executat.

## Regles arquitectòniques permanents

- El nucli és funcional, pur i sense efectes laterals.
- `core` no pot dependre de persistència, anàlisi, visualització, CLI ni infraestructura.
- Els agents es representen amb arrays poblacionals, no amb objectes Python mutables per individu.
- La població utilitza capacitat fixa i màscares.
- Les formes dels arrays no poden dependre de la població viva.
- La gestió de PRNG de JAX és explícita i reproduïble.
- No s'utilitza aleatorietat global amagada.
- Equinox no s'utilitza com una jerarquia tradicional d'objectes mutables.
- La viabilitat es comprova abans de permetre la reproducció.
- No hi ha funció externa de fitness que seleccioni agents.
- Mètriques, persistència i visualització no poden modificar la simulació.
- FastAPI, distribució multi-node, reproducció sexual i aprenentatge individual queden fora del prototip inicial.

## Convencions de llengua i codi

- Documentació, explicacions, resums de PR i comunicació del projecte: català.
- Noms de paquets, mòduls, classes, funcions, variables, tipus, APIs i identificadors tècnics: anglès.
- Docstrings i comentaris de codi: anglès, breus i útils.
- Noms establerts per llibreries o estàndards: conserva la forma original.
- Evita abreviatures opaques.

## Qualitat i proves

- El codi haurà d'estar tipat.
- Les llavors aleatòries seran explícites.
- Les funcions crítiques tindran proves.
- Caldrà executar les comprovacions definides pel repositori abans de donar una tasca per acabada.
- Els errors no s'han de corregir amb fallbacks silenciosos.
- Els resultats numèrics no s'han d'interpretar com a evolució si fallen invariants.

Nota: el PR-01 haurà d'actualitzar aquesta secció amb les ordres reals del projecte quan existeixin eines de qualitat configurades.

## Protocol de finalització

Codex ha d'informar sempre de:

- què ha canviat;
- quins fitxers s'han modificat;
- quines verificacions s'han executat;
- quines verificacions no s'han pogut executar;
- riscos, limitacions o decisions pendents.

## Review guidelines

Una revisió ha de detectar especialment:

- treball fora de l'abast del PR;
- duplicació de lògica;
- imports prohibits entre capes;
- efectes laterals dins `core`;
- aleatorietat no controlada;
- formes dinàmiques incompatibles amb JAX;
- absència de proves quan la funcionalitat les requereixi;
- canvis arquitectònics no documentats.
