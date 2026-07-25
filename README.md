# Projecte Evolució

Projecte Evolució és un prototip de vida artificial evolutiva orientat a estudiar poblacions d'agents simples en un món 2D amb simulacions repetibles i auditables.

## Objectiu del prototip inicial

El prototip inicial ha de validar un bucle evolutiu mínim en què agents neuronals simples sobreviuen, consumeixen recursos, es reprodueixen de manera asexual i transmeten mutacions heretables. La selecció ha d'emergir de l'entorn, no d'una funció externa de fitness.

## Requisits

- Python 3.12.
- [uv](https://docs.astral.sh/uv/) per gestionar el projecte, l'entorn i les dependències.

## Preparació de l'entorn

```console
uv sync --group dev
```

## Comprovació del paquet

```console
uv run python -c "import evolucio"
```

## Qualitat local

```console
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run lint-imports
uv lock --check
uv build
```

## Extres opcionals

Els extres `persistence`, `analysis` i `visualization` preparen les dependències
d'infraestructura corresponents, però no formen part de la instal·lació base. S'activaran quan
els PR posteriors implementin aquestes capes.

## Principis essencials

- Selecció ambiental emergent.
- Reproducció asexual.
- Mutació heretable.
- Agents neuronals simples.
- Experiments repetibles.

## Documentació principal

- [Arquitectura tècnica](docs/architecture/projecte_evolucio_arquitectura_tecnica_sistema_v1_0.md)
- [Roadmap de PR](docs/roadmap/projecte_evolucio_llista_pr_prototip_v1_0.md)
- [Guia de contribució](CONTRIBUTING.md)
- [Instruccions per a Codex](AGENTS.md)

## Estat d'implementació

El paquet `evolucio` existeix, és instal·lable i encara no inclou motor de simulació, CLI ni
configuració funcional. El PR-02 incorporarà les primeres proves i la integració contínua; la
implementació funcional començarà als PR següents.
