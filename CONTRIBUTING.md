# Guia de contribució

## Flux de treball

- Cada PR ha de tenir una única responsabilitat.
- No barregis refactoritzacions no relacionades amb l'objectiu del PR.
- Consulta l'[arquitectura tècnica](docs/architecture/projecte_evolucio_arquitectura_tecnica_sistema_v1_0.md) i el [roadmap de PR](docs/roadmap/projecte_evolucio_llista_pr_prototip_v1_0.md) abans de començar.
- Descriu explícitament què queda fora de l'abast de cada PR.
- Preserva la compatibilitat documental quan moguis o actualitzis documents.
- Afegeix o actualitza proves quan implementis comportament.
- Documenta qualsevol decisió que canviï contractes, límits entre capes o arquitectura.
- Evita notebooks com a estructura principal del projecte.

## Dependències, proves i traçabilitat

- Inicialitza l'entorn de desenvolupament amb `uv sync --locked --group dev`.
- Executa localment les portes equivalents a la CI abans de completar un PR:

  ```console
  uv lock --check
  uv sync --locked --group dev
  uv run ruff check .
  uv run ruff format --check .
  uv run pyright
  uv run lint-imports
  uv run coverage erase
  uv run coverage run -m pytest
  uv run coverage report
  uv build
  ```
- Els jobs de CI tenen els noms estables `quality` i `test-build`. Quan la configuració del
  repositori ho permeti, es recomana establir-los com a comprovacions obligatòries de branca;
  aquesta protecció s'ha de configurar externament.
- Els canvis de comportament han d'incloure proves deterministes. La cobertura global mínima és
  del 90 % i no es pot reduir, ignorar ni desactivar cap comprovació sense justificació explícita.
- Una CI verda valida una coherència tècnica mínima, però no demostra correcció científica.
- Gestiona les dependències amb `uv`, o modifica coherentment `pyproject.toml` i executa després
  `uv lock`.
- Versiona sempre `uv.lock` i no l'editis manualment.
- No afegeixis dependències sense justificar al PR si són obligatòries, opcionals o de
  desenvolupament.
- No actualitzis JAX o Equinox sense proves de regressió i benchmark quan aquests ja existeixin.
- No basis conclusions científiques en una sola execució.
- Revisa que configuració, seed i versions siguin traçables.

## Branques i commits

- Branca recomanada per aquest PR: `pr-00/documentation-foundation`.
- Branques futures: `pr-NN/short-description`.
- Commits amb format Conventional Commits.
- Tipus i scopes tècnics en anglès.
- Descripció breu i concreta.
- No cal reescriure l'historial si el repositori o la plataforma no ho permeten.

## Regles permanents de configuració

- Tot canvi d’esquema ha d’actualitzar la versió quan correspongui i mai no pot reinterpretar versions antigues.
- Els hashes es calculen sobre configuració canònica validada; no s’admeten variables d’entorn ni fallbacks silenciosos.
- Una configuració congelada no es modifica i la configuració host no pot importar JAX.
