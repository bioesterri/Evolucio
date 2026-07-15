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

- No afegeixis dependències sense justificació.
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
