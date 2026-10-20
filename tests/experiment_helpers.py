"""Small complete paired cases for real batch integration tests."""

import json
from pathlib import Path

from evolucio.experiments import ExperimentCase


def write_cases(tmp_path: Path) -> tuple[ExperimentCase, ...]:
    data = json.loads(Path("tests/fixtures/config/valid_v1.json").read_text())
    data["world"].update(width=4, height=4, resource_distribution="uniform")
    data["population"].update(initial_agents=2, max_agents=8, max_births_per_step=2)
    data["energy"].update(initial_energy=30.0, reproduction_threshold=25.0)
    data["evolution"].update(min_reproduction_age=1, max_age=4)
    cases = []
    for case_id, cost in (("A", 0.1), ("B", 0.2)):
        data["energy"]["basal_cost"] = cost
        path = tmp_path / f"{case_id}.json"
        path.write_text(json.dumps(data))
        cases.append(ExperimentCase(case_id, path))
    return tuple(cases)
