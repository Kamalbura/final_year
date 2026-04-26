from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import types
import unittest

from src.data.cities import ALL_MAJOR_CITIES, dag_id_for_city


class _FakeDAG:
    _active_stack: list["_FakeDAG"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tasks: list[object] = []

    def __enter__(self) -> "_FakeDAG":
        self._active_stack.append(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._active_stack.pop()
        return False

    @classmethod
    def current(cls) -> "_FakeDAG | None":
        return cls._active_stack[-1] if cls._active_stack else None


class _FakePythonOperator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        current_dag = _FakeDAG.current()
        if current_dag is not None:
            current_dag.tasks.append(self)


@contextmanager
def _fake_airflow_modules() -> types.ModuleType:
    airflow_module = types.ModuleType("airflow")
    airflow_module.DAG = _FakeDAG

    airflow_operators_module = types.ModuleType("airflow.operators")
    airflow_python_module = types.ModuleType("airflow.operators.python")
    airflow_python_module.PythonOperator = _FakePythonOperator

    original_modules = {
        name: sys.modules.get(name)
        for name in ("airflow", "airflow.operators", "airflow.operators.python")
    }

    sys.modules["airflow"] = airflow_module
    sys.modules["airflow.operators"] = airflow_operators_module
    sys.modules["airflow.operators.python"] = airflow_python_module
    try:
        yield airflow_module
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class AirflowCityFactoryTests(unittest.TestCase):
    def test_factory_registers_one_dag_per_city(self) -> None:
        factory_path = Path(__file__).resolve().parents[1] / "deployment" / "pi_airflow" / "dags" / "aq_city_factory.py"

        with _fake_airflow_modules():
            spec = importlib.util.spec_from_file_location("aq_city_factory_test", factory_path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        dag_ids = [dag_id_for_city(city) for city in ALL_MAJOR_CITIES]
        self.assertTrue(all(hasattr(module, dag_id) for dag_id in dag_ids))
        self.assertEqual(len(dag_ids), len(set(dag_ids)))

        for city in ALL_MAJOR_CITIES:
            dag_id = dag_id_for_city(city)
            dag = getattr(module, dag_id)
            self.assertEqual(dag.kwargs["dag_id"], dag_id)
            self.assertEqual(dag.kwargs["schedule"], "0 * * * *")
            self.assertEqual(dag.kwargs["max_active_runs"], 1)
            self.assertIn(city.country_code.lower(), dag.kwargs["tags"])
            self.assertIn(city.slug, dag.kwargs["tags"])
            self.assertEqual(len(dag.tasks), 1)

            task = dag.tasks[0]
            self.assertEqual(task.kwargs["task_id"], f"ingest_{city.slug}")
            self.assertEqual(task.kwargs["op_kwargs"]["city_slug"], city.slug)
            self.assertEqual(task.kwargs["op_kwargs"]["pipeline_name"], dag_id)


if __name__ == "__main__":
    unittest.main()