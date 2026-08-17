from __future__ import annotations

from src.executors.ImageStack import ImageStack


EXECUTORS = {"ImageStack": ImageStack}


def get_executor(name: str):
    return EXECUTORS[name]


def bootstrap_all() -> dict[str, dict]:
    return {name: executor.bootstrap() for name, executor in EXECUTORS.items()}


if __name__ == "__main__":
    for executor_name, bootstrap_data in bootstrap_all().items():
        print(f"{executor_name}: {bootstrap_data['status']}")

