"""Dependency-direction regression tests for the gateway architecture."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).parents[1] / "local_llm_server"
GATEWAY_MODULES = {
    "backend_core",
    "backend_runtime",
    "model_catalog",
    "process_control",
    "server_health",
    "gateway_runtime",
    "gateway_config",
    "gateway_manager",
    "gateway_http",
    "gateway_updates",
    "server",
    "daemon",
}


def _local_dependencies(module: str) -> set[str]:
    tree = ast.parse((PACKAGE / f"{module}.py").read_text(encoding="utf-8"))
    dependencies: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 1:
            continue
        if node.module:
            candidate = node.module.split(".", 1)[0]
            if candidate in GATEWAY_MODULES:
                dependencies.add(candidate)
        else:
            dependencies.update(
                alias.name for alias in node.names if alias.name in GATEWAY_MODULES
            )
    return dependencies


def test_gateway_module_graph_is_acyclic() -> None:
    """Composition may depend on foundations, never the other way around."""
    graph = {module: _local_dependencies(module) for module in GATEWAY_MODULES}
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(module: str) -> None:
        if module in visiting:
            cycle = visiting[visiting.index(module) :] + [module]
            raise AssertionError("gateway dependency cycle: " + " -> ".join(cycle))
        if module in visited:
            return
        visiting.append(module)
        for dependency in graph[module]:
            visit(dependency)
        visiting.pop()
        visited.add(module)

    for module in graph:
        visit(module)


def test_gateway_foundations_do_not_import_composition_modules() -> None:
    forbidden = {"server", "daemon"}
    foundations = {
        "backend_core",
        "backend_runtime",
        "model_catalog",
        "process_control",
        "server_health",
        "gateway_runtime",
        "gateway_config",
        "gateway_manager",
        "gateway_http",
        "gateway_updates",
    }
    offenders = {
        module: sorted(_local_dependencies(module) & forbidden)
        for module in foundations
        if _local_dependencies(module) & forbidden
    }
    assert offenders == {}
