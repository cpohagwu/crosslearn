from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path
import xml.etree.ElementTree as ET


def _as_int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def _as_float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _parse_junit_xml(path: Path) -> dict[str, float | int | str]:
    if not path.exists():
        return {
            "tests": 0,
            "failures": 0,
            "errors": 1,
            "skipped": 0,
            "time": 0.0,
            "status": "missing_results",
        }

    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))

    if not suites:
        suites = list(root.iter("testsuite"))

    metrics = {
        "tests": sum(_as_int(suite.get("tests")) for suite in suites),
        "failures": sum(_as_int(suite.get("failures")) for suite in suites),
        "errors": sum(_as_int(suite.get("errors")) for suite in suites),
        "skipped": sum(_as_int(suite.get("skipped")) for suite in suites),
        "time": sum(_as_float(suite.get("time")) for suite in suites),
    }
    metrics["status"] = (
        "failed" if metrics["failures"] or metrics["errors"] else "passed"
    )
    return metrics


def _github_ref_name() -> str:
    return (
        os.getenv("GITHUB_HEAD_REF")
        or os.getenv("GITHUB_REF_NAME")
        or os.getenv("GITHUB_REF")
        or "unknown"
    )


def _configure_mlflow_tracking(repo_root: Path) -> str:
    import mlflow

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        return tracking_uri

    offline_dir = repo_root / "mlruns"
    offline_dir.mkdir(parents=True, exist_ok=True)
    offline_uri = offline_dir.resolve().as_uri()
    mlflow.set_tracking_uri(offline_uri)
    return offline_uri


def log_results(junit_xml: Path) -> None:
    import mlflow

    repo_root = Path.cwd()
    tracking_uri = _configure_mlflow_tracking(repo_root)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME") or "crosslearn-ci"
    mlflow.set_experiment(experiment_name)

    parsed = _parse_junit_xml(junit_xml)
    status = str(parsed.pop("status"))
    run_name = (
        f"pytest-{os.getenv('PYTHON_VERSION') or platform.python_version()}-"
        f"{os.getenv('GITHUB_RUN_ID') or 'local'}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "status": status,
                "python_version": os.getenv("PYTHON_VERSION")
                or platform.python_version(),
                "github_sha": os.getenv("GITHUB_SHA", "local"),
                "github_branch": _github_ref_name(),
                "github_run_id": os.getenv("GITHUB_RUN_ID", "local"),
                "github_workflow": os.getenv("GITHUB_WORKFLOW", "local"),
                "tracking_mode": "online"
                if os.getenv("MLFLOW_TRACKING_URI")
                else "offline",
                "tracking_uri": tracking_uri,
            }
        )
        mlflow.log_metrics(
            {
                "pytest_tests": float(parsed["tests"]),
                "pytest_failures": float(parsed["failures"]),
                "pytest_errors": float(parsed["errors"]),
                "pytest_skipped": float(parsed["skipped"]),
                "pytest_time_seconds": float(parsed["time"]),
                "pytest_passed": 1.0 if status == "passed" else 0.0,
            }
        )

        if junit_xml.exists():
            mlflow.log_artifact(str(junit_xml), artifact_path="pytest")

    print(
        "Logged pytest results to MLflow "
        f"experiment={experiment_name!r} status={status!r}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Log pytest JUnit XML results to MLflow."
    )
    parser.add_argument(
        "junit_xml",
        nargs="?",
        default="test-results/pytest.xml",
        help="Path to the pytest JUnit XML file.",
    )
    args = parser.parse_args()

    log_results(Path(args.junit_xml))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
