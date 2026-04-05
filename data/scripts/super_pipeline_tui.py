"""Advanced Windows orchestrator for the MT5 pipeline with Rich TUI.

Features:
- Installs project packages
- Detects both configured MT5 terminals using Windows process data
- Runs full backfill/merge/build pipeline with progress bars and ETA
- Streams live command output into a TUI log panel
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


console = Console()


@dataclass
class Step:
    """Single orchestration step."""

    name: str
    command: list[str]
    critical: bool = True


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _run_powershell_json(command: str) -> Any:
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    if not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def _find_mt5_processes() -> list[dict[str, str]]:
    """Return running process info with executable paths when available."""
    data = _run_powershell_json(
        "Get-CimInstance Win32_Process "
        "| Select-Object Name,ExecutablePath,ProcessId "
        "| ConvertTo-Json -Compress"
    )
    if data is None:
        return []
    if isinstance(data, dict):
        data = [data]

    rows: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name") or "")
        if not name:
            continue
        rows.append(
            {
                "name": name,
                "path": str(item.get("ExecutablePath") or ""),
                "pid": str(item.get("ProcessId") or ""),
            }
        )
    return rows


def _normalize_win_path(path: str) -> str:
    return path.replace("\\", "/").rstrip("/").lower()


def _detect_broker_terminals(config_path: Path) -> tuple[bool, list[str], list[dict[str, str]]]:
    config = _load_yaml(config_path)
    brokers = config.get("brokers", {})
    if not isinstance(brokers, dict) or len(brokers) < 2:
        return False, ["Config must define at least two brokers under 'brokers'."], []

    expected: list[tuple[str, str]] = []
    for broker_id, broker_cfg in brokers.items():
        if not isinstance(broker_cfg, dict):
            continue
        terminal_path = str(broker_cfg.get("terminal_path") or "").strip()
        if not terminal_path:
            continue
        expected.append((str(broker_id), _normalize_win_path(terminal_path)))

    if len(expected) < 2:
        return False, ["At least two brokers must include terminal_path."], []

    running = _find_mt5_processes()
    if not running:
        return False, ["Could not read Windows process list (Win32_Process)."], []

    missing: list[str] = []
    for broker_id, exp_path in expected[:2]:
        matched = False
        for proc in running:
            proc_path = _normalize_win_path(proc.get("path", ""))
            if proc_path and proc_path == exp_path:
                matched = True
                break
        if not matched:
            basename = Path(exp_path).name
            same_name_count = sum(
                1 for p in running if p.get("name", "").lower() == basename.lower()
            )
            if basename.lower() == "terminal64.exe" and same_name_count >= 2:
                matched = True
        if not matched:
            missing.append(f"{broker_id} terminal not found: {exp_path}")

    ok = len(missing) == 0
    return ok, missing, running


def _build_steps(
    *,
    python_exe: str,
    config_path: Path,
    symbol: str,
    date_from: str,
    date_to: str,
    dataset_name: str,
    run_live_after: bool,
    live_broker: str,
    enable_book: bool,
) -> list[Step]:
    cfg = _load_yaml(config_path)
    brokers = list((cfg.get("brokers") or {}).keys())
    if len(brokers) < 2:
        raise ValueError("Need at least two brokers in config to run full pipeline.")

    broker_a, broker_b = brokers[0], brokers[1]

    steps: list[Step] = [
        Step(
            "Install project dependencies",
            [python_exe, "-m", "pip", "install", "-e", ".[dev]"],
            True,
        ),
    ]

    for broker in (broker_a, broker_b):
        steps.extend(
            [
                Step(
                    f"{broker}: capture symbol metadata",
                    [
                        python_exe,
                        "-m",
                        "mt5pipe.cli.app",
                        "backfill",
                        "symbol-metadata",
                        "--broker",
                        broker,
                        "--config",
                        str(config_path),
                    ],
                    True,
                ),
                Step(
                    f"{broker}: backfill ticks",
                    [
                        python_exe,
                        "-m",
                        "mt5pipe.cli.app",
                        "backfill",
                        "ticks",
                        "--broker",
                        broker,
                        "--symbol",
                        symbol,
                        "--from",
                        date_from,
                        "--to",
                        date_to,
                        "--config",
                        str(config_path),
                    ],
                    True,
                ),
                Step(
                    f"{broker}: backfill native bars M5",
                    [
                        python_exe,
                        "-m",
                        "mt5pipe.cli.app",
                        "backfill",
                        "bars",
                        "--broker",
                        broker,
                        "--symbol",
                        symbol,
                        "--timeframe",
                        "M5",
                        "--from",
                        date_from,
                        "--to",
                        date_to,
                        "--config",
                        str(config_path),
                    ],
                    True,
                ),
                Step(
                    f"{broker}: backfill history orders",
                    [
                        python_exe,
                        "-m",
                        "mt5pipe.cli.app",
                        "backfill",
                        "history-orders",
                        "--broker",
                        broker,
                        "--from",
                        date_from,
                        "--to",
                        date_to,
                        "--config",
                        str(config_path),
                    ],
                    False,
                ),
                Step(
                    f"{broker}: backfill history deals",
                    [
                        python_exe,
                        "-m",
                        "mt5pipe.cli.app",
                        "backfill",
                        "history-deals",
                        "--broker",
                        broker,
                        "--from",
                        date_from,
                        "--to",
                        date_to,
                        "--config",
                        str(config_path),
                    ],
                    False,
                ),
            ]
        )

    steps.extend(
        [
            Step(
                "Merge canonical ticks",
                [
                    python_exe,
                    "-m",
                    "mt5pipe.cli.app",
                    "merge",
                    "canonical",
                    "--symbol",
                    symbol,
                    "--broker-a",
                    broker_a,
                    "--broker-b",
                    broker_b,
                    "--from",
                    date_from,
                    "--to",
                    date_to,
                    "--config",
                    str(config_path),
                ],
                True,
            ),
            Step(
                "Build all bars from canonical ticks",
                [
                    python_exe,
                    "-m",
                    "mt5pipe.cli.app",
                    "bars",
                    "build",
                    "--symbol",
                    symbol,
                    "--from",
                    date_from,
                    "--to",
                    date_to,
                    "--config",
                    str(config_path),
                ],
                True,
            ),
            Step(
                "Build dataset",
                [
                    python_exe,
                    "-m",
                    "mt5pipe.cli.app",
                    "dataset",
                    "build",
                    "--symbol",
                    symbol,
                    "--name",
                    dataset_name,
                    "--from",
                    date_from,
                    "--to",
                    date_to,
                    "--config",
                    str(config_path),
                ],
                True,
            ),
            Step(
                "Validate storage",
                [
                    python_exe,
                    "-m",
                    "mt5pipe.cli.app",
                    "status",
                    "validate",
                    "--config",
                    str(config_path),
                ],
                True,
            ),
            Step(
                "Show status summary",
                [
                    python_exe,
                    "-m",
                    "mt5pipe.cli.app",
                    "status",
                    "show",
                    "--config",
                    str(config_path),
                ],
                True,
            ),
        ]
    )

    if run_live_after:
        book_flag = "--enable-book" if enable_book else "--no-enable-book"
        steps.append(
            Step(
                f"Start live collection ({live_broker})",
                [
                    python_exe,
                    "-m",
                    "mt5pipe.cli.app",
                    "live",
                    "collect",
                    "--broker",
                    live_broker,
                    "--symbol",
                    symbol,
                    book_flag,
                    "--duration",
                    "120",
                    "--config",
                    str(config_path),
                ],
                False,
            )
        )

    return steps


def _status_table(
    step_index: int,
    total_steps: int,
    step_name: str,
    started_at: float,
    success_count: int,
    fail_count: int,
) -> Table:
    table = Table.grid(expand=True)
    table.add_column(ratio=1)
    table.add_column(ratio=3)
    table.add_row("Step", f"{step_index}/{total_steps}")
    table.add_row("Current", step_name)
    table.add_row("Elapsed", f"{time.time() - started_at:,.1f}s")
    table.add_row("Succeeded", str(success_count))
    table.add_row("Failed", str(fail_count))
    return table


def _run_step(
    step: Step,
    overall: Progress,
    overall_task: int,
    current: Progress,
    current_task: int,
    log_lines: deque[str],
) -> tuple[int, list[str]]:
    log_lines.append(f"$ {' '.join(step.command)}")
    proc = subprocess.Popen(
        step.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    pulses = 0
    while True:
        line = ""
        if proc.stdout is not None:
            line = proc.stdout.readline()

        if line:
            msg = line.rstrip()
            if msg:
                log_lines.append(msg)
            pulses = min(95, pulses + 1)
            current.update(current_task, completed=pulses)

        code = proc.poll()
        if code is not None and not line:
            break

    remainder: list[str] = []
    if proc.stdout is not None:
        for line in proc.stdout.readlines():
            msg = line.rstrip()
            if msg:
                remainder.append(msg)
                log_lines.append(msg)

    current.update(current_task, completed=100)
    if proc.returncode == 0:
        overall.advance(overall_task, 1)

    return proc.returncode or 0, remainder


def run_orchestration(args: argparse.Namespace) -> int:
    if os.name != "nt":
        console.print("[red]This script is Windows-only.[/red]")
        return 2

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        console.print(f"[red]Config file not found:[/red] {config_path}")
        return 2

    ok, reasons, running = _detect_broker_terminals(config_path)
    if not ok:
        console.print("[bold red]MT5 terminal check failed[/bold red]")
        for r in reasons:
            console.print(f"  - {r}")
        if running:
            table = Table(title="Running MT-related Processes")
            table.add_column("PID")
            table.add_column("Name")
            table.add_column("ExecutablePath")
            for proc in running:
                name = proc.get("name", "")
                if "terminal" in name.lower() or "meta" in name.lower():
                    table.add_row(proc.get("pid", ""), name, proc.get("path", ""))
            console.print(table)
        console.print("\nOpen both broker terminals and login, then rerun.")
        return 3

    steps = _build_steps(
        python_exe=sys.executable,
        config_path=config_path,
        symbol=args.symbol,
        date_from=args.date_from,
        date_to=args.date_to,
        dataset_name=args.dataset_name,
        run_live_after=args.live_after,
        live_broker=args.live_broker,
        enable_book=args.enable_book,
    )

    overall = Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    )
    current = Progress(
        SpinnerColumn(style="green"),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        expand=True,
    )

    overall_task = overall.add_task("Overall pipeline", total=len(steps))
    current_task = current.add_task("Waiting", total=100)

    log_lines: deque[str] = deque(maxlen=args.log_lines)
    failures: list[tuple[str, int]] = []
    success_count = 0

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=8),
        Layout(name="progress", size=8),
        Layout(name="logs"),
    )

    started = time.time()

    def render(step_idx: int, step_name: str) -> Layout:
        status = _status_table(
            step_idx,
            len(steps),
            step_name,
            started,
            success_count,
            len(failures),
        )
        layout["header"].update(Panel(status, title="Pipeline Control Center", border_style="blue"))
        layout["progress"].update(Panel(Group(overall, current), title="Progress + ETA", border_style="cyan"))
        layout["logs"].update(
            Panel(
                "\n".join(log_lines) if log_lines else "No logs yet...",
                title="Live Logs",
                border_style="magenta",
            )
        )
        return layout

    with Live(render(0, "Bootstrap"), console=console, refresh_per_second=8, screen=True):
        for idx, step in enumerate(steps, start=1):
            current.update(current_task, description=step.name, completed=0)
            code, _ = _run_step(step, overall, overall_task, current, current_task, log_lines)

            if code == 0:
                success_count += 1
                log_lines.append(f"[ok] {step.name}")
            else:
                failures.append((step.name, code))
                log_lines.append(f"[failed:{code}] {step.name}")
                if step.critical:
                    break

            render(idx, step.name)

    summary = Table(title="Run Summary")
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("Total steps", str(len(steps)))
    summary.add_row("Succeeded", str(success_count))
    summary.add_row("Failed", str(len(failures)))
    summary.add_row("Elapsed", f"{time.time() - started:,.1f}s")
    console.print(summary)

    if failures:
        fail_tbl = Table(title="Failed Steps")
        fail_tbl.add_column("Step")
        fail_tbl.add_column("Exit Code")
        for name, code in failures:
            fail_tbl.add_row(name, str(code))
        console.print(fail_tbl)
        return 1

    console.print("[bold green]Pipeline completed successfully.[/bold green]")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Advanced MT5 pipeline orchestrator with Windows TUI.",
    )
    parser.add_argument("--config", default="config/pipeline.yaml", help="Pipeline config path")
    parser.add_argument("--symbol", default="XAUUSD", help="Trading symbol")
    parser.add_argument("--from", dest="date_from", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--dataset-name", default="default", help="Dataset output name")
    parser.add_argument("--log-lines", type=int, default=30, help="Lines kept in live log panel")
    parser.add_argument(
        "--live-after",
        action="store_true",
        help="Start live collection as final step after backfill/build",
    )
    parser.add_argument(
        "--live-broker",
        default="broker_a",
        help="Broker used for final live collection step",
    )
    parser.add_argument(
        "--enable-book",
        action="store_true",
        help="Enable market book in live collection step",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_orchestration(args)


if __name__ == "__main__":
    raise SystemExit(main())
