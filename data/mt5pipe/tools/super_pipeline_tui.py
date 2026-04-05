"""Zero-config MT5 pipeline orchestrator with advanced Rich TUI.

Run one file, touch nothing.  The orchestrator will:
 1. Ensure all Python dependencies are installed.
 2. Detect both open MT5 terminals from your config paths.
 3. Probe each broker: account, symbols, L2 support, MAX date range.
 4. Auto-plan the full pipeline (all ticks, all bars, all data).
 5. Execute every step with a fullscreen progress dashboard + live logs.
 6. Print a final summary.

Usage (normal user — double-click RUN_ME.bat)  or:
    python -m mt5pipe.tools.super_pipeline_tui
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.markup import escape
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
from rich.rule import Rule
from rich.table import Table

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION = "3.0.0"
NATIVE_BAR_TFS = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
STEP_LOG_TAIL_LINES = 2_000

console = Console()

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Step:
    name: str
    command: list[str]
    critical: bool = True
    timeout_seconds: int = 0  # 0 = no timeout
    max_retries: int = 0


@dataclass
class StepFailure:
    name: str
    code: int
    log_path: Path


@dataclass
class RunPlan:
    config_path: Path
    python_exe: str
    symbols: list[str]
    broker_ids: list[str] = field(default_factory=list)
    # broker -> symbol -> earliest date
    tick_start: dict[str, dict[str, dt.date]] = field(default_factory=dict)
    bar_start: dict[str, dict[str, dt.date]] = field(default_factory=dict)
    end_date: dt.date = field(default_factory=dt.date.today)
    # broker -> bool
    market_book: dict[str, dict[str, bool]] = field(default_factory=dict)
    native_bar_tfs: list[str] = field(default_factory=lambda: list(NATIVE_BAR_TFS))
    live_broker: str = ""
    live_book: bool = False
    live_warmup_seconds: int = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _find_config() -> Path:
    for candidate in ("config/pipeline.yaml", "pipeline.yaml", "config.yaml"):
        p = Path(candidate)
        if p.exists():
            return p.resolve()
    return Path("config/pipeline.yaml").resolve()


def _ensure_deps() -> None:
    """If MetaTrader5 is not yet importable, pip-install the project."""
    try:
        import MetaTrader5  # noqa: F401
    except ImportError:
        console.print("[yellow]  MetaTrader5 not found — installing dependencies...[/]")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", "."],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _sanitize_filename(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    return safe.strip("_")[:80] or "step"


def _write_step_failure_log(
    step: Step,
    code: int,
    output_lines: list[str],
    started_ts: float,
    ended_ts: float,
) -> Path:
    log_dir = Path("logs") / "pipeline_failures"
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_name = f"{stamp}_{_sanitize_filename(step.name)}.log"
    log_path = log_dir / file_name

    cmd = subprocess.list2cmdline(step.command)
    started_iso = dt.datetime.fromtimestamp(started_ts, tz=dt.timezone.utc).isoformat()
    ended_iso = dt.datetime.fromtimestamp(ended_ts, tz=dt.timezone.utc).isoformat()
    duration = max(0.0, ended_ts - started_ts)

    with log_path.open("w", encoding="utf-8") as f:
        f.write("MT5 PIPELINE STEP FAILURE LOG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Step: {step.name}\n")
        f.write(f"Exit code: {code}\n")
        f.write(f"Started (UTC): {started_iso}\n")
        f.write(f"Ended   (UTC): {ended_iso}\n")
        f.write(f"Duration: {duration:.2f}s\n")
        f.write(f"Command: {cmd}\n")
        f.write("\n")
        f.write("OUTPUT (tail)\n")
        f.write("-" * 80 + "\n")
        if output_lines:
            for line in output_lines:
                f.write(line)
                if not line.endswith("\n"):
                    f.write("\n")
        else:
            f.write("<no output captured>\n")

    return log_path


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------


def _show_banner() -> None:
    console.print()
    console.print(
        Panel.fit(
            "[bold bright_cyan]MT5 DATA PIPELINE[/]\n"
            "[dim]Atomic Writes  ·  Auto-Retry  ·  Step Timeouts  ·  Embargo Splits[/]\n"
            f"[dim]v{VERSION}[/]",
            border_style="bright_blue",
            padding=(1, 8),
        )
    )
    console.print()


# ---------------------------------------------------------------------------
# Detection phase
# ---------------------------------------------------------------------------


def _detect_phase(config_path: Path):
    """Connect to every configured broker, probe capabilities, disconnect."""
    from mt5pipe.config.loader import load_config
    from mt5pipe.mt5.detect import BrokerCaps, detect_broker

    cfg = load_config(config_path)
    symbols = cfg.symbols
    all_caps: list[BrokerCaps] = []

    for broker_id in cfg.broker_ids():
        broker_cfg = cfg.get_broker(broker_id)
        console.print(f"  Scanning [bold]{broker_id}[/] ...", end=" ")
        caps = detect_broker(broker_cfg, symbols)
        if caps.connected:
            console.print("[bold green]✓ Connected[/]")
        else:
            console.print(f"[bold red]✗ Failed[/]  ({caps.error})")
        all_caps.append(caps)

    return all_caps, cfg, symbols


# ---------------------------------------------------------------------------
# Broker panel rendering
# ---------------------------------------------------------------------------


def _broker_panel(caps) -> Panel:
    from mt5pipe.mt5.detect import BrokerCaps

    if not caps.connected:
        body = f"[bold red]✗ Connection failed[/]\n{caps.error or 'Unknown error'}"
        return Panel(body, title=f"[red]{caps.broker_id}[/]", border_style="red")

    lines: list[str] = []
    lines.append(f"[green]✓ Connected[/]  to  [bold]{caps.account_server}[/]")

    if caps.account_login:
        lines.append(
            f"Account [bold]#{caps.account_login}[/]  ·  {caps.account_name}  ·  "
            f"Balance: [green]{caps.account_balance:,.2f} {caps.account_currency}[/]  ·  "
            f"Leverage: 1:{caps.account_leverage}"
        )

    lines.append(
        f"Terminal: {caps.terminal_name}  Build {caps.terminal_build}  "
        f"(v{caps.terminal_version})"
    )
    lines.append(f"Symbols available: [bold]{caps.total_symbols}[/]")

    for sym_name, sc in caps.symbols.items():
        if not sc.available:
            lines.append(f"  [bold]{sym_name}[/]  [red]✗ Not found[/]")
            continue
        l2 = "[green]✓ L2[/]" if sc.market_book_ok else "[dim]✗ No L2[/]"
        tick_d = sc.first_tick_date.isoformat() if sc.first_tick_date else "—"
        bar_d = sc.first_bar_date.isoformat() if sc.first_bar_date else "—"
        days_tick = (
            f"({(dt.date.today() - sc.first_tick_date).days:,}d)"
            if sc.first_tick_date
            else ""
        )
        days_bar = (
            f"({(dt.date.today() - sc.first_bar_date).days:,}d)"
            if sc.first_bar_date
            else ""
        )
        lines.append(
            f"  [bold]{sym_name}[/]  {l2}  ·  "
            f"Ticks from [cyan]{tick_d}[/] {days_tick}  ·  "
            f"Bars from [cyan]{bar_d}[/] {days_bar}  ·  "
            f"Spread {sc.spread}  ·  Digits {sc.digits}"
        )

    return Panel(
        "\n".join(lines),
        title=f"[bold bright_cyan]{caps.broker_id}[/]",
        border_style="bright_blue",
    )


# ---------------------------------------------------------------------------
# Auto-plan
# ---------------------------------------------------------------------------


def _compute_plan(all_caps, config, config_path: Path, symbols: list[str]) -> RunPlan:
    today = dt.date.today()
    fallback = today - dt.timedelta(days=365)

    plan = RunPlan(
        config_path=config_path,
        python_exe=sys.executable,
        symbols=symbols,
        end_date=today,
    )

    for caps in all_caps:
        if not caps.connected:
            continue
        bid = caps.broker_id
        plan.broker_ids.append(bid)
        plan.tick_start[bid] = {}
        plan.bar_start[bid] = {}
        plan.market_book[bid] = {}

        for sym in symbols:
            sc = caps.symbols.get(sym)
            if sc and sc.available:
                plan.tick_start[bid][sym] = sc.first_tick_date or fallback
                plan.bar_start[bid][sym] = sc.first_bar_date or fallback
                plan.market_book[bid][sym] = sc.market_book_ok
            else:
                plan.tick_start[bid][sym] = fallback
                plan.bar_start[bid][sym] = fallback
                plan.market_book[bid][sym] = False

    if plan.broker_ids:
        plan.live_broker = plan.broker_ids[0]
    # Enable live-mode market book only if the live broker supports it
    if plan.live_broker:
        plan.live_book = any(plan.market_book.get(plan.live_broker, {}).values())

    # Pull warmup duration from config (default 120s so pipeline never blocks)
    plan.live_warmup_seconds = config.live.warmup_seconds if config.live.warmup_seconds > 0 else 120

    return plan


def _plan_panel(plan: RunPlan, step_count: int) -> Panel:
    lines: list[str] = []

    for sym in plan.symbols:
        all_dates: list[dt.date] = []
        for bid in plan.broker_ids:
            d = plan.tick_start.get(bid, {}).get(sym)
            if d:
                all_dates.append(d)
            d = plan.bar_start.get(bid, {}).get(sym)
            if d:
                all_dates.append(d)
        earliest = min(all_dates) if all_dates else plan.end_date - dt.timedelta(days=365)
        span = (plan.end_date - earliest).days

        lines.append(
            f"Symbol       [bold]{sym}[/]"
        )
        lines.append(
            f"Date Range   [bold cyan]{earliest}[/]  →  [bold cyan]{plan.end_date}[/]  "
            f"[dim](MAX · {span:,} days)[/]"
        )

    for bid in plan.broker_ids:
        books = plan.market_book.get(bid, {})
        book_syms = [s for s, v in books.items() if v]
        book_txt = (
            f"[green]✓ L2[/] ({', '.join(book_syms)})" if book_syms else "[dim]✗ No L2[/]"
        )
        lines.append(f"  {bid:14s} {book_txt}")

    lines.append(f"Native Bars  [bold]{' '.join(plan.native_bar_tfs)}[/]  per broker")
    lines.append(f"Canon. Bars  [bold]All 21 timeframes[/] from merged ticks")
    warmup_m = plan.live_warmup_seconds // 60
    warmup_s = plan.live_warmup_seconds % 60
    warmup_str = f"{warmup_m}m {warmup_s:02d}s" if warmup_m else f"{warmup_s}s"
    lines.append(f"Pipeline     [bold]{step_count}[/] operations  ·  Live after: [bold]{plan.live_broker or 'no'}[/]  ·  Warmup: [bold]{warmup_str}[/]")

    return Panel(
        "\n".join(lines),
        title="[bold bright_green]AUTO PILOT — MAXIMUM DATA[/]",
        border_style="bright_green",
    )


# ---------------------------------------------------------------------------
# Step builder
# ---------------------------------------------------------------------------


def _build_steps(plan: RunPlan) -> list[Step]:
    py = plan.python_exe
    cfg = str(plan.config_path)
    end = plan.end_date.isoformat()

    steps: list[Step] = []

    # ---- per-broker ingestion ----
    for bid in plan.broker_ids:
        # symbol metadata
        steps.append(
            Step(
                f"{bid}: symbol metadata",
                [py, "-m", "mt5pipe.cli.app", "backfill", "symbol-metadata",
                 "--broker", bid, "--config", cfg],
                timeout_seconds=120,
            )
        )

        for sym in plan.symbols:
            # tick backfill
            tfrom = plan.tick_start.get(bid, {}).get(sym)
            if tfrom:
                steps.append(
                    Step(
                        f"{bid}: ticks · {sym}",
                        [py, "-m", "mt5pipe.cli.app", "backfill", "ticks",
                         "--broker", bid, "--symbol", sym,
                         "--from", tfrom.isoformat(), "--to", end, "--config", cfg],
                        timeout_seconds=7200,  # 2h max for tick backfill
                        max_retries=1,
                    )
                )

            # native bars for key timeframes
            bfrom = plan.bar_start.get(bid, {}).get(sym)
            if bfrom:
                for tf in plan.native_bar_tfs:
                    steps.append(
                        Step(
                            f"{bid}: {tf} bars · {sym}",
                            [py, "-m", "mt5pipe.cli.app", "backfill", "bars",
                             "--broker", bid, "--symbol", sym, "--timeframe", tf,
                             "--from", bfrom.isoformat(), "--to", end, "--config", cfg],
                            timeout_seconds=3600,  # 1h max for bar backfill
                            max_retries=1,
                        )
                    )

        # history orders + deals (use earliest bar date across symbols)
        bar_dates = [d for d in plan.bar_start.get(bid, {}).values()]
        hfrom = min(bar_dates).isoformat() if bar_dates else (plan.end_date - dt.timedelta(days=365)).isoformat()

        steps.append(
            Step(
                f"{bid}: history orders",
                [py, "-m", "mt5pipe.cli.app", "backfill", "history-orders",
                 "--broker", bid, "--from", hfrom, "--to", end, "--config", cfg],
                critical=False,
                timeout_seconds=1800,
                max_retries=2,
            )
        )
        steps.append(
            Step(
                f"{bid}: history deals",
                [py, "-m", "mt5pipe.cli.app", "backfill", "history-deals",
                 "--broker", bid, "--from", hfrom, "--to", end, "--config", cfg],
                critical=False,
                timeout_seconds=1800,
                max_retries=2,
            )
        )

    # ---- canonical merge ----
    if len(plan.broker_ids) >= 2:
        ba, bb = plan.broker_ids[0], plan.broker_ids[1]
        for sym in plan.symbols:
            dates: list[dt.date] = []
            for bid in (ba, bb):
                d = plan.tick_start.get(bid, {}).get(sym)
                if d:
                    dates.append(d)
            mfrom = min(dates).isoformat() if dates else (plan.end_date - dt.timedelta(days=365)).isoformat()

            steps.append(
                Step(
                    f"Merge canonical ticks · {sym}",
                    [py, "-m", "mt5pipe.cli.app", "merge", "canonical",
                     "--symbol", sym, "--broker-a", ba, "--broker-b", bb,
                     "--from", mfrom, "--to", end, "--config", cfg],
                    timeout_seconds=7200,
                    max_retries=1,
                )
            )

    # ---- build all bars from canonical ticks ----
    for sym in plan.symbols:
        dates = []
        for bid in plan.broker_ids:
            d = plan.tick_start.get(bid, {}).get(sym)
            if d:
                dates.append(d)
        bf = min(dates).isoformat() if dates else (plan.end_date - dt.timedelta(days=365)).isoformat()

        steps.append(
            Step(
                f"Build all canonical bars · {sym}",
                [py, "-m", "mt5pipe.cli.app", "bars", "build",
                 "--symbol", sym, "--from", bf, "--to", end, "--config", cfg],
                timeout_seconds=3600,
                max_retries=1,
            )
        )

    # ---- dataset ----
    for sym in plan.symbols:
        dates = []
        for bid in plan.broker_ids:
            d = plan.tick_start.get(bid, {}).get(sym)
            if d:
                dates.append(d)
        df = min(dates).isoformat() if dates else (plan.end_date - dt.timedelta(days=365)).isoformat()

        steps.append(
            Step(
                f"Build ML dataset · {sym}",
                [py, "-m", "mt5pipe.cli.app", "dataset", "build",
                 "--symbol", sym, "--from", df, "--to", end, "--config", cfg],
                timeout_seconds=3600,
                max_retries=1,
            )
        )

    # ---- validate + status ----
    steps.append(
        Step("Validate storage integrity",
             [py, "-m", "mt5pipe.cli.app", "status", "validate", "--config", cfg])
    )
    steps.append(
        Step("Generate status report",
             [py, "-m", "mt5pipe.cli.app", "status", "show", "--config", cfg])
    )

    # ---- optional live ----
    if plan.live_broker:
        book_flag = "--enable-book" if plan.live_book else "--no-enable-book"
        for sym in plan.symbols:
            steps.append(
                Step(
                    f"Live collection · {plan.live_broker} · {sym} ({plan.live_warmup_seconds}s)",
                    [py, "-m", "mt5pipe.cli.app", "live", "collect",
                     "--broker", plan.live_broker, "--symbol", sym,
                     book_flag,
                     "--duration", str(plan.live_warmup_seconds),
                     "--config", cfg],
                    critical=False,
                )
            )

    return steps


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------


def _run_step(
    step: Step,
    overall: Progress,
    overall_task: int,
    current: Progress,
    current_task: int,
    log_lines: deque[str],
) -> tuple[int, list[str]]:
    ts = time.strftime("%H:%M:%S")
    cmd_text = subprocess.list2cmdline(step.command)
    log_lines.append(f"[dim][{ts}][/] [bold]$ {escape(cmd_text)}[/]")

    output_tail: deque[str] = deque(maxlen=STEP_LOG_TAIL_LINES)

    proc = subprocess.Popen(
        step.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env={
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        },
    )

    deadline = (time.monotonic() + step.timeout_seconds) if step.timeout_seconds > 0 else 0.0

    pulses = 0
    while True:
        # Check process-level timeout
        if deadline and time.monotonic() >= deadline:
            ts = time.strftime("%H:%M:%S")
            log_lines.append(
                f"[bold yellow][{ts}] ⏱ Step timed out after {step.timeout_seconds}s — killing[/]"
            )
            proc.kill()
            proc.wait(timeout=10)
            output_tail.append(f"KILLED: step timed out after {step.timeout_seconds}s")
            break

        line = ""
        if proc.stdout is not None:
            line = proc.stdout.readline()
        if line:
            msg = line.rstrip()
            if msg:
                output_tail.append(msg)
                ts = time.strftime("%H:%M:%S")
                log_lines.append(f"[dim][{ts}][/] {escape(msg)}")
            pulses = min(95, pulses + 1)
            current.update(current_task, completed=pulses)
        if proc.poll() is not None and not line:
            break

    # drain
    if proc.stdout is not None:
        for tail in proc.stdout.readlines():
            msg = tail.rstrip()
            if msg:
                output_tail.append(msg)
                ts = time.strftime("%H:%M:%S")
                log_lines.append(f"[dim][{ts}][/] {escape(msg)}")

    current.update(current_task, completed=100)
    if proc.returncode == 0:
        overall.advance(overall_task, 1)
    return (proc.returncode or 0, list(output_tail))


# ---------------------------------------------------------------------------
# TUI layout helpers
# ---------------------------------------------------------------------------


def _header_table(
    step_idx: int,
    total: int,
    step_name: str,
    started: float,
    ok: int,
    fail: int,
) -> Table:
    elapsed = time.time() - started
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    elapsed_str = f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

    t = Table.grid(expand=True)
    t.add_column(ratio=1)
    t.add_column(ratio=3)
    t.add_row("[bold]Step[/]", f"[bold]{step_idx}[/] / {total}")
    t.add_row("[bold]Current[/]", step_name)
    t.add_row("[bold]Elapsed[/]", elapsed_str)
    t.add_row("[bold]Succeeded[/]", f"[green]{ok}[/]")
    t.add_row("[bold]Failed[/]", f"[red]{fail}[/]" if fail else "[dim]0[/]")
    return t


def _make_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=9),
        Layout(name="progress", size=8),
        Layout(name="logs"),
    )
    return layout


# ---------------------------------------------------------------------------
# Main orchestration flow
# ---------------------------------------------------------------------------


def run() -> int:
    if os.name != "nt":
        console.print("[red]This pipeline requires Windows (MT5 is Windows-only).[/]")
        return 2

    _show_banner()

    # --- config ---
    config_path = _find_config()
    if not config_path.exists():
        console.print(f"[red]Config not found:[/] {config_path}")
        console.print("[dim]Create config/pipeline.yaml with your broker terminal_path entries.[/]")
        return 2
    console.print(f"  Config   [dim]{config_path}[/]")
    console.print()

    # --- deps ---
    _ensure_deps()

    # --- detection ---
    console.rule("[bold bright_blue] Broker Detection [/]")
    console.print()

    try:
        all_caps, cfg, symbols = _detect_phase(config_path)
    except Exception as exc:
        console.print(f"\n[bold red]Detection failed:[/] {exc}")
        console.print("[dim]Make sure both MT5 terminals are open and logged in.[/]")
        return 3

    console.print()

    for caps in all_caps:
        console.print(_broker_panel(caps))
        console.print()

    connected = [c for c in all_caps if c.connected]
    if len(connected) < 2:
        console.print("[bold red]✗ Need at least 2 connected broker terminals.[/]")
        console.print("  Open both MT5 terminals in Windows, log in, then run again.")
        return 3

    console.print("[bold green]✓ Successfully connected to both broker terminals![/]")
    console.print()

    # --- auto plan ---
    plan = _compute_plan(all_caps, cfg, config_path, symbols)
    steps = _build_steps(plan)

    console.print(_plan_panel(plan, len(steps)))
    console.print()

    # --- countdown ---
    console.rule("[bold bright_yellow] Launching Pipeline [/]")
    for tick in range(3, 0, -1):
        console.print(f"  [bold bright_yellow]{tick}...[/]", end="\r")
        time.sleep(1)
    console.print(f"  [bold bright_green]GO!{'':.<30}[/]")
    console.print()

    # --- execution TUI ---
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
    current_task = current.add_task("Preparing", total=100)

    log_lines: deque[str] = deque(maxlen=50)
    failures: list[StepFailure] = []
    success_count = 0
    started = time.time()

    layout = _make_layout()

    def render(idx: int, name: str) -> Layout:
        header = _header_table(idx, len(steps), name, started, success_count, len(failures))
        layout["header"].update(
            Panel(header, title="[bold bright_blue]Pipeline Control Center[/]", border_style="bright_blue")
        )
        layout["progress"].update(
            Panel(Group(overall, current), title="[bold cyan]Progress + ETA[/]", border_style="cyan")
        )
        layout["logs"].update(
            Panel(
                "\n".join(log_lines) if log_lines else "[dim]Waiting for output...[/]",
                title="[bold magenta]Live Logs[/]",
                border_style="magenta",
            )
        )
        return layout

    try:
        with Live(render(0, "Starting"), console=console, refresh_per_second=8, screen=True):
            for idx, step in enumerate(steps, 1):
                current.update(current_task, description=step.name, completed=0)

                # Retry loop for non-critical steps
                attempts = 1 + step.max_retries
                code = -1
                output_tail: list[str] = []
                step_started = time.time()

                for attempt in range(1, attempts + 1):
                    if attempt > 1:
                        ts = time.strftime("%H:%M:%S")
                        log_lines.append(
                            f"[yellow][{ts}] ↻ Retry {attempt}/{attempts}: {escape(step.name)}[/]"
                        )
                        current.update(current_task, completed=0)
                        time.sleep(min(attempt * 2, 10))  # backoff

                    code, output_tail = _run_step(step, overall, overall_task, current, current_task, log_lines)
                    if code == 0:
                        break

                step_ended = time.time()

                ts = time.strftime("%H:%M:%S")
                if code == 0:
                    success_count += 1
                    log_lines.append(f"[green][{ts}] ✓ {escape(step.name)}[/]")
                else:
                    failure_log = _write_step_failure_log(
                        step=step,
                        code=code,
                        output_lines=output_tail,
                        started_ts=step_started,
                        ended_ts=step_ended,
                    )
                    failures.append(StepFailure(name=step.name, code=code, log_path=failure_log))
                    log_lines.append(f"[red][{ts}] ✗ {escape(step.name)} (exit {code})[/]")
                    log_lines.append(
                        f"[bold yellow][{ts}] log saved:[/] {escape(str(failure_log))}"
                    )
                    if step.critical:
                        log_lines.append(f"[bold red]Critical step failed — stopping pipeline.[/]")
                        break

                render(idx, step.name)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Pipeline interrupted by user (Ctrl+C).[/]")

    # --- summary ---
    elapsed = time.time() - started
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    elapsed_str = f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

    console.print()
    summary = Table(title="[bold]Run Summary[/]", border_style="bright_blue")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("Total steps", str(len(steps)))
    summary.add_row("Succeeded", f"[green]{success_count}[/]")
    summary.add_row("Failed", f"[red]{len(failures)}[/]" if failures else "[dim]0[/]")
    summary.add_row("Duration", elapsed_str)
    console.print(summary)

    if failures:
        console.print()
        ftbl = Table(title="[bold red]Failed Steps[/]", border_style="red")
        ftbl.add_column("Step")
        ftbl.add_column("Exit Code", justify="right")
        ftbl.add_column("Failure Log")
        for fail in failures:
            ftbl.add_row(fail.name, str(fail.code), str(fail.log_path))
        console.print(ftbl)
        console.print("\n[bold yellow]Tip:[/] open the failure log file above for full traceback and command output.")
        return 1

    console.print()
    console.print("[bold bright_green]✓ Pipeline completed — all data downloaded and processed.[/]")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
