#!/usr/bin/env python3
"""
scan_processes.py

List running processes with executable path and useful metadata.

Usage:
    python scan_processes.py
    python scan_processes.py --filter mt5
    python scan_processes.py --csv processes.csv
    python scan_processes.py --all

Requires:
    pip install psutil
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
from typing import Any

import psutil


def safe_join_cmdline(cmdline: list[str] | None) -> str:
    if not cmdline:
        return ""
    return " ".join(cmdline)


def safe_time(ts: float | None) -> str:
    if not ts:
        return ""
    try:
        return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def get_process_info(proc: psutil.Process) -> dict[str, Any]:
    info: dict[str, Any] = {
        "pid": "",
        "name": "",
        "exe": "",
        "username": "",
        "status": "",
        "created": "",
        "cmdline": "",
    }

    try:
        with proc.oneshot():
            info["pid"] = proc.pid
            info["name"] = proc.name()
            info["status"] = proc.status()
            info["created"] = safe_time(proc.create_time())

            try:
                info["exe"] = proc.exe()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                info["exe"] = "<access denied>"
            except Exception:
                info["exe"] = "<unavailable>"

            try:
                info["username"] = proc.username()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                info["username"] = "<access denied>"
            except Exception:
                info["username"] = "<unavailable>"

            try:
                info["cmdline"] = safe_join_cmdline(proc.cmdline())
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                info["cmdline"] = "<access denied>"
            except Exception:
                info["cmdline"] = "<unavailable>"

    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        info["name"] = "<terminated>"
    except psutil.AccessDenied:
        info["name"] = "<access denied>"
    except Exception as exc:
        info["name"] = f"<error: {exc}>"

    return info


def collect_processes(name_filter: str | None, include_all: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for proc in psutil.process_iter():
        row = get_process_info(proc)

        if not include_all:
            # Skip obvious blank/error rows unless --all is used
            if not row["name"] or row["name"] in {"<terminated>"}:
                continue

        if name_filter:
            haystack = " ".join(
                [
                    str(row.get("name", "")),
                    str(row.get("exe", "")),
                    str(row.get("cmdline", "")),
                ]
            ).lower()
            if name_filter.lower() not in haystack:
                continue

        rows.append(row)

    rows.sort(key=lambda r: (str(r["name"]).lower(), int(r["pid"]) if str(r["pid"]).isdigit() else 0))
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No matching processes found.")
        return

    headers = ["PID", "Name", "Path", "User", "Status", "Started", "CommandLine"]
    widths = [8, 28, 60, 24, 14, 19, 80]

    def trim(value: Any, width: int) -> str:
        s = str(value)
        if len(s) <= width:
            return s
        return s[: width - 3] + "..."

    header_line = " | ".join(trim(h, w).ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        values = [
            row["pid"],
            row["name"],
            row["exe"],
            row["username"],
            row["status"],
            row["created"],
            row["cmdline"],
        ]
        print(" | ".join(trim(v, w).ljust(w) for v, w in zip(values, widths)))


def write_csv(rows: list[dict[str, Any]], path: str) -> None:
    fieldnames = ["pid", "name", "exe", "username", "status", "created", "cmdline"]
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan running Windows processes and show executable paths.")
    parser.add_argument("--filter", type=str, default=None, help="Filter by process name/path/cmdline substring.")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV export path.")
    parser.add_argument("--all", action="store_true", help="Include more inaccessible/system rows.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect_processes(name_filter=args.filter, include_all=args.all)

    print_table(rows)

    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nSaved CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())