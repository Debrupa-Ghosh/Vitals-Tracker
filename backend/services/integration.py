"""
Agent 6: Integration & Launch Agent
Orchestrates system startup, database migration, health checks, and monitoring.
"""

import sys
import time
import subprocess
import json
from datetime import datetime

_start_time = None


def _check_python_version() -> dict:
    """Check Python version >= 3.10."""
    ver = sys.version_info
    ok = ver.major >= 3 and ver.minor >= 10
    return {
        "name": "Python Version",
        "status": "pass" if ok else "fail",
        "detail": f"{ver.major}.{ver.minor}.{ver.micro}",
        "required": "3.10+",
    }


def _check_mysql_connection() -> dict:
    """Check MySQL connectivity."""
    try:
        from db import get_connection
        start = time.time()
        conn = get_connection(use_db=False)
        latency = round((time.time() - start) * 1000, 1)
        conn.close()
        return {
            "name": "MySQL Connection",
            "status": "pass",
            "detail": f"Connected ({latency}ms)",
        }
    except Exception as e:
        return {
            "name": "MySQL Connection",
            "status": "fail",
            "detail": str(e),
        }


def _check_database_exists() -> dict:
    """Check if the vitals_tracker database exists."""
    try:
        from db import get_connection
        from config import Config
        conn = get_connection(use_db=False)
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        dbs = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        exists = Config.DB_NAME in dbs
        return {
            "name": "Database",
            "status": "pass" if exists else "warn",
            "detail": f"'{Config.DB_NAME}' {'exists' if exists else 'will be created'}",
        }
    except Exception as e:
        return {
            "name": "Database",
            "status": "fail",
            "detail": str(e),
        }


def _check_tables() -> dict:
    """Check if required tables exist."""
    try:
        from db import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = {row[0] for row in cursor.fetchall()}
        cursor.close()
        conn.close()
        required = {"users", "health_records", "doctors", "ui_config"}
        missing = required - tables
        if not missing:
            return {"name": "Tables", "status": "pass", "detail": f"All {len(required)} tables present"}
        else:
            return {"name": "Tables", "status": "warn", "detail": f"Missing: {', '.join(missing)}"}
    except Exception:
        return {"name": "Tables", "status": "warn", "detail": "Will be created on startup"}


def _check_npm() -> dict:
    """Check if npm is available."""
    try:
        result = subprocess.run(
            ["npm", "--version"],
            capture_output=True, text=True, timeout=10,
            shell=True
        )
        if result.returncode == 0:
            return {"name": "npm", "status": "pass", "detail": f"v{result.stdout.strip()}"}
        return {"name": "npm", "status": "fail", "detail": "npm not found"}
    except Exception as e:
        return {"name": "npm", "status": "fail", "detail": str(e)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preflight_check() -> dict:
    """
    Run all preflight checks and return a comprehensive status report.
    """
    checks = [
        _check_python_version(),
        _check_mysql_connection(),
        _check_database_exists(),
        _check_tables(),
        _check_npm(),
    ]

    all_pass = all(c["status"] in ("pass", "warn") for c in checks)

    return {
        "overall": "ready" if all_pass else "blocked",
        "checks": checks,
        "timestamp": datetime.now().isoformat(),
    }


def migrate_db():
    """Create database, tables, and seed data if needed."""
    from db import init_db
    from seed import run_seed

    print("\n🔧 Running database migration...")
    init_db()
    run_seed()
    print("✅ Migration complete!\n")


def mark_started():
    """Mark the system as started (for uptime tracking)."""
    global _start_time
    _start_time = datetime.now()


def get_status() -> dict:
    """
    Return current system health status.
    Called by the /api/status endpoint.
    """
    global _start_time

    # Calculate uptime
    uptime = "unknown"
    if _start_time:
        delta = datetime.now() - _start_time
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)
        uptime = f"{hours}h {minutes}m"

    # Quick DB check
    db_status = _check_mysql_connection()

    # Count records
    stats = {"totalUsers": 0, "totalRecords": 0}
    try:
        from db import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        stats["totalUsers"] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM health_records")
        stats["totalRecords"] = cursor.fetchone()[0]
        cursor.close()
        conn.close()
    except Exception:
        pass

    return {
        "status": "healthy" if db_status["status"] == "pass" else "degraded",
        "uptime": uptime,
        "checks": {
            "database": db_status,
            "backend": {"status": "running", "port": 5000},
        },
        "agents": {
            "input_agent": "ready",
            "analysis_agent": "ready",
            "reporting_agent": "ready",
            "assistant_agent": "ready",
            "frontend_agent": "ready",
            "integration_agent": "ready",
        },
        "stats": stats,
    }


def print_preflight_report(report: dict):
    """Pretty-print the preflight check report to console."""
    icons = {"pass": "✅", "fail": "❌", "warn": "⚠️"}
    print("\n" + "=" * 55)
    print("  🏥 Vitals Tracker — Preflight Check")
    print("=" * 55)
    for check in report["checks"]:
        icon = icons.get(check["status"], "❓")
        print(f"  {icon}  {check['name']:<22} {check['detail']}")
    print("-" * 55)
    overall_icon = "🚀" if report["overall"] == "ready" else "🛑"
    print(f"  {overall_icon}  Overall: {report['overall'].upper()}")
    print("=" * 55 + "\n")
