"""
Vitals Tracker 
Usage:
    python launch.py
"""

import sys
import os
import subprocess
import signal
import time
import webbrowser

# Force UTF-8 encoding for Windows terminal to support emojis
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
os.environ["PYTHONUTF8"] = "1"

# Ensure backend is in path
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
sys.path.insert(0, BACKEND_DIR)

from services.integration import preflight_check, migrate_db, print_preflight_report
from config import Config

processes = []


def cleanup(signum=None, frame=None):
    """Gracefully shut down all subprocesses."""
    print("\n\n🛑 Shutting down Vitals Tracker...")
    for name, proc in processes:
        if proc.poll() is None:
            print(f"  Stopping {name}...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    print("👋 Goodbye!\n")
    sys.exit(0)


def main():
    global processes

    print("\n" + "=" * 55)
    print("  🏥 Vitals Tracker — Multi-Agent Health Monitor")
    print("  Starting 6-agent system...")
    print("=" * 55)

    # --- Step 1: Preflight checks (Agent 6) ---
    report = preflight_check()
    print_preflight_report(report)

    if report["overall"] == "blocked":
        failed = [c for c in report["checks"] if c["status"] == "fail"]
        print("❌ Cannot start. Fix these issues first:")
        for f in failed:
            print(f"   • {f['name']}: {f['detail']}")
        sys.exit(1)

    # --- Step 2: Database migration (Agent 6) ---
    migrate_db()

    # --- Step 3: Install frontend dependencies if needed ---
    if not os.path.exists(os.path.join(FRONTEND_DIR, "node_modules")):
        print("📦 Installing frontend dependencies...")
        subprocess.run(
            ["npm", "install"],
            cwd=FRONTEND_DIR,
            shell=True,
            check=True,
        )
        print("✅ Frontend dependencies installed!\n")

    # --- Step 4: Register signal handler ---
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # --- Step 5: Start Flask backend ---
    print("🐍 Starting Flask backend on port", Config.PORT, "...")
    backend_proc = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=BACKEND_DIR
    )
    processes.append(("Backend (Flask)", backend_proc))

    # Give Flask a moment to start
    time.sleep(2)

    # --- Step 6: Start Vite dev server ---
    print("⚡ Starting Vite frontend on port", Config.FRONTEND_PORT, "...")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_DIR,
        shell=True
    )
    processes.append(("Frontend (Vite)", frontend_proc))

    # Give Vite a moment to start
    time.sleep(3)

    # --- Step 7: Open browser ---
    url = f"http://localhost:{Config.FRONTEND_PORT}"
    print(f"\n{'=' * 55}")
    print(f"  ✅ Vitals Tracker is running!")
    print(f"  🌐 Frontend:  {url}")
    print(f"  🐍 Backend:   http://localhost:{Config.PORT}")
    print(f"  📊 Status:    http://localhost:{Config.PORT}/api/status")
    print(f"  Press Ctrl+C to stop")
    print(f"{'=' * 55}\n")

    webbrowser.open(url)

    # --- Step 8: Monitor processes ---
    try:
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"⚠️  {name} exited with code {proc.returncode}")
                    cleanup()
            time.sleep(2)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
