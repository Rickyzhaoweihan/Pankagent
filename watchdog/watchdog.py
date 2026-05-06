"""
Cron-driven watchdog for the PanKLLM agent server on :8001.

Probe → two-strike rule → restart → email+buglog on failure.

Intended crontab entry (run as the server-owning user):
    * * * * * /usr/bin/flock -n /db/usr/rickyhan/PanKLLM_implementation/logs/.watchdog.lock \
        /db/usr/rickyhan/envs/agent/bin/python \
        /db/usr/rickyhan/PanKLLM_implementation/watchdog/watchdog.py \
        >> /db/usr/rickyhan/PanKLLM_implementation/logs/watchdog.log 2>&1

Lock acquisition order in this file: state file lock → never held concurrently with anything else.
"""

import email.message
import json
import os
import signal
import smtplib
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = Path("/db/usr/rickyhan/PanKLLM_implementation")
PYTHON = "/db/usr/rickyhan/envs/agent/bin/python"
HEALTH_URL = "http://127.0.0.1:8001/health"
HEALTH_TIMEOUT = 5          # seconds per probe
STRIKE_THRESHOLD = 2        # consecutive failures before restart
RESTART_WAIT = 60           # seconds to wait for server to become healthy after restart
RESTART_POLL = 2            # poll interval during wait

EMAIL_THROTTLE = 3600       # seconds between emails for the same ongoing incident
SMTP_HOST = os.environ.get("PANK_SMTP_HOST", "smtp.mail.umich.edu")
SMTP_PORT = int(os.environ.get("PANK_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("PANK_SMTP_USER", "")      # empty = try unauthenticated
SMTP_PASS = os.environ.get("PANK_SMTP_PASS", "")
EMAIL_FROM = os.environ.get("PANK_EMAIL_FROM", f"pank-watchdog@{socket.getfqdn()}")
EMAIL_TO = ["rickyhan@umich.edu", "runbomao@umich.edu"]

LOGS = BASE / "logs"
STATE_FILE = LOGS / "watchdog_state.json"
FAILURE_DIR = LOGS / "restart_failures"
WATCHDOG_LOG = LOGS / "watchdog.log"
PID_FILE = BASE / "server.pid"
SERVER_LOG = BASE / "server.log"
RESTART_SCRIPT = BASE / "watchdog" / "restart_server.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(event: str, detail: str = "") -> None:
    line = f"{_ts()} {event}{(' ' + detail) if detail else ''}"
    print(line, flush=True)


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"consecutive_failures": 0, "last_email_ts": 0.0, "last_incident_id": ""}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _probe() -> tuple[bool, str]:
    """Returns (healthy: bool, detail: str)."""
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=HEALTH_TIMEOUT) as resp:
            if resp.status != 200:
                return False, f"HTTP {resp.status}"
            body = json.loads(resp.read().decode())
            if body.get("status") == "healthy":
                return True, "ok"
            return False, f"status={body.get('status')!r}"
    except urllib.error.URLError as e:
        return False, str(e.reason)
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _pid_alive_and_is_server(pid: int) -> bool:
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").replace("\0", " ")
        return "server.py" in cmdline
    except (FileNotFoundError, PermissionError):
        return False


def _read_pid() -> int | None:
    try:
        return int(PID_FILE.read_text().strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


def _sh(cmd: str, capture: bool = True) -> str:
    """Run a shell command, return stdout+stderr (or empty string on error)."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=capture,
            text=True, timeout=30
        )
        return (r.stdout + r.stderr).strip() if capture else ""
    except Exception:  # noqa: BLE001
        return "(command failed)"


def _tcp_reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Bug log builder
# ---------------------------------------------------------------------------
def _build_bug_log(
    incident_id: str,
    last_probe_detail: str,
    old_pid: int | None,
    new_pid: int | None,
    new_pid_alive_after_wait: bool,
) -> str:
    lines: list[str] = [
        f"# Agent Server Restart Failure — {incident_id}",
        "",
        "## Summary",
        f"- Incident ID: `{incident_id}`",
        f"- Health probe last result: `{last_probe_detail}`",
        f"- Old PID: `{old_pid}`",
        f"- New PID after restart attempt: `{new_pid}`",
        f"- New PID alive after {RESTART_WAIT}s wait: `{new_pid_alive_after_wait}`",
        "",
        "## Process state",
    ]

    if old_pid:
        alive = _pid_alive_and_is_server(old_pid)
        try:
            cmd = Path(f"/proc/{old_pid}/cmdline").read_bytes().decode(errors="replace").replace("\0", " ").strip()
        except Exception:
            cmd = "(unreadable)"
        lines.append(f"- Old PID {old_pid}: alive={alive}, cmdline=`{cmd[:200]}`")
    else:
        lines.append("- Old PID: not found")

    if new_pid:
        alive = _pid_alive_and_is_server(new_pid)
        try:
            cmd = Path(f"/proc/{new_pid}/cmdline").read_bytes().decode(errors="replace").replace("\0", " ").strip()
        except Exception:
            cmd = "(unreadable)"
        lines.append(f"- New PID {new_pid}: alive={alive}, cmdline=`{cmd[:200]}`")
    else:
        lines.append("- New PID: not spawned")

    lines += ["", "## Server log tail (last 200 lines)"]
    log_path = SERVER_LOG
    # try to find the rotated log from this restart (newest .log.<ts> file)
    rotated = sorted(BASE.glob("server.log.*"), reverse=True)
    if rotated:
        log_path = rotated[0]
    lines.append(f"_(from `{log_path}`)_")
    lines.append("```")
    lines.append(_sh(f"tail -200 {log_path}") or "(empty)")
    lines.append("```")

    lines += ["", "## dmesg — last 10 min (oom_kill / segfault)"]
    lines.append("```")
    dmesg = _sh("dmesg --time-format iso 2>/dev/null | tail -200 | grep -E 'oom_kill|segfault|Out of memory' | tail -50")
    lines.append(dmesg or "(nothing found)")
    lines.append("```")

    lines += ["", "## Disk / FD / memory snapshot"]
    lines.append(f"- `df -h /db /home`: {_sh('df -h /db /home 2>/dev/null')}")
    lines.append(f"- `free -h`: {_sh('free -h')}")
    lines.append(f"- `ulimit -n` (open files): {_sh('ulimit -n')}")
    if new_pid and _pid_alive_and_is_server(new_pid):
        fd_count = _sh(f"lsof -p {new_pid} 2>/dev/null | wc -l")
        lines.append(f"- open FDs for new PID {new_pid}: {fd_count}")

    lines += ["", "## Environment sanity"]
    py_stat = _sh(f"ls -ld {PYTHON} 2>&1")
    lines.append(f"- `{PYTHON}` stat: `{py_stat}`")
    import_check = _sh(
        f"{PYTHON} -c \"import anthropic, fastapi, uvicorn, neo4j; print('imports OK')\" 2>&1"
    )
    lines.append(f"- Python import check: `{import_check}`")
    lines.append(f"- vLLM :8002 reachable: {_tcp_reachable('localhost', 8002)}")
    lines.append(f"- Neo4j Bolt :8687 reachable: {_tcp_reachable('localhost', 8687)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Email sender
# ---------------------------------------------------------------------------
def _send_email(subject: str, body: str) -> bool:
    """Returns True if sent successfully.

    SMTP config via env vars:
      PANK_SMTP_HOST   (default: smtp.mail.umich.edu)
      PANK_SMTP_PORT   (default: 587)
      PANK_SMTP_USER   (optional — enables STARTTLS + AUTH LOGIN when set)
      PANK_SMTP_PASS   (required when PANK_SMTP_USER is set)
      PANK_EMAIL_FROM  (default: pank-watchdog@<hostname>)

    UMich relay (smtp.mail.umich.edu) only accepts AUTH GSSAPI from campus hosts,
    so credentials or an external relay are required.  On failure the bug log file
    is still written and email_send_failed is logged — the watchdog never crashes.
    """
    msg = email.message.EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(EMAIL_TO)
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
            s.ehlo()
            try:
                s.starttls()
                s.ehlo()
            except smtplib.SMTPException:
                pass  # server may not advertise STARTTLS on port 25
            if SMTP_USER:
                s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        return True
    except Exception as exc:  # noqa: BLE001
        _log("email_send_failed", str(exc))
        return False


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------
def main() -> None:
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)

    state = _load_state()
    healthy, probe_detail = _probe()

    if healthy:
        if state["consecutive_failures"] > 0:
            _log("probe_ok_recovered", f"was at failures={state['consecutive_failures']}")
            state["consecutive_failures"] = 0
            state["last_email_ts"] = 0.0   # reset throttle so next incident pages immediately
            state["last_incident_id"] = ""
        else:
            _log("probe_ok")
        _save_state(state)
        return

    # --- probe failed ---
    state["consecutive_failures"] += 1
    _log("probe_failed", f"failures={state['consecutive_failures']} detail={probe_detail!r}")

    if state["consecutive_failures"] < STRIKE_THRESHOLD:
        _save_state(state)
        return

    # --- two-strike threshold reached: attempt restart ---
    incident_id = _ts()
    state["last_incident_id"] = incident_id
    _log("restart_attempt", f"incident={incident_id}")

    old_pid = _read_pid()
    new_pid: int | None = None

    restart_ok = subprocess.run(
        ["bash", str(RESTART_SCRIPT)],
        capture_output=True, text=True, timeout=30
    )
    if restart_ok.stdout:
        _log("restart_script_stdout", restart_ok.stdout.replace("\n", " | "))
    if restart_ok.stderr:
        _log("restart_script_stderr", restart_ok.stderr.replace("\n", " | "))

    new_pid = _read_pid()  # restart_server.sh wrote this

    # Wait for server to become healthy
    deadline = time.monotonic() + RESTART_WAIT
    came_up = False
    while time.monotonic() < deadline:
        time.sleep(RESTART_POLL)
        ok, _ = _probe()
        if ok:
            came_up = True
            break

    if came_up:
        _log("restart_succeeded", f"incident={incident_id} new_pid={new_pid}")
        state["consecutive_failures"] = 0
        state["last_email_ts"] = 0.0
        state["last_incident_id"] = ""
        _save_state(state)
        return

    # --- restart failed ---
    new_pid_alive = _pid_alive_and_is_server(new_pid) if new_pid else False
    _log("restart_failed", f"incident={incident_id} new_pid={new_pid} alive={new_pid_alive}")

    bug_log = _build_bug_log(
        incident_id=incident_id,
        last_probe_detail=probe_detail,
        old_pid=old_pid,
        new_pid=new_pid,
        new_pid_alive_after_wait=new_pid_alive,
    )

    bug_log_path = FAILURE_DIR / f"{incident_id.replace(':', '-')}.md"
    try:
        bug_log_path.write_text(bug_log)
        _log("bug_log_written", str(bug_log_path))
    except OSError as exc:
        _log("bug_log_write_failed", str(exc))

    # Email throttle
    now = time.time()
    if now - state["last_email_ts"] >= EMAIL_THROTTLE:
        hostname = socket.getfqdn()
        subject = f"[PanKLLM watchdog] Agent server :8001 restart FAILED on {hostname} at {incident_id}"
        body = f"Bug log saved to {bug_log_path}\n\n{bug_log}"
        sent = _send_email(subject, body)
        if sent:
            _log("email_sent", f"to={EMAIL_TO}")
            state["last_email_ts"] = now
        # If not sent, last_email_ts stays old — don't update so we retry next hour
    else:
        remaining = int(EMAIL_THROTTLE - (now - state["last_email_ts"]))
        _log("email_throttled", f"next_in={remaining}s bug_log={bug_log_path}")

    _save_state(state)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"{_ts()} watchdog_crash {exc}", flush=True)
        sys.exit(1)
