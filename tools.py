"""
Tool definitions and execution handlers for the CLI AI Agent.

Each tool has:
  1. A schema dict (Ollama tool-calling format)
  2. An execute_* function that performs the action and returns a string result
  3. A risk level: "safe", "moderate", or "dangerous"
"""

import os
import sys
import json
import glob
import shutil
import platform
import subprocess
import tempfile
import traceback
import re
from urllib.parse import quote_plus

import requests

from config import COMMAND_TIMEOUT, MAX_OUTPUT_CHARS


# ═══════════════════════════════════════════════════════════════
#  RISK LEVELS — used by the agent to decide approval behavior
# ═══════════════════════════════════════════════════════════════
# "safe"      → auto-approved always (read-only, no side effects)
# "moderate"  → auto-approved in --yolo mode, confirmed otherwise
# "dangerous" → ALWAYS asks for confirmation, even in --yolo mode

TOOL_RISK = {
    # ── Core: File & System ────────────────────────
    "run_command":       "moderate",
    "read_file":         "safe",
    "write_file":        "moderate",
    "append_file":       "moderate",
    "list_directory":    "safe",
    "search_files":      "safe",
    "search_in_files":   "safe",
    "get_system_info":   "safe",
    "fetch_url":         "safe",
    "python_exec":       "moderate",
    # ── Web ────────────────────────────────────────
    "web_search":        "safe",
    # ── Package Management ─────────────────────────
    "pkg_install":       "dangerous",
    "pkg_uninstall":     "dangerous",
    "pkg_list":          "safe",
    "pip_install":       "moderate",
    "pip_list":          "safe",
    # ── Termux API ─────────────────────────────────
    "termux_notification":  "moderate",
    "termux_vibrate":       "safe",
    "termux_torch":         "safe",
    "termux_battery":       "safe",
    "termux_clipboard_get": "safe",
    "termux_clipboard_set": "moderate",
    "termux_tts":           "moderate",
    "termux_sms_send":      "dangerous",
    "termux_sms_list":      "moderate",
    "termux_camera_photo":  "moderate",
    "termux_location":      "moderate",
    "termux_share":         "moderate",
    "termux_toast":         "safe",
    "termux_wifi_info":     "safe",
    "termux_open_url":      "moderate",
    "termux_volume":        "moderate",
    "termux_contact_list":  "safe",
    "termux_download":      "moderate",
    # ── File Management ────────────────────────────
    "delete_file":       "dangerous",
    "move_file":         "moderate",
    "copy_file":         "moderate",
}


# ═══════════════════════════════════════════════════════════════
#  TOOL SCHEMAS  (passed to Ollama via `tools=`)
# ═══════════════════════════════════════════════════════════════

TOOL_SCHEMAS = [
    # ────────────────────────────────────────────────
    #  CORE TOOLS (original 10)
    # ────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Execute a shell command on the user's device and return its "
                "stdout and stderr. Use this for running scripts, git operations, "
                "system tasks, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Optional working directory. Defaults to current dir.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": f"Timeout in seconds (default {COMMAND_TIMEOUT}).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file and return it as text. "
                "Supports optional line range to read only part of the file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional 1-based start line.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional 1-based end line (inclusive).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite a file with the given content. "
                "Parent directories are created automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Append content to the end of an existing file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to append.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": (
                "List files and directories in the given path. "
                "Returns names, types (file/dir), and sizes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list. Defaults to '.'",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "If true, list recursively. Default false.",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Max recursion depth (default 3).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files by name pattern (glob) in a directory tree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. '*.py' or '**/*.json'.",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Root directory to search from. Default '.'",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": (
                "Search for a text pattern inside files (like grep). "
                "Returns matching lines with file paths and line numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text or regex pattern to search for.",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in. Default '.'",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Optional glob to filter files, e.g. '*.py'.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Case-sensitive search. Default true.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": (
                "Get system information: OS, architecture, Python version, "
                "disk usage, memory, environment variables, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch content from a URL via HTTP GET. "
                "Returns the response body as text (useful for APIs, web pages)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch.",
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers as key-value pairs.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": (
                "Execute a Python code snippet and return its stdout output. "
                "Use for calculations, data processing, or quick scripts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    },
                },
                "required": ["code"],
            },
        },
    },

    # ────────────────────────────────────────────────
    #  WEB SEARCH
    # ────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using DuckDuckGo and return results with titles, "
                "URLs, and snippets. Use this when you need to look up information, "
                "find documentation, or answer questions about current events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 15).",
                    },
                },
                "required": ["query"],
            },
        },
    },

    # ────────────────────────────────────────────────
    #  PACKAGE MANAGEMENT
    # ────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "pkg_install",
            "description": (
                "Install a system package using the package manager (pkg on Termux, "
                "apt on Debian/Ubuntu). DANGEROUS: always asks for confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "packages": {
                        "type": "string",
                        "description": "Space-separated package names to install.",
                    },
                },
                "required": ["packages"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pkg_uninstall",
            "description": "Uninstall a system package. DANGEROUS: always asks for confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "packages": {
                        "type": "string",
                        "description": "Space-separated package names to uninstall.",
                    },
                },
                "required": ["packages"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pkg_list",
            "description": "List installed system packages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": "Optional filter to grep package names.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pip_install",
            "description": "Install Python packages using pip.",
            "parameters": {
                "type": "object",
                "properties": {
                    "packages": {
                        "type": "string",
                        "description": "Space-separated Python package names.",
                    },
                },
                "required": ["packages"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pip_list",
            "description": "List installed Python packages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": "Optional filter to search package names.",
                    },
                },
                "required": [],
            },
        },
    },

    # ────────────────────────────────────────────────
    #  FILE MANAGEMENT (extended)
    # ────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": (
                "Delete a file or directory. DANGEROUS: always asks for confirmation. "
                "Use recursive=true for directories."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file or directory to delete.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "If true, delete directory recursively. Default false.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "Move or rename a file or directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source path.",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination path.",
                    },
                },
                "required": ["source", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "copy_file",
            "description": "Copy a file or directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source path.",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination path.",
                    },
                },
                "required": ["source", "destination"],
            },
        },
    },

    # ────────────────────────────────────────────────
    #  TERMUX API TOOLS
    #  (require: pkg install termux-api)
    # ────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "termux_notification",
            "description": (
                "Show an Android notification. Requires termux-api package."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Notification title.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Notification body text.",
                    },
                    "id": {
                        "type": "integer",
                        "description": "Optional notification ID (to update existing).",
                    },
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "termux_vibrate",
            "description": "Vibrate the device. Requires termux-api.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_ms": {
                        "type": "integer",
                        "description": "Vibration duration in milliseconds (default 1000).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "termux_torch",
            "description": "Turn the device flashlight/torch on or off. Requires termux-api.",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "True to turn on, false to turn off.",
                    },
                },
                "required": ["enabled"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "termux_battery",
            "description": "Get battery status (level, charging, temperature). Requires termux-api.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "termux_clipboard_get",
            "description": "Get the current clipboard content. Requires termux-api.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "termux_clipboard_set",
            "description": "Set the device clipboard content. Requires termux-api.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to copy to clipboard.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "termux_tts",
            "description": "Speak text aloud using text-to-speech. Requires termux-api.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to speak.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "termux_sms_send",
            "description": (
                "Send an SMS message. DANGEROUS: always asks for confirmation. "
                "Requires termux-api."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "string",
                        "description": "Phone number to send SMS to.",
                    },
                    "message": {
                        "type": "string",
                        "description": "SMS message body.",
                    },
                },
                "required": ["number", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "termux_sms_list",
            "description": (
                "List recent SMS messages from inbox. Requires termux-api."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max number of messages to return (default 10).",
                    },
                    "type": {
                        "type": "string",
                        "description": "SMS type: 'inbox', 'sent', 'draft', 'all'. Default 'inbox'.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "termux_camera_photo",
            "description": (
                "Take a photo using the device camera. Requires termux-api."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "camera_id": {
                        "type": "integer",
                        "description": "Camera ID (0=back, 1=front). Default 0.",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save the photo. Default 'photo.jpg'.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "termux_location",
            "description": (
                "Get the device GPS location (lat, lon, altitude). Requires termux-api."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Location provider: 'gps', 'network', 'passive'. Default 'gps'.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "termux_share",
            "description": (
                "Share a file or text via Android's share intent. Requires termux-api."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to share (if sharing text).",
                    },
                    "file": {
                        "type": "string",
                        "description": "File path to share (if sharing a file).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "termux_toast",
            "description": "Show a short Android toast message. Requires termux-api.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Toast message text.",
                    },
                    "position": {
                        "type": "string",
                        "description": "Position: 'top', 'middle', 'bottom'. Default 'middle'.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "termux_wifi_info",
            "description": "Get current WiFi connection info. Requires termux-api.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "termux_open_url",
            "description": "Open a URL in the default Android browser. Requires termux-api.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to open.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "termux_volume",
            "description": "Get or set device volume levels. Requires termux-api.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stream": {
                        "type": "string",
                        "description": "Audio stream: 'music', 'ring', 'alarm', 'notification'. Default 'music'.",
                    },
                    "volume": {
                        "type": "integer",
                        "description": "Volume level to set (0-15). Omit to just read current volume.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "termux_contact_list",
            "description": "List contacts from the device. Requires termux-api.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "termux_download",
            "description": (
                "Download a file using the Android download manager. Requires termux-api."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to download.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional download notification title.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional download notification description.",
                    },
                },
                "required": ["url"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════
#  HELPER
# ═══════════════════════════════════════════════════════════════

def _truncate(text: str) -> str:
    """Truncate output to MAX_OUTPUT_CHARS."""
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n\n... [truncated, {len(text)} total chars]"
    return text


def _fmt_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} TB"


def _run_termux_cmd(cmd: list, timeout: int = 30) -> str:
    """Run a termux-* command and return output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout.strip()
        if result.stderr:
            output += f"\nSTDERR: {result.stderr.strip()}"
        if result.returncode != 0:
            output += f"\nEXIT CODE: {result.returncode}"
        return output if output else "(no output)"
    except FileNotFoundError:
        return (
            f"ERROR: '{cmd[0]}' not found. Install termux-api:\n"
            "  pkg install termux-api\n"
            "Also install the Termux:API app from F-Droid."
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {timeout}s."
    except Exception as e:
        return f"ERROR: {e}"


# ═══════════════════════════════════════════════════════════════
#  CORE TOOL EXECUTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def execute_run_command(
    command: str,
    working_dir: str = ".",
    timeout: int = COMMAND_TIMEOUT,
    **_,
) -> str:
    """Execute a shell command and return combined output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=working_dir or ".",
            timeout=timeout,
        )
        output_parts = []
        if result.stdout:
            output_parts.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output_parts.append(f"STDERR:\n{result.stderr}")
        output_parts.append(f"EXIT CODE: {result.returncode}")
        return _truncate("\n".join(output_parts))
    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {timeout} seconds."
    except Exception as e:
        return f"ERROR: {e}"


def execute_read_file(
    path: str,
    start_line: int = None,
    end_line: int = None,
    **_,
) -> str:
    """Read file contents, optionally a line range."""
    try:
        path = os.path.expanduser(path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total = len(lines)
        if start_line or end_line:
            s = max(1, start_line or 1) - 1
            e = min(total, end_line or total)
            lines = lines[s:e]
            header = f"[Lines {s+1}-{e} of {total}]\n"
        else:
            header = f"[{total} lines]\n"

        return _truncate(header + "".join(lines))
    except FileNotFoundError:
        return f"ERROR: File not found: {path}"
    except Exception as e:
        return f"ERROR: {e}"


def execute_write_file(path: str, content: str, **_) -> str:
    """Write content to a file, creating parent dirs."""
    try:
        path = os.path.expanduser(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"ERROR: {e}"


def execute_append_file(path: str, content: str, **_) -> str:
    """Append content to a file."""
    try:
        path = os.path.expanduser(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully appended {len(content)} chars to {path}"
    except Exception as e:
        return f"ERROR: {e}"


def execute_list_directory(
    path: str = ".",
    recursive: bool = False,
    max_depth: int = 3,
    **_,
) -> str:
    """List directory contents."""
    try:
        path = os.path.expanduser(path or ".")
        entries = []

        if recursive:
            for root, dirs, files in os.walk(path):
                depth = root.replace(path, "").count(os.sep)
                if depth >= max_depth:
                    dirs.clear()
                    continue
                indent = "  " * depth
                entries.append(f"{indent}📁 {os.path.basename(root)}/")
                for f in sorted(files):
                    fp = os.path.join(root, f)
                    try:
                        size = os.path.getsize(fp)
                    except OSError:
                        size = 0
                    entries.append(f"{indent}  📄 {f}  ({_fmt_size(size)})")
                if len(entries) > 500:
                    entries.append("... [truncated]")
                    break
        else:
            for item in sorted(os.listdir(path)):
                fp = os.path.join(path, item)
                if os.path.isdir(fp):
                    entries.append(f"📁 {item}/")
                else:
                    try:
                        size = os.path.getsize(fp)
                    except OSError:
                        size = 0
                    entries.append(f"📄 {item}  ({_fmt_size(size)})")

        return "\n".join(entries) if entries else "(empty directory)"
    except FileNotFoundError:
        return f"ERROR: Directory not found: {path}"
    except Exception as e:
        return f"ERROR: {e}"


def execute_search_files(pattern: str, directory: str = ".", **_) -> str:
    """Search for files matching a glob pattern."""
    try:
        directory = os.path.expanduser(directory or ".")
        full_pattern = os.path.join(directory, pattern)
        matches = glob.glob(full_pattern, recursive=True)
        if not matches:
            return f"No files matching '{pattern}' in {directory}"
        result = f"Found {len(matches)} match(es):\n"
        for m in matches[:200]:
            result += f"  {m}\n"
        if len(matches) > 200:
            result += f"  ... and {len(matches) - 200} more"
        return _truncate(result)
    except Exception as e:
        return f"ERROR: {e}"


def execute_search_in_files(
    query: str,
    directory: str = ".",
    file_pattern: str = None,
    case_sensitive: bool = True,
    **_,
) -> str:
    """Grep-like search inside files."""
    try:
        directory = os.path.expanduser(directory or ".")
        cmd_parts = ["grep", "-rn"]
        if not case_sensitive:
            cmd_parts.append("-i")
        if file_pattern:
            cmd_parts.extend(["--include", file_pattern])
        cmd_parts.extend([query, directory])

        try:
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout or "(no matches)"
            return _truncate(output)
        except FileNotFoundError:
            pass

        # Python fallback
        matches = []
        glob_pat = os.path.join(directory, file_pattern or "**/*")
        for fp in glob.glob(glob_pat, recursive=True):
            if not os.path.isfile(fp):
                continue
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        check_line = line if case_sensitive else line.lower()
                        check_query = query if case_sensitive else query.lower()
                        if check_query in check_line:
                            matches.append(f"{fp}:{i}: {line.rstrip()}")
                            if len(matches) >= 100:
                                break
            except (OSError, UnicodeDecodeError):
                continue
            if len(matches) >= 100:
                break

        if not matches:
            return f"No matches for '{query}' in {directory}"
        return _truncate("\n".join(matches))
    except Exception as e:
        return f"ERROR: {e}"


def execute_get_system_info(**_) -> str:
    """Return system information."""
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "current_directory": os.getcwd(),
        "home_directory": os.path.expanduser("~"),
        "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
    }

    try:
        total, used, free = shutil.disk_usage("/")
        info["disk_total"] = _fmt_size(total)
        info["disk_used"] = _fmt_size(used)
        info["disk_free"] = _fmt_size(free)
    except Exception:
        pass

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal:", "MemAvailable:", "MemFree:")):
                    key = line.split(":")[0].strip()
                    val = line.split(":")[1].strip()
                    info[key] = val
    except (FileNotFoundError, PermissionError):
        pass

    lines = [f"  {k}: {v}" for k, v in info.items()]
    return "System Information:\n" + "\n".join(lines)


def execute_fetch_url(url: str, headers: dict = None, **_) -> str:
    """Fetch content from a URL."""
    try:
        resp = requests.get(
            url,
            headers=headers or {},
            timeout=30,
            allow_redirects=True,
        )
        result = f"Status: {resp.status_code}\n"
        content_type = resp.headers.get("content-type", "")
        result += f"Content-Type: {content_type}\n\n"

        if "json" in content_type:
            try:
                result += json.dumps(resp.json(), indent=2)
            except Exception:
                result += resp.text
        else:
            result += resp.text

        return _truncate(result)
    except requests.exceptions.Timeout:
        return "ERROR: Request timed out (30s)."
    except requests.exceptions.ConnectionError:
        return f"ERROR: Could not connect to {url}"
    except Exception as e:
        return f"ERROR: {e}"


def execute_python_exec(code: str, **_) -> str:
    """Execute Python code in a subprocess."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
        )
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"STDERR:\n{result.stderr}")
        output_parts.append(f"EXIT CODE: {result.returncode}")
        return _truncate("\n".join(output_parts)) if output_parts else "(no output)"
    except subprocess.TimeoutExpired:
        return f"ERROR: Code execution timed out after {COMMAND_TIMEOUT}s."
    except Exception as e:
        return f"ERROR: {e}"


# ═══════════════════════════════════════════════════════════════
#  WEB SEARCH
# ═══════════════════════════════════════════════════════════════

def execute_web_search(query: str, num_results: int = 5, **_) -> str:
    """Search the web using DuckDuckGo HTML (no API key needed)."""
    try:
        num_results = min(num_results or 5, 15)
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Linux; Android 13; Pixel 7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Mobile Safari/537.36"
            ),
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        # Parse results from HTML using regex (avoid dependency on bs4)
        results = []
        # DuckDuckGo HTML results are in <a class="result__a" ...>
        links = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            resp.text,
            re.DOTALL,
        )
        snippets = re.findall(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            resp.text,
            re.DOTALL,
        )

        for i, (link, title) in enumerate(links[:num_results]):
            # Clean HTML tags from title and snippet
            title_clean = re.sub(r'<[^>]+>', '', title).strip()
            snippet_clean = ""
            if i < len(snippets):
                snippet_clean = re.sub(r'<[^>]+>', '', snippets[i]).strip()

            # DuckDuckGo wraps URLs in a redirect, extract actual URL
            actual_url = link
            if "uddg=" in link:
                match = re.search(r'uddg=([^&]+)', link)
                if match:
                    from urllib.parse import unquote
                    actual_url = unquote(match.group(1))

            results.append(
                f"{i+1}. {title_clean}\n"
                f"   URL: {actual_url}\n"
                f"   {snippet_clean}\n"
            )

        if not results:
            return f"No search results found for: {query}"

        return f"Search results for: {query}\n\n" + "\n".join(results)
    except Exception as e:
        return f"ERROR: Web search failed: {e}"


# ═══════════════════════════════════════════════════════════════
#  PACKAGE MANAGEMENT
# ═══════════════════════════════════════════════════════════════

def _detect_pkg_manager() -> str:
    """Detect available package manager."""
    for mgr in ("pkg", "apt", "apt-get"):
        try:
            subprocess.run(
                [mgr, "--version"],
                capture_output=True,
                timeout=5,
            )
            return mgr
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return "pkg"  # default for Termux


def execute_pkg_install(packages: str, **_) -> str:
    """Install system packages."""
    mgr = _detect_pkg_manager()
    cmd = f"{mgr} install -y {packages}"
    return execute_run_command(cmd, timeout=120)


def execute_pkg_uninstall(packages: str, **_) -> str:
    """Uninstall system packages."""
    mgr = _detect_pkg_manager()
    cmd = f"{mgr} remove -y {packages}"
    return execute_run_command(cmd, timeout=60)


def execute_pkg_list(filter: str = None, **_) -> str:
    """List installed packages."""
    mgr = _detect_pkg_manager()
    cmd = f"{mgr} list --installed"
    if filter:
        cmd += f" 2>/dev/null | grep -i {filter}"
    return execute_run_command(cmd)


def execute_pip_install(packages: str, **_) -> str:
    """Install Python packages."""
    cmd = f"{sys.executable} -m pip install {packages}"
    return execute_run_command(cmd, timeout=120)


def execute_pip_list(filter: str = None, **_) -> str:
    """List installed Python packages."""
    cmd = f"{sys.executable} -m pip list"
    if filter:
        cmd += f" 2>/dev/null | grep -i {filter}"
    return execute_run_command(cmd)


# ═══════════════════════════════════════════════════════════════
#  FILE MANAGEMENT (extended)
# ═══════════════════════════════════════════════════════════════

def execute_delete_file(path: str, recursive: bool = False, **_) -> str:
    """Delete a file or directory."""
    try:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            if recursive:
                shutil.rmtree(path)
                return f"Deleted directory (recursive): {path}"
            else:
                os.rmdir(path)
                return f"Deleted empty directory: {path}"
        elif os.path.exists(path):
            os.remove(path)
            return f"Deleted file: {path}"
        else:
            return f"ERROR: Path not found: {path}"
    except Exception as e:
        return f"ERROR: {e}"


def execute_move_file(source: str, destination: str, **_) -> str:
    """Move/rename a file or directory."""
    try:
        source = os.path.expanduser(source)
        destination = os.path.expanduser(destination)
        shutil.move(source, destination)
        return f"Moved: {source} → {destination}"
    except Exception as e:
        return f"ERROR: {e}"


def execute_copy_file(source: str, destination: str, **_) -> str:
    """Copy a file or directory."""
    try:
        source = os.path.expanduser(source)
        destination = os.path.expanduser(destination)
        if os.path.isdir(source):
            shutil.copytree(source, destination)
        else:
            parent = os.path.dirname(destination)
            if parent:
                os.makedirs(parent, exist_ok=True)
            shutil.copy2(source, destination)
        return f"Copied: {source} → {destination}"
    except Exception as e:
        return f"ERROR: {e}"


# ═══════════════════════════════════════════════════════════════
#  TERMUX API TOOLS
# ═══════════════════════════════════════════════════════════════

def execute_termux_notification(
    title: str, content: str, id: int = None, **_
) -> str:
    """Show an Android notification."""
    cmd = ["termux-notification", "--title", title, "--content", content]
    if id is not None:
        cmd.extend(["--id", str(id)])
    return _run_termux_cmd(cmd)


def execute_termux_vibrate(duration_ms: int = 1000, **_) -> str:
    """Vibrate the device."""
    return _run_termux_cmd(
        ["termux-vibrate", "-d", str(duration_ms or 1000)]
    )


def execute_termux_torch(enabled: bool = True, **_) -> str:
    """Toggle the flashlight."""
    state = "on" if enabled else "off"
    return _run_termux_cmd(["termux-torch", state])


def execute_termux_battery(**_) -> str:
    """Get battery info."""
    return _run_termux_cmd(["termux-battery-status"])


def execute_termux_clipboard_get(**_) -> str:
    """Read clipboard."""
    return _run_termux_cmd(["termux-clipboard-get"])


def execute_termux_clipboard_set(text: str, **_) -> str:
    """Set clipboard content."""
    try:
        proc = subprocess.run(
            ["termux-clipboard-set"],
            input=text,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "Clipboard set successfully." if proc.returncode == 0 else f"ERROR: {proc.stderr}"
    except FileNotFoundError:
        return "ERROR: termux-clipboard-set not found. Install: pkg install termux-api"
    except Exception as e:
        return f"ERROR: {e}"


def execute_termux_tts(text: str, **_) -> str:
    """Text-to-speech."""
    try:
        proc = subprocess.run(
            ["termux-tts-speak"],
            input=text,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return "Spoke text successfully." if proc.returncode == 0 else f"ERROR: {proc.stderr}"
    except FileNotFoundError:
        return "ERROR: termux-tts-speak not found. Install: pkg install termux-api"
    except Exception as e:
        return f"ERROR: {e}"


def execute_termux_sms_send(number: str, message: str, **_) -> str:
    """Send an SMS."""
    return _run_termux_cmd(
        ["termux-sms-send", "-n", number, message],
        timeout=30,
    )


def execute_termux_sms_list(limit: int = 10, type: str = "inbox", **_) -> str:
    """List SMS messages."""
    cmd = ["termux-sms-list", "-l", str(limit or 10)]
    if type and type != "inbox":
        cmd.extend(["-t", type])
    return _run_termux_cmd(cmd)


def execute_termux_camera_photo(
    camera_id: int = 0, output_path: str = "photo.jpg", **_
) -> str:
    """Take a photo."""
    output_path = output_path or "photo.jpg"
    return _run_termux_cmd(
        ["termux-camera-photo", "-c", str(camera_id or 0), output_path],
        timeout=30,
    )


def execute_termux_location(provider: str = "gps", **_) -> str:
    """Get GPS location."""
    return _run_termux_cmd(
        ["termux-location", "-p", provider or "gps"],
        timeout=30,
    )


def execute_termux_share(text: str = None, file: str = None, **_) -> str:
    """Share content via Android intent."""
    if file:
        file = os.path.expanduser(file)
        return _run_termux_cmd(["termux-share", file])
    elif text:
        try:
            proc = subprocess.run(
                ["termux-share", "-a", "send"],
                input=text,
                capture_output=True,
                text=True,
                timeout=15,
            )
            return "Shared successfully." if proc.returncode == 0 else f"ERROR: {proc.stderr}"
        except FileNotFoundError:
            return "ERROR: termux-share not found. Install: pkg install termux-api"
        except Exception as e:
            return f"ERROR: {e}"
    return "ERROR: Provide either 'text' or 'file' to share."


def execute_termux_toast(text: str, position: str = "middle", **_) -> str:
    """Show a toast message."""
    cmd = ["termux-toast", "-g", position or "middle", text]
    return _run_termux_cmd(cmd)


def execute_termux_wifi_info(**_) -> str:
    """Get WiFi info."""
    return _run_termux_cmd(["termux-wifi-connectioninfo"])


def execute_termux_open_url(url: str, **_) -> str:
    """Open URL in browser."""
    return _run_termux_cmd(["termux-open-url", url])


def execute_termux_volume(stream: str = None, volume: int = None, **_) -> str:
    """Get or set volume."""
    if volume is not None and stream:
        return _run_termux_cmd(
            ["termux-volume", stream, str(volume)]
        )
    return _run_termux_cmd(["termux-volume"])


def execute_termux_contact_list(**_) -> str:
    """List contacts."""
    return _run_termux_cmd(["termux-contact-list"])


def execute_termux_download(
    url: str, title: str = None, description: str = None, **_
) -> str:
    """Download a file via Android download manager."""
    cmd = ["termux-download", url]
    if title:
        cmd.extend(["-t", title])
    if description:
        cmd.extend(["-d", description])
    return _run_termux_cmd(cmd)


# ═══════════════════════════════════════════════════════════════
#  DISPATCHER
# ═══════════════════════════════════════════════════════════════

TOOL_HANDLERS = {
    # Core
    "run_command":          execute_run_command,
    "read_file":            execute_read_file,
    "write_file":           execute_write_file,
    "append_file":          execute_append_file,
    "list_directory":       execute_list_directory,
    "search_files":         execute_search_files,
    "search_in_files":      execute_search_in_files,
    "get_system_info":      execute_get_system_info,
    "fetch_url":            execute_fetch_url,
    "python_exec":          execute_python_exec,
    # Web
    "web_search":           execute_web_search,
    # Package Management
    "pkg_install":          execute_pkg_install,
    "pkg_uninstall":        execute_pkg_uninstall,
    "pkg_list":             execute_pkg_list,
    "pip_install":          execute_pip_install,
    "pip_list":             execute_pip_list,
    # File Management
    "delete_file":          execute_delete_file,
    "move_file":            execute_move_file,
    "copy_file":            execute_copy_file,
    # Termux API
    "termux_notification":  execute_termux_notification,
    "termux_vibrate":       execute_termux_vibrate,
    "termux_torch":         execute_termux_torch,
    "termux_battery":       execute_termux_battery,
    "termux_clipboard_get": execute_termux_clipboard_get,
    "termux_clipboard_set": execute_termux_clipboard_set,
    "termux_tts":           execute_termux_tts,
    "termux_sms_send":      execute_termux_sms_send,
    "termux_sms_list":      execute_termux_sms_list,
    "termux_camera_photo":  execute_termux_camera_photo,
    "termux_location":      execute_termux_location,
    "termux_share":         execute_termux_share,
    "termux_toast":         execute_termux_toast,
    "termux_wifi_info":     execute_termux_wifi_info,
    "termux_open_url":      execute_termux_open_url,
    "termux_volume":        execute_termux_volume,
    "termux_contact_list":  execute_termux_contact_list,
    "termux_download":      execute_termux_download,
}


def get_tool_risk(name: str) -> str:
    """Get the risk level of a tool."""
    return TOOL_RISK.get(name, "moderate")


def execute_tool(name: str, arguments: dict) -> str:
    """
    Execute a tool by name with the given arguments.
    Returns the tool's output as a string.
    """
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return f"ERROR: Unknown tool '{name}'. Available: {', '.join(TOOL_HANDLERS.keys())}"
    try:
        return handler(**arguments)
    except TypeError as e:
        return f"ERROR: Bad arguments for tool '{name}': {e}"
    except Exception as e:
        return f"ERROR executing '{name}': {traceback.format_exc()}"
