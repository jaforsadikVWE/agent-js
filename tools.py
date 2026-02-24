"""
Tool definitions and execution handlers for the CLI AI Agent.

Each tool has:
  1. A schema dict (Ollama tool-calling format)
  2. An execute_* function that performs the action and returns a string result
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

import requests

from config import COMMAND_TIMEOUT, MAX_OUTPUT_CHARS


# ═══════════════════════════════════════════════════════════════
#  TOOL SCHEMAS  (passed to Ollama via `tools=`)
# ═══════════════════════════════════════════════════════════════

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Execute a shell command on the user's device and return its "
                "stdout and stderr. Use this for installing packages, running "
                "scripts, git operations, system tasks, etc."
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
                        "description": (
                            "Optional working directory. Defaults to current dir."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            f"Timeout in seconds (default {COMMAND_TIMEOUT})."
                        ),
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
            "description": (
                "Search for files by name pattern (glob) in a directory tree."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Glob pattern, e.g. '*.py' or '**/*.json'."
                        ),
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
                        "description": (
                            "Optional glob to filter files, e.g. '*.py'."
                        ),
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
                "Use for calculations, data processing, or quick scripts. "
                "The code runs in a subprocess with full access to installed packages."
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
]


# ═══════════════════════════════════════════════════════════════
#  TOOL EXECUTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _truncate(text: str) -> str:
    """Truncate output to MAX_OUTPUT_CHARS."""
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n\n... [truncated, {len(text)} total chars]"
    return text


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
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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


def _fmt_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} TB"


def execute_search_files(
    pattern: str,
    directory: str = ".",
    **_,
) -> str:
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
        # Try using grep if available (much faster on Termux/Linux)
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
            # grep not available, fallback to Python
            pass

        # Python fallback
        matches = []
        glob_pattern = os.path.join(directory, file_pattern or "**/*")
        for fp in glob.glob(glob_pattern, recursive=True):
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

    # Disk usage
    try:
        total, used, free = shutil.disk_usage("/")
        info["disk_total"] = _fmt_size(total)
        info["disk_used"] = _fmt_size(used)
        info["disk_free"] = _fmt_size(free)
    except Exception:
        pass

    # Memory - try reading /proc/meminfo (Linux/Termux)
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
#  DISPATCHER
# ═══════════════════════════════════════════════════════════════

TOOL_HANDLERS = {
    "run_command":      execute_run_command,
    "read_file":        execute_read_file,
    "write_file":       execute_write_file,
    "append_file":      execute_append_file,
    "list_directory":   execute_list_directory,
    "search_files":     execute_search_files,
    "search_in_files":  execute_search_in_files,
    "get_system_info":  execute_get_system_info,
    "fetch_url":        execute_fetch_url,
    "python_exec":      execute_python_exec,
}


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
