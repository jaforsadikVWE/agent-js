#!/usr/bin/env python3
"""
CLI AI Agent — a tool-calling assistant powered by Ollama cloud API.

Usage:
    python agent.py                 # interactive mode
    python agent.py --yolo          # auto-approve commands (no confirmation)
    python agent.py --model NAME    # override model name
    python agent.py "do something"  # single-shot mode: run one prompt and exit
"""

import os
import sys
import json
import argparse
import traceback

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.live import Live

from config import MODEL, API_HOST, API_KEY_ENV, SYSTEM_PROMPT, MAX_HISTORY, MAX_TOOL_ITERATIONS
from tools import TOOL_SCHEMAS, execute_tool, get_tool_risk

# ─── Globals ─────────────────────────────────────────────────
console = Console()
auto_approve = False


# ═══════════════════════════════════════════════════════════════
#  OLLAMA CLIENT SETUP
# ═══════════════════════════════════════════════════════════════

def create_client():
    """Create and return an Ollama client pointed at the cloud API."""
    try:
        from ollama import Client
    except ImportError:
        console.print(
            "[bold red]Error:[/] ollama package not installed.\n"
            "Run: pip install ollama",
        )
        sys.exit(1)

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        console.print(
            f"[bold red]Error:[/] {API_KEY_ENV} environment variable not set.\n"
            "Get an API key from https://ollama.com and set it:\n"
            f"  export {API_KEY_ENV}=your_api_key",
        )
        sys.exit(1)

    return Client(
        host=API_HOST,
        headers={"Authorization": f"Bearer {api_key}"},
    )


# ═══════════════════════════════════════════════════════════════
#  TOOL CALL HANDLING
# ═══════════════════════════════════════════════════════════════

def confirm_tool_call(name: str, args: dict) -> bool:
    """Decide whether to run a tool based on risk level.

    Risk levels:
      safe      → always auto-approved (read-only, no side effects)
      moderate  → auto-approved in --yolo mode, confirmed otherwise
      dangerous → ALWAYS asks for confirmation, even in --yolo mode
    """
    global auto_approve
    risk = get_tool_risk(name)

    # Safe tools always run
    if risk == "safe":
        return True

    # In yolo mode, moderate tools auto-approve; dangerous still asks
    if auto_approve and risk == "moderate":
        return True

    # Show the confirmation panel
    risk_color = {"moderate": "yellow", "dangerous": "red bold"}.get(risk, "yellow")
    risk_label = f"[{risk_color}]{risk.upper()}[/{risk_color}]"

    console.print()
    console.print(
        Panel(
            f"[bold cyan]Tool:[/] {name}  ({risk_label})\n"
            f"[bold cyan]Args:[/] {json.dumps(args, indent=2, ensure_ascii=False)}",
            title="🔧 Tool Call",
            border_style="red" if risk == "dangerous" else "yellow",
        )
    )

    while True:
        choice = console.input("[yellow]Allow? [y/n/a(lways)]: [/]").strip().lower()
        if choice in ("y", "yes"):
            return True
        if choice in ("n", "no"):
            return False
        if choice in ("a", "always"):
            auto_approve = True
            return True
        console.print("[dim]Enter y, n, or a[/]")


def process_tool_calls(client, model: str, messages: list, response) -> str:
    """
    Handle the tool-calling loop.
    If the model requests tool calls, execute them, feed results back,
    and repeat until the model returns a plain text response.
    Returns the final text response.
    """
    iteration = 0

    while iteration < MAX_TOOL_ITERATIONS:
        # Check if response has tool calls
        msg = response.message if hasattr(response, 'message') else response.get('message', {})
        tool_calls = getattr(msg, 'tool_calls', None) or (msg.get('tool_calls') if isinstance(msg, dict) else None)

        if not tool_calls:
            # No tool calls — return the text content
            content = getattr(msg, 'content', None) or (msg.get('content', '') if isinstance(msg, dict) else '')
            return content or ""

        # Add the assistant's message (with tool calls) to history
        messages.append({
            "role": "assistant",
            "content": getattr(msg, 'content', '') or (msg.get('content', '') if isinstance(msg, dict) else ''),
            "tool_calls": _serialize_tool_calls(tool_calls),
        })

        # Execute each tool call
        for tc in tool_calls:
            func_name = _get_tc_field(tc, 'function', 'name')
            func_args = _get_tc_field(tc, 'function', 'arguments')

            # Parse arguments if they're a string
            if isinstance(func_args, str):
                try:
                    func_args = json.loads(func_args)
                except json.JSONDecodeError:
                    func_args = {}

            if not isinstance(func_args, dict):
                func_args = {}

            # Confirm with user
            if not confirm_tool_call(func_name, func_args):
                result = "Tool call was denied by the user."
                console.print(f"  [dim]↳ Denied[/]")
            else:
                console.print(f"  [bold green]▶ Executing:[/] {func_name}")
                result = execute_tool(func_name, func_args)

                # Show truncated result
                preview = result[:300] + "..." if len(result) > 300 else result
                console.print(f"  [dim]↳ {preview}[/]")

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "content": result,
            })

        iteration += 1

        # Send the updated conversation back to the model
        console.print(f"  [dim italic]Thinking... (step {iteration})[/]")
        try:
            response = client.chat(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                stream=False,
            )
        except Exception as e:
            return f"Error during tool iteration: {e}"

    return "(Reached maximum tool iterations. The task may be incomplete.)"


def _serialize_tool_calls(tool_calls):
    """Convert tool calls to serializable dicts."""
    result = []
    for tc in tool_calls:
        entry = {}
        if hasattr(tc, 'function'):
            fn = tc.function
            entry['function'] = {
                'name': getattr(fn, 'name', ''),
                'arguments': getattr(fn, 'arguments', {}),
            }
        elif isinstance(tc, dict):
            entry = tc
        result.append(entry)
    return result


def _get_tc_field(tc, *keys):
    """Safely get nested fields from a tool call (object or dict)."""
    current = tc
    for key in keys:
        if hasattr(current, key):
            current = getattr(current, key)
        elif isinstance(current, dict):
            current = current.get(key, {})
        else:
            return {}
    return current


# ═══════════════════════════════════════════════════════════════
#  MAIN CHAT LOOP
# ═══════════════════════════════════════════════════════════════

def chat_loop(client, model: str, single_prompt: str = None):
    """Main interactive chat loop."""
    global auto_approve
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if single_prompt:
        # Single-shot mode
        handle_user_message(client, model, messages, single_prompt)
        return

    # Interactive mode
    print_welcome(model)

    while True:
        try:
            console.print()
            user_input = console.input("[bold green]You ❯ [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye! 👋[/]")
            break

        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            if cmd in ("/exit", "/quit", "/q"):
                console.print("[dim]Goodbye! 👋[/]")
                break
            elif cmd == "/clear":
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                console.print("[dim]Conversation cleared.[/]")
                continue
            elif cmd == "/history":
                show_history(messages)
                continue
            elif cmd == "/help":
                print_help()
                continue
            elif cmd == "/tools":
                show_tools()
                continue
            elif cmd == "/model":
                console.print(f"[dim]Current model: {model}[/]")
                continue
            elif cmd == "/yolo":
                auto_approve = not auto_approve
                state = "ON" if auto_approve else "OFF"
                console.print(f"[dim]Auto-approve is now {state}[/]")
                continue
            else:
                console.print(f"[dim]Unknown command: {cmd}. Type /help for commands.[/]")
                continue

        handle_user_message(client, model, messages, user_input)


def handle_user_message(client, model: str, messages: list, user_input: str):
    """Process a single user message through the agent."""
    messages.append({"role": "user", "content": user_input})

    # Trim history if too long
    trim_history(messages)

    try:
        console.print()
        console.print("[bold blue]Agent[/] [dim]is thinking...[/]")

        response = client.chat(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            stream=False,
        )

        # Process tool calls (if any) and get final response
        final_text = process_tool_calls(client, model, messages, response)

        if final_text:
            messages.append({"role": "assistant", "content": final_text})
            console.print()
            console.print(Panel(
                Markdown(final_text),
                title="🤖 Agent",
                border_style="blue",
                padding=(1, 2),
            ))

    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/] {e}")
        if os.environ.get("AGENT_DEBUG"):
            console.print(traceback.format_exc())


def trim_history(messages: list):
    """Keep conversation history within MAX_HISTORY limit."""
    # Always keep the system message (index 0)
    if len(messages) <= MAX_HISTORY * 2 + 1:
        return
    # Keep system prompt + last MAX_HISTORY pairs
    keep = MAX_HISTORY * 2
    messages[1:] = messages[-(keep):]


# ═══════════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════════

def print_welcome(model: str):
    """Print the welcome banner."""
    console.print()
    console.print(Panel(
        "[bold cyan]🚀 CLI AI Agent[/]\n\n"
        f"Model: [green]{model}[/]\n"
        f"Tools: [green]{len(TOOL_SCHEMAS)} available[/]\n\n"
        "[dim]Type your request and the agent will use tools to help you.\n"
        "Type /help for commands, /tools to see available tools.[/]",
        border_style="cyan",
        padding=(1, 2),
    ))


def print_help():
    """Print available commands."""
    console.print(Panel(
        "[bold]Commands:[/]\n\n"
        "  [cyan]/help[/]     — Show this help message\n"
        "  [cyan]/tools[/]    — List available tools\n"
        "  [cyan]/model[/]    — Show current model\n"
        "  [cyan]/history[/]  — Show conversation history summary\n"
        "  [cyan]/clear[/]    — Clear conversation history\n"
        "  [cyan]/yolo[/]     — Toggle auto-approve for tool calls\n"
        "  [cyan]/exit[/]     — Exit the agent\n",
        title="Help",
        border_style="cyan",
    ))


def show_tools():
    """List all available tools."""
    lines = ["[bold]Available Tools:[/]\n"]
    for schema in TOOL_SCHEMAS:
        func = schema["function"]
        name = func["name"]
        desc = func["description"]
        params = func.get("parameters", {}).get("properties", {})
        param_names = ", ".join(params.keys()) if params else "none"
        lines.append(f"  [cyan]{name}[/]({param_names})")
        lines.append(f"    [dim]{desc}[/]\n")
    console.print(Panel("\n".join(lines), title="🔧 Tools", border_style="cyan"))


def show_history(messages: list):
    """Show a summary of conversation history."""
    count = len([m for m in messages if m["role"] in ("user", "assistant")])
    tool_count = len([m for m in messages if m["role"] == "tool"])
    console.print(
        f"[dim]Messages: {count} (user+assistant) | Tool calls: {tool_count} | "
        f"Total entries: {len(messages)}[/]"
    )


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CLI AI Agent — your terminal assistant",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Single prompt to run (non-interactive mode)",
    )
    parser.add_argument(
        "--model", "-m",
        default=os.environ.get("AGENT_MODEL", MODEL),
        help=f"Model to use (default: {MODEL})",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Auto-approve all tool calls (no confirmation prompts)",
    )
    args = parser.parse_args()

    global auto_approve
    auto_approve = args.yolo

    client = create_client()
    chat_loop(client, args.model, single_prompt=args.prompt)


if __name__ == "__main__":
    main()
