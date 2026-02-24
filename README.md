# 🚀 CLI AI Agent for Termux

A powerful command-line AI agent that uses the Ollama cloud API with tool-calling capabilities. It can execute shell commands, read/write files, search your filesystem, fetch URLs, run Python code, and more — all through natural language.

## Setup (Termux)

### 1. Install Python
```bash
pkg update && pkg upgrade
pkg install python
```

### 2. Clone / copy the files
Put `agent.py`, `tools.py`, `config.py`, and `requirements.txt` in a folder.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your API key
Get an API key from [ollama.com](https://ollama.com), then:
```bash
export OLLAMA_API_KEY=your_api_key_here
```

To make it permanent, add the line above to your `~/.bashrc` or `~/.zshrc`.

### 5. Run the agent
```bash
python agent.py
```

## Usage

### Interactive mode
```bash
python agent.py
```

### Single-shot mode
```bash
python agent.py "list all Python files in the current directory"
```

### Auto-approve tool calls
```bash
python agent.py --yolo
```

### Use a different model
```bash
python agent.py --model gpt-oss:120b-cloud
```

## Slash Commands

| Command    | Description                          |
|------------|--------------------------------------|
| `/help`    | Show available commands              |
| `/tools`   | List all available tools             |
| `/model`   | Show current model name              |
| `/history` | Show conversation history summary    |
| `/clear`   | Clear conversation history           |
| `/yolo`    | Toggle auto-approve for tool calls   |
| `/exit`    | Exit the agent                       |

## Available Tools

| Tool              | Description                                    |
|-------------------|------------------------------------------------|
| `run_command`     | Execute shell commands with timeout            |
| `read_file`       | Read file contents (supports line ranges)      |
| `write_file`      | Create or overwrite files                      |
| `append_file`     | Append content to files                        |
| `list_directory`  | List files and directories                     |
| `search_files`    | Search files by name pattern (glob)            |
| `search_in_files` | Grep-like text search inside files             |
| `get_system_info` | Get OS, memory, disk, Python version info      |
| `fetch_url`       | HTTP GET requests for web content/APIs         |
| `python_exec`     | Execute Python code snippets                   |

## Environment Variables

| Variable        | Description                                | Default                  |
|-----------------|--------------------------------------------|--------------------------|
| `OLLAMA_API_KEY` | Your Ollama API key (required)            | —                        |
| `AGENT_MODEL`   | Override the default model                 | `qwen3-coder:480b-cloud` |
| `AGENT_DEBUG`   | Show full error tracebacks when set        | —                        |

## Examples

```
You ❯ Create a script that downloads a webpage and counts the words

You ❯ What's my disk usage and available memory?

You ❯ Find all .json files in my home directory larger than 1MB

You ❯ Install the requests library and test it by fetching example.com
```
