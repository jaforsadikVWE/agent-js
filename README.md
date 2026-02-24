# 🚀 CLI AI Agent for Termux

A powerful command-line AI agent that uses the Ollama cloud API with tool-calling capabilities. It can execute shell commands, read/write files, search the web, manage packages, control your Android device via Termux API, and even work in **voice mode** — all through natural language.

## Setup (Termux)

### 1. Install Python
```bash
pkg update && pkg upgrade
pkg install python
```

### 2. Clone / copy the files
Put all `.py` files, `requirements.txt`, and `setup.sh` in a folder.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your API key
Get an API key from [ollama.com](https://ollama.com), then:
```bash
export OLLAMA_API_KEY=your_api_key_here
```

To make it permanent:
```bash
echo 'export OLLAMA_API_KEY=your_key_here' >> ~/.bashrc
source ~/.bashrc
```

### 5. (Optional) Install globally
```bash
bash setup.sh
```
This creates an `ai` command you can use from anywhere.

### 6. (Optional) Install Termux API
For Android-specific features + voice mode:
```bash
pkg install termux-api
```
Also install the **Termux:API** app from F-Droid.

### 7. Run the agent
```bash
python agent.py        # or just: ai
```

## Usage

```bash
ai                              # interactive mode
ai "list all .py files here"    # single-shot mode
ai --yolo                       # auto-approve moderate tools
ai --voice                      # voice mode (speak & listen)
ai --voice --yolo               # voice + auto-approve
ai --model gpt-oss:120b-cloud   # different model
```

## 🎤 Voice Mode

Talk to the agent with your voice and hear responses spoken aloud.

```bash
ai --voice          # or: ai -v
```

**How it works:**
1. 🎤 Listens via `termux-speech-to-text` (Google speech recognition)
2. Shows what it heard so you can verify
3. Agent processes your request (uses tools etc.)
4. 🔊 Speaks the response via `termux-tts-speak`

**Toggle mid-session:** Type `/voice` to switch on/off anytime.

**Requires:** `pkg install termux-api` + Termux:API app from F-Droid.

## Slash Commands

| Command    | Description                          |
|------------|--------------------------------------|
| `/help`    | Show available commands              |
| `/tools`   | List all available tools             |
| `/model`   | Show current model name              |
| `/history` | Show conversation history summary    |
| `/clear`   | Clear conversation history           |
| `/yolo`    | Toggle auto-approve for tool calls   |
| `/voice`   | Toggle voice mode (speak & listen)   |
| `/exit`    | Exit the agent                       |

## Available Tools (29 total)

### Core (10 tools)
| Tool              | Risk     | Description                              |
|-------------------|----------|------------------------------------------|
| `run_command`     | moderate | Execute shell commands                   |
| `read_file`       | safe     | Read file contents (line ranges)         |
| `write_file`      | moderate | Create or overwrite files                |
| `append_file`     | moderate | Append content to files                  |
| `list_directory`  | safe     | List files and directories               |
| `search_files`    | safe     | Search files by name (glob)              |
| `search_in_files` | safe     | Grep-like text search in files           |
| `get_system_info` | safe     | OS, memory, disk, Python version         |
| `fetch_url`       | safe     | HTTP GET for web content/APIs            |
| `python_exec`     | moderate | Execute Python code snippets             |

### Web & Package Management (6 tools)
| Tool              | Risk      | Description                             |
|-------------------|-----------|-----------------------------------------|
| `web_search`      | safe      | Search the web (DuckDuckGo)             |
| `pkg_install`     | dangerous | Install system packages                 |
| `pkg_uninstall`   | dangerous | Uninstall system packages               |
| `pkg_list`        | safe      | List installed system packages          |
| `pip_install`     | moderate  | Install Python packages                 |
| `pip_list`        | safe      | List installed Python packages          |

### File Management (3 tools)
| Tool              | Risk      | Description                             |
|-------------------|-----------|-----------------------------------------|
| `delete_file`     | dangerous | Delete files or directories             |
| `move_file`       | moderate  | Move or rename files                    |
| `copy_file`       | moderate  | Copy files or directories               |

### Termux API (15 tools)
| Tool                   | Risk      | Description                        |
|------------------------|-----------|------------------------------------|
| `termux_notification`  | moderate  | Show Android notifications         |
| `termux_vibrate`       | safe      | Vibrate the device                 |
| `termux_torch`         | safe      | Toggle flashlight                  |
| `termux_battery`       | safe      | Get battery status                 |
| `termux_clipboard_get` | safe      | Read clipboard                     |
| `termux_clipboard_set` | moderate  | Set clipboard content              |
| `termux_tts`           | moderate  | Text-to-speech                     |
| `termux_sms_send`      | dangerous | Send SMS messages                  |
| `termux_sms_list`      | moderate  | Read SMS inbox                     |
| `termux_camera_photo`  | moderate  | Take photos                        |
| `termux_location`      | moderate  | Get GPS location                   |
| `termux_share`         | moderate  | Share via Android intent           |
| `termux_toast`         | safe      | Show toast messages                |
| `termux_wifi_info`     | safe      | Get WiFi connection info           |
| `termux_open_url`      | moderate  | Open URL in browser                |
| `termux_volume`        | moderate  | Get/set volume levels              |
| `termux_contact_list`  | safe      | List device contacts               |
| `termux_download`      | moderate  | Download files                     |

## 🛡️ Risk-Based Approval System

| Risk Level  | Normal Mode        | `--yolo` Mode       |
|-------------|--------------------|--------------------|
| 🟢 **safe**      | Auto-approved      | Auto-approved      |
| 🟡 **moderate**  | Asks confirmation  | Auto-approved      |
| 🔴 **dangerous** | Asks confirmation  | **Still asks!**    |

Dangerous tools (delete files, install/uninstall packages, send SMS) **always** require your explicit approval, even with `--yolo`.

## Environment Variables

| Variable        | Description                                | Default                  |
|-----------------|--------------------------------------------|--------------------------|
| `OLLAMA_API_KEY` | Your Ollama API key (required)            | —                        |
| `AGENT_MODEL`   | Override the default model                 | `qwen3-coder:480b-cloud` |
| `AGENT_DEBUG`   | Show full error tracebacks when set        | —                        |

## Examples

```
You ❯ Search the web for "best Python libraries for data analysis"

You ❯ Send a notification saying "Task complete!"

You ❯ What's my battery level and WiFi connection?

You ❯ Install ffmpeg and convert video.mp4 to audio

You ❯ Take a photo with the front camera

You ❯ Read my last 5 SMS messages

You ❯ Find all files larger than 10MB in my home directory
```
