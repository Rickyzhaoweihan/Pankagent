import anthropic
import os
import re
import sys
import json
import traceback
from threading import Lock
from copy import deepcopy
from typing import Tuple

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from stream_events import emit
# -------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------
file_path = os.path.abspath(__file__)
file_path = os.path.dirname(file_path) + '/'

prompt = open(file_path + 'prompts/general_prompt.txt').read()
error_prompt = open(file_path + 'prompts/error_prompt.txt').read()

# Anthropic API key: prefer environment variable, fall back to config.py
_ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not _ANTHROPIC_API_KEY:
    try:
        import config
        _ANTHROPIC_API_KEY = config.ANTHROPIC_API_KEY
    except (ImportError, AttributeError):
        pass
if not _ANTHROPIC_API_KEY or _ANTHROPIC_API_KEY == 'YOUR_ANTHROPIC_API_KEY_HERE':
    raise EnvironmentError("ANTHROPIC_API_KEY not set. Export it or set it in config.py")
_ANTHROPIC_API_KEY = _ANTHROPIC_API_KEY.strip()
client = anthropic.Anthropic(api_key=_ANTHROPIC_API_KEY)

# Model name for Claude Opus 4.6
CLAUDE_MODEL = "claude-sonnet-4-6"

__all__ = ['chat_and_get_formatted', 'set_log_enable']

LOG_ENABLE = False
log_file = open(file_path + 'logs/claude_log.txt', 'a')
log_file_lock = Lock()


def _extract_json_from_response(text: str) -> str:
    """
    Extract JSON from Claude's response, handling cases where it wraps
    JSON in markdown code blocks like ```json ... ```
    """
    text = text.strip()
    
    if text.startswith('{') or text.startswith('['):
        return text
    
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    brace_start = text.find('{')
    if brace_start != -1:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return text[brace_start:i+1]
    
    return text


def _convert_messages_for_claude(messages: list) -> tuple:
    """
    Convert OpenAI-style messages to Claude format.
    Returns: (system_prompt, claude_messages)
    """
    system_prompt = ""
    claude_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        else:
            claude_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    return system_prompt, claude_messages


# -------------------------------------------------------------------------
# Logging control
# -------------------------------------------------------------------------
def set_log_enable(enable: bool):
    global LOG_ENABLE
    if not LOG_ENABLE and enable:
        log_file.write('\n\nLOG START\n\n')
    LOG_ENABLE = enable

# -------------------------------------------------------------------------
# Core chat function
# -------------------------------------------------------------------------
def chat(messages: list) -> Tuple[bool, str]:
    """Send messages to Claude, return (success, text_response)."""
    max_trail = 3
    while max_trail > 0:
        try:
            system_prompt, claude_messages = _convert_messages_for_claude(messages)
            
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                temperature=0.6,
                system=system_prompt,
                messages=claude_messages
            )

            content = response.content[0].text

            _in_tok = response.usage.input_tokens
            _out_tok = response.usage.output_tokens
            _cost_input = _in_tok * 5.0 / 1_000_000
            _cost_output = _out_tok * 25.0 / 1_000_000
            _cost_total = _cost_input + _cost_output
            emit("template_agent_claude_done", {
                "agent": "TemplateToolAgent",
                "input_tokens": _in_tok,
                "output_tokens": _out_tok,
                "cost_input_usd": round(_cost_input, 6),
                "cost_output_usd": round(_cost_output, 6),
                "cost_total_usd": round(_cost_total, 6),
            })

            if LOG_ENABLE:
                with log_file_lock:
                    log_file.write('\nRequest:\n')
                    log_file.write(json.dumps(messages, indent=2, ensure_ascii=False))
                    log_file.write('\n\nResponse:\n')
                    log_file.write(content)
                    log_file.write('\n\n')
                    log_file.flush()

            return True, content

        except Exception as e:
            traceback.print_exc()
            max_trail -= 1

    return False, ''

# -------------------------------------------------------------------------
# Simplified wrapper
# -------------------------------------------------------------------------
def chat_and_get_formatted(messages: list) -> Tuple[list, dict]:
    messages = deepcopy(messages)
    messages.insert(0, {"role": "system", "content": prompt})

    success, response_text = chat(messages)
    if not success:
        raise RuntimeError("Failed to get response from Claude after 3 retries")

    messages.append({"role": "assistant", "content": response_text})
    messages.pop(0)

    # Wrap the plain string in the expected dict format
    response = {
        "to": "user",
        "text": response_text
    }

    return messages, response


# -------------------------------------------------------------------------
# Manual test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    messages, response = chat_and_get_formatted([
        {"role": "user", "content": "====== From User ======\nWhat is TP53?"}
    ])
