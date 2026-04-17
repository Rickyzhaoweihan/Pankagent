import anthropic
import config
import os
import re
import sys
from copy import deepcopy
from typing import Tuple
import json
import traceback
from threading import Lock

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from stream_events import emit

file_path = os.path.abspath(__file__)
file_path = os.path.dirname(file_path) + '/'

prompt = open(file_path + 'prompts/general_prompt.txt').read()
# json_prompt = open(file_path + 'prompts/json_prompt.txt').read()
error_prompt = open(file_path + 'prompts/error_prompt.txt').read()

# Anthropic client for Claude Opus 4.6
_ANTHROPIC_API_KEY = config.ANTHROPIC_API_KEY.strip() if isinstance(config.ANTHROPIC_API_KEY, str) else config.ANTHROPIC_API_KEY
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
    
    # If it already starts with { or [, it's raw JSON
    if text.startswith('{') or text.startswith('['):
        return text
    
    # Try to extract from markdown code block
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    # Last resort: find the first { ... } block
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


def _convert_messages_for_claude(messages: list) -> Tuple[str, list]:
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


def set_log_enable(enable: bool):
    global LOG_ENABLE
    if (not LOG_ENABLE and enable):
        log_file.write('\n\nLOG START\n\n')
    LOG_ENABLE = enable


def chat(messages: list) -> Tuple[bool, str]:
    max_trail = 3
    while (max_trail > 0):
        try:
            system_prompt, claude_messages = _convert_messages_for_claude(messages)
            
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                temperature=0.6,
                system=system_prompt + "\n\nIMPORTANT: Your response MUST be a valid JSON object. Output ONLY the JSON, no additional text, no markdown code blocks.",
                messages=claude_messages
            )
            
            raw_text = response.content[0].text
            content = _extract_json_from_response(raw_text)

            _in_tok = response.usage.input_tokens
            _out_tok = response.usage.output_tokens
            _cost_input = _in_tok * 5.0 / 1_000_000
            _cost_output = _out_tok * 25.0 / 1_000_000
            _cost_total = _cost_input + _cost_output
            emit("glkb_agent_claude_done", {
                "agent": "GLKBAgent",
                "input_tokens": _in_tok,
                "output_tokens": _out_tok,
                "cost_input_usd": round(_cost_input, 6),
                "cost_output_usd": round(_cost_output, 6),
                "cost_total_usd": round(_cost_total, 6),
            })

            if (LOG_ENABLE):
                with log_file_lock:
                    log_file.write('\nRequest:\n')
                    log_file.write(json.dumps(messages, indent=2, ensure_ascii=False))
                    log_file.write('\n\nResponse:\n')
                    log_file.write(content)
                    log_file.write('\n\n')
                    log_file.flush()
            return (True, content)
        except:
            traceback.print_exc()
            max_trail -= 1
    return (False, '')


def check_json(response: str) -> Tuple[bool, dict]:
    try:
        response = json.loads(response, strict=False)
        assert (type(response) == dict)
        assert ('draft' in response)
        assert (response['to'] in ('system', 'user'))
        if (response['to'] == 'user'):
            assert (type(response['text']) == str)
        else:
            assert (type(response['functions'] == list))
            assert (len(response['functions']) > 0)
            for function in response['functions']:
                assert (type(function) == dict)
                assert ('name' in function)
                assert (type(function['name']) == str)
                assert ('input' in function)
                assert (type(function['input']) == str)
                assert (function['name'] in ('text_embedding'))
        return (True, response)
    except:
        err_msg = traceback.format_exc()
        return (False, err_msg)


def chat_and_get_formatted(messages: list) -> Tuple[list, dict]:
    '''
    Use messages to chat with Claude, and append the response to messages, but return
    json format with another chat

    messages don't include system message
    '''
    trials = 1
    messages = deepcopy(messages)
    messages.insert(0, {"role": "system", "content": prompt})
    valid = False
    response = ''
    json_response = {}
    for _ in range(0, trials):
        success, response = chat(messages)
        assert (success), 'Failed to get response from Claude'
        messages.append({"role": "assistant", "content": response})
        valid, json_response = check_json(response)
        if (valid):
            break
        messages.append({"role": "user", "content": error_prompt.replace('$err-msg$', json_response)})
    assert (valid), f"Claude continuously returned invalid format, for {trials} times"
    messages.pop(0)
    return (messages, json_response)


if __name__ == "__main__":
    a = chat_and_get_formatted([{"role": "user", "content": "====== From User ======\nWhat is TP 53"}])
    print(a)
