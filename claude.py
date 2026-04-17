import anthropic
import os
from copy import deepcopy
from typing import Tuple
import json
import re
import traceback
from threading import Lock
import sys

from stream_events import emit
file_path = os.path.abspath(__file__)
file_path = os.path.dirname(file_path) + '/'

prompt = open(file_path + 'prompts/general_prompt.txt').read()
# json_prompt = open(file_path + 'prompts/json_prompt.txt').read()
error_prompt = open(file_path + 'prompts/error_prompt.txt').read()

# Anthropic API key: prefer environment variable, fall back to config.py
_ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not _ANTHROPIC_API_KEY:
    try:
        from config import ANTHROPIC_API_KEY as _cfg_key
        _ANTHROPIC_API_KEY = _cfg_key
    except ImportError:
        pass
if not _ANTHROPIC_API_KEY or _ANTHROPIC_API_KEY == 'YOUR_ANTHROPIC_API_KEY_HERE':
    raise EnvironmentError(
        "ANTHROPIC_API_KEY not set. Either:\n"
        "  export ANTHROPIC_API_KEY='sk-ant-...'\n"
        "or set it in config.py"
    )
_ANTHROPIC_API_KEY = _ANTHROPIC_API_KEY.strip()
client = anthropic.Anthropic(api_key=_ANTHROPIC_API_KEY)

# Model name for Claude Opus 4.6
CLAUDE_MODEL = "claude-sonnet-4-6"

# Import FormatAgent skill
sys.path.insert(0, os.path.join(file_path, 'skills', 'format-agent', 'scripts'))
from format_response import format_response as _skill_format_response
from format_response import run_format_pipeline

# Import ReasoningAgent skill
sys.path.insert(0, os.path.join(file_path, 'skills', 'reasoning-agent', 'scripts'))
from reasoning_response import reasoning_response as _skill_reasoning_response
from reasoning_response import run_reasoning_pipeline

# Import Rigor-FormatAgent skill
sys.path.insert(0, os.path.join(file_path, 'skills', 'rigor-format-agent', 'scripts'))
from rigor_format_response import run_rigor_format_pipeline

# Import Rigor-ReasoningAgent skill
sys.path.insert(0, os.path.join(file_path, 'skills', 'rigor-reasoning-agent', 'scripts'))
from rigor_reasoning_response import run_rigor_reasoning_pipeline

# Import QueryPlanner skill
sys.path.insert(0, os.path.join(file_path, 'skills', 'query-planner', 'scripts'))
from qp_query_planner import run_query_planner_pipeline

__all__ = ['chat_and_get_formatted', 'set_log_enable', 'format_agent', 'reasoning_agent',
           'run_format_pipeline', 'run_reasoning_pipeline',
           'run_rigor_format_pipeline', 'run_rigor_reasoning_pipeline',
           'run_query_planner_pipeline']


LOG_ENABLE = False
log_file = open(file_path + 'logs/claude_log.txt', 'a')
log_file_lock = Lock()

def format_agent(user_input: str, use_literature: bool = False) -> str:
    """
    Call FormatAgent skill with the appropriate prompt using Claude Opus 4.6.
    
    This is a thin wrapper that delegates to the FormatAgent Claude Skill
    located in skills/format-agent/scripts/format_response.py.
    
    The skill handles:
    - Prompt selection (WITH-LITERATURE vs NO-LITERATURE)
    - Claude API call with system prompt
    - JSON extraction from Claude's response
    
    For the full pipeline (compress + format + hallucination check), use
    run_format_pipeline() instead.
    
    Args:
        user_input: The formatted input string for the FormatAgent
        use_literature: If True, use the full prompt (with HIRN literature/PubMed support).
                        If False, use the no-literature prompt that forbids ALL PubMed IDs.
    """
    # Delegate to the FormatAgent skill's format_response function
    # We pass empty compressed_neo4j and cypher_queries since user_input
    # already contains the fully formatted prompt text
    return _skill_format_response(
        human_query=user_input,
        compressed_neo4j=[],
        cypher_queries=[],
        use_literature=use_literature,
    )


def reasoning_agent(user_input: str, use_literature: bool = False) -> str:
    """
    Call ReasoningAgent skill with the appropriate prompt using Claude Opus 4.6.
    
    This is a thin wrapper that delegates to the ReasoningAgent Claude Skill
    located in skills/reasoning-agent/scripts/reasoning_response.py.
    
    The skill handles:
    - Prompt selection (WITH-LITERATURE vs NO-LITERATURE)
    - Multi-hop reasoning with chain-of-thought
    - Claude API call with reasoning system prompt
    - JSON extraction from Claude's response
    
    For the full pipeline (compress + reason + hallucination check), use
    run_reasoning_pipeline() instead.
    
    Args:
        user_input: The formatted input string for the ReasoningAgent
        use_literature: If True, use the full prompt (with HIRN literature/PubMed support).
                        If False, use the no-literature prompt that forbids ALL PubMed IDs.
    """
    return _skill_reasoning_response(
        human_query=user_input,
        neo4j_results=[],
        cypher_queries=[],
        use_literature=use_literature,
    )


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
        # Find matching closing brace
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return text[brace_start:i+1]
    
    # Return as-is if nothing found
    return text


def set_log_enable(enable: bool):
    global LOG_ENABLE
    if (not LOG_ENABLE and enable):
        log_file.write('\n\nLOG START\n\n')
    LOG_ENABLE = enable


def _convert_messages_for_claude(messages: list) -> Tuple[str, list]:
    """
    Convert OpenAI-style messages to Claude format.
    
    OpenAI format: [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]
    Claude format: system parameter + messages list (no "system" role in messages)
    
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


def chat(messages: list) -> Tuple[bool, str]:
    max_trail = 3
    while (max_trail > 0):
        try:
            system_prompt, claude_messages = _convert_messages_for_claude(messages)
            
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                temperature=0.2,
                system=system_prompt + "\n\nIMPORTANT: Your response MUST be a valid JSON object. Output ONLY the JSON, no additional text, no markdown code blocks.",
                messages=claude_messages
            )
            
            raw_text = response.content[0].text
            # Extract JSON from potential markdown wrapping
            content = _extract_json_from_response(raw_text)

            _in_tok = response.usage.input_tokens
            _out_tok = response.usage.output_tokens
            _cost_input = _in_tok * 5.0 / 1_000_000
            _cost_output = _out_tok * 25.0 / 1_000_000
            _cost_total = _cost_input + _cost_output
            emit("planner_agent_claude_done", {
                "agent": "PlannerAgent",
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
            # assert (type(response['text']) == dict)
            pass
        else:
            assert (type(response['functions'] == list))
            assert (len(response['functions']) > 0)
            for function in response['functions']:
                assert (type(function) == dict)
                assert ('name' in function)
                assert (type(function['name']) == str)
                assert ('input' in function)
                assert (type(function['input']) == str)
                assert (function['name'] in ('pankbase_chat_one_round', 'hirn_chat_one_round', 'template_chat_one_round')) #Function Names here
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
    trials = 3
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
