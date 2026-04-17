import json
import traceback
from _thread import start_new_thread
from queue import Queue
import time
import sys
import requests
import os 
import re
from utils import add_cypher_query

from performance_monitor import instrument_module_functions
from profiling_tools import profile_to_file

_TEXT2CYPHER_AGENT = None


def force_limit_on_unconstrained(cypher: str, limit: int = 50) -> str:
    """
    FORCE add LIMIT to queries WITHOUT WHERE clause.
    No conditions, no score checks - just adds it.
    """
    if not cypher:
        return cypher
    
    # Skip if already has LIMIT
    if re.search(r'\bLIMIT\s+\d+', cypher, re.IGNORECASE):
        return cypher
    
    # Skip if has WHERE clause (constrained query)
    if re.search(r'\bWHERE\b', cypher, re.IGNORECASE):
        return cypher
    
    # Extract ONLY the MATCH clause (before WITH or RETURN)
    match_clause = re.search(r'\bMATCH\b(.*?)(?=\bWITH\b|\bRETURN\b|$)', cypher, re.IGNORECASE | re.DOTALL)
    if not match_clause:
        return cypher
    
    match_part = match_clause.group(1)
    
    # Find all variables from MATCH clause ONLY
    node_vars = set(re.findall(r'\((\w+)(?::\w+)?[^)]*\)', match_part))
    rel_vars = set(re.findall(r'\[(\w+):\w+[^\]]*\]', match_part))
    all_vars = node_vars | rel_vars
    
    if not all_vars:
        return cypher
    
    # Insert LIMIT before WITH collect()
    collect_match = re.search(r'\bWITH\s+collect\s*\(', cypher, re.IGNORECASE)
    if collect_match:
        vars_str = ', '.join(sorted(all_vars))
        insert_pos = collect_match.start()
        return (cypher[:insert_pos] + 
                f"WITH {vars_str} LIMIT {limit} " + 
                cypher[insert_pos:])
    
    return cypher
_PANKBASE_SESSION: requests.Session | None = None

# Thread-local storage for per-thread sessions (enables true parallel HTTP calls)
import threading
_thread_local = threading.local()


def _get_text2cypher_agent():
    global _TEXT2CYPHER_AGENT
    if _TEXT2CYPHER_AGENT is None:
        os.environ['NEO4J_SCHEMA_PATH'] = 'PankBaseAgent/text_to_cypher/data/input/neo4j_schema_ada.json'
        sys.path.append('text_to_cypher/src')
        from .text_to_cypher.src.text2cypher_agent import Text2CypherAgent
        _TEXT2CYPHER_AGENT = Text2CypherAgent()
    return _TEXT2CYPHER_AGENT


def _get_pankbase_session() -> requests.Session:
    """Get a thread-local session for parallel HTTP requests."""
    if not hasattr(_thread_local, 'session'):
        session = requests.Session()
        session.headers.update({'Content-Type': 'application/json'})
        _thread_local.session = session
    return _thread_local.session


def process_document(content: str, metadata: dict = None) -> dict:
    result = {
        'abstract': None,
        'title': None,
        'pubmedid': None,
        'score': metadata.get('score', 0) if metadata else 0
    }
    lines = content.split('\n')
    for line in lines:
        if line.startswith('abstract: '):
            result['abstract'] = line.replace('abstract: ', '', 1)
        elif line.startswith('title: '):
            result['title'] = line.replace('title: ', '', 1)
        elif line.startswith('pubmedid: '):
            result['pubmedid'] = line.replace('pubmedid: ', '', 1)
            
    return result


def semantic_search(query: str, limit: int = 10) -> list:
    results = abstract_store.similarity_search(query, k=limit)
    results = [process_document(result.page_content, result.metadata) for result in results]
    return results


def run_cypher_2(command: str, parameters = None, timeout: int = 60):
    config = {"timeout": timeout}
    return driver_2.execute_query(command, parameters, **config)



__all__ = ['run_functions']


def run_functions(functions: list[dict]) -> str:
    q = Queue()
    results = []
    num = len(functions)
    for i in range(0, len(functions)):
        name = functions[i]['name']
        input = functions[i]['input']
        func = eval(name)
        start_new_thread(lambda _func, _input, _index: q.put(_func(_input, _index + 1)), (func, input, i))
    while (q.qsize() < num):
        time.sleep(0.2)
    while (q.qsize() > 0):
        results.append(q.get())
    results.sort(key=lambda x: int(x.split('.')[0]))
    result = ''.join(results)
    return result    

def pankbase_api_query(input: str, index: int) -> str:
    '''
    Output the messages needed to return to claude, including function name and
    input, and status, and error (if any). Will handle error and timeout, promise
    to return in 60 seconds.
    '''
    q = Queue()
    start_new_thread(_pankbase_api_query, (input, q))
    start = time.time()
    while (time.time() - start < 60):
        time.sleep(0.2)
        if (q.qsize() == 1):
            break
    size = q.qsize()
    result = f'{index}. PankbaseAPI query: {str([input])[1:-1]}\n'
    if (size == 0):
        result += f'Status: timeout\n'
        result += f'Error: Cannot get the result from PankbaseAPI in 60 seconds\n\n'
        return result
    success, res = q.get(block=False)
    if (success == False):
        result += f'Status: error\n'
        result += f'Error: {str([res])[1:-1]}\n\n'
        return result
    result += f'Status: success\n'
    # No truncation - GPT-4o has 128K token context window
    result += f'Result: {res}\n\n'
    return result

def clean_cypher_for_json(cypher: str) -> str:
    """
    Clean Cypher query for JSON submission to Pankbase API.
    """
    
    cleaned = ' '.join(cypher.split())
    cleaned = cleaned.replace('"', '\"').replace("'", '\"')
    return cleaned

def _pankbase_api_query_core(input: str, q: Queue) -> None:
    try:
        agent = _get_text2cypher_agent()
        
        # Use refinement with test-time scaling (adaptive approach)
        # First try simple generation
        print(f"\n{'─'*60}")
        print(f"TEXT-TO-CYPHER INPUT: {input}")
        print(f"{'─'*60}")
        
        vllm_start_time = time.time()
        cypher_result = agent.respond(input)
        vllm_elapsed = time.time() - vllm_start_time
        
        print(f"\n🔍 GENERATED CYPHER QUERY (vLLM took {vllm_elapsed:.2f}s) for: {input[:40]}...")
        print(f"{'─'*60}")
        print(cypher_result)
        print(f"{'─'*60}")
        
        # Import validator and auto-fix to check and fix quality
        import sys
        from .text_to_cypher.src.cypher_validator import validate_cypher, auto_fix_cypher
        validation = validate_cypher(cypher_result)
        
        # ALWAYS run auto-fix to apply safety fixes (like LIMIT 50 for unconstrained queries)
        fixed_cypher, fixes = auto_fix_cypher(cypher_result, validation)
        if fixes:
            cypher_result = fixed_cypher
            validation = validate_cypher(cypher_result)
            with open('log.txt', 'a') as log_file:
                log_file.write(f"Query: {input}\n")
                log_file.write(f"Auto-fixes applied (new score: {validation['score']}/100)\n")
                log_file.write(f"Fixes applied: {', '.join(fixes)}\n")
                log_file.write(f"Fixed Cypher: {cypher_result}\n")
        
        # If score is still low after auto-fix, use iterative refinement
        if validation['score'] < 90:
            refinement_result = agent.respond_with_refinement(input, max_iterations=5)
            cypher_result = refinement_result['cypher']
            final_score = refinement_result['score']
            
            # Log refinement metrics to JSONL
            try:
                from .text_to_cypher.src.refinement_logger import log_refinement_metrics
                log_refinement_metrics(input, refinement_result)
            except Exception as e:
                pass
            
            # Log refinement details to text log
            with open('log.txt', 'a') as log_file:
                log_file.write(f"Query: {input}\n")
                log_file.write(f"Refinement used: Best from iteration {refinement_result['iteration']}\n")
                log_file.write(f"Score: {final_score}/100\n")
                log_file.write(f"All attempts:\n")
                for attempt in refinement_result['all_attempts']:
                    log_file.write(f"  Iteration {attempt['iteration']}: score={attempt['score']}\n")
                log_file.write(f"Final Cypher: {cypher_result}\n")
                
                # Check if final score is still too low after refinement
                if final_score < 90 or refinement_result['validation_report'].get('errors'):
                    log_file.write(f"REJECTED: Score {final_score} is below threshold (70) or has critical errors.\n")
                    from .text_to_cypher.src.cypher_validator import format_validation_report
                    log_file.write(f"Validation Report:\n{format_validation_report(refinement_result['validation_report'])}\n")
                    log_file.write("##########################\n")
                    q.put({
                        "error": f"Query quality too low after {refinement_result['iteration']} refinement iterations. "
                                f"Score: {final_score}/100. Please rephrase your question or be more specific.",
                        "validation": refinement_result['validation_report']
                    })
                    return
                
                log_file.write("##########################\n")
        else:
            final_score = validation['score']
            print(f"📝 Writing to log for: {input[:40]}...", flush=True)
            try:
                with open('log.txt', 'a') as log_file:
                    log_file.write(f"Query: {input}\n")
                    log_file.write(f"Score: {final_score}/100 (no refinement needed)\n")
                    log_file.write(f"Cypher: {cypher_result}\n")
                    log_file.write("##########################\n")
                print(f"📝 Log write complete for: {input[:40]}...", flush=True)
            except Exception as e:
                print(f"❌ Log write FAILED for {input[:40]}: {e}", flush=True)
        
        print(f"\n✅ VALIDATION SCORE: {final_score}/100", flush=True)
        if validation.get('errors'):
            print(f"❌ Validation ERRORS: {validation['errors']}")
        if validation.get('warnings'):
            print(f"⚠️  Validation WARNINGS: {validation['warnings']}")
        
        cleaned_cypher = clean_cypher_for_json(cypher_result)
        
        # FORCE add LIMIT 50 to unconstrained queries (no WHERE clause)
        cleaned_cypher = force_limit_on_unconstrained(cleaned_cypher, limit=50)
        
        # Debug: Show we're about to send to API
        print(f"\n🔄 PREPARING TO SEND: {input[:50]}...")
        
        print(f"\n📤 SENDING TO NEO4J API:")
        print(f"{'─'*60}")
        print(cleaned_cypher)
        print(f"{'─'*60}")

        from neo4j import GraphDatabase as _GDB
        _bolt_uri = os.environ.get("NEO4J_BOLT_URI", "bolt://localhost:8687")
        _neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
        _neo4j_pass = os.environ.get("NEO4J_PASSWORD", "password")
        _neo4j_db = os.environ.get("NEO4J_DATABASE", "pankgraph")

        neo4j_start_time = time.time()
        _driver = _GDB.driver(_bolt_uri, auth=(_neo4j_user, _neo4j_pass))
        with _driver.session(database=_neo4j_db) as _sess:
            _raw = _sess.run(cleaned_cypher)
            _keys = list(_raw.keys())
            _records = []
            for _rec in _raw:
                _row = {}
                for _k in _keys:
                    _v = _rec[_k]
                    if isinstance(_v, list):
                        _row[_k] = []
                        for _item in _v:
                            if hasattr(_item, 'labels'):
                                _row[_k].append({"__type__": "node", "id": str(_item.element_id), "element_id": _item.element_id, "labels": list(_item.labels), "properties": dict(_item)})
                            elif hasattr(_item, 'type'):
                                _row[_k].append({"__type__": "relationship", "id": str(_item.element_id), "element_id": _item.element_id, "type": _item.type, "properties": dict(_item)})
                            else:
                                _row[_k].append(_item)
                    else:
                        _row[_k] = _v
                _records.append(_row)
        _driver.close()
        neo4j_elapsed = time.time() - neo4j_start_time
        print(f"\n  NEO4J BOLT QUERY TOOK: {neo4j_elapsed:.2f} seconds (Query: {input[:30]}...)")

        try:
            result = {"records": _records, "keys": _keys}
            
            print(f"\n📥 NEO4J API RESPONSE:")
            print(f"{'─'*60}")
            result_str = json.dumps(result, indent=2, ensure_ascii=False)
            print(result_str)
            print(f"{'─'*60}")
            
            # Check if result has actual data
            # Old API format: {"results": "...string...", "query": "..."}
            # New API format: {"records": [...], "keys": [...], "truncated": bool}
            has_data = True
            if isinstance(result, dict):
                # New API format
                if "records" in result:
                    records = result.get("records", [])
                    if not records:
                        has_data = False
                    else:
                        # Check if all records have empty nodes and edges
                        all_empty = all(
                            not rec.get("nodes") and not rec.get("edges")
                            for rec in records
                            if isinstance(rec, dict)
                        )
                        if all_empty:
                            has_data = False
                # Old API format
                else:
                    results_value = result.get('results', '')
                    if isinstance(results_value, str) and results_value.strip().lower() == "no results":
                        has_data = False
                    elif isinstance(results_value, str) and not results_value.strip():
                        has_data = False
                    elif isinstance(results_value, str):
                        normalized = ' '.join(results_value.split())
                        if 'nodes, edges' in normalized.lower() and ('[], []' in normalized or '[][]' in normalized.replace(' ', '')):
                            has_data = False
            
            # Track query with data status AND store the actual Neo4j result
            add_cypher_query(cleaned_cypher, returned_data=has_data, neo4j_result=result)
            
            combined = {
                "cypher_query": cleaned_cypher,
                "api_result": result
            }
            q.put((True, json.dumps(combined, ensure_ascii=False)))
        except json.JSONDecodeError:
            add_cypher_query(cleaned_cypher, returned_data=False)  # Invalid JSON
            q.put((False, f"Invalid JSON response from API: {response.text}"))
    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"\n❌❌❌ EXCEPTION IN _pankbase_api_query_core for '{input[:40]}...':")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)[:500]}")
        print(f"Traceback: {err_msg[:1000]}")
        if (len(err_msg) > 2000):
            first = err_msg[:1000]
            second = err_msg[-1000:]
            err_msg = first + ' ...  ' + second
        q.put((False, err_msg))


def _pankbase_api_query(input: str, q: Queue) -> None:
    # TEMPORARILY DISABLED profiling - it uses sys.settrace() which can cause issues with threads
    # profile_to_file(
    #     _pankbase_api_query_core,
    #     args=(input, q),
    #     output_path="logs/pankbase_line_profile.txt",
    # )
    # Call the function directly without profiling
    _pankbase_api_query_core(input, q)

def test_a():
    a = 0
    b = 1
    c = b / a


def test_b():
    x = 4
    test_a()
    y = 5


def test_c():
    v = 8
    try:
        test_b()
    except:
        error_msg = traceback.format_exc()
    u = 10


instrument_module_functions(globals(), include_private=True, exclude={'_pankbase_api_query_core'})


if __name__ == "__main__":
    # Test Pankbase API functionality
    result = pankbase_api_query("Get detailed information for gene CFTR", 1)
    print(result)
