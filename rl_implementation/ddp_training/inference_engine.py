"""
Inference Engine using external vLLM OpenAI-compatible servers.

Architecture:
- Orchestrator: Single vLLM server handles all 4 modes (same model, different prompts)
- Cypher Generator: Separate vLLM server
- Servers are started externally via command line with CUDA_VISIBLE_DEVICES
- This engine connects to the servers via HTTP using OpenAI-compatible API

Server Setup (run before training):
  # GPU 0: Orchestrator (handles Question Gen, Data Eval, Synthesis, Answer Eval)
  CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model <orch_model> --port 8001 &
  # GPU 1: Cypher Generator
  CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model <cypher_model> --port 8002 &
"""

import logging
import time
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference engine using external vLLM servers."""
    # Model paths (used for model name in API calls)
    cypher_model_path: str = ""
    orchestrator_model_path: str = ""
    
    # Server ports
    # All orchestrator modes use the same server (same model, different prompts)
    orchestrator_port_question: int = 8001
    orchestrator_port_data_eval: int = 8001  # Same as question
    orchestrator_port_synthesis: int = 8001  # Same as question
    orchestrator_port_answer_eval: int = 8001  # Same as question
    cypher_inference_port: int = 8002
    
    # Server host (default localhost)
    server_host: str = "localhost"
    
    # API settings
    api_timeout: float = 180.0  # seconds (increased for batch requests)
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    
    # Batch settings
    batch_size: int = 64  # Max prompts per batch API request
    
    # Generation settings
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Legacy GPU settings (kept for backward compatibility, not used)
    orchestrator_gpu_question: int = 0
    orchestrator_gpu_data_eval: int = 1
    orchestrator_gpu_synthesis: int = 2
    orchestrator_gpu_answer_eval: int = 3
    cypher_inference_gpu: int = 4
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90  # Increased for batch processing
    max_num_seqs: int = 128  # Increased to handle larger batches


class VLLMServerClient:
    """Client for a vLLM OpenAI-compatible server with batch support."""
    
    def __init__(
        self,
        host: str,
        port: int,
        model_name: str,
        name: str,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        batch_size: int = 32,  # Max prompts per batch request
    ):
        self.host = host
        self.port = port
        self.model_name = model_name  # Original model path (may not match vLLM's registered ID)
        self.registered_model_id = None  # Will be auto-detected from /v1/models
        self.name = name
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size
        
        self.base_url = f"http://{host}:{port}"
        self.completions_url = f"{self.base_url}/v1/completions"
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.health_url = f"{self.base_url}/health"
        self.models_url = f"{self.base_url}/v1/models"
    
    def wait_for_server(self, timeout: float = 300.0) -> bool:
        """Wait for the server to be ready and auto-detect model ID."""
        start_time = time.time()
        logger.info(f"  [{self.name}] Waiting for server at {self.base_url}...")
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(self.health_url, timeout=5)
                if response.status_code == 200:
                    # Server is healthy, now get the actual model ID
                    if self._detect_model_id():
                        logger.info(f"  [{self.name}] ✓ Server ready at port {self.port}")
                        logger.info(f"  [{self.name}]   Model ID: {self.registered_model_id}")
                        return True
                    else:
                        logger.warning(f"  [{self.name}] Server healthy but could not detect model ID, using original: {self.model_name}")
                        self.registered_model_id = self.model_name
                        return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2.0)
        
        logger.error(f"  [{self.name}] ✗ Server not available after {timeout}s")
        return False
    
    def _detect_model_id(self) -> bool:
        """Query /v1/models to get the actual registered model ID."""
        try:
            response = requests.get(self.models_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("data") and len(data["data"]) > 0:
                    self.registered_model_id = data["data"][0]["id"]
                    logger.info(f"  [{self.name}] Auto-detected model ID: {self.registered_model_id}")
                    return True
        except Exception as e:
            logger.warning(f"  [{self.name}] Failed to detect model ID: {e}")
        return False
    
    def get_effective_model_id(self) -> str:
        """Get the model ID to use in API requests."""
        return self.registered_model_id if self.registered_model_id else self.model_name
    
    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: List[str] = None,
        max_prompt_chars: int = 24000,  # ~6000 tokens, safe limit for 8192 context
    ) -> List[str]:
        """
        Generate responses for the given prompts using TRUE BATCH requests.
        
        vLLM supports batch requests where multiple prompts are sent in a single
        API call. This is much more efficient than sequential calls.
        
        Args:
            prompts: List of prompts to generate responses for
            max_tokens: Maximum tokens to generate per response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            max_prompt_chars: Maximum characters per prompt (truncate if exceeded)
            
        Returns:
            List of generated responses (same order as prompts)
        """
        if stop is None:
            stop = ["<|im_end|>", "<|endoftext|>"]
        
        if not prompts:
            return []
        
        # Validate and truncate prompts
        processed_prompts = []
        for i, prompt in enumerate(prompts):
            if len(prompt) > max_prompt_chars:
                logger.warning(
                    f"  [{self.name}] Prompt {i} too long ({len(prompt)} chars), "
                    f"truncating to {max_prompt_chars} chars"
                )
                prompt = prompt[:max_prompt_chars] + "\n...(truncated)"
            processed_prompts.append(prompt)
        
        # Process in batches for very large requests
        all_responses = []
        for batch_start in range(0, len(processed_prompts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(processed_prompts))
            batch_prompts = processed_prompts[batch_start:batch_end]
            
            batch_responses = self._generate_batch(
                batch_prompts, max_tokens, temperature, top_p, stop
            )
            all_responses.extend(batch_responses)
            
            if batch_end < len(processed_prompts):
                logger.info(f"  [{self.name}] Processed {batch_end}/{len(processed_prompts)} prompts...")
        
        return all_responses
    
    def _generate_batch(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: List[str],
    ) -> List[str]:
        """
        Generate responses for a batch of prompts in a single API call.
        
        vLLM's /v1/completions endpoint accepts a list of prompts and returns
        all responses together, processing them in parallel internally.
        """
        last_error = None
        model_id = self.get_effective_model_id()
        
        # Calculate timeout based on batch size (more prompts = longer timeout)
        batch_timeout = self.timeout * max(1, len(prompts) / 8)
        
        for attempt in range(self.max_retries):
            try:
                # vLLM accepts a list of prompts in a single request
                payload = {
                    "model": model_id,
                    "prompt": prompts,  # List of prompts for batch processing
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop": stop,
                }
                
                logger.debug(f"  [{self.name}] Sending batch request with {len(prompts)} prompts...")
                
                response = requests.post(
                    self.completions_url,
                    json=payload,
                    timeout=batch_timeout,
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # vLLM returns one choice per prompt in batch mode
                    # Choices are indexed to match the input prompts
                    choices = result.get("choices", [])
                    
                    # Sort by index to ensure correct order
                    sorted_choices = sorted(choices, key=lambda c: c.get("index", 0))
                    
                    responses = [c.get("text", "") for c in sorted_choices]
                    
                    # Verify we got the right number of responses
                    if len(responses) != len(prompts):
                        logger.warning(
                            f"  [{self.name}] Expected {len(prompts)} responses, "
                            f"got {len(responses)}. Padding with empty strings."
                        )
                        while len(responses) < len(prompts):
                            responses.append("")
                    
                    logger.debug(f"  [{self.name}] Batch complete: {len(responses)} responses")
                    return responses
                else:
                    error_detail = self._parse_error_response(response)
                    last_error = f"HTTP {response.status_code}: {error_detail}"
                    logger.warning(f"  [{self.name}] Batch request failed (attempt {attempt + 1}): {last_error}")
                    
                    # If model ID mismatch, try to re-detect
                    if response.status_code == 400 and "model" in error_detail.lower():
                        logger.info(f"  [{self.name}] Possible model ID mismatch, re-detecting...")
                        if self._detect_model_id():
                            model_id = self.get_effective_model_id()
                            logger.info(f"  [{self.name}] Updated model ID: {model_id}")
                    
            except requests.exceptions.Timeout:
                last_error = f"Request timeout (batch of {len(prompts)} prompts, timeout={batch_timeout:.0f}s)"
                logger.warning(f"  [{self.name}] {last_error} (attempt {attempt + 1})")
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.warning(f"  [{self.name}] Request error (attempt {attempt + 1}): {e}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"  [{self.name}] Unexpected error (attempt {attempt + 1}): {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        logger.error(f"  [{self.name}] All retries failed for batch of {len(prompts)}: {last_error}")
        # Return empty strings for all prompts on failure
        return [""] * len(prompts)
    
    def generate_single(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: List[str] = None,
    ) -> str:
        """Generate a single response (convenience wrapper around batch)."""
        responses = self.generate([prompt], max_tokens, temperature, top_p, stop)
        return responses[0] if responses else ""
    
    def _parse_error_response(self, response: requests.Response) -> str:
        """Parse error details from vLLM response."""
        try:
            # Try to parse JSON error
            error_data = response.json()
            if isinstance(error_data, dict):
                # vLLM error format
                if "error" in error_data:
                    error_obj = error_data["error"]
                    if isinstance(error_obj, dict):
                        return error_obj.get("message", str(error_obj))
                    return str(error_obj)
                # OpenAI-style error
                if "detail" in error_data:
                    return str(error_data["detail"])
                return str(error_data)
        except:
            pass
        # Fall back to raw text
        return response.text[:500] if response.text else "No error details"
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: List[str] = None,
    ) -> List[str]:
        """
        Generate responses for batch of prompts (alias for generate).
        
        Now uses true batch processing internally.
        """
        return self.generate(prompts, max_tokens, temperature, top_p, stop)


class InferenceEngine:
    """
    Inference engine connecting to external vLLM OpenAI-compatible servers.
    
    Each server runs on a separate GPU (started externally).
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        
        # Server clients (not connected yet)
        self.orch_question: Optional[VLLMServerClient] = None
        self.orch_data_eval: Optional[VLLMServerClient] = None
        self.orch_synthesis: Optional[VLLMServerClient] = None
        self.orch_answer_eval: Optional[VLLMServerClient] = None
        self.cypher_instance: Optional[VLLMServerClient] = None
        
        # Tokenizers (loaded in main process)
        self.orchestrator_tokenizer = None
        self.cypher_tokenizer = None
        
        logger.info("InferenceEngine initialized (external vLLM server mode)")
        logger.info(f"  Orchestrator model: {config.orchestrator_model_path}")
        logger.info(f"  Cypher Generator model: {config.cypher_model_path}")
        logger.info(f"  Server host: {config.server_host}")
    
    def initialize(self):
        """Initialize connections to all vLLM servers."""
        from transformers import AutoTokenizer
        
        logger.info("=" * 60)
        logger.info("INFERENCE ENGINE INITIALIZATION")
        logger.info("Connecting to external vLLM OpenAI servers")
        logger.info("=" * 60)
        logger.info(f"Orchestrator model: {self.config.orchestrator_model_path}")
        logger.info(f"Cypher model: {self.config.cypher_model_path}")
        logger.info(f"Server ports:")
        logger.info(f"  - Orch Question: {self.config.orchestrator_port_question}")
        logger.info(f"  - Orch DataEval: {self.config.orchestrator_port_data_eval}")
        logger.info(f"  - Orch Synthesis: {self.config.orchestrator_port_synthesis}")
        logger.info(f"  - Orch AnswerEval: {self.config.orchestrator_port_answer_eval}")
        logger.info(f"  - Cypher Inference: {self.config.cypher_inference_port}")
        logger.info("=" * 60)
        
        # Load tokenizers first (CPU only)
        logger.info("")
        logger.info("Step 1: Loading tokenizers...")
        self.orchestrator_tokenizer = AutoTokenizer.from_pretrained(
            self.config.orchestrator_model_path,
            trust_remote_code=True,
        )
        self.cypher_tokenizer = AutoTokenizer.from_pretrained(
            self.config.cypher_model_path,
            trust_remote_code=True,
        )
        logger.info("  ✓ Tokenizers loaded")
        
        # Create server clients
        clients_config = [
            ("orch_question", self.config.orchestrator_port_question,
             self.config.orchestrator_model_path, "Orch-Question"),
            ("orch_data_eval", self.config.orchestrator_port_data_eval,
             self.config.orchestrator_model_path, "Orch-DataEval"),
            ("orch_synthesis", self.config.orchestrator_port_synthesis,
             self.config.orchestrator_model_path, "Orch-Synthesis"),
            ("orch_answer_eval", self.config.orchestrator_port_answer_eval,
             self.config.orchestrator_model_path, "Orch-AnswerEval"),
            ("cypher_instance", self.config.cypher_inference_port,
             self.config.cypher_model_path, "CypherGen"),
        ]
        
        # Create and verify connections to each server
        logger.info("")
        logger.info("Step 2: Connecting to vLLM servers...")
        
        for attr_name, port, model_path, display_name in clients_config:
            logger.info(f"  Connecting to {display_name} at port {port}...")
            
            client = VLLMServerClient(
                host=self.config.server_host,
                port=port,
                model_name=model_path,
                name=display_name,
                timeout=self.config.api_timeout,
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay,
                batch_size=self.config.batch_size,
            )
            
            # Wait for server to be ready
            if not client.wait_for_server(timeout=300.0):
                raise RuntimeError(
                    f"Failed to connect to {display_name} server at port {port}. "
                    f"Make sure the server is started with: "
                    f"CUDA_VISIBLE_DEVICES=<gpu> python -m vllm.entrypoints.openai.api_server "
                    f"--model {model_path} --port {port}"
                )
            
            setattr(self, attr_name, client)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("All vLLM server connections established!")
        logger.info("=" * 60)
        
        # Run a quick test to verify all servers work
        logger.info("")
        logger.info("Step 3: Running quick validation tests...")
        self._validate_servers()
    
    def _validate_servers(self):
        """Run quick tests to validate all servers are working."""
        test_prompt = "What is 2+2? Answer with just the number."
        
        # Only test unique servers (orchestrator modes share the same server)
        # Check if orchestrator clients share the same port
        orch_port = self.config.orchestrator_port_question
        clients = []
        
        # Add orchestrator (only once since all modes use same server)
        clients.append((self.orch_question, "Orchestrator"))
        
        # Add cypher generator
        clients.append((self.cypher_instance, "CypherGen"))
        
        all_ok = True
        for client, name in clients:
            try:
                responses = client.generate([test_prompt], max_tokens=10, temperature=0.1)
                if responses and responses[0]:
                    logger.info(f"  ✓ {name}: OK (response: '{responses[0][:30]}...')")
                else:
                    logger.warning(f"  ⚠ {name}: Empty response")
                    all_ok = False
            except Exception as e:
                logger.error(f"  ✗ {name}: Failed - {e}")
                all_ok = False
        
        if all_ok:
            logger.info("  All servers validated successfully!")
        else:
            logger.warning("  Some servers may have issues. Check logs for details.")
    
    # =========================================================================
    # Orchestrator Methods (4 modes)
    # =========================================================================
    
    def generate_questions(
        self,
        prompts: List[str],
        temperature: float = 0.8,
    ) -> List[str]:
        """Generate questions using Orchestrator (Question Generator mode)."""
        logger.debug(f"Generating {len(prompts)} questions...")
        if prompts:
            logger.debug(f"First prompt preview ({len(prompts[0])} chars): {prompts[0][:200]}...")
        
        responses = self.orch_question.generate(
            prompts,
            max_tokens=self.config.max_new_tokens,
            temperature=temperature,
            top_p=0.95,
        )
        
        # Log any empty responses
        empty_count = sum(1 for r in responses if not r)
        if empty_count > 0:
            logger.warning(f"Got {empty_count}/{len(responses)} empty responses from question generation")
        
        return [self._parse_question(r) for r in responses]
    
    def evaluate_data_quality(
        self,
        prompts: List[str],
    ) -> List[Dict[str, Any]]:
        """Evaluate data quality using Orchestrator (Data Evaluator mode)."""
        responses = self.orch_data_eval.generate(
            prompts,
            max_tokens=self.config.max_new_tokens,
            temperature=0.3,
            top_p=0.9,
        )
        return [self._parse_data_quality_eval(r) for r in responses]
    
    def synthesize_answers(
        self,
        prompts: List[str],
    ) -> List[str]:
        """Synthesize answers using Orchestrator (Synthesis mode)."""
        responses = self.orch_synthesis.generate(
            prompts,
            max_tokens=self.config.max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )
        return [self._parse_answer(r) for r in responses]
    
    def evaluate_answer_quality(
        self,
        prompts: List[str],
    ) -> List[Dict[str, Any]]:
        """Evaluate answer quality using Orchestrator (Answer Evaluator mode)."""
        responses = self.orch_answer_eval.generate(
            prompts,
            max_tokens=self.config.max_new_tokens,
            temperature=0.3,
            top_p=0.9,
        )
        return [self._parse_answer_quality_eval(r) for r in responses]
    
    # =========================================================================
    # Cypher Generator Methods
    # =========================================================================
    
    def generate_cypher(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """Generate Cypher queries using Cypher Generator."""
        return self.cypher_instance.generate(
            prompts,
            max_tokens=max_new_tokens or self.config.max_new_tokens,
            temperature=temperature or self.config.temperature,
        )
    
    # =========================================================================
    # Response Parsing Helpers
    # =========================================================================
    
    def _parse_question(self, response: str) -> str:
        """
        Extract EXACTLY ONE biomedical question from Orchestrator's response.
        
        The Orchestrator often outputs:
        - Conversational text: "Sure! Here's a question: **What genes...?**"
        - Multiple questions: "Which genes...? Which diseases...?"
        - Formatted questions: "Question:** Which diseases..."
        
        We need to extract just ONE actual biomedical question.
        """
        import re
        
        if not response:
            return ""
        
        response = response.strip()
        
        # First, clean common prefixes that appear in the logs
        prefixes_to_remove = [
            r'^#+\s*Question:\s*',       # ### Question:
            r'^Question:\*\*\s*',         # Question:**
            r'^Question:\s*',             # Question:
            r'^```\s*',                   # ``` at start
            r'^\*\*Question:\*\*\s*',     # **Question:**
            r'^,\s*',                     # Leading comma
        ]
        for prefix in prefixes_to_remove:
            response = re.sub(prefix, '', response, flags=re.IGNORECASE | re.MULTILINE)
        
        # Patterns that indicate conversational/meta responses (NOT actual questions)
        conversational_patterns = [
            r'would you like',
            r'do you want',
            r'shall i',
            r'let me know',
            r'here\'s a question',
            r'here is a question',
            r'i can generate',
            r'i\'ll generate',
            r'adjust anything',
            r'another question',
            r'proceed with',
            r'confirm this',
            r'further refinements',
            r'any other',
            r'🚀',
            r'🤔',
        ]
        
        def is_conversational(text: str) -> bool:
            text_lower = text.lower()
            for pattern in conversational_patterns:
                if re.search(pattern, text_lower):
                    return True
            return False
        
        def extract_first_question(text: str) -> str:
            """Extract only the FIRST question (ending with ?) from text."""
            # Split on ? and take the first complete question
            # Be careful: some questions contain ? in quoted text
            parts = re.split(r'\?(?=\s|$)', text)
            if parts and parts[0].strip():
                first_q = parts[0].strip() + '?'
                return first_q
            return text
        
        biomedical_keywords = [
            'gene', 'protein', 'disease', 'cell', 'snp', 'variant', 'expression',
            'diabetes', 'chromosome', 'ontology', 'pathway', 'interaction',
            'regulate', 'associate', 'located', 'express', 'signal', 'gwas',
            'alpha', 'beta', 'pancrea', 'insulin', 'type 1', 'type 2'
        ]
        
        def is_biomedical_question(text: str) -> bool:
            text_lower = text.lower()
            return any(kw in text_lower for kw in biomedical_keywords)
        
        # Method 1: Look for quoted questions (between ** ** or " ")
        quoted_patterns = [
            r'\*\*["\']?([^*]+\?)["\']?\*\*',  # **"Question?"** or **Question?**
            r'["\']([^"\']+\?)["\']',           # "Question?" or 'Question?'
            r'\*\*([^*]+\?)\*\*',               # **Question?**
        ]
        
        for pattern in quoted_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                cleaned = match.strip().strip('"\'*').strip()
                # Extract only FIRST question if multiple
                first_q = extract_first_question(cleaned)
                if first_q.endswith('?') and len(first_q) > 20 and not is_conversational(first_q):
                    if is_biomedical_question(first_q):
                        return first_q
        
        # Method 2: Look for lines that look like biomedical questions
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip().strip('*"\'').strip()
            if not '?' in line:
                continue
            if len(line) < 20:
                continue
            if is_conversational(line):
                continue
            
            # Extract only FIRST question from the line
            first_q = extract_first_question(line)
            if first_q.endswith('?') and is_biomedical_question(first_q):
                return first_q
        
        # Method 3: Find first sentence ending with ? that's not conversational
        # Use more robust sentence splitting
        sentences = re.split(r'(?<=[?])\s+', response)
        for sentence in sentences:
            sentence = sentence.strip().strip('*"\'').strip()
            if not sentence.endswith('?'):
                # Maybe it ends with ? followed by other chars
                q_match = re.match(r'^(.+?\?)', sentence)
                if q_match:
                    sentence = q_match.group(1)
                else:
                    continue
            if len(sentence) > 20 and not is_conversational(sentence):
                if is_biomedical_question(sentence):
                    return sentence
        
        # Method 4: Extract first question using regex
        # Look for text starting with question words
        q_match = re.search(
            r'((?:Which|What|How|Where|When|Who|Why|Are|Is|Do|Does|Can|Could)[^?]*\?)',
            response,
            re.IGNORECASE
        )
        if q_match:
            question = q_match.group(1).strip()
            if len(question) > 20 and not is_conversational(question):
                return question
        
        # Method 5: Fallback - find any question-like text
        for line in lines:
            line = line.strip().strip('*"\'').strip()
            if '?' in line and len(line) > 15:
                # Extract just the first question part
                q_match = re.search(r'([A-Z][^?]+\?)', line)
                if q_match:
                    q = q_match.group(1).strip()
                    if not is_conversational(q):
                        return q
        
        # Last resort: return first non-empty line (cleaned), truncated
        for line in lines:
            line = line.strip().strip('*"\'').strip()
            if line and len(line) > 10:
                # If it contains ?, extract just that
                if '?' in line:
                    first_q = extract_first_question(line)
                    return first_q
                return line[:200] if len(line) > 200 else line
        
        return response[:200] if len(response) > 200 else response
    
    def _parse_data_quality_eval(self, response: str) -> Dict[str, Any]:
        """Parse data quality evaluation, handling various JSON formats."""
        import json
        import re
        
        # Try multiple code block formats
        # Format 1: ```json ... ```
        json_blocks = re.findall(r'```json\s*(.*?)```', response, re.DOTALL | re.IGNORECASE)
        if not json_blocks:
            # Format 2: ``` ... ``` (generic code block with JSON inside)
            json_blocks = re.findall(r'```\s*(\{.*?\})\s*```', response, re.DOTALL)
        
        for block in json_blocks:
            try:
                data = json.loads(block.strip())
                return self._validate_data_quality(data)
            except json.JSONDecodeError:
                continue
        
        # Try to find JSON objects directly in the response
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_objects = re.findall(json_pattern, response, re.DOTALL)
        
        for json_str in reversed(json_objects):  # Try from last to first
            try:
                data = json.loads(json_str)
                if 'data_quality_score' in data or 'relevance_score' in data or 'score' in data:
                    return self._validate_data_quality(data)
            except json.JSONDecodeError:
                continue
        
        # If parsing failed, return LOW scores (not medium 0.5!)
        logger.warning(f"Failed to parse data quality JSON from response: {response[:200]}...")
        return {
            'data_quality_score': 0.2,
            'relevance_score': 0.2,
            'completeness_score': 0.2,
            'trajectory_quality_score': 0.2,
            'doubt_level': 0.8,  # High doubt when parsing fails
            'could_answer_question': False,
        }
    
    def _validate_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fill missing fields in data quality evaluation."""
        return {
            'data_quality_score': data.get('data_quality_score', 0.5),
            'relevance_score': data.get('relevance_score', 0.5),
            'completeness_score': data.get('completeness_score', 0.5),
            'consistency_score': data.get('consistency_score', 0.5),
            'trajectory_quality_score': data.get('trajectory_quality_score', 0.5),
            'reasoning': data.get('reasoning', ''),
            'semantic_issues': data.get('semantic_issues', []),
            'problematic_regions': data.get('problematic_regions', []),
            'could_answer_question': data.get('could_answer_question', True),
            'doubt_level': data.get('doubt_level', 0.0),
        }
    
    def _parse_answer(self, response: str) -> str:
        lines = response.strip().split('\n')
        answer_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.lower().startswith(('question:', 'data:', 'task:')):
                answer_lines.append(line)
        if answer_lines:
            return ' '.join(answer_lines)
        return response.strip()
    
    def _parse_answer_quality_eval(self, response: str) -> Dict[str, Any]:
        """Parse answer quality evaluation, handling various JSON formats."""
        import json
        import re
        
        # Try multiple code block formats
        json_blocks = re.findall(r'```json\s*(.*?)```', response, re.DOTALL | re.IGNORECASE)
        if not json_blocks:
            json_blocks = re.findall(r'```\s*(\{.*?\})\s*```', response, re.DOTALL)
        
        for block in json_blocks:
            try:
                data = json.loads(block.strip())
                return self._validate_answer_quality(data)
            except json.JSONDecodeError:
                continue
        
        # Try to find JSON objects directly
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_objects = re.findall(json_pattern, response, re.DOTALL)
        
        for json_str in reversed(json_objects):
            try:
                data = json.loads(json_str)
                if 'score' in data or 'correctness' in data:
                    return self._validate_answer_quality(data)
            except json.JSONDecodeError:
                continue
        
        # Return low scores on parsing failure
        logger.warning(f"Failed to parse answer quality JSON from response: {response[:200]}...")
        return {
            'score': 0.3,
            'correctness': 0.3,
            'completeness': 0.3,
            'clarity': 0.3,
            'accuracy': 0.3,
        }
    
    def _validate_answer_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fill missing fields in answer quality evaluation."""
        return {
            'score': data.get('score', 0.5),
            'correctness': data.get('correctness', 0.5),
            'completeness': data.get('completeness', 0.5),
            'clarity': data.get('clarity', 0.5),
            'accuracy': data.get('accuracy', 0.5),
            'reasoning': data.get('reasoning', ''),
            'strengths': data.get('strengths', ''),
            'weaknesses': data.get('weaknesses', ''),
        }
    
    def shutdown(self):
        """Cleanup resources."""
        logger.info("Shutting down InferenceEngine...")
        # With external servers, we just clear the client references
        # The servers continue running independently
        self.orch_question = None
        self.orch_data_eval = None
        self.orch_synthesis = None
        self.orch_answer_eval = None
        self.cypher_instance = None
        logger.info("InferenceEngine shutdown complete")
        logger.info("Note: External vLLM servers are still running. Stop them manually if needed.")
