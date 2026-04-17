#!/usr/bin/env python3
import json
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

from .text2cypher_utils import get_env_variable
from .schema_loader import get_schema, get_schema_hints, get_simplified_schema, get_minimal_schema_for_llm
from .cypher_validator import validate_cypher, format_validation_report


load_dotenv()

SYSTEM_RULES = (
    "Generate Cypher statement to query a biomedical graph database.\n"
    "Use only the provided relationship types and properties in the schema.\n"
    "\n"
    "CRITICAL RULES:\n"
    "1. Every relationship MUST have a variable name: WRONG: [:regulation] RIGHT: [r:regulation]\n"
    "2. Always return format: WITH collect(DISTINCT nodes...) AS nodes, collect(DISTINCT edges...) AS edges RETURN nodes, edges;\n"
    "3. Disease name: ALWAYS use 'type 1 diabetes' (lowercase, never T1D or Type 1 Diabetes)\n"
    "4. Name ALL variables in MATCH clause\n"
    "5. Use DISTINCT in all collect()\n"
    "6. ALWAYS use WHERE constraints to filter results (by name, id, or properties)\n"
    "   - If query mentions specific entities (gene name, SNP name, etc.), add WHERE clause\n"
    "   - Avoid unconstrained queries that return ALL nodes (e.g., MATCH (sn:snp) without WHERE)\n"
    "   - Use properties like .name, .id, or relationship properties to filter\n"
    "7. ALL matched nodes and relationships MUST be collected and returned\n"
    "   - If you MATCH multiple nodes (g1, g2, g3), collect ALL of them: collect(DISTINCT g1)+collect(DISTINCT g2)+collect(DISTINCT g3)\n"
    "   - If you MATCH multiple relationships (r1, r2), collect ALL of them: collect(DISTINCT r1)+collect(DISTINCT r2)\n"
    "   - WRONG: MATCH (g1)-[r1]->(g2)-[r2]->(g3) WITH collect(DISTINCT g1) AS nodes... (missing g2, g3, r1, r2!)\n"
    "   - RIGHT: MATCH (g1)-[r1]->(g2)-[r2]->(g3) WITH collect(DISTINCT g1)+collect(DISTINCT g2)+collect(DISTINCT g3) AS nodes, collect(DISTINCT r1)+collect(DISTINCT r2) AS edges\n"
    "\n"
    "GOOD EXAMPLES:\n"
    "Query: 'Find gene with name INS'\n"
    "MATCH (g:gene) WHERE g.name = 'INS'\n"
    "WITH collect(DISTINCT g) AS nodes, [] AS edges\n"
    "RETURN nodes, edges;\n"
    "\n"
    "Query: 'Get SNPs that have part_of_QTL_signal relationships with gene MAFA'\n"
    "MATCH (sn:snp)-[r:part_of_QTL_signal]->(g:gene) WHERE g.name = 'MAFA'\n"
    "WITH collect(DISTINCT sn)+collect(DISTINCT g) AS nodes, collect(DISTINCT r) AS edges\n"
    "RETURN nodes, edges;\n"
    "\n"
    "Query: 'Get upregulated genes in Alpha Cell'\n"
    "MATCH (g:gene)-[deg:DEG_in]->(ct:cell_type) WHERE ct.name = 'Alpha Cell' AND deg.UpOrDownRegulation = 'up'\n"
    "WITH collect(DISTINCT g)+collect(DISTINCT ct) AS nodes, collect(DISTINCT deg) AS edges\n"
    "RETURN nodes, edges;\n"
    "\n"
    "BAD EXAMPLES:\n"
    "WRONG: MATCH (sn:snp)-[r:part_of_QTL_signal]->(g:gene) (no WHERE - returns ALL SNPs!)\n"
    "WRONG: MATCH (g:gene)-[:function_annotation]->(fo:gene_ontology) (missing variable name)\n"
    "WRONG: MATCH (g:gene) (no WHERE - returns ALL genes!)\n"
    "WRONG: MATCH (g:gene)-[r1:regulation]->(g2:gene)-[r2:DEG_in]->(ct:cell_type) WITH collect(DISTINCT g) AS nodes... (missing g2, ct, r1, r2!)\n"
    "\n"
    "Schema:\n{schema}\n"
)


def make_llm(provider: str = "local"):
    """Return a Chat instance for the specified provider."""
    provider = "local"
    if provider == "openai":
        return ChatOpenAI(
            base_url=get_env_variable("OPENAI_API_BASE_URL"),  # e.g. "https://api.openai.com/v1"
            api_key=get_env_variable("OPENAI_API_KEY"),
            model=get_env_variable("OPENAI_API_MODEL"),
            temperature=float(get_env_variable("OPENAI_API_TEMP", 0))
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=get_env_variable("GOOGLE_MODEL"),
            google_api_key=get_env_variable("GOOGLE_API_KEY"),
            temperature=0
        )

    elif provider == "local":
        # self-hosted vLLM instance (qwen2.5-coder-14b)
        return ChatOpenAI(
            base_url="http://localhost:8001/v1",
            api_key="EMPTY",  # vLLM ignores this field but LangChain requires it
            model="cypher-writer",
            temperature=0
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")
    

class Text2CypherAgent:
    """Single‑LLM agent that remembers conversation context + schema."""

    def __init__(self, provider: str = "local",
                 enable_refinement: bool = True,
                 max_refinement_iterations: int = 5,
                 min_acceptable_score: int = 90):
        self.provider = provider
        self.enable_refinement = enable_refinement
        self.max_refinement_iterations = max_refinement_iterations
        self.min_acceptable_score = min_acceptable_score
        
        # Use ultra-minimal schema optimized for small models (9B with 8k context)
        self.minimal_schema = get_minimal_schema_for_llm()
        
        # Build prompt using the model's training format: Schema + Question + Cypher output
        system_prompt = SYSTEM_RULES.format(schema=self.minimal_schema)

        self.llm = make_llm(provider)
        
        # Use the exact format the model was trained on
        self.prompt = ChatPromptTemplate.from_messages([
            ("human", system_prompt + "\nQuestion: {user_input}\n\n")
        ])

        self.chain = self.prompt | self.llm

    # def respond(self, user_text: str) -> str:
    #     result = self.chain.invoke(
    #         {"user_input": user_text},
    #         config={"configurable": {"session_id": self.session_id}}
    #     )
    #     return result.content.strip().strip("` ")
    def respond(self, user_text: str) -> str:
        result = self.chain.invoke({"user_input": user_text})
        cypher = result.content.strip().strip("` ")
        # Remove common prefixes that models add
        if cypher.lower().startswith("cypher"):
            cypher = cypher[6:].strip()
        return cypher

    def respond_with_refinement(self, user_text: str, max_iterations: int = None) -> dict:
        """
        Generate Cypher with iterative refinement using test-time scaling.
        
        Args:
            user_text: The user's natural language query
            max_iterations: Maximum refinement iterations (uses self.max_refinement_iterations if None)
            
        Returns:
            {
                'cypher': str,              # Best Cypher query
                'score': int,               # Validation score (0-100)
                'iteration': int,           # Which iteration produced the best result
                'all_attempts': list,       # All attempts with scores and validation
                'validation_report': dict   # Detailed validation of best query
            }
        """
        if max_iterations is None:
            max_iterations = self.max_refinement_iterations
        
        all_attempts = []
        best_result = None
        best_score = -1
        
        # Iteration 1: Initial generation
        cypher = self.respond(user_text)
        validation = validate_cypher(cypher)
        
        attempt = {
            'iteration': 1,
            'cypher': cypher,
            'score': validation['score'],
            'validation': validation
        }
        all_attempts.append(attempt)
        
        if validation['score'] > best_score:
            best_score = validation['score']
            best_result = attempt
        
        # Early stopping if score is excellent
        if validation['score'] >= 90:
            return {
                'cypher': cypher,
                'score': validation['score'],
                'iteration': 1,
                'all_attempts': all_attempts,
                'validation_report': validation
            }
        
        # Iterations 2-N: Refinement loop
        for iteration in range(2, max_iterations + 1):
            # Build refinement prompt with previous attempt and errors
            prev_validation = all_attempts[-1]['validation']
            
            if prev_validation['score'] >= self.min_acceptable_score and not prev_validation['errors']:
                # Good enough, stop early
                break
            
            refinement_prompt = self._build_refinement_prompt(
                user_text,
                all_attempts[-1]['cypher'],
                prev_validation
            )
            
            # Generate refined Cypher
            cypher = self._generate_with_refinement_prompt(refinement_prompt)
            validation = validate_cypher(cypher)
            
            attempt = {
                'iteration': iteration,
                'cypher': cypher,
                'score': validation['score'],
                'validation': validation
            }
            all_attempts.append(attempt)
            
            # Track best result
            if validation['score'] > best_score:
                best_score = validation['score']
                best_result = attempt
            
            # Early stopping if score is excellent
            if validation['score'] >= 90:
                break
        
        return {
            'cypher': best_result['cypher'],
            'score': best_result['score'],
            'iteration': best_result['iteration'],
            'all_attempts': all_attempts,
            'validation_report': best_result['validation']
        }
    
    def _build_refinement_prompt(self, original_question: str, previous_cypher: str, 
                                  validation: dict) -> str:
        """Build a focused refinement prompt with only essential error feedback."""
        
        # Build concise error list
        errors_text = "\n".join(f"  - {error}" for error in validation['errors'])
        if not errors_text:
            errors_text = "  - Low score but no specific errors detected"
        
        # Build compact prompt focusing on what's wrong
        prompt = (
            f"Previous attempt (Score: {validation['score']}/100):\n"
            f"{previous_cypher}\n\n"
            f"Fix these errors:\n{errors_text}\n\n"
            f"Question: {original_question}\n\n"
            f"Generate corrected Cypher query."
        )
        
        with open('log.txt', 'a') as log_file:
            log_file.write(f"Refinement prompt:\n{prompt}\n")
        return prompt
    
    def _generate_with_refinement_prompt(self, refinement_prompt: str) -> str:
        """Generate Cypher using refinement prompt."""
        # Build a simplified prompt for refinement
        schema_section = f"Schema:\n{self.minimal_schema}\n\n"
        full_prompt = schema_section + refinement_prompt + "\n\n"
        
        result = self.llm.invoke(full_prompt)
        cypher = result.content.strip().strip("` ")
        # Remove common prefixes that models add
        if cypher.lower().startswith("cypher"):
            cypher = cypher[6:].strip()
        return cypher

    def get_history(self) -> list[dict[str, str]]:
        """Return chat history as list of {role, content} dicts."""
        return []

    def clear_history(self) -> None:
        """Clear the shared history."""
        return None

if __name__ == "__main__":
    try:
        agent = Text2CypherAgent()
    except EnvironmentError as e:
        sys.exit(1)

    try:
        while True:
            txt = input("You> ").strip()
            if not txt:
                continue
            print(agent.respond(txt) + "\n")
    except (KeyboardInterrupt, EOFError):
        pass