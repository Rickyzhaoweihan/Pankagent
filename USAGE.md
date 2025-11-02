# PlannerAgent Usage Guide

## Two Modes of Operation

### 1. Single Round Mode (Production Use)

Run the agent with a question as a command-line argument. The program will process the question, return the answer, and exit.

```bash
# Basic usage
python3 main.py "What is the function of gene TP53?"

# Multi-word questions
python3 main.py "Find all genes related to type 1 diabetes"

# Complex queries
python3 main.py "What are the SNPs associated with gene INS and their QTL relationships?"
```

**Output**: The program will print only the final formatted answer and then exit.

### 2. Interactive Mode (Development/Testing)

Run the agent without arguments to enter interactive mode for continuous conversation.

```bash
python3 main.py
```

This will start an interactive session:
```
Your question: What is gene CFTR?
Response:

[Answer here]

Your question: Tell me more about its mutations
Response:

[Answer here]

Your question: [Ctrl+C to exit]
```

## Examples

### Single Round Examples

```bash
# Gene information query
python3 main.py "Get detailed information for gene CFTR"

# Disease-related query
python3 main.py "What genes are associated with type 1 diabetes?"

# Relationship query
python3 main.py "Find SNPs that have QTL relationships with gene MAFA"

# Cell type expression query
python3 main.py "Which genes are upregulated in Alpha Cell?"
```

### Integration Example

```python
# Using in a Python script
import subprocess

question = "What is the function of gene INS?"
result = subprocess.run(
    ["python3", "main.py", question],
    capture_output=True,
    text=True
)

answer = result.stdout.strip()
print(f"Answer: {answer}")
```

## Output Behavior

- **Single Round Mode**: Only prints the final answer
- **Interactive Mode**: Prints each response after the user's question
- **Logs**: All execution details are written to `log.txt` for debugging
- **No Debug Output**: The program runs silently without intermediate print statements

## Exit Codes

- `0`: Successful execution
- Non-zero: Error occurred (check error message)

