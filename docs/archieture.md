# Debug Agent Architecture

## Overview

The Debug Agent is an AI-powered system designed to analyze software bugs, retrieve relevant context from codebases, and generate Root Cause Analysis (RCA) using Large Language Models (LLMs). The system employs semantic search via embeddings, security controls, and structured workflow orchestration to provide accurate bug diagnostics.

### Key Features

- **Semantic Code Search**: Uses vector embeddings (FAISS) to find relevant code sections
- **Security-First Design**: Sanitizes secrets and detects prompt injection attempts
- **Contextual Analysis**: Combines bug details, logs, and codebase information
- **LLM-Powered RCA**: Generates comprehensive root cause analysis using AI models
- **Modular Architecture**: Clear separation of concerns across embedding, security, and workflow modules

---

## System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          Debug Agent                             │
│                                                                   │
│  ┌────────────┐     ┌──────────────┐     ┌─────────────┐       │
│  │   Data     │────▶│  Embeddings  │────▶│  Security   │       │
│  │ Collection │     │   & Vector   │     │   & Utils   │       │
│  │            │     │   Database   │     │             │       │
│  └────────────┘     └──────────────┘     └─────────────┘       │
│                            │                      │              │
│                            │                      │              │
│                            ▼                      ▼              │
│                     ┌─────────────────────────────────┐         │
│                     │    Workflow Orchestration       │         │
│                     │    & LLM Interaction            │         │
│                     └─────────────────────────────────┘         │
│                                  │                               │
│                                  ▼                               │
│                          ┌──────────────┐                       │
│                          │   RCA Output │                       │
│                          └──────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
debug_agent/
├── data/                          # Data storage
│   ├── codebase/                  # Source code files to analyze
│   └── bugs/                      # Bug information storage
│       └── {bug_id}/              # Individual bug directories
│           ├── detail.md          # Bug description and details
│           ├── logs.txt           # Application/system logs
│           └── rca.md             # Generated root cause analysis
├── src/                           # Source code
│   ├── main.py                    # Application entry point
│   ├── embeddings/                # Embedding creation and retrieval
│   │   └── __init__.py
│   ├── utils/                     # Security and utility functions
│   │   └── __init__.py
│   └── workflow/                  # LLM workflow orchestration
│       └── __init__.py
└── docs/
    └── architecture.md            # This document
```

---

## Detailed Component Architecture

### 1. Data Collection Layer

**Responsibility**: Organize and structure bug-related data for processing

#### Data Storage Structure

| Directory | Purpose | File Format |
|-----------|---------|-------------|
| `/data/codebase/` | Source code files | Any text-based format (.py, .js, .java, etc.) |
| `/data/bugs/{bug_id}/` | Bug-specific information | Markdown, text files |

#### Bug Data Format

Each bug must have a unique identifier (`bug_id`) with the following files:

- **`detail.md`**: Bug description, reproduction steps, expected vs. actual behavior
- **`logs.txt`**: Application logs, stack traces, error messages
- **`rca.md`**: Generated root cause analysis (output)

**Example Bug Structure:**

```
/data/bugs/bug001/
  ├── detail.md      # "User authentication fails on login"
  ├── logs.txt       # "ERROR: NullPointerException at AuthService.java:45"
  └── rca.md         # Generated analysis
```

---

### 2. Embeddings & Retrieval Module (`src/embeddings/`)

**Responsibility**: Create semantic embeddings and retrieve contextually relevant code

#### Architecture Components

```python
┌─────────────────────────────────────────────────────┐
│           Embeddings Module                          │
│                                                       │
│  ┌──────────────┐     ┌──────────────┐             │
│  │  Embedding   │────▶│    FAISS     │             │
│  │  Generator   │     │Vector Store  │             │
│  └──────────────┘     └──────────────┘             │
│         │                     │                      │
│         │                     │                      │
│         ▼                     ▼                      │
│  ┌──────────────────────────────────┐              │
│  │   Semantic Retrieval Engine      │              │
│  └──────────────────────────────────┘              │
└─────────────────────────────────────────────────────┘
```

#### Core Functionality

##### Primary Function: `get_context(bug_id: str) -> str`

**Purpose**: Retrieve comprehensive context for bug analysis by combining bug information with relevant code sections

**Algorithm Flow**:

```python
def get_context(bug_id: str) -> str:
    """
    Retrieves contextualized information for bug analysis.
    
    Args:
        bug_id: Unique identifier for the bug
        
    Returns:
        Combined context string containing bug details, logs, 
        and relevant codebase excerpts
        
    Raises:
        FileNotFoundError: If bug data files are missing
        EmbeddingError: If embedding generation fails
    """
    # Step 1: Load bug-specific data
    bug_detail = load_file(f"/data/bugs/{bug_id}/detail.md")
    bug_logs = load_file(f"/data/bugs/{bug_id}/logs.txt")
    
    # Step 2: Generate semantic embeddings
    bug_detail_embedding = create_embedding(bug_detail)
    bug_logs_embedding = create_embedding(bug_logs)
    
    # Step 3: Retrieve semantically similar code sections
    # Uses cosine similarity search in FAISS vector store
    relevant_code_files = retrieve_relevant_codebase(
        query_embeddings=[bug_detail_embedding, bug_logs_embedding],
        top_k=10,  # Retrieve top 10 most relevant files
        similarity_threshold=0.7
    )
    
    # Step 4: Combine all information into structured context
    context = combine_context(
        bug_detail=bug_detail,
        bug_logs=bug_logs,
        relevant_code=relevant_code_files
    )
    
    return context
```

#### Vector Database: FAISS

**Why FAISS?**
- High-performance similarity search
- Efficient memory usage for large codebases
- Support for multiple indexing algorithms
- No external database server required

**Embedding Strategy:**
- **Model**: Uses OpenAI embeddings or sentence-transformers
- **Chunking**: Code files split into logical chunks (functions, classes)
- **Indexing**: Pre-computed embeddings stored in FAISS index
- **Retrieval**: Cosine similarity search with configurable top-k results

**Index Structure:**
```python
{
    "file_path": "src/auth/login.py",
    "chunk_id": "func_authenticate_user",
    "embedding": [0.123, -0.456, ...],  # 768/1536-dim vector
    "metadata": {
        "line_start": 45,
        "line_end": 78,
        "function_name": "authenticate_user"
    }
}
```

---

### 3. Security & Privacy Module (`src/utils/`)

**Responsibility**: Ensure safe and secure data handling before LLM processing

#### Security Architecture

```python
┌─────────────────────────────────────────────────────┐
│              Security Module                         │
│                                                       │
│  ┌──────────────────┐   ┌─────────────────────┐    │
│  │ Secret Scanner   │   │ Prompt Injection    │    │
│  │ & Sanitizer      │   │ Detector            │    │
│  └──────────────────┘   └─────────────────────┘    │
│         │                         │                  │
│         └───────┬─────────────────┘                 │
│                 ▼                                    │
│         ┌──────────────┐                            │
│         │ Cleaned Data │                            │
│         └──────────────┘                            │
└─────────────────────────────────────────────────────┘
```

#### Core Security Functions

##### 1. Secret Sanitization

**Function**: `sanitize_secrets(text: str) -> str`

**Purpose**: Remove sensitive information before sending to LLM

**Detection Patterns**:
- API keys (AWS, OpenAI, GitHub tokens)
- Passwords and credentials
- Private keys and certificates
- Database connection strings
- Email addresses and phone numbers
- IP addresses and internal URLs

**Implementation Strategy**:
```python
def sanitize_secrets(text: str) -> str:
    """
    Redacts sensitive information from text.
    
    Args:
        text: Input text potentially containing secrets
        
    Returns:
        Sanitized text with secrets replaced by placeholders
    """
    patterns = {
        'api_key': r'[A-Za-z0-9_-]{32,}',
        'password': r'password\s*=\s*["\']([^"\']+)["\']',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        # ... more patterns
    }
    
    for secret_type, pattern in patterns.items():
        text = re.sub(pattern, f'[REDACTED_{secret_type.upper()}]', text)
    
    return text
```

##### 2. Prompt Injection Detection

**Function**: `detect_prompt_injection(text: str) -> bool`

**Purpose**: Identify malicious prompt manipulation attempts

**Detection Techniques**:
- Instruction override patterns ("Ignore previous instructions")
- Role manipulation attempts ("You are now a...")
- System prompt extraction attempts
- Jailbreak patterns
- Encoding tricks (Base64, Unicode obfuscation)

**Response Strategy**:
```python
def detect_prompt_injection(text: str) -> bool:
    """
    Detects potential prompt injection attacks.
    
    Args:
        text: Input text to analyze
        
    Returns:
        True if injection attempt detected, False otherwise
    """
    injection_patterns = [
        r'ignore\s+(previous|above|prior)\s+instructions',
        r'you\s+are\s+now\s+a',
        r'system\s*:\s*',
        r'<\|im_start\|>',
        # ... more patterns
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            logger.warning(f"Prompt injection detected: {pattern}")
            return True
    
    return False
```

#### Security Workflow Integration

Both security functions are **mandatory** in the main processing pipeline:

```python
# Main workflow integration
context = get_context(bug_id)

# Security checkpoint
if detect_prompt_injection(context):
    raise SecurityException("Prompt injection detected")

sanitized_context = sanitize_secrets(context)

# Proceed to LLM
rca = generate_rca(sanitized_context, bug_detail)
```

---

### 4. LLM Workflow Module (`src/workflow/`)

**Responsibility**: Orchestrate LLM interaction for RCA generation

#### Workflow Architecture

```python
┌─────────────────────────────────────────────────────┐
│           Workflow Orchestration                     │
│                                                       │
│  Input: Context + Bug Details                        │
│           │                                           │
│           ▼                                           │
│  ┌────────────────────┐                              │
│  │  Prompt Template   │                              │
│  │  Construction      │                              │
│  └────────────────────┘                              │
│           │                                           │
│           ▼                                           │
│  ┌────────────────────┐                              │
│  │  LLM API Call      │                              │
│  │  (GPT-4, Claude)   │                              │
│  └────────────────────┘                              │
│           │                                           │
│           ▼                                           │
│  ┌────────────────────┐                              │
│  │ Response Parser    │                              │
│  │ & Validator        │                              │
│  └────────────────────┘                              │
│           │                                           │
│           ▼                                           │
│  Output: Structured RCA                               │
└─────────────────────────────────────────────────────┘
```

#### Core Function: `generate_rca(context: str, bug_detail: str) -> str`

**Purpose**: Generate comprehensive Root Cause Analysis using LLM

**Implementation**:

```python
def generate_rca(context: str, bug_detail: str) -> str:
    """
    Generates Root Cause Analysis using LLM.
    
    Args:
        context: Sanitized context from get_context() and security checks
        bug_detail: Original bug description
        
    Returns:
        Structured RCA markdown document
        
    Raises:
        LLMAPIError: If API call fails
        ValidationError: If response format is invalid
    """
    # Construct prompt with structured template
    prompt = f"""
    You are a senior software engineer conducting a root cause analysis.
    
    Bug Description:
    {bug_detail}
    
    Relevant Context:
    {context}
    
    Provide a detailed RCA including:
    1. Root Cause Identification
    2. Affected Components
    3. Impact Analysis
    4. Recommended Fix
    5. Prevention Strategies
    """
    
    # Call LLM API
    response = llm_client.generate(
        prompt=prompt,
        model="gpt-4",
        temperature=0.3,  # Lower for more deterministic output
        max_tokens=2000
    )
    
    # Parse and validate response
    rca = parse_rca_response(response)
    validate_rca_structure(rca)
    
    return rca
```

#### LLM Configuration

**Supported Models**:
- OpenAI GPT-4 (primary)
- OpenAI GPT-3.5-turbo (fallback)
- Anthropic Claude (alternative)

**Parameters**:
- **Temperature**: 0.3 (balanced between creativity and consistency)
- **Max Tokens**: 2000-4000 (depending on bug complexity)
- **Top-p**: 0.9
- **Frequency Penalty**: 0.2

---

## Complete System Workflow

### End-to-End Process Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Bug Data Collection                                       │
│    - Store bug details, logs in /data/bugs/{bug_id}/        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Context Retrieval (embeddings.get_context)               │
│    - Load bug details and logs                               │
│    - Generate embeddings                                     │
│    - Retrieve relevant code via FAISS similarity search     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Security Checks (utils.security)                         │
│    - Detect prompt injection attempts                        │
│    - Sanitize secrets and sensitive data                    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. LLM Processing (workflow.generate_rca)                   │
│    - Construct structured prompt                             │
│    - Call LLM API                                            │
│    - Parse and validate response                            │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. RCA Output                                                │
│    - Save to /data/bugs/{bug_id}/rca.md                     │
│    - Return structured analysis                              │
└─────────────────────────────────────────────────────────────┘
```

### Example Usage

```python
from src.embeddings import get_context
from src.utils import sanitize_secrets, detect_prompt_injection
from src.workflow import generate_rca

# Main execution
def analyze_bug(bug_id: str) -> str:
    """Complete bug analysis workflow."""
    
    # Step 1: Retrieve context
    context = get_context(bug_id)
    
    # Step 2: Security validation
    if detect_prompt_injection(context):
        raise SecurityError("Potential prompt injection detected")
    
    sanitized_context = sanitize_secrets(context)
    
    # Step 3: Load bug detail
    bug_detail = load_file(f"/data/bugs/{bug_id}/detail.md")
    
    # Step 4: Generate RCA
    rca = generate_rca(sanitized_context, bug_detail)
    
    # Step 5: Save output
    save_file(f"/data/bugs/{bug_id}/rca.md", rca)
    
    return rca

# Execute
result = analyze_bug("bug001")
```

---

## Configuration & Dependencies

### System Requirements

- **Python**: 3.9+
- **Memory**: Minimum 4GB RAM (8GB recommended for large codebases)
- **Storage**: Depends on codebase size (vector index requires ~2-3x source size)

### Key Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| `langchain` | LLM orchestration framework | 1.2.0 |
| `openai` | OpenAI API client | 2.13.0 |
| `faiss-cpu` | Vector similarity search | Latest |
| `numpy` | Numerical computations | 2.3.5 |
| `autogen-agentchat` | Agent workflows | 0.7.5 |

See `requirements.txt` for complete dependency list.

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...                    # OpenAI API key
EMBEDDING_MODEL=text-embedding-ada-002   # Embedding model

# Optional
LLM_MODEL=gpt-4                          # LLM model choice
FAISS_INDEX_PATH=/data/faiss_index       # Vector store path
MAX_CONTEXT_LENGTH=8000                  # Max tokens for context
SIMILARITY_THRESHOLD=0.7                 # Minimum similarity score
```

---

## Error Handling & Resilience

### Error Types & Mitigation

| Error Type | Cause | Mitigation Strategy |
|------------|-------|---------------------|
| `FileNotFoundError` | Missing bug data files | Validate file existence before processing |
| `EmbeddingError` | Embedding generation failure | Retry with exponential backoff |
| `SecurityException` | Prompt injection detected | Log incident and reject request |
| `LLMAPIError` | API rate limit or failure | Implement retry logic with fallback models |
| `ValidationError` | Invalid RCA format | Re-prompt with format instructions |

### Logging & Monitoring

```python
import logging

# Structured logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug_agent.log'),
        logging.StreamHandler()
    ]
)

# Key metrics to monitor
- Embedding generation time
- FAISS retrieval latency
- LLM API response time
- Security event frequency
- RCA generation success rate
```

---

## Performance Considerations

### Optimization Strategies

1. **Embedding Caching**: Pre-compute and store codebase embeddings
2. **Batch Processing**: Process multiple bugs concurrently
3. **FAISS Index Optimization**: Use IVF (Inverted File Index) for large codebases
4. **Context Window Management**: Truncate to fit LLM token limits
5. **Async API Calls**: Use async/await for I/O operations

### Scalability

- **Small Codebases** (<1000 files): Single-machine deployment
- **Medium Codebases** (1K-10K files): Distributed FAISS indexes
- **Large Codebases** (>10K files): Sharded vector stores with load balancing

---

## Example Use Case: Apache HTTP Server

### Sample Configuration

- **Codebase**: Apache HTTP Server
- **Repository**: https://github.com/apache/httpd
- **Issue Tracker**: https://issues.apache.org/jira/projects/HTTPD
- **Bug Example**: [HDDS-14214](https://issues.apache.org/jira/browse/HDDS-14214)

### Workflow Example

```bash
# 1. Clone Apache HTTPD codebase
git clone https://github.com/apache/httpd /data/codebase/httpd

# 2. Create bug entry
mkdir -p /data/bugs/HDDS-14214
echo "Memory leak in worker thread pool" > /data/bugs/HDDS-14214/detail.md

# 3. Add logs
cp /var/log/apache2/error.log /data/bugs/HDDS-14214/logs.txt

# 4. Run analysis
python src/main.py analyze HDDS-14214

# 5. Review RCA
cat /data/bugs/HDDS-14214/rca.md
```

---

## Future Enhancements

### Roadmap

- [ ] Multi-repository support
- [ ] Real-time bug monitoring integration
- [ ] Interactive RCA refinement with user feedback
- [ ] Support for additional LLM providers (Azure OpenAI, Google Vertex AI)
- [ ] Automated fix suggestion with code generation
- [ ] Integration with CI/CD pipelines
- [ ] Web-based UI for bug submission and RCA viewing

### Extensibility Points

1. **Custom Embedding Models**: Plug in domain-specific encoders
2. **Alternative Vector Databases**: Support for Pinecone, Weaviate, Milvus
3. **LLM Provider Abstraction**: Easy switching between providers
4. **Custom Security Rules**: Configurable secret patterns and injection detectors

---

## Troubleshooting

### Common Issues

**Issue**: FAISS index not found
```bash
Solution: Build index first using `python src/embeddings/build_index.py`
```

**Issue**: LLM API rate limit exceeded
```bash
Solution: Implement exponential backoff or use lower-tier model as fallback
```

**Issue**: Out of memory during embedding generation
```bash
Solution: Process files in batches, reduce chunk size, or increase system memory
```

---

## References

- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Prompt Injection Prevention](https://learnprompting.org/docs/prompt_hacking/injection)

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-19 | Initial architecture documentation |

---

**Document Owner**: Development Team  
**Last Updated**: December 19, 2025  
**Review Cycle**: Quarterly