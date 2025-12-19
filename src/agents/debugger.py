from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
)


def generate_rca(context: str) -> str:
    """
    Generates Root Cause Analysis using LLM.
    
    Args:
        context: Sanitized context from get_context() and security checks
                 (includes bug details, logs, and relevant code)
        bug_detail: Original bug description
        
    Returns:
        Structured RCA markdown document
        
    Raises:
        Exception: If API call fails or response format is invalid
    """
    # Construct prompt with structured template
    prompt = f"""
    You are a senior software engineer conducting a root cause analysis.
    Relevant Context (includes logs and relevant code):
    {context}

    Provide a detailed RCA in markdown format including:

    ## 1. Root Cause Identification
    Identify the primary cause of the bug based on the provided context.

    ## 2. Affected Components
    List all components, files, or modules affected by this bug.

    ## 3. Impact Analysis
    Describe the impact of this bug on system functionality, users, and operations.

    ## 4. Recommended Fix
    Provide specific, actionable steps to fix this bug.

    ## 5. Prevention Strategies
    Suggest measures to prevent similar bugs in the future.

    Please provide your analysis in a clear, structured markdown format.
    """
    
    try:
        # Call LLM API using the initialized model
        response = model.invoke(prompt)
        
        # Extract the content from the response
        rca = response.content
        
        # Basic validation - ensure we got a non-empty response
        if not rca or len(rca.strip()) < 100:
            raise ValueError("Generated RCA is too short or empty")
        
        return rca
        
    except Exception as e:
        raise Exception(f"LLM API Error during RCA generation: {str(e)}")

