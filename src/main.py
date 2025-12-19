from embeddings import get_context
from utils.security import GuardRail
from agents import generate_rca



def process(bug_id: str):
    # 1. Get the context for the given bug ID
    context = get_context(bug_id)

    # 2. Check for prompt injection attacks
    guard = GuardRail()
    if guard.detect_prompt_injection(context):
        raise ValueError("Prompt injection detected in the context!")
    
    # 3. Sanitize secret information from the context
    context = guard.sanitize_secrets(context)

    # 4. Generate RCA using LLM``
    rca = generate_rca(context)

    return rca


if __name__ == "__main__":
    bug_id = "HDDS-14214"
    print(process(bug_id))
