Debug agent architecture


Process flow
1. Data collection
   - Data is placed in location `/data`
   - `/data/codebase` contains the codebase files
   - `/data/bugs` contains details of each bug.
   - Each bug has three files:
     - `detail.md`: Description of the bug
     - `logs.txt`: Logs related to the bug
     - `rca.md`: The fix for the bug

2. Create Embeddings and retrieval for bug detail, logs and codebase
   - The module `src/embedding` is responsible for creating embeddings for the bug details, logs, and codebase files.
   - It will expose a function which takes bug id as input and returns relevant information from the codebase, bug details, and logs as context for LLM.
   ```
   # Sudo code of the function
    def get_context(bug_id: str) -> str:
         # Load bug detail and logs
         bug_detail = load_file(f"/data/bugs/{bug_id}/detail.md")
         bug_logs = load_file(f"/data/bugs/{bug_id}/logs.txt")
    
         # Create embeddings for bug detail and logs
         bug_detail_embedding = create_embedding(bug_detail)
         bug_logs_embedding = create_embedding(bug_logs)
    
         # Retrieve relevant codebase files using embeddings
         relevant_code_files = retrieve_relevant_codebase([bug_detail_embedding, bug_logs_embedding])
    
         # Combine all context information
         context = combine_context(bug_detail, bug_logs, relevant_code_files)
    
         return context
   ```
   - It uses vector database to store and retrieve embeddings. here we use FAISS as vector database.

3. Ensure security and privacy
   - The module `src/utils` expose a function to sanitize secrets from the context before passing it to LLM.
   - It also exposes a function to identify prompt injection attempts and handle them appropriately.
   - both functions are used in the main flow before sending context to LLM.

4. LLM interaction
   - The module `src/workflow` is responsible for interacting with the LLM.
   - It exposes a function which takes the context and bug detail as input and generates the RCA using LLM.


Sample:
- Codebase: Apache HTTP Server codebase
- git hub repo: https://github.com/apache/httpd
- Jira Borad: https://issues.apache.org/jira/browse/HDDS-14214?filter=-4