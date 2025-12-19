import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from .generation import EmbeddingGenerator

class Retrieval:
    def __init__(self):
        load_dotenv()
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        self.generator = EmbeddingGenerator()

    def get_context(self, bug_id: str) -> str:
        try:
            details_path = os.path.join(self.project_root, "data", "bugs", bug_id, "bug_details.md")
            if not os.path.exists(details_path):
                return f"Bug details file not found for {bug_id}"
            with open(details_path, 'r') as f:
                bug_details = f.read()
            query_embedding = self.generator.get_embedding(bug_details)
            bug_vectorstore = FAISS.load_local(os.path.join(self.project_root, "faiss_bug_index"), self.embeddings, allow_dangerous_deserialization=True)
            relevant_logs_results = bug_vectorstore.similarity_search_by_vector(query_embedding, k=5, filter={"BugId": bug_id})
            relevant_logs = " ".join([doc.page_content for doc in relevant_logs_results])
            code_vectorstore = FAISS.load_local(os.path.join(self.project_root, "faiss_code_index"), self.embeddings, allow_dangerous_deserialization=True)
            logs_embedding = self.generator.get_embedding(relevant_logs)
            relevant_code_results = code_vectorstore.similarity_search_by_vector(logs_embedding, k=5)
            relevant_code = " ".join([doc.page_content for doc in relevant_code_results])
            return f"{bug_details} /n {relevant_logs} /n {relevant_code}"
        except Exception as e:
            return f"Error in get_context: {str(e)}"


obj = Retrieval()
res = obj.get_context(bug_id="bug1")

print(res)