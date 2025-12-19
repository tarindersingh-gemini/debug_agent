import os
import glob
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

class EmbeddingGenerator:
    def __init__(self):
        load_dotenv()
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        self.code_splitter = RecursiveCharacterTextSplitter.from_language(language="python", chunk_size=1000, chunk_overlap=200)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def generate_code_embeddings(self, code_dir=None):
        if code_dir is None:
            code_dir = os.path.join(self.project_root, "data", "Code")
        py_files = glob.glob(os.path.join(code_dir, "**", "*.py"), recursive=True)
        code_chunks = []
        for file in py_files:
            with open(file, 'r') as f:
                code = f.read()
            chunks = self.code_splitter.split_text(code)
            print(f"{file}: {len(chunks)} chunks")
            code_chunks.extend(chunks)
        if not code_chunks:
            print("No code chunks found. Skipping FAISS creation for code.")
            return
        print(f"Code: {len(code_chunks)} chunks created.")
        code_vectorstore = FAISS.from_texts(code_chunks, self.embeddings)
        code_vectorstore.save_local(os.path.join(self.project_root, "faiss_code_index"))
        print("Code embeddings stored in FAISS.")

    def generate_bug_embeddings(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(self.project_root, "data", "bugs")
        bug_dirs = [d for d in os.listdir(data_dir) if d.startswith("bug") and os.path.isdir(os.path.join(data_dir, d))]
        bug_chunks = []
        bug_metadatas = []
        for bug_dir in bug_dirs:
            bug_id = bug_dir
            log_file = os.path.join(data_dir, bug_dir, "bug_logs.txt")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = f.read()
                chunks = self.text_splitter.split_text(logs)
                print(f"{bug_dir}: {len(chunks)} chunks")
                bug_chunks.extend(chunks)
                bug_metadatas.extend([{"BugId": bug_id} for _ in chunks])
        if not bug_chunks:
            print("No bug chunks found. Skipping FAISS creation for bugs.")
            return
        print(f"Bugs: {len(bug_chunks)} chunks created.")
        bug_vectorstore = FAISS.from_texts(bug_chunks, self.embeddings, metadatas=bug_metadatas)
        bug_vectorstore.save_local(os.path.join(self.project_root, "faiss_bug_index"))
        print("Bug embeddings stored in FAISS.")

    def get_embedding(self, text):
        return self.embeddings.embed_query(text)

    def run(self):
        self.generate_code_embeddings()
        self.generate_bug_embeddings()

if __name__ == "__main__":
    generator = EmbeddingGenerator()
    generator.run()
    generator.run()