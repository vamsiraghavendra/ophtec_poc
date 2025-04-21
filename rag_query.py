import os
import pickle
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from query_rewriter import QueryRewriter
import time  # Add at the top with other imports

# Load environment variables
load_dotenv()

class RAGQuery:
    def __init__(self, index_path: str = ".", debug: bool = False):
        self.index_path = index_path
        self.metadata_path = "metadata.pkl"  # Changed to direct file name
        self.debug = debug
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or not (api_key.startswith("sk-") or api_key.startswith("sk-proj-")):
                raise ValueError("Invalid OpenAI API key format. Key should start with 'sk-' or 'sk-proj-'")
            
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=api_key
            )
            self.client = OpenAI(api_key=api_key)
            self.vector_store = None
            self.metadata = None
            self.query_rewriter = QueryRewriter(api_key)
            self.chat_history = []
            self.load_resources()
        except Exception as e:
            print(f"Error initializing RAGQuery: {str(e)}")
            raise

    def load_resources(self):
        """Load existing vector store and metadata"""
        try:
            # Check if running on Streamlit Cloud
            is_streamlit = os.getenv('STREAMLIT_RUNTIME_ENV') is not None
            
            if is_streamlit:
                # Use relative paths for Streamlit Cloud
                vector_path = "."
                metadata_path = "metadata.pkl"
            else:
                # Use absolute paths for local development
                vector_path = os.path.abspath(".")
                metadata_path = os.path.abspath("metadata.pkl")
            
            # Load vector store
            self.vector_store = FAISS.load_local(
                vector_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Load metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            else:
                print(f"Warning: Metadata file not found at {metadata_path}")
                self.metadata = {}
                
        except Exception as e:
            print(f"Error loading resources: {str(e)}")
            raise

    def query(self, query_text: str, category: str = None, k: int = 6, skip_rewrite: bool = False):
        """Query the vector store and get response from ChatGPT"""
        try:
            start_time = time.time()
            
            if not self.vector_store:
                print("\n⚠️ Vector store not loaded, attempting to reload...")
                try:
                    self.load_resources()
                except Exception as e:
                    print(f"\n❌ Failed to load vector store: {str(e)}")
                    return None  # Return None instead of raising error
            
            # Time the rewrite step    
            if not skip_rewrite:
                rewrite_start = time.time()
                query_text = self.query_rewriter.rewrite_query(query_text, self.chat_history)
                print(f"\n⏱️ Query rewrite took: {time.time() - rewrite_start:.2f} seconds")
            
            # Time the document retrieval
            retrieval_start = time.time()
            try:
                if category:
                    docs = self.vector_store.similarity_search(
                        query_text,
                        k=k,
                        filter={"category": category}
                    )
                else:
                    docs = self.vector_store.similarity_search(query_text, k=k)
                    
                if not docs:
                    print("\n⚠️ No relevant documents found")
                    return None
                    
            except Exception as e:
                print(f"\n❌ Error during document retrieval: {str(e)}")
                return None
            
            print(f"\n⏱️ Document retrieval took: {time.time() - retrieval_start:.2f} seconds")
            
            # Debug output if enabled
            if self.debug:
                print("\nRetrieving relevant chunks:")
                for i, doc in enumerate(docs):
                    print(f"\nChunk {i+1}/{len(docs)}:")
                    print("-" * 40)
                    print(doc.page_content)
                    print(f"Source: {doc.metadata.get('source', 'unknown')}")
                    print(f"Category: {doc.metadata.get('category', 'unknown')}")
                    print("-" * 40)

            # Prepare context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Time the ChatGPT query
            gpt_start = time.time()
            messages = [
                {"role": "system", "content": "You are a helpful ophthalmology assistant. Answer questions based only on the following context:"},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query_text}"}
            ]

            try:
                print(f"\n🤖 Sending RAG query to ChatGPT: {query_text[:100]}...")
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.3
                )
                print(f"\n⏱️ ChatGPT response took: {time.time() - gpt_start:.2f} seconds")
                print(f"\n⏱️ Total query time: {time.time() - start_time:.2f} seconds")
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"\n❌ Error querying ChatGPT: {str(e)}")
                return None
                
        except Exception as e:
            print(f"\n❌ Error in RAG query: {str(e)}")
            return None

def interactive_query():
    rag = RAGQuery()
    
    print("\nWelcome to the Medical Knowledge Base Query System")
    print("Available categories: 'ctr', 'iols'")
    print("Type 'quit' to exit\n")
    
    while True:
        # Get category
        category = input("\nEnter category (or press Enter to search all): ").lower().strip()
        if category == 'quit':
            break
        if category and category not in ['ctr', 'iols']:
            print("Invalid category. Please choose 'ctr' or 'iols'")
            continue
            
        # Get query
        query = input("Enter your question: ").strip()
        if query.lower() == 'quit':
            break
            
        # Get response
        try:
            print("\nSearching...")
            response = rag.query(
                query,
                category=category if category else None
            )
            print("\nAnswer:")
            print(response)
            print("-" * 80)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    interactive_query() 