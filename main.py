from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import pickle
import os
from query_rewriter import QueryRewriter
from rag_query import RAGQuery
from query_merger import QueryMerger

class QueryEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.metadata = None
        self.load_resources()
    
    def load_resources(self):
        # Load the vector store from current directory
        self.vector_store = FAISS.load_local(
            ".",  # Changed from "vector_index" to "."
            self.embeddings
        )
        
        # Load metadata from current directory
        with open("metadata.pkl", 'rb') as f:  # Changed path
            self.metadata = pickle.load(f)
    
    def query(self, query_text: str, category: str = None, k: int = 4):
        search_kwargs = {}
        if category:
            search_kwargs["filter"] = {"category": category}
            
        results = self.vector_store.similarity_search(
            query_text,
            k=k,
            **search_kwargs
        )
        return results

class MedicalQuerySystem:
    def __init__(self, debug: bool = False):
        try:
            # Verify API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")
                
            self.rag = RAGQuery(index_path=".", debug=debug)
            self.current_category = None
            self.categories = ['ctr', 'iols', 'gen']
            self.category_aliases = {
                'iol': 'iols',
                'iols': 'iols',
                'ctr': 'ctr',
                'gen': None,
                'general': None
            }
            self.chat_histories = {
                'ctr': [],
                'iols': [],
                None: []  # For general mode
            }
            self.query_rewriter = QueryRewriter(api_key)
            self.query_merger = QueryMerger(api_key)
            self.current_role = "doctor"
            self.valid_roles = ["doctor", "sales"]
        except Exception as e:
            print(f"Error initializing MedicalQuerySystem: {str(e)}")
            raise
    
    def get_current_history(self):
        """Get chat history for current category"""
        return self.chat_histories[self.current_category]
    
    def switch_category(self, command):
        """Handle category switching commands"""
        if command.startswith('switch '):
            new_category = command.split()[1].lower()
            # Check if the category or its alias exists
            if new_category in self.category_aliases:
                old_category = self.current_category
                self.current_category = self.category_aliases[new_category]
                return True
            else:
                print(f"\nInvalid category. Available categories: {', '.join(['ctr', 'iols', 'gen'])}")
                return True
        return False
    
    def switch_role(self, command):
        """Handle role switching commands"""
        if command.startswith('role '):
            new_role = command.split()[1].lower()
            if new_role in self.valid_roles:
                self.current_role = new_role
                print(f"\nSwitched to {new_role.upper()} role")
                return True
            else:
                print(f"\nInvalid role. Available roles: {', '.join(self.valid_roles)}")
                return True
        return False
    
    def process_query(self, query: str) -> str:
        """Process query and get appropriate response"""
        try:
            # Get current category's history
            current_history = self.get_current_history()
            
            # Single rewrite for both RAG and merger
            rewritten_query = self.query_rewriter.rewrite_query(query, current_history)
            
            # If query was rewritten, show the rewrite
            if rewritten_query != query:
                print(f"Rewritten query: {rewritten_query}")
            
            # Update chat history for current category
            current_history.append({
                "role": "user",
                "content": query
            })
            
            # Get response based on category
            if self.current_category:
                # Get KB response using rewritten query
                kb_response = self.rag.query(
                    rewritten_query,
                    category=self.current_category,
                    skip_rewrite=True
                )
                # Process KB response according to role
                final_response = self.query_merger.get_response(
                    rewritten_query,
                    category=self.current_category,
                    kb_response=kb_response,
                    role=self.current_role
                )
            else:
                # For general queries, use the merger with role
                final_response = self.query_merger.get_response(
                    rewritten_query,
                    role=self.current_role
                )
            
            # Update chat history for current category
            current_history.append({
                "role": "assistant",
                "content": final_response
            })
            
            return final_response
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return "I apologize, but I encountered an error. Could you please try again?"
    
    def run(self):
        print("\nWelcome to the Medical Knowledge Base Query System")
        print("Available commands:")
        print("- 'switch iol' or 'switch iols': Switch to IOL category")
        print("- 'switch ctr': Switch to CTR category")
        print("- 'switch gen': Switch to general mode")
        print("- 'role doctor': Switch to doctor role")
        print("- 'role sales': Switch to sales representative role")
        print("- 'quit': Exit the system")
        print("\nStart asking questions! Use switch commands to focus on specific categories.")
        
        # Main query loop
        while True:
            # Show current context
            context = f"[{self.current_category.upper()}]" if self.current_category else "[GENERAL]"
            query = input(f"\n{context} Enter your question: ").strip()
            
            # Handle exit command
            if query.lower() == 'quit':
                break
                
            # Handle category switching
            if self.switch_category(query.lower()):
                continue
            
            # Handle role switching
            if self.switch_role(query.lower()):
                continue
            
            # Process query
            try:
                print("\nSearching...")
                response = self.process_query(query)
                print("\nAnswer:")
                print(response)
                print("-" * 80)
            except Exception as e:
                print(f"Error: {e}")

def main(debug: bool = True):
    system = MedicalQuerySystem(debug=debug)
    system.run()

if __name__ == "__main__":
    # Can be controlled via command line argument or environment variable
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    main(debug=debug_mode) 