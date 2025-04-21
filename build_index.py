from typing import Dict, List
import os
from pathlib import Path
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import fitz  # PyMuPDF
import tiktoken
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

class KnowledgeBaseBuilder:
    def __init__(self, embeddings_model: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(
            model=embeddings_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
    def num_tokens_from_string(self, string: str, encoding_name: str = "cl100k_base") -> int:
        """Count the number of tokens in a text string"""
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(string))
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = "\n".join([page.get_text() for page in doc])
            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def process_directory(self, base_dir: str) -> List[Dict]:
        """Process all PDFs in the directory structure"""
        documents = []
        base_path = Path(base_dir)
        
        # Process each category directory
        for category_dir in base_path.iterdir():
            if not category_dir.is_dir():
                continue
                
            category = category_dir.name.lower()
            print(f"\nProcessing category: {category}")
            
            # Process all PDFs in the category directory
            for pdf_path in category_dir.glob("*.pdf"):
                try:
                    print(f"\nProcessing {pdf_path}")
                    
                    # Extract text from PDF
                    text = self.extract_text_from_pdf(str(pdf_path))
                    if not text:
                        continue
                    
                    # Split text into chunks
                    chunks = self.text_splitter.split_text(text)
                    
                    # Log chunks for debugging
                    print(f"Generated {len(chunks)} chunks from {pdf_path.name}")
                    for i, chunk in enumerate(chunks):
                        print(f"\nChunk {i+1}/{len(chunks)}:")
                        print("-" * 40)
                        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                        print("-" * 40)
                        
                        # Create document with metadata
                        documents.append({
                            "page_content": chunk,
                            "metadata": {
                                "source": str(pdf_path),
                                "category": category,
                                "filename": pdf_path.name
                            }
                        })
                        
                except Exception as e:
                    print(f"Error processing {pdf_path}: {e}")
                    continue
        
        return documents

    def build_index(self, pdfs_dir: str = "pdfs", output_dir: str = "vector_index"):
        """Build vector store index from PDF directory"""
        # Process PDFs
        documents_dict = self.process_directory(pdfs_dir)
        if not documents_dict:
            raise ValueError("No documents were successfully processed!")
            
        # Convert dictionaries to Document objects
        documents = [
            Document(
                page_content=doc["page_content"],
                metadata=doc["metadata"]
            ) for doc in documents_dict
        ]
            
        # Create metadata list
        metadata_list = [doc.metadata for doc in documents]
        
        # Create vector store
        vector_store = FAISS.from_documents(
            documents,
            self.embeddings
        )
        
        # Save vector store and metadata
        os.makedirs(output_dir, exist_ok=True)
        vector_store.save_local(output_dir)
        
        with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata_list, f)
            
        print(f"\nIndex built successfully!")
        print(f"Total documents indexed: {len(documents)}")
        print(f"Documents by category:")
        categories = {}
        for doc in documents:
            cat = doc.metadata["category"]
            categories[cat] = categories.get(cat, 0) + 1
        for cat, count in categories.items():
            print(f"  - {cat}: {count} chunks")
        print(f"Index saved to: {output_dir}")

def main():
    # Build index from KB/pdfs directory
    builder = KnowledgeBaseBuilder()
    builder.build_index("KB/pdfs")  # Updated path to KB/pdfs directory

if __name__ == "__main__":
    main() 