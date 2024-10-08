import streamlit as st
import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Generator
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DocumentStore:
    def __init__(self, embeddings_path: str, texts_path: str, faiss_index_path: str, batch_size: int = 1000):
        self.embeddings_path = embeddings_path
        self.texts_path = texts_path
        self.faiss_index_path = faiss_index_path
        self.batch_size = batch_size
        
        # Load FAISS index
        self.index = faiss.read_index(faiss_index_path)
        
        # Initialize sentence transformer for query encoding
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create ID mapping
        self._setup_id_mapping()
        
    def _setup_id_mapping(self):
        """Create mappings between string IDs and integer indices"""
        self.str_to_int_id = {}
        self.int_to_str_id = {}
        
        with open(self.embeddings_path, 'r') as f:
            embeddings_data = json.load(f)
        
        for idx, str_id in enumerate(embeddings_data.keys()):
            self.str_to_int_id[str_id] = idx
            self.int_to_str_id[idx] = str_id

    def document_generator(self) -> Generator[Dict[str, Any], None, None]:
        with open(self.embeddings_path, 'r') as emb_file, open(self.texts_path, 'r') as text_file:
            embeddings_data = json.load(emb_file)
            texts_data = json.load(text_file)
            
            batch = []
            for str_id, embedding in embeddings_data.items():
                if str_id in texts_data:
                    doc = {
                        "str_id": str_id,
                        "int_id": self.str_to_int_id[str_id],
                        "embedding": embedding,
                        "text": texts_data[str_id]
                    }
                    batch.append(doc)
                    
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
            
            if batch:
                yield batch

class RAGModel:
    def __init__(self, document_store: DocumentStore):
        self.document_store = document_store

    def get_relevant_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = self.document_store.model.encode([query])[0].astype(np.float32)
        
        distances, indices = self.document_store.index.search(
            query_embedding.reshape(1, -1), 
            top_k
        )
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            try:
                str_id = self.document_store.int_to_str_id[idx]
                
                for batch in self.document_store.document_generator():
                    doc_dict = {doc['str_id']: doc for doc in batch}
                    if str_id in doc_dict:
                        similarity = 1 - (distance ** 2) / 2
                        doc = {
                            "id": str_id,
                            "text": doc_dict[str_id]['text'],
                            "similarity": similarity
                        }
                        results.append(doc)
                        break
            except Exception as e:
                logger.error(f"Error processing document index {idx}: {e}")
        
        return results

class ClaudeEnhancer:
    def __init__(self):
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        self.client = Anthropic(api_key=api_key)

    def generate_response(self, query: str, relevant_docs: List[Dict[str, Any]]) -> str:
        context = "\n\n".join([
            f"Document {i+1} ({doc['id']}, Similarity: {doc['similarity']:.4f}):\n{doc['text']}" 
            for i, doc in enumerate(relevant_docs)
        ])
        
        prompt = f"""Based on the following context and query, provide a relevant and informative response.
        Prioritize information from documents with higher similarity scores.

Context:
{context}

Query: {query}

Response:"""

        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

class RAGSystem:
    def __init__(self, embeddings_path: str, texts_path: str, faiss_index_path: str, batch_size: int = 1000):
        self.document_store = DocumentStore(embeddings_path, texts_path, faiss_index_path, batch_size)
        self.rag_model = RAGModel(self.document_store)
        self.claude_enhancer = ClaudeEnhancer()

    def process_query(self, query: str, top_k: int) -> Dict[str, Any]:
        try:
            relevant_docs = self.rag_model.get_relevant_documents(query, top_k)
            if not relevant_docs:
                return {
                    "query": query,
                    "error": "No relevant documents found"
                }
            
            enhanced_response = self.claude_enhancer.generate_response(query, relevant_docs)
            
            return {
                "query": query,
                "relevant_documents": relevant_docs,
                "enhanced_response": enhanced_response
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": query,
                "error": str(e)
            }

# Streamlit UI
def main():
    st.set_page_config(page_title="Legal Research Assistant", page_icon="⚖️", layout="wide")
    
    st.title("🔍 Legal Research Assistant")
    
    # Sidebar
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Number of relevant documents", min_value=1, max_value=10, value=3)
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        with st.spinner("Initializing the system..."):
            try:
                st.session_state.rag_system = RAGSystem(
                    embeddings_path=r"C:\themis\pdf_embeddings.json",
                    texts_path=r"C:\themis\pdf_texts.json",
                    faiss_index_path=r"C:\themis\legal_docs_index.faiss"
                )
                st.sidebar.success("System initialized successfully!")
            except Exception as e:
                st.sidebar.error(f"Error initializing system: {str(e)}")
                return
    
    # Query input
    query = st.text_area("Enter your legal research query:", height=100)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("Search", type="primary")
    with col2:
        if st.button("Clear Results"):
            st.session_state.pop('last_results', None)
    
    if search_button and query:
        with st.spinner("Searching and analyzing..."):
            start_time = time.time()
            results = st.session_state.rag_system.process_query(query, top_k)
            end_time = time.time()
            
            st.session_state.last_results = results
    
    # Display results
    if 'last_results' in st.session_state:
        results = st.session_state.last_results
        
        if "error" in results:
            st.error(f"Error: {results['error']}")
        else:
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents Analyzed", len(results['relevant_documents']))
            with col2:
                avg_similarity = np.mean([doc['similarity'] for doc in results['relevant_documents']])
                st.metric("Average Similarity", f"{avg_similarity:.4f}")
            with col3:
                st.metric("Processing Time", f"{end_time - start_time:.2f} seconds")
            
            # Enhanced Response
            st.header("📋 Summary")
            st.write(results['enhanced_response'])
            
            # Relevant Documents
            st.header("📚 Relevant Documents")
            for i, doc in enumerate(results['relevant_documents'], 1):
                with st.expander(f"Document {i}: {doc['id']} (Similarity: {doc['similarity']:.4f})"):
                    st.text_area("Content:", doc['text'], height=200)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made with ❤️ by Your Themis Legal Tech Team")

if __name__ == "__main__":
    main()