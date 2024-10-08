import streamlit as st
import os
import json
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

# Try importing faiss
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    st.error("FAISS is not installed. Please check the logs for more information.")

class DocumentStore:
    def __init__(self, embeddings_data: Dict, texts_data: Dict, faiss_index):
        self.embeddings_data = embeddings_data
        self.texts_data = texts_data
        self.index = faiss_index
        
        # Initialize sentence transformer for query encoding
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create ID mapping
        self._setup_id_mapping()
        
    def _setup_id_mapping(self):
        self.str_to_int_id = {str_id: idx for idx, str_id in enumerate(self.embeddings_data.keys())}
        self.int_to_str_id = {idx: str_id for str_id, idx in self.str_to_int_id.items()}

    def document_generator(self) -> Generator[Dict[str, Any], None, None]:
        batch = []
        for str_id, embedding in self.embeddings_data.items():
            if str_id in self.texts_data:
                doc = {
                    "str_id": str_id,
                    "int_id": self.str_to_int_id[str_id],
                    "embedding": embedding,
                    "text": self.texts_data[str_id]
                }
                batch.append(doc)
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
        api_key = st.secrets["ANTHROPIC_API_KEY"]
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
    def __init__(self, embeddings_data: Dict, texts_data: Dict, faiss_index):
        self.document_store = DocumentStore(embeddings_data, texts_data, faiss_index)
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

@st.cache_resource
def load_data():
    try:
        # Load embeddings
        embeddings_data = st.session_state.embeddings_data
        texts_data = st.session_state.texts_data
        faiss_index = st.session_state.faiss_index
        
        return embeddings_data, texts_data, faiss_index
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def main():
    st.set_page_config(page_title="Legal Research Assistant", page_icon="⚖️", layout="wide")
    
    st.title("🔍 Legal Research Assistant")
    
    # File upload section in sidebar
    with st.sidebar:
        st.header("📁 File Upload")
        embeddings_file = st.file_uploader("Upload embeddings JSON", type="json")
        texts_file = st.file_uploader("Upload texts JSON", type="json")
        faiss_file = st.file_uploader("Upload FAISS index")
        
        if embeddings_file and texts_file and faiss_file:
            try:
                # Save uploads to session state
                st.session_state.embeddings_data = json.load(embeddings_file)
                st.session_state.texts_data = json.load(texts_file)
                
                # Save FAISS index to a temporary file and load it
                faiss_bytes = faiss_file.read()
                temp_faiss_path = "temp_index.faiss"
                with open(temp_faiss_path, "wb") as f:
                    f.write(faiss_bytes)
                st.session_state.faiss_index = faiss.read_index(temp_faiss_path)
                os.remove(temp_faiss_path)  # Clean up
                
                st.success("All files loaded successfully!")
            except Exception as e:
                st.error(f"Error loading files: {str(e)}")
    
    # Main content
    if not all(hasattr(st.session_state, attr) for attr in ['embeddings_data', 'texts_data', 'faiss_index']):
        st.warning("Please upload all required files in the sidebar to continue.")
        return
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        embeddings_data, texts_data, faiss_index = load_data()
        if all([embeddings_data, texts_data, faiss_index]):
            st.session_state.rag_system = RAGSystem(embeddings_data, texts_data, faiss_index)
    
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
            results = st.session_state.rag_system.process_query(query, top_k=3)
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

if __name__ == "__main__":
    main()