import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any
import logging
from pathlib import Path
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForSequenceClassification
import streamlit as st
import time
from rank_bm25 import BM25Okapi
from anthropic import Anthropic
from dotenv import load_dotenv
import boto3

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and Configuration
VALID_YEARS = list(range(2011, 2023))
VALID_COURT_TYPES = ["federal", "supreme", "appeal"]

@dataclass
class S3Config:
    bucket_name: str = 'themiscore22bucket'
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY")

@st.cache_resource
def initialize_s3_client(config: S3Config):
    """Initialize S3 client with caching."""
    return boto3.client(
        's3',
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key
    )

@st.cache_data
def load_json_from_s3(s3_client, bucket: str, key: str):
    """Load JSON data from S3 with caching."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        logger.error(f"Error loading {key} from S3: {e}")
        return None

@st.cache_data
def load_faiss_index_from_s3(s3_client, bucket: str, key: str):
    """Load FAISS index from S3 with caching."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        index_bytes = response['Body'].read()
        return faiss.deserialize_index(index_bytes)
    except Exception as e:
        logger.error(f"Error loading FAISS index {key} from S3: {e}")
        return None

class MetadataManager:
    def __init__(self, s3_client, bucket: str):
        self.s3_client = s3_client
        self.bucket = bucket
        self.metadata_by_year = {}
        self.load_metadata()

    def load_metadata(self):
        for year in VALID_YEARS:
            metadata = load_json_from_s3(
                self.s3_client,
                self.bucket,
                f"metadata/cleaned_{year}_data.json"
            )
            if metadata:
                self.metadata_by_year[year] = {
                    entry["Citation"].replace(" ", "_") + ".pdf": entry 
                    for entry in metadata
                }
                logger.info(f"Loaded metadata for {year}: {len(metadata)} entries")

    def get_metadata_for_year(self, year: int) -> Dict[str, Any]:
        return self.metadata_by_year.get(year, {})

class DocumentStore:
    def __init__(self, s3_client, bucket: str):
        self.s3_client = s3_client
        self.bucket = bucket
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.full_texts = load_json_from_s3(s3_client, bucket, "pdf_texts.json") or {}
        logger.info("Document store initialized")

    def load_faiss_index(self, year: int, court_type: str):
        key = f"embeddings/{year}/{court_type}/{year}_{court_type}_index.faiss"
        return load_faiss_index_from_s3(self.s3_client, self.bucket, key)

    def load_texts_data(self, year: int, court_type: str):
        key = f"embeddings/{year}/{court_type}/{year}_{court_type}_doc_ids.json"
        return load_json_from_s3(self.s3_client, self.bucket, key)

@st.cache_resource
def load_legalbert_model(s3_client, bucket: str):
    """Load LegalBERT model with caching."""
    try:
        # Download necessary model files to temporary storage
        model_files = [
            "config.json", 
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.txt"
        ]
        
        model_path = "legalbert_model"
        os.makedirs(model_path, exist_ok=True)
        
        for file in model_files:
            response = s3_client.get_object(
                Bucket=bucket,
                Key=f"legalbert_finetuned/{file}"
            )
            with open(f"{model_path}/{file}", 'wb') as f:
                f.write(response['Body'].read())
        
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading LegalBERT model: {e}")
        return None, None

class ClaudeAPIHandler:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    def summarize(self, text: str, max_length: int = 100) -> str:
        prompt = f"""Please provide a concise summary of the following legal text in approximately {max_length} words:\n\n{text}\n\nSummary:"""
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=max_length,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error in Claude API: {e}")
            return "Summary unavailable"

class RAGModel:
    def __init__(self, document_store: DocumentStore):
        self.document_store = document_store

    def get_relevant_documents(self, query: str, court_type: str, top_k: int = 3, year: int = None) -> List[Dict[str, Any]]:
        index = self.document_store.load_faiss_index(year, court_type)
        doc_ids = self.document_store.load_texts_data(year, court_type)
        
        if not index or not doc_ids:
            return []
        
        query_embedding = self.document_store.model.encode(query).astype("float32")
        distances, indices = index.search(np.array([query_embedding]), k=top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            doc_id = doc_ids[idx]
            full_text = self.document_store.full_texts.get(doc_id, "Full document text not available.")
            results.append({
                "id": doc_id,
                "text": full_text,
                "similarity": 1 - dist
            })
        return results

class LegalBERTResponseGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_response(self, query: str, relevant_docs: List[Dict[str, Any]]) -> str:
        if not self.model or not self.tokenizer:
            return "Model initialization failed. Please check logs."

        context = "\n\n".join([
            f"Document {i+1} ({doc['id']}, Similarity: {doc['similarity']:.4f}):\n{doc['text']}" 
            for i, doc in enumerate(relevant_docs)
        ])
        prompt = f"{context}\nQuery: {query}\nResponse:"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            output = self.model(**inputs)
            logits = output.logits
            probabilities = logits.softmax(dim=-1)
            predicted_class = probabilities.argmax().item()
            confidence = probabilities[0, predicted_class].item()
            
            class_names = ["Irrelevant", "Relevant"]
            return f"The query is classified as {class_names[predicted_class]} with confidence {confidence:.2f}"
        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            return "Response generation failed. Please try again."

def initialize_rag_system():
    """Initialize the RAG system with S3 resources."""
    config = S3Config()
    s3_client = initialize_s3_client(config)
    
    document_store = DocumentStore(s3_client, config.bucket_name)
    rag_model = RAGModel(document_store)
    
    model, tokenizer = load_legalbert_model(s3_client, config.bucket_name)
    legalbert_generator = LegalBERTResponseGenerator(model, tokenizer)
    
    claude_api = ClaudeAPIHandler()
    
    return rag_model, legalbert_generator, claude_api

def main():
    st.title("Canadian Law RAG System")
    
    # Initialize system
    with st.spinner("Initializing system..."):
        rag_model, legalbert_generator, claude_api = initialize_rag_system()

    st.header("Chatbot Interface")
    user_input = st.text_input("Ask a legal question:")
    year_input = st.selectbox("Select Year", VALID_YEARS)
    court_type_input = st.selectbox("Select Court Type", VALID_COURT_TYPES)

    if st.button("Submit"):
        if not user_input.strip():
            st.warning("Please enter a valid question.")
        else:
            with st.spinner("Processing your query..."):
                start_time = time.time()
                
                # Get relevant documents
                relevant_docs = rag_model.get_relevant_documents(
                    query=user_input,
                    court_type=court_type_input,
                    year=year_input
                )
                
                if not relevant_docs:
                    st.error(f"No data found for {year_input} - {court_type_input}")
                    return

                # Generate summaries
                for doc in relevant_docs:
                    doc['summary'] = claude_api.summarize(doc['text'])

                # Generate response
                response = legalbert_generator.generate_response(user_input, relevant_docs)
                
                end_time = time.time()

                # Display results
                st.success("Query processed successfully!")
                st.markdown(f"**Query:** {user_input}")

                st.markdown("### Relevant Documents:")
                for i, doc in enumerate(relevant_docs, 1):
                    st.markdown(f"**Document {i}:**")
                    st.markdown(f"*Similarity Score:* {doc['similarity']:.4f}")
                    st.markdown(f"*Summary:* {doc.get('summary', 'No summary available')}")

                    with st.expander(f"View Full Document {i}"):
                        st.markdown(f"*Content:* {doc.get('text', 'No full document available')}")

                st.markdown("### Response:")
                st.write(response)
                st.markdown(f"**Processing Time:** {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()