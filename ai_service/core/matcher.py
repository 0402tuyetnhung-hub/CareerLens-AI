import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class SemanticMatcher:
    def __init__(self, model_name='keepitreal/vietnamese-sbert'):
        # using a multilingual or vietnamese specific model
        # For fast local testing, we can use 'paraphrase-multilingual-MiniLM-L12-v2'
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.index = None
        self.jobs = []

    def load_jobs(self, jobs_path):
        with open(jobs_path, 'r', encoding='utf-8') as f:
            self.jobs = json.load(f)
            
    def build_index(self):
        if not self.jobs:
            return
        
        # Combine job title and description for better context
        job_texts = []
        for job in self.jobs:
            text = f"{job.get('job_title', '')}. {job.get('requirements_text', '')}"
            job_texts.append(text)
            
        embeddings = self.model.encode(job_texts, convert_to_numpy=True)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
    def match_cv(self, cv_text, top_k=3):
        if not self.index:
            raise ValueError("Index not built. Call build_index() first.")
            
        cv_vector = self.model.encode([cv_text], convert_to_numpy=True)
        distances, indices = self.index.search(cv_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            job = self.jobs[idx]
            # Convert L2 distance to a simple similarity score (0 to 1) for demo
            score = 1 / (1 + distances[0][i])
            results.append({
                'job_title': job.get('job_title'),
                'company': job.get('company', 'Unknown'),
                'similarity_score': round(float(score), 4),
                'requirements': job.get('requirements_text', '')[:200] + '...'
            })
            
        return results
