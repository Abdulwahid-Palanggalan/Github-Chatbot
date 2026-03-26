import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load a small, highly efficient local embedding model
# This runs 100% offline on your CPU and requires no API keys
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_kb(filepath="github_kb.json"):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_embedding(text):
    # Generates the vector embedding locally
    return model.encode(text)

def prepare_kb_embeddings(kb):
    for entry in kb:
        # Build a rich search string from the feature name, description, and keywords
        search_text = f"{entry['feature']} {entry['description']} {' '.join(entry['keywords'])}"
        entry['embedding'] = get_embedding(search_text)
    return kb

def search_knowledge_base(user_question, kb_indexed, threshold=0.45):
    # Determine the intent of the user's natural language question
    question_embedding = get_embedding(user_question)
    
    similarities = []
    for entry in kb_indexed:
        score = cosine_similarity(
            [question_embedding], 
            [entry['embedding']]
        )[0][0]
        similarities.append((score, entry))
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    if not similarities:
        return None
        
    best_score, best_match = similarities[0]
    
    # If the closest match is below the confidence threshold, assume it's missing from KB
    if best_score >= threshold:
        return best_match
    return None

def generate_response(user_question, matched_entry):
    if not matched_entry:
        return "Sorry, I don't have information on that. Please check GitHub Docs: [https://docs.github.com/](https://docs.github.com/)"
    
    # We construct the response deterministically directly from the JSON
    # Since it's a structural knowledge base, we don't need a generative LLM 
    # to format a simple step-by-step list!
    
    response = f"### {matched_entry['feature']}\n\n"
    response += f"Here are the instructions to help you:\n"
    
    for i, step in enumerate(matched_entry['steps'], start=1):
        response += f"{i}. {step}\n"
    
    response += f"\n**Official Documentation**: [Link]({matched_entry['url']})"
    
    return response
