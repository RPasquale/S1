"""
Test the trained CFA embedding model
"""
import os
import sys
import logging
from sentence_transformers import SentenceTransformer, util
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # First check if we have the trained model
    model_path = "./trained_models/cfa_models"
    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}")
        return
    
    logger.info(f"Loading trained model from {model_path}")
    model = SentenceTransformer(model_path)
    
    # Some test CFA-related queries
    queries = [
        "What are portfolio management techniques?",
        "Explain financial statement analysis",
        "What is the Capital Asset Pricing Model?",
        "How to evaluate fixed income securities",
        "What are ethical considerations in investment management?"
    ]
    
    # Examples from CFA documents - these are just placeholders 
    # usually you would get these from documents, we're just testing the model here
    corpus = [
        "Portfolio management involves the allocation of assets to meet investment objectives while managing risk.",
        "Financial statement analysis is the process of evaluating a company's financial statements to assess performance and value.",
        "The Capital Asset Pricing Model (CAPM) relates expected return to systematic risk through beta coefficient.",
        "Fixed income securities evaluation requires assessment of yield, duration, convexity and credit risk metrics.",
        "Ethical considerations in investment management include fiduciary duty, conflicts of interest, and client preference alignment."
    ]
    
    # Encode queries and corpus
    logger.info("Encoding queries and corpus")
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    
    # Find the closest 3 sentences for each query
    top_k = min(3, len(corpus))
    
    # Compute cosine similarity
    logger.info("Computing similarity scores")
    for i, query in enumerate(queries):
        logger.info(f"\nQuery: {query}")
        
        # Get similarity scores
        cos_scores = util.cos_sim(query_embeddings[i], corpus_embeddings)[0]
        
        # Sort results by score
        top_results = torch.topk(cos_scores, k=top_k)
        
        logger.info(f"Top {top_k} most similar sentences:")
        for score, idx in zip(top_results[0], top_results[1]):
            logger.info(f"Score: {score:.4f}\tText: {corpus[idx]}")

if __name__ == "__main__":
    main()
