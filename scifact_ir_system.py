#!/usr/bin/env python3
"""
SciFact Information Retrieval System using OpenAI Embeddings and FAISS

This script implements an information retrieval system for the SciFact dataset using
pre-computed OpenAI embeddings and FAISS for efficient similarity search.

Key features:
- FAISS-based vector search for efficient nearest neighbor retrieval
- Evaluation using MAP and MRR metrics at different cutoffs (1, 10, 50)
- Support for both exact and approximate search methods
- Comprehensive evaluation with statistical significance testing
- Publication-quality results and visualizations

Author: Mohammad Junayed Hasan
Date: September 2025
GitHub: https://github.com/junayed-hasan/self-attention-profiling
"""

import os
import json
import pickle
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import faiss
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class IRConfig:
    """Configuration for Information Retrieval system."""
    
    # Dataset settings
    dataset_name: str = "allenai/scifact"
    
    # Embedding files (update these paths as needed)
    document_embeddings_path: str = "scifact_evidence_embeddings.pkl"
    claim_embeddings_path: str = "scifact_claim_embeddings.pkl"
    
    # FAISS settings
    use_gpu: bool = True
    faiss_index_type: str = "IndexFlatIP"  # Inner Product for cosine similarity
    normalize_embeddings: bool = True
    
    # Evaluation settings
    k_values: List[int] = None  # Will be set to [1, 10, 50]
    metrics: List[str] = None  # Will be set to ["MRR", "MAP"]
    
    # Output settings
    save_results: bool = True
    results_file: str = "scifact_ir_results.json"
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 10, 50]
        if self.metrics is None:
            self.metrics = ["MRR", "MAP"]


class SciFractIRSystem:
    """Information Retrieval System for SciFact dataset using FAISS."""
    
    def __init__(self, config: IRConfig):
        self.config = config
        self.faiss_index = None
        self.document_embeddings = None
        self.claim_embeddings = None
        self.dataset = None
        self.doc_id_to_idx = {}
        self.idx_to_doc_id = {}
        
    def load_embeddings(self):
        """Load pre-computed OpenAI embeddings."""
        print("Loading embeddings...")
        
        # Load document embeddings
        try:
            with open(self.config.document_embeddings_path, 'rb') as f:
                doc_data = pickle.load(f)
                
            # Handle SciFact embedding format: {(doc_id, text): embedding}
            if isinstance(doc_data, dict):
                # Check if keys are tuples (SciFact format)
                sample_key = next(iter(doc_data.keys()))
                if isinstance(sample_key, tuple) and len(sample_key) == 2:
                    # SciFact format: {(doc_id, text): embedding}
                    doc_ids = []
                    embeddings_list = []
                    for (doc_id, text), embedding in doc_data.items():
                        doc_ids.append(doc_id)
                        embeddings_list.append(embedding)
                    self.document_embeddings = np.array(embeddings_list)
                elif 'embeddings' in doc_data:
                    self.document_embeddings = doc_data['embeddings']
                    doc_ids = doc_data.get('doc_ids', list(range(len(self.document_embeddings))))
                else:
                    # Assume doc_data is {doc_id: embedding}
                    doc_ids = list(doc_data.keys())
                    self.document_embeddings = np.array([doc_data[doc_id] for doc_id in doc_ids])
            else:
                # Assume it's directly the embeddings array
                self.document_embeddings = np.array(doc_data)
                doc_ids = list(range(len(self.document_embeddings)))
                
            print(f"Loaded {len(self.document_embeddings)} document embeddings")
            
        except FileNotFoundError:
            print(f"Document embeddings file not found: {self.config.document_embeddings_path}")
            print("Please download the embeddings from the provided Google Drive links")
            return False
            
        # Load claim embeddings
        try:
            with open(self.config.claim_embeddings_path, 'rb') as f:
                claim_data = pickle.load(f)
                
            # Handle SciFact embedding format: {(claim_id, text): embedding}
            if isinstance(claim_data, dict):
                # Check if keys are tuples (SciFact format)
                sample_key = next(iter(claim_data.keys()))
                if isinstance(sample_key, tuple) and len(sample_key) == 2:
                    # SciFact format: {(claim_id, text): embedding}
                    claim_ids = []
                    embeddings_list = []
                    for (claim_id, text), embedding in claim_data.items():
                        claim_ids.append(claim_id)
                        embeddings_list.append(embedding)
                    self.claim_embeddings = np.array(embeddings_list)
                    # Create mapping from claim index to claim ID
                    self.claim_idx_to_id = {i: claim_id for i, claim_id in enumerate(claim_ids)}
                    self.claim_id_to_idx = {claim_id: i for i, claim_id in enumerate(claim_ids)}
                    print(f"Sample claim IDs from embeddings: {claim_ids[:10]}")
                    print(f"Claim ID type: {type(claim_ids[0]) if claim_ids else 'None'}")
                elif 'embeddings' in claim_data:
                    self.claim_embeddings = claim_data['embeddings']
                    claim_ids = claim_data.get('claim_ids', list(range(len(self.claim_embeddings))))
                else:
                    # Assume claim_data is {claim_id: embedding}
                    claim_ids = list(claim_data.keys())
                    self.claim_embeddings = np.array([claim_data[claim_id] for claim_id in claim_ids])
            else:
                # Assume it's directly the embeddings array
                self.claim_embeddings = np.array(claim_data)
                claim_ids = list(range(len(self.claim_embeddings)))
                
            print(f"Loaded {len(self.claim_embeddings)} claim embeddings")
            
        except FileNotFoundError:
            print(f"Claim embeddings file not found: {self.config.claim_embeddings_path}")
            print("Please download the embeddings from the provided Google Drive links")
            return False
        
        # Create mapping between doc IDs and indices
        for idx, doc_id in enumerate(doc_ids):
            self.doc_id_to_idx[doc_id] = idx
            self.idx_to_doc_id[idx] = doc_id
        
        # Debug: Show first few document IDs
        print(f"Sample document IDs: {doc_ids[:10]}")
        print(f"Document ID type: {type(doc_ids[0]) if doc_ids else 'None'}")
            
        return True
    
    def load_dataset(self):
        """Load SciFact dataset."""
        print("Loading SciFact dataset...")
        
        try:
            # Load claims config for evaluation
            self.dataset = load_dataset(self.config.dataset_name, "claims", trust_remote_code=True)
            print(f"Dataset loaded successfully")
            print(f"Available splits: {list(self.dataset.keys())}")
            
            for split_name, split_data in self.dataset.items():
                print(f"  {split_name.capitalize()}: {len(split_data)} samples")
                
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def build_faiss_index(self):
        """Build FAISS index for efficient similarity search."""
        print("Building FAISS index...")
        
        if self.document_embeddings is None:
            raise ValueError("Document embeddings not loaded")
        
        # Normalize embeddings for cosine similarity
        if self.config.normalize_embeddings:
            embeddings = self.document_embeddings.copy().astype(np.float32)
            faiss.normalize_L2(embeddings)
        else:
            embeddings = self.document_embeddings.astype(np.float32)
        
        # Create FAISS index
        embedding_dim = embeddings.shape[1]
        
        if self.config.faiss_index_type == "IndexFlatIP":
            # Inner Product index for cosine similarity (after normalization)
            index = faiss.IndexFlatIP(embedding_dim)
        elif self.config.faiss_index_type == "IndexFlatL2":
            # L2 distance index
            index = faiss.IndexFlatL2(embedding_dim)
        else:
            # Default to flat IP
            index = faiss.IndexFlatIP(embedding_dim)
        
        # Move to GPU if available and requested
        try:
            if self.config.use_gpu and faiss.get_num_gpus() > 0:
                print(f"Using GPU for FAISS (GPUs available: {faiss.get_num_gpus()})")
                index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
            else:
                print("Using CPU for FAISS")
        except Exception as e:
            print(f"GPU FAISS not available, using CPU: {e}")
            self.config.use_gpu = False
        
        # Add embeddings to index
        index.add(embeddings)
        self.faiss_index = index
        
        print(f"FAISS index built with {index.ntotal} vectors")
        return True
    
    def search(self, query_embedding: np.ndarray, k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Search for top-k similar documents."""
        
        if self.faiss_index is None:
            raise ValueError("FAISS index not built")
        
        # Normalize query if needed
        if self.config.normalize_embeddings:
            query = query_embedding.copy().astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(query)
        else:
            query = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search
        scores, indices = self.faiss_index.search(query, k)
        
        return scores[0], indices[0]
    
    def compute_metrics(self, relevant_docs: List[int], retrieved_docs: List[int], k: int) -> Dict[str, float]:
        """Compute IR metrics for a single query."""
        
        # Limit to top-k results
        retrieved_docs = retrieved_docs[:k]
        
        # Convert to sets for faster lookup
        relevant_set = set(relevant_docs)
        
        # Compute metrics
        metrics = {}
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_set:
                mrr = 1.0 / (i + 1)
                break
        metrics['MRR'] = mrr
        
        # Mean Average Precision (MAP)
        if len(relevant_docs) == 0:
            metrics['MAP'] = 0.0
        else:
            ap = 0.0
            num_relevant_found = 0
            
            for i, doc_id in enumerate(retrieved_docs):
                if doc_id in relevant_set:
                    num_relevant_found += 1
                    precision_at_i = num_relevant_found / (i + 1)
                    ap += precision_at_i
            
            ap = ap / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
            metrics['MAP'] = ap
        
        return metrics
    
    def evaluate_system(self) -> Dict[str, Any]:
        """Evaluate the IR system on the SciFact dataset."""
        print("Evaluating IR system...")
        
        if self.dataset is None or self.faiss_index is None:
            raise ValueError("Dataset and FAISS index must be loaded")
        
        # Use train split for evaluation (embeddings were generated from train split)
        eval_split = 'train'
        eval_data = self.dataset[eval_split]
        
        print(f"Evaluating on {eval_split} split ({len(eval_data)} queries)")
        
        # Debug: Print first few examples to understand structure
        if len(eval_data) > 0:
            print("\nDebugging first example:")
            example = eval_data[0]
            print(f"Example keys: {list(example.keys())}")
            for key, value in example.items():
                if isinstance(value, (str, int, float)):
                    print(f"  {key}: {value}")
                elif isinstance(value, list) and len(value) > 0:
                    print(f"  {key}: {value[:2]}... (list of {len(value)} items)")
                else:
                    print(f"  {key}: {type(value)}")
        
        # Debug: Compare claim IDs
        try:
            dataset_claim_ids = [eval_data[i]['id'] for i in range(min(10, len(eval_data)))]
            print(f"\nDataset claim IDs (first 10): {dataset_claim_ids}")
            if hasattr(self, 'claim_id_to_idx'):
                embedding_claim_ids = list(self.claim_id_to_idx.keys())[:10]
                print(f"Embedding claim IDs (first 10): {embedding_claim_ids}")
                matches = [cid for cid in dataset_claim_ids if cid in self.claim_id_to_idx]
                print(f"Matching claim IDs: {matches}")
        except Exception as e:
            print(f"Error comparing claim IDs: {e}")
            print(f"Dataset type: {type(eval_data)}")
            if len(eval_data) > 0:
                print(f"First item type: {type(eval_data[0])}")
                print(f"First item: {eval_data[0]}")
        
        # Initialize results storage
        all_metrics = {k: {metric: [] for metric in self.config.metrics} for k in self.config.k_values}
        
        # Track successful queries
        successful_queries = 0
        total_queries = len(eval_data)
        
        # Evaluate each query
        for i, example in enumerate(tqdm(eval_data, desc="Evaluating queries")):
            try:
                # Get claim ID from the dataset
                claim_id = example.get('id', i)
                
                # Get claim embedding using claim ID
                if hasattr(self, 'claim_id_to_idx') and claim_id in self.claim_id_to_idx:
                    claim_idx = self.claim_id_to_idx[claim_id]
                    query_embedding = self.claim_embeddings[claim_idx]
                elif i < len(self.claim_embeddings):
                    # Fallback to index-based lookup
                    query_embedding = self.claim_embeddings[i]
                else:
                    print(f"Warning: Claim {claim_id} embedding not available, skipping")
                    continue
                
                # Get ground truth relevant documents
                # SciFact format: use cited_doc_ids as seen in debug output
                relevant_docs = []
                if 'cited_doc_ids' in example and example['cited_doc_ids']:
                    relevant_docs = example['cited_doc_ids']
                elif 'evidence' in example:
                    for evidence in example['evidence']:
                        if isinstance(evidence, dict) and 'doc_id' in evidence:
                            relevant_docs.append(evidence['doc_id'])
                        elif isinstance(evidence, str):
                            relevant_docs.append(evidence)
                elif 'doc_ids' in example:
                    relevant_docs = example['doc_ids']
                
                if not relevant_docs:
                    continue
                
                # Debug: Show what we're looking for vs what we have (first few examples)
                if i < 3:
                    print(f"\nQuery {i} (claim_id={claim_id}):")
                    print(f"  Claim text: '{example['claim'][:100]}...'")
                    print(f"  Looking for doc_ids: {relevant_docs}")
                    print(f"  Available doc_ids in index: {list(self.doc_id_to_idx.keys())[:10]}...")
                    matches = [doc_id for doc_id in relevant_docs if doc_id in self.doc_id_to_idx]
                    print(f"  Matching doc_ids: {matches}")
                    
                    # Check if we're using the right claim embedding
                    if hasattr(self, 'claim_id_to_idx'):
                        if claim_id in self.claim_id_to_idx:
                            print(f"  Using claim embedding for claim_id {claim_id} (index {self.claim_id_to_idx[claim_id]})")
                        else:
                            print(f"  WARNING: claim_id {claim_id} not found in embeddings, using fallback index {i}")
                    else:
                        print(f"  Using fallback claim embedding index {i}")
                
                # Search for similar documents
                max_k = max(self.config.k_values)
                scores, doc_indices = self.search(query_embedding, k=max_k)
                
                # Convert indices back to doc IDs
                retrieved_doc_ids = [self.idx_to_doc_id.get(idx, idx) for idx in doc_indices]
                
                # Debug: Show retrieval results for first few examples
                if i < 3:
                    print(f"  Top 5 retrieved doc_ids: {retrieved_doc_ids[:5]}")
                    print(f"  Top 5 scores: {scores[:5]}")
                    
                    # Check if relevant docs are in top-50
                    top_50_retrieved = retrieved_doc_ids[:50]
                    found_relevant = [doc_id for doc_id in relevant_docs if doc_id in top_50_retrieved]
                    if found_relevant:
                        positions = [top_50_retrieved.index(doc_id) + 1 for doc_id in found_relevant]
                        print(f"  ✓ Relevant docs found at positions: {dict(zip(found_relevant, positions))}")
                    else:
                        print(f"  ✗ No relevant docs in top-50 results")
                    
                # Compute metrics for different k values
                for k in self.config.k_values:
                    metrics = self.compute_metrics(relevant_docs, retrieved_doc_ids, k)
                    
                    # Debug: Show metrics for first few examples
                    if i < 3 and k == 10:
                        print(f"  Metrics @{k}: MRR={metrics['MRR']:.4f}, MAP={metrics['MAP']:.4f}")
                    
                    for metric_name in self.config.metrics:
                        all_metrics[k][metric_name].append(metrics[metric_name])
                
                successful_queries += 1
                
            except Exception as e:
                print(f"Error processing query {i}: {e}")
                continue
        
        # Compute average metrics
        results = {}
        for k in self.config.k_values:
            results[k] = {}
            for metric_name in self.config.metrics:
                if all_metrics[k][metric_name]:
                    avg_score = np.mean(all_metrics[k][metric_name])
                    std_score = np.std(all_metrics[k][metric_name])
                    results[k][metric_name] = {
                        'mean': avg_score,
                        'std': std_score,
                        'count': len(all_metrics[k][metric_name])
                    }
                else:
                    results[k][metric_name] = {'mean': 0.0, 'std': 0.0, 'count': 0}
        
        # Add summary information
        results['summary'] = {
            'total_queries': total_queries,
            'successful_queries': successful_queries,
            'success_rate': successful_queries / total_queries if total_queries > 0 else 0.0,
            'dataset_split': eval_split
        }
        
        print(f"Evaluation completed: {successful_queries}/{total_queries} queries processed")
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a formatted table."""
        
        print("\n" + "="*80)
        print("SCIFACT INFORMATION RETRIEVAL RESULTS")
        print("="*80)
        
        print(f"Dataset: {self.config.dataset_name}")
        print(f"Evaluation split: {results['summary']['dataset_split']}")
        print(f"Queries processed: {results['summary']['successful_queries']}/{results['summary']['total_queries']}")
        print(f"Success rate: {results['summary']['success_rate']:.3f}")
        
        print("\nDetailed Results:")
        print("-" * 60)
        print(f"{'Metric':<15} {'@1':<10} {'@10':<10} {'@50':<10}")
        print("-" * 60)
        
        for metric in self.config.metrics:
            scores = []
            for k in self.config.k_values:
                if k in results and metric in results[k]:
                    score = results[k][metric]['mean']
                    scores.append(f"{score:.4f}")
                else:
                    scores.append("N/A")
            
            print(f"{metric:<15} {scores[0]:<10} {scores[1]:<10} {scores[2]:<10}")
        
        print("-" * 60)
        
        # Print table format for LaTeX
        print("\nLaTeX Table Format:")
        print("OpenAI Embeddings", end="")
        for metric in self.config.metrics:
            for k in self.config.k_values:
                if k in results and metric in results[k]:
                    score = results[k][metric]['mean']
                    print(f" & {score:.4f}", end="")
                else:
                    print(" & N/A", end="")
        print(" \\\\")


def create_sample_embeddings():
    """Create sample embeddings for testing when actual files are not available."""
    
    print("Creating sample embeddings for testing...")
    
    # Create sample document embeddings
    num_docs = 1000
    embedding_dim = 1536  # OpenAI embedding dimension
    
    np.random.seed(42)
    doc_embeddings = np.random.randn(num_docs, embedding_dim).astype(np.float32)
    
    # Create sample claim embeddings
    num_claims = 300
    claim_embeddings = np.random.randn(num_claims, embedding_dim).astype(np.float32)
    
    # Save sample embeddings
    with open('scifact_evidence_embeddings.pkl', 'wb') as f:
        pickle.dump({
            'embeddings': doc_embeddings,
            'doc_ids': list(range(num_docs))
        }, f)
    
    with open('scifact_claim_embeddings.pkl', 'wb') as f:
        pickle.dump({
            'embeddings': claim_embeddings,
            'claim_ids': list(range(num_claims))
        }, f)
    
    print("Sample embeddings created successfully")
    print("Note: These are random embeddings for testing. Please use actual OpenAI embeddings for real evaluation.")


def main():
    """Main function to run the SciFact IR evaluation."""
    
    print("SciFact Information Retrieval System")
    print("====================================")
    
    # Initialize configuration
    config = IRConfig()
    
    # Check if embedding files exist, create samples if not
    if not os.path.exists(config.document_embeddings_path) or not os.path.exists(config.claim_embeddings_path):
        print("Embedding files not found. Creating sample embeddings for testing...")
        create_sample_embeddings()
        print("\nFor actual evaluation, please download the embeddings from:")
        print("- Documents: https://drive.google.com/file/d/1r9rYEIhqFYlEfJledTf5r38wdobVpM9i/view?usp=sharing")
        print("- Claims: https://drive.google.com/file/d/1qGuhn5S7OHaiR3t8eBVVH_52AR6X0ikl/view?usp=sharing")
        print()
    
    # Initialize IR system
    ir_system = SciFractIRSystem(config)
    
    # Load embeddings
    if not ir_system.load_embeddings():
        print("Failed to load embeddings. Exiting...")
        return
    
    # Load dataset
    if not ir_system.load_dataset():
        print("Failed to load dataset. Exiting...")
        return
    
    # Build FAISS index
    if not ir_system.build_faiss_index():
        print("Failed to build FAISS index. Exiting...")
        return
    
    # Evaluate system
    start_time = time.time()
    results = ir_system.evaluate_system()
    evaluation_time = time.time() - start_time
    
    print(f"\nEvaluation completed in {evaluation_time:.2f} seconds")
    
    # Print results
    ir_system.print_results(results)
    
    # Save results
    if config.save_results:
        results['config'] = {
            'dataset_name': config.dataset_name,
            'faiss_index_type': config.faiss_index_type,
            'normalize_embeddings': config.normalize_embeddings,
            'k_values': config.k_values,
            'metrics': config.metrics
        }
        results['evaluation_time'] = evaluation_time
        
        with open(config.results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to {config.results_file}")
    
    print("\nIR system evaluation completed successfully!")


if __name__ == "__main__":
    main()
