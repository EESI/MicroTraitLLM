"""
RAG-Optimized Article Prompt Compression System
Filters semantic chunks based on relevance to user queries for efficient retrieval
"""

import re
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass

# Required installations:
# pip install sentence-transformers scikit-learn nltk

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class CompressionConfig:
    """Configuration for compression strategies"""
    similarity_threshold: float = 0.75  # For semantic chunking
    max_chunk_size: int = 500  # Maximum tokens per chunk
    compression_ratio: float = 0.3  # Target compression (0.3 = keep 30%)
    min_sentence_length: int = 20  # Minimum characters for a sentence
    query_relevance_threshold: float = 0.5  # NEW: Minimum similarity to query
    top_k_chunks: Optional[int] = None  # NEW: Keep only top K chunks (None = use threshold)
    max_output_tokens: Optional[int] = None  # NEW: Hard limit on output tokens (e.g., 3000)


class RAGArticleCompressor:
    """Article compression system optimized for RAG with query-based filtering"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", config: Optional[CompressionConfig] = None):
        """
        Initialize compressor with embedding model
        
        Args:
            model_name: HuggingFace model name
                       'all-MiniLM-L6-v2' (fast, good for most cases)
                       'all-mpnet-base-v2' (better quality, slower)
                       'multi-qa-MiniLM-L6-cos-v1' (optimized for Q&A retrieval)
            config: Compression configuration
        """
        #print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.config = config or CompressionConfig()
        #print("Model loaded successfully!")
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count for text
        Uses approximation: ~1.3 tokens per word for English text
        This is a conservative estimate that works well for most LLM tokenizers
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        words = len(text.split())
        # Conservative estimate: 1.3 tokens per word
        # (GPT models average ~1.3, Claude similar)
        return int(words * 1.3)
    
    @staticmethod
    def estimate_tokens_precise(text: str) -> int:
        """
        More precise token estimation using character count
        Rule of thumb: ~4 characters per token for English
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Remove excessive whitespace for better estimate
        text = re.sub(r'\s+', ' ', text).strip()
        chars = len(text)
        # 4 characters per token is typical for English
        return int(chars / 4)
    
    def check_token_limit(self, text: str, method: str = "word") -> Dict:
        """
        Check if text exceeds token limits
        
        Args:
            text: Input text
            method: 'word' (faster, ~1.3 tokens/word) or 'char' (more precise, 4 chars/token)
            
        Returns:
            Dict with token count and limit status
        """
        if method == "word":
            tokens = self.estimate_tokens(text)
        else:
            tokens = self.estimate_tokens_precise(text)
        
        result = {
            'estimated_tokens': tokens,
            'max_limit': self.config.max_output_tokens,
            'within_limit': True
        }
        
        if self.config.max_output_tokens:
            result['within_limit'] = tokens <= self.config.max_output_tokens
            result['tokens_over'] = max(0, tokens - self.config.max_output_tokens)
            result['percentage_of_limit'] = (tokens / self.config.max_output_tokens) * 100
        
        return result
    
    def preprocess_text(self, text: str) -> List[str]:
        """Clean and split text into sentences"""
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = nltk.sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= self.config.min_sentence_length]
        return sentences
    
    def semantic_chunking(self, text: str, method: str = "average") -> List[Dict]:
        """
        Pure semantic chunking using cosine similarity between sentence embeddings
        
        Args:
            method: 'consecutive', 'average', or 'centroid'
        """
        sentences = self.preprocess_text(text)
        
        if not sentences:
            return []
        
        print(f"Encoding {len(sentences)} sentences...")
        embeddings = self.model.encode(sentences, show_progress_bar=False)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_embeddings = [embeddings[0]]
        similarities = []
        
        for i in range(1, len(sentences)):
            if method == "consecutive":
                similarity = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
            elif method == "average":
                chunk_avg = np.mean(current_embeddings, axis=0).reshape(1, -1)
                similarity = cosine_similarity(chunk_avg, [embeddings[i]])[0][0]
            elif method == "centroid":
                chunk_centroid = np.mean(current_embeddings, axis=0)
                chunk_centroid = chunk_centroid / np.linalg.norm(chunk_centroid)
                similarity = cosine_similarity([chunk_centroid], [embeddings[i]])[0][0]
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if similarity >= self.config.similarity_threshold:
                current_chunk.append(sentences[i])
                current_embeddings.append(embeddings[i])
                similarities.append(similarity)
            else:
                avg_sim = np.mean(similarities) if similarities else 1.0
                chunk_text = ' '.join(current_chunk)
                chunk_embedding = np.mean(current_embeddings, axis=0)
                chunks.append({
                    'text': chunk_text,
                    'sentence_count': len(current_chunk),
                    'avg_similarity': float(avg_sim),
                    'token_count': len(chunk_text.split()),
                    'embedding': chunk_embedding  # Store for query matching
                })
                current_chunk = [sentences[i]]
                current_embeddings = [embeddings[i]]
                similarities = []
        
        # Add final chunk
        if current_chunk:
            avg_sim = np.mean(similarities) if similarities else 1.0
            chunk_text = ' '.join(current_chunk)
            chunk_embedding = np.mean(current_embeddings, axis=0)
            chunks.append({
                'text': chunk_text,
                'sentence_count': len(current_chunk),
                'avg_similarity': float(avg_sim),
                'token_count': len(chunk_text.split()),
                'embedding': chunk_embedding
            })
        
        #print(f"Created {len(chunks)} semantic chunks using '{method}' method")
        return chunks
    
    def query_filter_chunks(self, chunks: List[Dict], query: str, 
                           return_scores: bool = True) -> List[Dict]:
        """
        Filter chunks based on semantic similarity to query
        If max_output_tokens is set, ensures output stays within limit
        
        Args:
            chunks: List of chunk dictionaries (must have 'embedding' key)
            query: User query/question
            return_scores: Include relevance scores in output
        
        Returns:
            Filtered list of chunks, sorted by relevance (highest first)
        """
        if not chunks:
            return []
        
        # Encode query
        #print(f"Encoding query: '{query[:100]}...'")
        query_embedding = self.model.encode([query], show_progress_bar=False)[0]
        
        # Calculate similarity scores
        chunk_embeddings = np.array([c['embedding'] for c in chunks])
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        
        # Add scores to chunks
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            chunk_copy['query_relevance'] = float(similarities[i])
            scored_chunks.append(chunk_copy)
        
        # Sort by relevance (highest first)
        scored_chunks.sort(key=lambda x: x['query_relevance'], reverse=True)
        
        # Filter based on threshold or top-k
        if self.config.top_k_chunks is not None:
            filtered = scored_chunks[:self.config.top_k_chunks]
            #print(f"Kept top {len(filtered)} chunks (top_k={self.config.top_k_chunks})")
        else:
            filtered = [c for c in scored_chunks 
                       if c['query_relevance'] >= self.config.query_relevance_threshold]
            #print(f"Kept {len(filtered)}/{len(chunks)} chunks above threshold "
            #      f"({self.config.query_relevance_threshold})")
        
        # NEW: Enforce token limit if specified
        if self.config.max_output_tokens and filtered:
            filtered = self._enforce_token_limit(filtered)
        
        # Optionally remove embeddings and scores for cleaner output
        if not return_scores:
            for chunk in filtered:
                chunk.pop('embedding', None)
                chunk.pop('query_relevance', None)
        
        return filtered
    
    def _enforce_token_limit(self, chunks: List[Dict]) -> List[Dict]:
        """
        Internal method to trim chunks to fit within max_output_tokens
        Removes lowest-relevance chunks until within limit
        
        Args:
            chunks: List of chunks sorted by relevance (highest first)
            
        Returns:
            Trimmed list of chunks
        """
        combined_text = ' '.join([c['text'] for c in chunks])
        current_tokens = self.estimate_tokens(combined_text)
        
        if current_tokens <= self.config.max_output_tokens:
            #print(f"Token count: {current_tokens}/{self.config.max_output_tokens} ✓")
            return chunks
        
        #print(f"Token count: {current_tokens}/{self.config.max_output_tokens} - trimming...")
        
        # Remove chunks from the end (lowest relevance) until within limit
        trimmed = chunks.copy()
        while trimmed and current_tokens > self.config.max_output_tokens:
            removed = trimmed.pop()
            combined_text = ' '.join([c['text'] for c in trimmed])
            current_tokens = self.estimate_tokens(combined_text)
            #print(f"  Removed chunk (relevance: {removed['query_relevance']:.3f}), "
            #      f"now at {current_tokens} tokens")
        
        #print(f"Final token count: {current_tokens}/{self.config.max_output_tokens} ✓")
        return trimmed
    
    def rag_compress(self, text: str, query: str, 
                     chunking_method: str = "average",
                     return_metadata: bool = True) -> Union[str, Dict]:
        """
        Main RAG compression pipeline: chunk -> filter by query -> combine
        Automatically enforces max_output_tokens if configured
        
        Args:
            text: Article text to compress
            query: User query/question
            chunking_method: Semantic chunking method
            return_metadata: Return detailed metadata or just compressed text
        
        Returns:
            Compressed text (str) or dict with text and metadata
        """
        # Step 1: Create semantic chunks
        chunks = self.semantic_chunking(text, method=chunking_method)
        
        if not chunks:
            return "" if not return_metadata else {
                'compressed_text': "",
                'chunks_kept': 0,
                'total_chunks': 0,
                'estimated_tokens': 0
            }
        
        # Step 2: Filter by query relevance (includes token limit enforcement)
        relevant_chunks = self.query_filter_chunks(chunks, query, return_scores=True)
        
        if not relevant_chunks:
            print("Warning: No chunks met relevance threshold!")
            if not return_metadata:
                return ""
            return {
                'compressed_text': "",
                'chunks_kept': 0,
                'total_chunks': len(chunks),
                'relevance_scores': [],
                'estimated_tokens': 0
            }
        
        # Step 3: Combine relevant chunks
        compressed_text = ' '.join([c['text'] for c in relevant_chunks])
        
        # Calculate token estimate
        token_estimate = self.estimate_tokens(compressed_text)
        
        if not return_metadata:
            return compressed_text
        
        result = {
            'compressed_text': compressed_text,
            'chunks_kept': len(relevant_chunks),
            'total_chunks': len(chunks),
            'compression_ratio': len(compressed_text) / len(text),
            'relevance_scores': [c['query_relevance'] for c in relevant_chunks],
            'avg_relevance': np.mean([c['query_relevance'] for c in relevant_chunks]),
            'chunks': relevant_chunks,
            'original_length': len(text),
            'compressed_length': len(compressed_text),
            'estimated_tokens': token_estimate,
            'token_limit_status': self.check_token_limit(compressed_text)
        }
        
        return result
    
    def batch_rag_compress(self, articles: List[Dict], query: str) -> List[Dict]:
        """
        Process multiple articles for a single query (typical RAG scenario)
        
        Args:
            articles: List of dicts with 'text' and optional 'meta' keys
            query: User query/question
        
        Returns:
            List of compressed articles with relevance scores
        """
        results = []
        
        for i, article in enumerate(articles):
            print(f"\nProcessing article {i+1}/{len(articles)}...")
            text = article.get('text', '')
            meta = article.get('meta', {})
            
            result = self.rag_compress(text, query, return_metadata=True)
            result['meta'] = meta
            result['article_index'] = i
            
            results.append(result)
        
        # Sort by average relevance
        results.sort(key=lambda x: x['avg_relevance'], reverse=True)
        
        print(f"\nProcessed {len(articles)} articles")
        return results
    
    def hybrid_rag_compress(self, text: str, query: str) -> Dict:
        """
        Hybrid approach: chunk -> filter by query -> extractive compress each chunk
        More aggressive compression for very long documents
        """
        # Step 1: Create and filter chunks
        chunks = self.semantic_chunking(text)
        relevant_chunks = self.query_filter_chunks(chunks, query, return_scores=True)
        
        if not relevant_chunks:
            return {
                'compressed_text': "",
                'chunks_kept': 0,
                'total_chunks': len(chunks)
            }
        
        # Step 2: Apply extractive compression to each relevant chunk
        compressed_chunks = []
        for chunk in relevant_chunks:
            chunk_sentences = self.preprocess_text(chunk['text'])
            
            if len(chunk_sentences) <= 2:
                compressed_chunks.append({
                    'text': chunk['text'],
                    'query_relevance': chunk['query_relevance']
                })
            else:
                # Compress chunk
                embeddings = self.model.encode(chunk_sentences, show_progress_bar=False)
                similarity_matrix = cosine_similarity(embeddings)
                centrality_scores = similarity_matrix.mean(axis=1)
                
                num_keep = max(1, int(len(chunk_sentences) * self.config.compression_ratio))
                top_indices = sorted(np.argsort(centrality_scores)[-num_keep:])
                
                compressed = ' '.join([chunk_sentences[i] for i in top_indices])
                compressed_chunks.append({
                    'text': compressed,
                    'query_relevance': chunk['query_relevance']
                })
        
        final_text = ' '.join([c['text'] for c in compressed_chunks])
        
        return {
            'compressed_text': final_text,
            'chunks_kept': len(compressed_chunks),
            'total_chunks': len(chunks),
            'compression_ratio': len(final_text) / len(text),
            'avg_relevance': np.mean([c['query_relevance'] for c in compressed_chunks])
        }


def main():
    """Example usage for RAG applications"""
    
    # Sample articles
    articles = [
        {
            "text": """
            Artificial intelligence has transformed the technology landscape in unprecedented ways. 
            Machine learning algorithms now power everything from recommendation systems to autonomous vehicles.
            The field of natural language processing has seen particularly rapid advancement in recent years.
            Large language models can now generate human-like text with remarkable coherence and accuracy.
            However, these models require significant computational resources to train and deploy.
            The environmental impact of training large AI models has become a topic of concern.
            Researchers are exploring more efficient training methods to reduce energy consumption.
            Transfer learning allows models to leverage knowledge from previous tasks.
            This approach significantly reduces the data and compute requirements for new applications.
            Edge computing brings AI capabilities directly to devices, reducing latency and privacy concerns.
            """ * 20,  # Make it longer to test token limits
            "meta": {
                "title": "AI Technology Overview",
                "source": "Tech Journal"
            }
        },
        {
            "text": """
            Climate change continues to accelerate with global temperatures rising steadily.
            The Arctic ice sheets are melting at unprecedented rates.
            Scientists warn that we have a narrow window to prevent catastrophic warming.
            Renewable energy adoption is growing but not fast enough to meet climate goals.
            Solar and wind power have become cost-competitive with fossil fuels.
            Carbon capture technology shows promise but faces scalability challenges.
            International cooperation is essential to address this global crisis.
            Many countries have committed to net-zero emissions by 2050.
            Individual actions like reducing consumption can make a meaningful difference.
            The transition to a sustainable economy requires systemic changes.
            """,
            "meta": {
                "title": "Climate Change Report",
                "source": "Environmental Science"
            }
        }
    ]
    
    # Initialize compressor for RAG with token limit
    compressor = RAGArticleCompressor(
        model_name="all-MiniLM-L6-v2",
        config=CompressionConfig(
            similarity_threshold=0.75,
            query_relevance_threshold=0.4,
            compression_ratio=0.5,
            top_k_chunks=None,
            max_output_tokens=3000  # NEW: Hard limit at 3k tokens
        )
    )
    
    print("="*70)
    print("RAG COMPRESSION DEMO WITH TOKEN LIMITS")
    print("="*70)
    
    # Example 1: Token estimation
    print("\n" + "="*70)
    print("EXAMPLE 1: Token Estimation")
    print("="*70)
    
    sample_text = articles[0]['text'][:500]
    print(f"\nSample text: {sample_text[:100]}...")
    
    token_check = compressor.check_token_limit(sample_text)
    print(f"\nEstimated tokens (word method): {token_check['estimated_tokens']}")
    
    token_check_char = compressor.check_token_limit(sample_text, method="char")
    print(f"Estimated tokens (char method): {token_check_char['estimated_tokens']}")
    
    # Example 2: Compression with automatic token limiting
    print("\n" + "="*70)
    print("EXAMPLE 2: Compression with Token Limit Enforcement")
    print("="*70)
    
    query1 = "How does transfer learning work in machine learning?"
    print(f"\nQuery: {query1}")
    print(f"Max output tokens: {compressor.config.max_output_tokens}")
    print(f"\nOriginal text length: {len(articles[0]['text'])} chars")
    print(f"Original estimated tokens: {compressor.estimate_tokens(articles[0]['text'])}")
    
    result = compressor.rag_compress(articles[0]['text'], query1, return_metadata=True)
    
    print(f"\n--- RESULTS ---")
    print(f"Chunks kept: {result['chunks_kept']}/{result['total_chunks']}")
    print(f"Average relevance: {result['avg_relevance']:.3f}")
    print(f"Compression ratio: {result['compression_ratio']:.2%}")
    print(f"Estimated tokens: {result['estimated_tokens']}")
    print(f"Within limit: {result['token_limit_status']['within_limit']}")
    if result['token_limit_status']['max_limit']:
        print(f"Percentage of limit: {result['token_limit_status']['percentage_of_limit']:.1f}%")
    print(f"\nCompressed text preview:\n{result['compressed_text'][:300]}...")
    
    # Example 3: Without token limit
    print("\n" + "="*70)
    print("EXAMPLE 3: Compression WITHOUT Token Limit")
    print("="*70)
    
    compressor_no_limit = RAGArticleCompressor(
        model_name="all-MiniLM-L6-v2",
        config=CompressionConfig(
            similarity_threshold=0.75,
            query_relevance_threshold=0.4,
            max_output_tokens=None  # No limit
        )
    )
    
    print(f"\nQuery: {query1}")
    print("No token limit set")
    
    result_no_limit = compressor_no_limit.rag_compress(
        articles[0]['text'], query1, return_metadata=True
    )
    
    print(f"\n--- RESULTS ---")
    print(f"Chunks kept: {result_no_limit['chunks_kept']}")
    print(f"Estimated tokens: {result_no_limit['estimated_tokens']}")
    print(f"(Would exceed 3k limit by {result_no_limit['estimated_tokens'] - 3000} tokens)")
    
    # Example 4: Batch processing with token limits
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Processing with Token Limits")
    print("="*70)
    
    query3 = "What are the energy costs of AI models?"
    print(f"\nQuery: {query3}")
    print(f"Max tokens per article: {compressor.config.max_output_tokens}")
    
    batch_results = compressor.batch_rag_compress(articles, query3)
    
    print(f"\nResults ranked by relevance:")
    for i, result in enumerate(batch_results, 1):
        print(f"\n{i}. {result['meta']['title']}")
        print(f"   Relevance: {result['avg_relevance']:.3f}")
        print(f"   Chunks: {result['chunks_kept']}/{result['total_chunks']}")
        print(f"   Tokens: {result['estimated_tokens']}/{compressor.config.max_output_tokens}")
        print(f"   Preview: {result['compressed_text'][:100]}...")
    
    # Example 5: Very aggressive token limit
    print("\n" + "="*70)
    print("EXAMPLE 5: Very Aggressive Token Limit (500 tokens)")
    print("="*70)
    
    compressor_aggressive = RAGArticleCompressor(
        model_name="all-MiniLM-L6-v2",
        config=CompressionConfig(
            similarity_threshold=0.75,
            query_relevance_threshold=0.3,  # Lower threshold
            max_output_tokens=500  # Very strict limit
        )
    )
    
    query5 = "Tell me about AI and machine learning"
    print(f"\nQuery: {query5}")
    print(f"Aggressive limit: 500 tokens")
    
    result5 = compressor_aggressive.rag_compress(
        articles[0]['text'], query5, return_metadata=True
    )
    
    print(f"\n--- RESULTS ---")
    print(f"Chunks kept: {result5['chunks_kept']}/{result5['total_chunks']}")
    print(f"Estimated tokens: {result5['estimated_tokens']}/500")
    print(f"Average relevance: {result5['avg_relevance']:.3f}")
    print(f"\nCompressed text:\n{result5['compressed_text']}")


if __name__ == "__main__":
    main()
