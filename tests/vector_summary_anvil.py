import os
from openai import OpenAI
from ollama import chat
from sentence_transformers import SentenceTransformer, util
import numpy as np
from nltk.tokenize import sent_tokenize
from src.citations import APA_citation, MLA_citation, NLM_citation
import time
# Initialize model once (reuse it, don't recreate each time!)
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
def embedder(chunk):
    """
    Embed text chunks for semantic search.
    Returns normalized embeddings ready for cosine similarity.
    """
    return MODEL.encode(chunk, normalize_embeddings=True)


def search(query, data_source, k):
    """
    Search function to find top k similar chunks to the query.
    
    Args:
        query: Search query string
        data_source: List of tuples [(embedding, chunk, metadata), ...]
                    where metadata is a dict with article info
        k: Number of results to return
    
    Returns:
        List of tuples [(similarity_score, chunk, metadata), ...] sorted by relevance
    """
    # Get query embedding
    query_embedding = embedder(query)
    
    # Check for dimension consistency
    embeddings = []
    chunks = []
    metadatas = []
    expected_dim = 384  # Dimension for all-MiniLM-L6-v2
    
    for idx, item in enumerate(data_source):
        # Handle both old format (emb, chunk) and new format (emb, chunk, metadata)
        if len(item) == 2:
            emb, chunk = item
            metadata = {}
        else:
            emb, chunk, metadata = item
            
        emb_array = np.array(emb).flatten()
        
        if emb_array.shape[0] != expected_dim:
            raise ValueError(
                f"Embedding at index {idx} has {emb_array.shape[0]} dimensions, "
                f"expected {expected_dim}. You have mixed embeddings from different models! "
            )
        
        embeddings.append(emb_array)
        chunks.append(chunk)
        metadatas.append(metadata)
    
    embeddings = np.vstack(embeddings)
    
    # Compute cosine similarities (vectorized - fast!)
    similarities = util.cos_sim(query_embedding, embeddings)[0]
    
    # Get top k results (ensure k doesn't exceed number of chunks)
    k = min(k, len(chunks))
    top_k_indices = similarities.argsort(descending=True)[:k].tolist()
    top_k = [(similarities[i].item(), chunks[i], metadatas[i]) for i in top_k_indices]
    
    return top_k
    

def prepare_article_data(articles,citation_format):
    """
    Prepare articles with metadata for vector space creation from extractor output.
    
    Args:
        articles: List of dicts from PMCArticleExtractor.extract_articles():
                  [
                      {
                          'text': "article text...",
                          'meta': {
                              'title': '...',
                              'authors': [...],
                              'journal': '...',
                              'publication_date': '...',
                              'volume': '...',
                              'issue': '...',
                              'first_page': '...',
                              'pages': '...',
                              'doi': '...'
                          }
                      }
                  ]
    
    Returns:
        chunks: List of text chunks (sentences)
        metadatas: List of corresponding metadata dicts
    """
    chunks = []
    metadatas = []
    citations = []
    for article in articles:
        # Extract metadata and text
        article_text = article.get('text', '')
        metadata = article.get('meta', {})
        
        # Create metadata dict for this article
        article_metadata = {
            'title': metadata.get('title', 'Unknown'),
            'authors': metadata.get('authors', []),
            'journal': metadata.get('journal', ''),
            'publication_date': metadata.get('publication_date', ''),
            'volume': metadata.get('volume', ''),
            'issue': metadata.get('issue', ''),
            'first_page': metadata.get('first_page', ''),
            'pages': metadata.get('pages', ''),
            'doi': metadata.get('doi', ''),
        }
        if citation_format == 'APA':
            citation = APA_citation(article_metadata)
        elif citation_format == 'MLA':
            citation = MLA_citation(article_metadata)
        elif citation_format == 'NLM':
            citation = NLM_citation(article_metadata)   
        # Split text into sentences
        citations.append(citation)
        if article_text.strip():
            sentences = sent_tokenize(article_text)
            for sentence in sentences:
                if sentence.strip():  # Skip empty sentences
                    chunks.append(sentence)
                    metadatas.append(article_metadata.copy())
    
    return chunks, metadatas, citations


def vector_space_creation(json_data, citation_format):
    """
    Create vector space from JSON data.
    
    Args:
        json_data: Dict loaded from JSON file
    
    Returns:
        data_source: List of tuples [(embedding, chunk, metadata), ...]
    """
    # Prepare data
    chunks, metadatas, citations = prepare_article_data(json_data, citation_format)
    
    # Create embeddings
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = MODEL.encode(chunks, normalize_embeddings=True, show_progress_bar=True)
    
    # Create data source with metadata included
    data_source = list(zip(embeddings, chunks, metadatas))
    
    return data_source, citations

def vector_space_search(question, data_source, top_k):
    # Search
    top_k_results = search(question, data_source, top_k)
    
    # Format results
    results = []
    for similarity, chunk, metadata in top_k_results:
        results.append({
            'text': chunk,
            'similarity': similarity,
            'citation': metadata
        })
    
    return results

def vector_summary(results, user_question_prompt,model,citation_format,temperature,citation,trait_info,api_key):
    # Function to generate a summary and citation for a given article URL using requested model
    # Define the system context and examples
    Context = f"""
    Assume you are a helpful researcher with no memory. You are tasked with answering questions based solely on the information provided to you. This includes article summaries and grades for how well the article answers the user’s question. The entries provided will be formatted in a Python list, with each entry in the list containing the following format:
    - An article summary
    - A grade indicating the relevance of the article to the question
    - The citation of the article in {citation_format}

    **Important Instructions:**
    1. **Do not use any information outside of what is provided in the current task.** You are not allowed to use any knowledge beyond the articles provided to you.
    2. **Do not invent or hallucinate citations.** Only use citations explicitly included in the provided articles.
    3. **Cite all information correctly** using the {citation_format} provided. 
    - If information comes from an article, cite it in **in-text citation format** as specified in {citation_format} (i.e., using the article citation and not the grade).
    4. **Do not include the grade in your response**. The grade indicates the relevance of the article but should not appear in the final response. Only include the article citation when referencing data.
    5. **If you cannot find a citation for a particular piece of information in the provided data, leave that piece out of your answer.** Do not attempt to fabricate or invent a citation.

    Your response should be a detailed paragraph of 5-10 sentences answering the user’s question, **using only the articles**. You should prioritize the articles with the highest grade. If you cannot find any relevant data to answer the question, clearly state that you do not have enough information to answer.

    Please ensure your response is fully based on the data presented to you in this session and **does not include any references that are not directly provided.**
    """

    EX1 = 'Question: What are some bacterial strains associated with ear infections?\n'
    
    A1 = """Ear infections, also known as otitis media, can be caused by various bacterial strains. Some common bacterial strains associated with ear infections include Streptococcus pneumoniae, Haemophilus influenzae, and Moraxella catarrhalis (Schilder et al. 2016). These bacteria are often found in the upper respiratory tract and can migrate to the middle ear, leading to infection. Streptococcus pneumoniae is one of the most common bacterial pathogens causing ear infections, particularly in children. Moraxella catarrhalis is also known to be involved in ear infections, particularly in cases of chronic otitis media. Understanding the bacterial strains associated with ear infections is crucial for appropriate diagnosis and treatment strategies.\n

    """
    # Extract API key if provided

    # Prepare messages for the chat models
    all_messages = [
                        {"role": "system", "content": Context},
                        {"role": "user", "content": EX1},
                        {"role": "assistant", "content": A1},
                        {"role": "user", "content": f"The information to summarize is: {results}"},
                        {"role": "user", "content": f"The metadata to make the in-text citations in {citation_format} citation is {citation}"},
                        {"role": "user", "content": f"The trait information to use from metaTraits is {trait_info}"},
                        {"role": "user", "content": f"The user question is: {user_question_prompt}"}
                        ]
            
            # Call the appropriate model
    if model == "ChatGPT-4o-mini":
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", api_key))
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=all_messages,
                    temperature=temperature,
                )
                summary_text = response.choices[0].message.content
                return summary_text, citation

    elif model == "llama-3.3-70b-versatile":
                client = OpenAI(base_url = "https://api.groq.com/openai/v1",api_key=os.environ.get("GROQ_API_KEY", api_key))
                response = client.chat.completions.create(
                    model=model,
                    messages=all_messages,
                    temperature=temperature,
                    )
                summary_text = response.choices[0].message.content
                return summary_text, citation

    elif model == "gpt-oss:120b" or model == "llama4:latest" or model == "gemma:latest":
        import requests
        data = None
        attempt = 0
        url = "https://anvilgpt.rcac.purdue.edu/api/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": model,
            "messages":all_messages,
            "stream": False
        }

        # Include incremental backoff for Anvil in case of timeouts or empty responses
        while data is None:
            response = requests.post(url, headers=headers, json=body,timeout=120)
            if response.status_code == 200:
                data = response.json()
                if data is None:
                    if attempt == 3:
                        break
                    attempt = attempt+1
            else:
                raise Exception(f"Error: {response.status_code}, {response.text}")
            time.sleep(2)
        if data is None:
            summary_text = "No data could be generated"
        else:
            summary_text = data['choices'][0]['message']['content']
        return summary_text, citation
    else:
                response = chat(
                    model=model,
                    messages=all_messages,
                    options = {'temperature': temperature, 'num_predict': 4096})
                summary_text = response.message.content
                return summary_text, citation
