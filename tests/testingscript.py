#!/usr/bin/env python
import src.pubmed_central_search as pmc
import traceback
import src.compile_supplement_generation as csg
import src.pool_summary as pool_summary
from timer import Timer
import os
import gc
from src.read_api_keys import load_api_keys
from vector_summary_anvil import vector_summary, vector_space_creation, vector_space_search
from sentence_transformers import SentenceTransformer
from src.citations import APA_citation, MLA_citation, NLM_citation
import src.call_api as call_api
import json
import re
from src.parse_xml import PMCArticleExtractor
import psutil
import torch
import time
import requests
from src.compression import RAGArticleCompressor, CompressionConfig
from eval_logging import ScriptLogger
from src.metaTraits_MiMeDB import load_jsonl, load_mapping
import ast
import csv
import ahocorasick
from collections import defaultdict

"""
Testing script to evaluate summary performace for LLMs in MicroTraitLLM. Requirements:
    - Local storage of PMC articles (can be set up with the download script in src)
    - API keys for OpenAI, Groq, and Anvil in an api_keys.txt file
    - test_questions.json file with questions and answer keywords for evaluation
"""
class ResourceTracker:
    def __init__(self):
        self.process = psutil.Process()
        self.has_gpu = torch.cuda.is_available()
        
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        memory_info = self.process.memory_info()
        ram_mb = memory_info.rss / 1024 / 1024
        
        gpu_mb = 0
        if self.has_gpu:
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
            
        return ram_mb, gpu_mb
    
    def log_usage(self, stage, log_file="resource_usage.csv"):
        """Log current resource usage"""
        ram_mb, gpu_mb = self.get_memory_usage()
        cpu_percent = self.process.cpu_percent()
        
        write_header = not os.path.exists(log_file) or os.path.getsize(log_file) == 0
        with open(log_file, 'a') as f:
            if write_header:
                f.write("timestamp,stage,ram_mb,gpu_mb,cpu_percent\n")
            f.write(f"{time.time()},{stage},{ram_mb:.2f},{gpu_mb:.2f},{cpu_percent}\n")
        
        return ram_mb, gpu_mb

# Get user input once
# Initialize model once (reuse it, don't recreate each time!)
MODEL = SentenceTransformer('paraphrase-MiniLM-L6-v2')
def get_next_filename(base_name, folder, model, question_number,ans_iter, method):
    filename = f"{model}_{question_number}_{ans_iter}_{method}_{base_name}.txt"
    file_path = os.path.join(folder, filename)
    return file_path
    #if not os.path.exists(file_path):
    #    return file_path

def create_new_file(model, question_number, ans_iter,method):
    if fileloc.lower() == "reference" or fileloc.lower() == "r":
        folder = "ref"
        #base_name = "100ref_output"
        timer_base_name = "100ref_timer_output"
    else:
        folder = "ee"
        base_name = "20_eval"
        timer_base_name = "20_eval"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_path = get_next_filename(base_name, folder,model, question_number,ans_iter, method)
    with open(file_path, 'w') as f:
        f.write("")  # Create an empty file
    return file_path


def append_paragraph_to_file(paragraph, citation, filename):
    with open(filename, "a", encoding="utf-8") as file:
    # Add a blank line if the file already has content
        if file.tell() > 0:
            file.write("\n\n\n\n\n")
        file.write(paragraph)
        file.write("\n")
        file.write("Reference:\n")
        for line in citation:
            file.write(f"{line}\n")



def check_answer(answer_to_check, correct_keywords):
    """
    Check a single answer against a list of correct keywords.
    Handles variations like hyphens, dashes, and word stems.
    
    Args:
        answer_to_check: String answer to validate
        correct_keywords: Single keyword (str) or list of keywords
    
    Returns:
        Dictionary with check results
    """
    # Normalize the answer (lowercase, strip whitespace)
    answer_normalized = answer_to_check.lower().strip() if answer_to_check else ""
    
    # Normalize special characters (replace various dashes with regular hyphen)
    answer_normalized = re.sub(r'[—–−]', '-', answer_normalized)
    
    correct_keywords = correct_keywords.split(';')
    # Handle correct_keywords as either a single keyword or list of keywords
    if isinstance(correct_keywords, str):
        keywords = [correct_keywords.lower().strip()]
    else:
        keywords = [k.lower().strip() for k in correct_keywords]
    
    # Normalize keywords too
    keywords = [re.sub(r'[—–−]', '-', kw) for kw in keywords]
    
    # Check if any keyword is in the answer
    matched_keywords = []
    for keyword in keywords:
        # Check for exact substring match OR as a word stem
        if keyword in answer_normalized:
            matched_keywords.append(keyword)
        # Also check if the keyword appears as a stem (e.g., "cluster" matches "clusters" or "clustered")
        elif any(word.startswith(keyword) for word in answer_normalized.split()):
            matched_keywords.append(keyword)
    
    return len(matched_keywords)

def system_run(question, article_number, model, temperature, api_key, ncbi_api_key):
    # first segment: article URLs
    first_url, search_term = pmc.question_formation(question,model,temperature,api_key,ncbi_api_key)
    idlist = pmc.idlist_confirm(first_url, question, article_number,model,temperature,api_key, search_term,ncbi_api_key)
    art_ids = []
    for NCBIid in idlist:
        PMCid = 'PMC'+NCBIid
        art_ids.append(PMCid)
    extractor = PMCArticleExtractor(r"/var/www/html/data/FTP_Downloads")
    extractor.extract_articles(art_ids)

def load_csv_data(csv_file):
    """Load species data from CSV file"""
    data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def list_files(directory, extension):
    """List all files with given extension in directory"""
    try:
        files = os.listdir(directory)
        return [f for f in files if f.endswith(extension)]
    except FileNotFoundError:
        print(f"Directory {directory} not found")
        return []

def extract_traits(record):
    """Extract description from a record"""
    traits = []
    if record and 'description' in record and record['description'].strip():
        return [record['description']]
    elif record and 'summaries' in record:
        for item in record['summaries']:
            if "true" in item['majority_label']:
                traits.append(item['name'])
        return traits
    return []

def build_species_automaton(csv_files, csv_directory):
    """Build Aho-Corasick automaton from all CSV files"""
    all_species_data = {}
    
    for csv_file in csv_files:
        csv_path = os.path.join(csv_directory, csv_file)
        data = load_csv_data(csv_path)
        
        for record in data:
            # Use 'name' column (second column) as the species name
            if 'name' in record and record['name'].strip():
                species_name = record['name'].strip().lower()
                all_species_data[species_name] = record
    
    
    
    # Build Aho-Corasick automaton
    automaton = ahocorasick.Automaton()
    
    for species_name in all_species_data.keys():
        automaton.add_word(species_name, species_name)
    
    ncbi_file = "/var/www/html/data/metaTraits/ncbi_species_summary.jsonl"
    gtdb_file = "/var/www/html/data/metaTraits/gtdb_species_summary.jsonl"
    mapping_file = "/var/www/html/data/metaTraits/GTDB2NCBI.tsv"

    # Load data

    ncbi_data = load_jsonl(ncbi_file)
    gtdb_data = load_jsonl(gtdb_file)

    # PRE-PROCESSING: Build lookup dictionaries

    ncbi_lookup = {record['tax_name'].lower(): record for record in ncbi_data}
    gtdb_lookup = {record['tax_name'].lower(): record for record in gtdb_data}

    # Build Aho-Corasick automaton for ultra-fast multi-pattern matching

    for species_name in ncbi_lookup.keys():
        if species_name not in all_species_data.keys():
            automaton.add_word(species_name, ('ncbi', species_name))

    for species_name in gtdb_lookup.keys():
        if species_name not in ncbi_lookup and species_name not in all_species_data.keys():  # Avoid duplicates
            automaton.add_word(species_name, ('gtdb', species_name))

    automaton.make_automaton()
    print(f"Automaton built with {len(all_species_data)} patterns")
    
    return automaton, all_species_data

def find_species_in_text(text, automaton, use_context_filter=False, context_words=None, context_window=100):
    """Find all species mentions in text using Aho-Corasick with optional context filtering"""
    text_lower = text.lower()
    found_species = {}
    
    for end_index, species_name in automaton.iter(text_lower):
        # Check word boundaries
        start_index = end_index - len(species_name) + 1
        
        if start_index > 0 and text_lower[start_index - 1].isalnum():
            continue
        if end_index < len(text_lower) - 1 and text_lower[end_index + 1].isalnum():
            continue
        
        # Context-based filtering
        if use_context_filter and context_words:
            # Extract surrounding text
            context_start = max(0, start_index - context_window)
            context_end = min(len(text_lower), end_index + context_window)
            surrounding_text = text_lower[context_start:context_end]
            
            # Check if any context word appears nearby
            has_context = any(word in surrounding_text for word in context_words)
            if not has_context:
                continue
        
        # Track position for counting mentions
        if species_name not in found_species:
            found_species[species_name] = []
        found_species[species_name].append(start_index)
    
    return found_species

#question = #input("Question: ")
article_number = 8 #int(input("Number of Articles: "))
temperature = 0.0 #float(input("Temperature: "))
citation_format = "APA" #input("Citation Format: ")
#model = "gemma2:2b" #input("Model: ")
fileloc = "eval"#input("Reference or Evaluation?: ")
nog = 50#int(input("How many generated answers?: ")) #Number of answers Generated


ans_iter = list(range(1, nog+1))
# Load API keys
api_keys = load_api_keys('api_keys.txt')

# Use the API keys
api_key_openai = api_keys.get('API_KEY_OPENAI')
api_key_groq = api_keys.get('API_KEY_GROQ')
api_key_anvil = api_keys.get('API_KEY_ANVIL')
ncbi_api_key = api_keys.get('API_KEY_NCBI')

# Define questions and answers to test
with open("test_questions.json", "r") as f:
    questions = json.load(f)

csv_directory = "/var/www/html/data/MiMeDB2.0"
# Initialize tracker
tracker = ResourceTracker()
# Initialize logging
logger = ScriptLogger(log_file='metaTraits.logger.log', log_dir='logs')
modelnum = [6]
csv_files = list_files(csv_directory, '.csv')
# Build automaton from CSV data
automaton, species_data = build_species_automaton(csv_files, csv_directory)
"""
compressor = RAGArticleCompressor(
    model_name="paraphrase-MiniLM-L6-v2",
    config=CompressionConfig(
        similarity_threshold=0.75,
        query_relevance_threshold=0.4,  # Keep chunks with >40% relevance
        compression_ratio=0.6,
        top_k_chunks=3, # Use threshold instead of top-k
        max_output_tokens=500
    )
)
"""
# Filtering Configuration
MIN_MENTIONS = 2  # Minimum times a species must appear to be included (set to 1 to disable)
USE_CONTEXT_FILTER = False  # Enable/disable context-based filtering
CONTEXT_WORDS = [
    'bacteria', 'microbe', 'strain', 'species', 'infection', 'pathogen',
    'culture', 'isolate', 'gut', 'microbiome', 'probiotic', 'flora',
    'genus', 'family', 'phylum', 'organism', 'microbial', 'bacterial'
]
CONTEXT_WINDOW = 100  # Characters before/after the match to check for context

for modeli in modelnum:
    # Model selection
    if modeli == 1:
        model = "ChatGPT-4o-mini"
    elif modeli == 2:
        model = "llama-3.3-70b-versatile"
    elif modeli == 3:
        model = "llama3.2"
    elif modeli == 4:
        model = "gemma2:2b"
    elif modeli == 5:
        model = "gpt-oss:120b"
    elif modeli == 6:
        model = "llama4:latest"
    elif modeli == 7:
        model = "gemma:latest"
    else:
        model = None


    logger.section(f"Starting Model {model} for vector version")
    tracker.log_usage(f"start_{model}")
    
    if model == "ChatGPT-4o-mini":
        api_key = api_key_openai
    elif model == "llama-3.3-70b-versatile":
        api_key = api_key_groq
    elif model == "gpt-oss:120b" or model == "llama4:latest" or model == "gemma:latest":
        api_key = api_key_anvil
    else:
        api_key = None

    for q in questions:
        question = q["question"]
        answer_keywords = q["answer_keywords"]
        vector_space = None  # Initialize per question
        

        for i in ans_iter:
            t = Timer()
            top_k = 8
            n = 8
            if i == 1:
                # Clear previous vector space
                if vector_space is not None:
                    del vector_space
                    gc.collect()
                
                t.start()
                system_run(question, n, 'llama3.2', citation_format, temperature, api_key,ncbi_api_key)
                with open('all_articles.json', 'r') as f:
                    json_data = json.load(f)
                vector_space, citation = vector_space_creation(json_data,citation_format)
                initial_time = t.stop()
                
                # Clean up embeddings immediately
                if os.path.exists('all_articles.json'):
                    os.remove('all_articles.json')
                gc.collect()
                
                if ":" in model:
                    # If colon is present, remove the colon and concatenate the parts
                    parts = model.split(":")
                    model_name = "".join(parts)
                else:
                    model_name = model
                
            try:
                file_path = create_new_file(model_name, questions.index(q)+1,i,"vector")
                t.start()
                top_k_vectors = vector_space_search(question, vector_space, top_k)
                trait_info = []
                all_trait_info = []
                global_species_mentions = defaultdict(int)
                global_species_positions = defaultdict(list)
                processed_organisms = set()
                for h, article in enumerate(top_k_vectors):

                # Combine all text from the article
                    if isinstance(article, dict) and 'text' in article:
                        full_text = article['text']
                    elif isinstance(article, list):
                        full_text = ' '.join(
                            item['text'] if isinstance(item, dict) else str(item)
                            for item in article
                        )
                    else:
                        full_text = str(article)
                    # Find all species mentioned
                    found_species_raw = find_species_in_text(
                        full_text,
                        automaton,
                        use_context_filter=USE_CONTEXT_FILTER,
                        context_words=CONTEXT_WORDS,
                        context_window=CONTEXT_WINDOW
                    )

                    # Accumulate mentions across all articles
                    for species_name, positions in found_species_raw.items():
                        global_species_mentions[species_name] += len(positions)
                        global_species_positions[species_name].extend(positions)

                filtered_species = {
                    name: count for name, count in global_species_mentions.items()
                    if count >= MIN_MENTIONS
                }

                sorted_species = sorted(filtered_species.items(), key=lambda x: x[1], reverse=True)

                for species_name, mention_count in sorted_species:
                    record = species_data[species_name]
                    description = extract_traits(record)

                    if description:
                        # Get the original casing from the record
                        original_name = record.get('name', species_name)
                        all_trait_info.append(f"Description for {original_name} (mentioned {mention_count} times across all articles):")
                        all_trait_info.extend(description)
                
                response, citation = vector_summary(top_k_vectors, question, model, citation_format, temperature, citation, all_trait_info,api_key)
                elapsed_time = t.stop()
                final_time = initial_time + elapsed_time
                
                logger.info(f"Completed Sample {i} for Question {questions.index(q)+1} using model {model_name} vector summaries in {final_time:.2f} seconds.")
                
                score = check_answer(response, answer_keywords)
                scores_file = os.path.join(os.path.dirname(file_path), "vector_scores.csv")
                timer_file = os.path.join(os.path.dirname(file_path), "vector_timer.csv")
                
                write_header = not os.path.exists(scores_file) or os.path.getsize(scores_file) == 0
                safe_model = '"' + str(model_name).replace('"', '""') + '"'
                
                with open(scores_file, 'a', encoding='utf-8') as sf:
                    if write_header:
                        sf.write("model,question,sample,score\n")
                    sf.write(f"{safe_model},{questions.index(q)+1},{i},{score}\n")
                    
                with open(timer_file, 'a', encoding='utf-8') as sf:
                    if write_header:
                        sf.write("model,question,sample,time\n")
                    sf.write(f"{safe_model},{questions.index(q)+1},{i},{final_time}\n")
                    
                append_paragraph_to_file(response, citation, file_path)
                
                # Clean up response data
                del top_k_vectors, response, trait_info, processed_organisms
                gc.collect()
    
            except Exception as e:
                logger.error(f"Error processing Sample {i}: {e}")
                traceback.print_exc()
            
        time.sleep(180)
        # Clean up vector space after question
        if vector_space is not None:
            del vector_space
            gc.collect()

    
    # Unload Ollama models to free VRAM
    if model in ["llama3.2", "gemma2:2b"]:
        requests.post('http://localhost:11434/api/generate', 
                     json={"model": model, "keep_alive": 0})
        time.sleep(2)
    
    tracker.log_usage(f"end_{model}")
    logger.success(f"Finished Model {model}\n")
logger.success("Finished vector summary testing.")

# Clean up all loop variables
del model, api_key, q, question, answer_keywords, i, t, top_k, n, model_name
del file_path, citation, elapsed_time, final_time, initial_time
del score, scores_file, timer_file, write_header, safe_model, json_data
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# base version run
for modeli in modelnum:
    # Model selection
    if modeli == 1:
        model = "ChatGPT-4o-mini"
    elif modeli == 2:
        model = "llama-3.3-70b-versatile"
    elif modeli == 3:
        model = "llama3.2"
    elif modeli == 4:
        model = "gemma2:2b"
    elif modeli == 5:
        model = "gpt-oss:120b"
    elif modeli == 6: 
        model = "llama4:latest"
    else:
        model = None

    logger.section(f"Starting Model {model} for base version")
    tracker.log_usage(f"start_{model}")
    
    if model == "ChatGPT-4o-mini":
        api_key = api_key_openai
    elif model == "llama-3.3-70b-versatile":
        api_key = api_key_groq
    elif model == "gpt-oss:120b" or model == "llama4:latest":
        api_key = api_key_anvil
    else:
        api_key = None

    for q in questions:
        question = q["question"]
        answer_keywords = q["answer_keywords"]
        

        for i in ans_iter:
            t = Timer()
            top_k = 8
            n = 8
            if i == 1:
                
                t.start()
                system_run(question, n, 'llama3.2', citation_format, temperature, api_key, ncbi_api_key)
                with open('all_articles.json', 'r') as f:
                    json_data = json.load(f)
                summaries, citations = pool_summary.spawn(q, json_data, model, citation_format, temperature, api_key)
                initial_time = t.stop()
                
                # Clean up embeddings immediately
                if os.path.exists('all_articles.json'):
                    os.remove('all_articles.json')
                gc.collect()
                
            if ":" in model:
                # If colon is present, remove the colon and concatenate the parts
                parts = model.split(":")
                model_name = "".join(parts)
            else:
                model_name = model
                
            try:
                method = "base"
                file_path = create_new_file(model_name, questions.index(q)+1,i,method)
                t.start()
                trait_info = []
                all_trait_info = []
                global_species_mentions = defaultdict(int)
                global_species_positions = defaultdict(list)
                processed_organisms = set()
                for h, article in enumerate(summaries):

                # Combine all text from the article
                    if isinstance(article, dict) and 'text' in article:
                        full_text = article['text']
                    elif isinstance(article, list):
                        full_text = ' '.join(
                            item['text'] if isinstance(item, dict) else str(item)
                            for item in article
                        )
                    else:
                        full_text = str(article)
                    # Find all species mentioned
                    found_species_raw = find_species_in_text(
                        full_text,
                        automaton,
                        use_context_filter=USE_CONTEXT_FILTER,
                        context_words=CONTEXT_WORDS,
                        context_window=CONTEXT_WINDOW
                    )
                    # Accumulate mentions across all articles
                    for species_name, positions in found_species_raw.items():
                        global_species_mentions[species_name] += len(positions)
                        global_species_positions[species_name].extend(positions)
            
                filtered_species = {
                    name: count for name, count in global_species_mentions.items()
                    if count >= MIN_MENTIONS
                }
            
                sorted_species = sorted(filtered_species.items(), key=lambda x: x[1], reverse=True)
                for species_name, mention_count in sorted_species:
                    record = species_data[species_name]
                    description = extract_traits(record)

                    if description:
                        # Get the original casing from the record
                        original_name = record.get('name', species_name)
                        all_trait_info.append(f"Description for {original_name}:")
                        all_trait_info.extend(description)
                response = csg.generate_summary(summaries,question,model,citation_format,citations,temperature,trait_info,api_key)
                elapsed_time = t.stop()
                final_time = initial_time + elapsed_time
                logger.info(f"Completed Sample {i} for Question {questions.index(q)+1} base testing in {final_time:.2f} seconds.")
                
                score = check_answer(response, answer_keywords)
                scores_file = os.path.join(os.path.dirname(file_path), "base_scores.csv")
                timer_file = os.path.join(os.path.dirname(file_path), "base_timer.csv")
                
                write_header = not os.path.exists(scores_file) or os.path.getsize(scores_file) == 0
                safe_model = '"' + str(model_name).replace('"', '""') + '"'
                
                with open(scores_file, 'a', encoding='utf-8') as sf:
                    if write_header:
                        sf.write("model,question,sample,score\n")
                    sf.write(f"{safe_model},{questions.index(q)+1},{i},{score}\n")
                    
                with open(timer_file, 'a', encoding='utf-8') as sf:
                    if write_header:
                        sf.write("model,question,sample,time\n")
                    sf.write(f"{safe_model},{questions.index(q)+1},{i},{final_time}\n")    
                append_paragraph_to_file(response, citations, file_path)
                
                # Clear all important variables
                del response, trait_info, processed_organisms
                gc.collect()
            except Exception as e:
                logger.error(f"Error processing Sample {i}: {e}")
                traceback.print_exc()
    
    # Unload Ollama models to free VRAM
    if model in ["llama3.2", "gemma2:2b"]:
        requests.post('http://localhost:11434/api/generate', 
                     json={"model": model, "keep_alive": 0})
        time.sleep(2)
    
    tracker.log_usage(f"end_{model}")
    logger.success(f"Finished Model {model}\n")
logger.success("Finished base summary testing.")

# Clean up all loop variables
del model, api_key, q, question, answer_keywords, i, t, n, model_name
del file_path, citations, final_time, initial_time
del score, scores_file, timer_file, write_header, safe_model, json_data
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# compression version run
# Initialize compressor with lightweight model
    # Initialize compressor for RAG
compressor = RAGArticleCompressor(
    model_name="paraphrase-MiniLM-L6-v2",
    config=CompressionConfig(
        similarity_threshold=0.75,
        query_relevance_threshold=0.4,  # Keep chunks with >40% relevance
        compression_ratio=0.6,
        top_k_chunks=3, # Use threshold instead of top-k
        max_output_tokens=500
    )
)

# Filtering Configuration
MIN_MENTIONS = 2  # Minimum times a species must appear to be included (set to 1 to disable)
USE_CONTEXT_FILTER = False  # Enable/disable context-based filtering
CONTEXT_WORDS = [
    'bacteria', 'microbe', 'strain', 'species', 'infection', 'pathogen',
    'culture', 'isolate', 'gut', 'microbiome', 'probiotic', 'flora',
    'genus', 'family', 'phylum', 'organism', 'microbial', 'bacterial'
]
CONTEXT_WINDOW = 100  # Characters before/after the match to check for context


for modeli in modelnum:
    # Model selection
    if modeli == 1:
        model = "ChatGPT-4o-mini"
    elif modeli == 2:
        model = "llama-3.3-70b-versatile"
    elif modeli == 3:
        model = "llama3.2"
    elif modeli == 4:
        model = "gemma2:2b"
    elif modeli == 5:
        model = "gpt-oss:120b"
    elif modeli == 6:
        model = "llama4:latest"
    elif modeli == 7:
        model = "gemma:latest"
    else:
        model = None

    logger.section(f"Starting Model {model} for compressed verison")
    tracker.log_usage(f"start_{model}")
    
    if model == "ChatGPT-4o-mini":
        api_key = api_key_openai
    elif model == "llama-3.3-70b-versatile":
        api_key = api_key_groq
    elif model == "gpt-oss:120b" or model == "llama4:latest" or model == "gemma:latest":
        api_key = api_key_anvil
    else:
        api_key = None

    for q in questions:
        question = q["question"]
        answer_keywords = q["answer_keywords"]
        

        for i in ans_iter:
            t = Timer()
            top_k = 8
            n = 8
            if i == 1:
                
                t.start()
                system_run(question, n, 'llama3.2', citation_format, temperature, api_key, ncbi_api_key)
                # Load your articles
                with open('all_articles.json', 'r') as f:
                    json_data = json.load(f)
                    
                citations = []
                all_chunks = []
                for article in json_data:
                    article_text = article['text']
                    if citation_format == "APA":
                        citation = APA_citation(article['meta'])
                        citations.append(citation)
                    elif citation_format == "MLA":
                        citation = MLA_citation(article['meta'])
                        citations.append(citation)
                    elif citation_format == "NLM":
                        citation = NLM_citation(article['meta'])
                        citations.append(citation)
                    chunk = compressor.rag_compress(article_text, question, return_metadata=True)
                    print(f"\nChunks kept: {chunk['chunks_kept']}/{chunk['total_chunks']}")
                    print(f"Estimated tokens: {chunk['estimated_tokens']}")
                    all_chunks.append({'text': chunk['compressed_text'], 'citation': citation})
                initial_time = t.stop()
                del chunk
                # Clean up embeddings immediately
                if os.path.exists('all_articles.json'):
                    os.remove('all_articles.json')
                gc.collect()
                
            if ":" in model:
                # If colon is present, remove the colon and concatenate the parts
                parts = model.split(":")
                model_name = "".join(parts)
            else:
                model_name = model
# This isn't working right now because text gets returned as a list of each semantic chunk                
            try:
                method = "compressed"
                file_path = create_new_file(model_name, questions.index(q)+1,i,method)
                t.start()
                trait_info = []
                all_trait_info = []
                global_species_mentions = defaultdict(int)
                global_species_positions = defaultdict(list)
                processed_organisms = set()
                for h, article in enumerate(all_chunks):

                # Combine all text from the article
                    if isinstance(article, dict) and 'text' in article:
                        full_text = article['text']
                    elif isinstance(article, list):
                        full_text = ' '.join(
                            item['text'] if isinstance(item, dict) else str(item)
                            for item in article
                        )
                    else:
                        full_text = str(article)
                    # Find all species mentioned
                    found_species_raw = find_species_in_text(
                        full_text,
                        automaton,
                        use_context_filter=USE_CONTEXT_FILTER,
                        context_words=CONTEXT_WORDS,
                        context_window=CONTEXT_WINDOW
                    )

                    # Accumulate mentions across all articles
                    for species_name, positions in found_species_raw.items():
                        global_species_mentions[species_name] += len(positions)
                        global_species_positions[species_name].extend(positions)

                filtered_species = {
                    name: count for name, count in global_species_mentions.items()
                    if count >= MIN_MENTIONS
                }

                sorted_species = sorted(filtered_species.items(), key=lambda x: x[1], reverse=True)

                for species_name, mention_count in sorted_species:
                    record = species_data[species_name]
                    description = extract_traits(record)

                    if description:
                        # Get the original casing from the record
                        original_name = record.get('name', species_name)
                        all_trait_info.append(f"Description for {original_name} (mentioned {mention_count} times across all articles):")
                        all_trait_info.extend(description)
                response = csg.generate_summary(all_chunks,question,model,citation_format,citations,temperature,all_trait_info,api_key)
                elapsed_time = t.stop()
                final_time = initial_time + elapsed_time
                
                logger.info(f"Completed Sample {i} for Question {questions.index(q)+1} using model {model_name} compressed summaries in {final_time:.2f} seconds.")

                score = check_answer(response, answer_keywords)
                scores_file = os.path.join(os.path.dirname(file_path), "compressed_scores.csv")
                timer_file = os.path.join(os.path.dirname(file_path), "compressed_timer.csv")
                
                write_header = not os.path.exists(scores_file) or os.path.getsize(scores_file) == 0
                safe_model = '"' + str(model_name).replace('"', '""') + '"'
                
                with open(scores_file, 'a', encoding='utf-8') as sf:
                    if write_header:
                        sf.write("model,question,sample,score\n")
                    sf.write(f"{safe_model},{questions.index(q)+1},{i},{score}\n")
                    
                with open(timer_file, 'a', encoding='utf-8') as sf:
                    if write_header:
                        sf.write("model,question,sample,time\n")
                    sf.write(f"{safe_model},{questions.index(q)+1},{i},{final_time}\n")
                    
                append_paragraph_to_file(response, citations, file_path)
                
                del response, processed_organisms, trait_info 
            except Exception as e:
                logger.error(f"Error processing Sample {i}: {e}")
                traceback.print_exc()
            time.sleep(10)
    # Unload Ollama models to free VRAM
    if model in ["llama3.2", "gemma2:2b"]:
        requests.post('http://localhost:11434/api/generate', 
                     json={"model": model, "keep_alive": 0})
        time.sleep(2)
    
    tracker.log_usage(f"end_{model}")
    logger.success(f"Finished Model {model}\n")
logger.success("Finished compressed summary testing.")

