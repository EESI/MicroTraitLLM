import json
import csv
from collections import defaultdict
import os
import ujson
import pickle

def load_jsonl(filename, use_cache=True):
    """Optimized JSONL loading with caching and ujson"""
    cache_file = filename + '.pickle'
    
    # Try cache first
    if use_cache and os.path.exists(cache_file):
        if os.path.getmtime(cache_file) >= os.path.getmtime(filename):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    
    # Load with ujson
    print(f"Loading {filename}...")
    with open(filename, 'r', encoding='utf-8') as f:
        data = [ujson.loads(line) for line in f if line.strip()]
    
    # Save cache
    if use_cache:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return data
# Load mapping file to identify GTDB-NCBI equivalents
def load_mapping(mapping_file):
    """Load GTDB to NCBI mapping and create a lookup dictionary."""
    gtdb_to_ncbi = {}
    try:
        with open(mapping_file, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                # Adjust column names based on your TSV structure
                # Common formats: gtdb_name -> ncbi_name or gtdb -> ncbi
                gtdb_name = row.get('gtdb_name') or row.get('gtdb') or row.get('GTDB')
                ncbi_name = row.get('ncbi_name') or row.get('ncbi') or row.get('NCBI')
                if gtdb_name and ncbi_name:
                    gtdb_to_ncbi[gtdb_name.lower()] = ncbi_name.lower()
    except FileNotFoundError:
        print(f"Warning: Mapping file {mapping_file} not found. Proceeding without deduplication.")
    return gtdb_to_ncbi

import csv

def mimedb_names(filepath):
    """
    Search the MiMeDB database for entries matching the given name.

    Parameters:
    name (str): The name to search for in the MiMeDB database.
    filepath (str): The path to the MiMeDB CSV file.

    Returns:
    list: A list of dictionaries containing the matching entries.
    """
    results = []
    
    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            results.append(row)
    
    return results