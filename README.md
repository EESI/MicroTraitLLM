# MicroTraitLLM

A Retrieval Augmented Generation (RAG) tool for querying prokaryote functional traits, grounded in the PubMedCentral Open Access Subset to reduce hallucination and improve answer reliability.

## Background

Single cell sequencing and metagenomics have generated vast amounts of prokaryotic genomic data — far more than researchers can manually process. While LLMs offer powerful capabilities for information retrieval and synthesis in bioinformatics, they are prone to hallucination. MicroTraitLLM addresses both challenges by combining LLM inference with retrieval from curated biomedical literature.

## Installation

```bash
git clone https://github.com/yourusername/MicroTraitLLM.git
cd MicroTraitLLM
pip install -r requirements.txt
```

> Python 3.8+ recommended. Consider using a virtual environment:
> ```bash
> python -m venv venv
> source venv/bin/activate  # Windows: venv\Scripts\activate
> ```

## Usage

```bash
python main.py
```
or
```bash
waitress-serve --host=127.0.0.1 --port=8080 main:app
```
Note: You must use the 'remote' option if not downloading the PubMedCentral Open Access Subset.
## License

MIT
