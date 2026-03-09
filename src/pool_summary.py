import os
import time
import inspect
import requests
from openai import OpenAI
from ollama import chat
from src.citations import APA_citation, MLA_citation, NLM_citation
from src import call_api
import json
from src.pmc_text_api import find_text, extract_info

class BaseAPIClient:
    """Minimal interface every API client must implement."""

    def complete(self, messages: list[dict], temperature: float) -> str:
        raise NotImplementedError


class OpenAIClient(BaseAPIClient):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str | None = None):
        self.model = model
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", api_key),
            **({"base_url": base_url} if base_url else {}),
        )

    def complete(self, messages: list[dict], temperature: float) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content


class GroqClient(OpenAIClient):

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        super().__init__(
            api_key=os.environ.get("GROQ_API_KEY", api_key),
            model=model,
            base_url="https://api.groq.com/openai/v1",
        )


class AnvilGPTClient(BaseAPIClient):

    BASE_URL = "https://anvilgpt.rcac.purdue.edu/api/chat/completions"
    MAX_RETRIES = 3

    def __init__(self, api_key: str, model: str):
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def complete(self, messages: list[dict], temperature: float) -> str:
        body = {"model": self.model, "messages": messages, "stream": False}

        for attempt in range(1, self.MAX_RETRIES + 1):
            response = requests.post(self.BASE_URL, headers=self.headers, json=body, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data:
                    return data["choices"][0]["message"]["content"]
            else:
                raise RuntimeError(f"AnvilGPT error {response.status_code}: {response.text}")

            if attempt < self.MAX_RETRIES:
                time.sleep(10)

        return "No data could be generated"


class OllamaClient(BaseAPIClient):
    def __init__(self, model: str):
        self.model = model

    def complete(self, messages: list[dict], temperature: float) -> str:
        response = chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature, "num_predict": 4096},
        )
        return response.message.content


OPENAI_MODELS = {"ChatGPT-4o-mini"}
GROQ_MODELS = {"llama-3.3-70b-versatile"}
ANVIL_MODELS = {"gpt-oss:120b", "llama4:latest", "gemma:latest"}

def get_client(model: str, api_key: str) -> BaseAPIClient:
    if model in OPENAI_MODELS:
        return OpenAIClient(api_key=api_key, model="gpt-4o-mini")
    if model in GROQ_MODELS:
        return GroqClient(api_key=api_key, model=model)
    if model in ANVIL_MODELS:
        return AnvilGPTClient(api_key=api_key, model=model)
    # Fall back to local Ollama for any other model name
    return OllamaClient(model=model)


def build_citation(metadata: dict, citation_format: str) -> str:
    formatters = {"APA": APA_citation, "MLA": MLA_citation, "NLM": NLM_citation}
    formatter = formatters.get(citation_format)
    if formatter is None:
        raise ValueError(f"Unsupported citation format: {citation_format}")
    return formatter(metadata)

SYSTEM_CONTEXT = """
    Assume you are a helpful researcher with no memory. You are tasked with summarizing the article text presented to you, touching upon all key points within the article. Futhermore, when creating this summary, you are tasked with keeping any important details relevant to the provided question within your summary. 

    **Important Instructions:**
    1. **Do not use any information outside of what is provided in the current task.** You are not allowed to use any knowledge beyond the articles provided to you.
    2. **Do not invent or hallucinate citations.** Only use citations explicitly included in the provided articles.
    3. **Cite all information correctly** using the {citation_format} provided. 
    - If information comes from an article, cite it in **in-text citation format** as specified in {citation_format} (i.e., using the article citation and not the grade).
    4. **Do not paraphrase or shorten any biological terms, including taxanomic names, protein names, or compound names.** If a word is shortened in the article text, attempt to identify the term and use the full name of the biological term.
    5. **If you cannot find a citation for a particular piece of information in the provided data, leave that piece out of your answer.** Do not attempt to fabricate or invent a citation.

    Your response should be a detailed paragraph of 5-10 sentences answering the user’s question, **using only the articles**. You should prioritize the articles with the highest grade. If you cannot find any relevant data to answer the question, clearly state that you do not have enough information to answer.

    Please ensure your response is fully based on the data presented to you in this session and **does not include any references that are not directly provided.**
    """

def build_summary_messages(article_text: str, citation: str, citation_format: str, user_question: str) -> list[dict]:
    """Default message list for the summary task. Pass a custom list to `summary()` to override."""
    return [
        {"role": "user", "content": SYSTEM_CONTEXT.format(citation_format=citation_format)},
        {"role": "user", "content": f"The paper to summarize is: {article_text}"},
        {"role": "user", "content": f"The metadata for in-text citations ({citation_format}): {citation}"},
        {"role": "user", "content": f"The user question is: {user_question}"},
    ]

def summary(article: dict, user_question: str, model: str, citation_format: str, temperature: float, api_key: str, messages: list[dict] | None = None) -> tuple[str, str]:
    metadata_raw = article.get("meta", {})
    article_metadata = {
        "title":            metadata_raw.get("title", "Unknown"),
        "authors":          metadata_raw.get("authors", []),
        "journal":          metadata_raw.get("journal", ""),
        "publication_date": metadata_raw.get("publication_date", ""),
        "volume":           metadata_raw.get("volume", ""),
        "issue":            metadata_raw.get("issue", ""),
        "first_page":       metadata_raw.get("first_page", ""),
        "pages":            metadata_raw.get("pages", ""),
        "doi":              metadata_raw.get("doi", ""),
    }

    citation = build_citation(article_metadata, citation_format)

    if messages is None:
        messages = build_summary_messages(
            article_text=article.get("text", ""),
            citation=citation,
            citation_format=citation_format,
            user_question=user_question,
        )

    client = get_client(model, api_key)
    result = client.complete(messages, temperature)
    return result + f" Reference: {citation}", citation

def spawn(user_question: str, articles: list[dict], model: str, citation_format: str, temperature: float, api_key: str, messages_list: list[list[dict]] | None = None) -> tuple[list[str], list[str]]:
    papers, citations = [], []
    call_fun = inspect.stack()[-2].function
    sep = "<br>" if call_fun == "ask" else "\n"

    for i, article in enumerate(articles):
        custom_messages = messages_list[i] if messages_list else None
        try:
            summary_text, citation = summary(
                article, user_question, model, citation_format, temperature, api_key,
                messages=custom_messages,
            )
            print("Process successful!")
            papers.append(summary_text)
            citations.append(citation + sep)
        except Exception as e:
            print(f"Task failed: {e}")

        if i < len(articles) - 1:
            time.sleep(10)

    return papers, citations

def summary_remote(url: list, user_question: str, model: str, citation_format: str, temperature: float, api_key: str, messages: list[dict] | None = None) -> tuple[str, str]:

    string = call_api.call_api(url)
    if string and not string.startswith('[Error] : No result'):
        # extract the JSON content from the string and parse it
        api_call = json.loads(string[1:-1])
        article_metadata = extract_info(api_call)
        article = find_text(api_call)
        
        citation = build_citation(article_metadata, citation_format)

        if messages is None:
            messages = build_summary_messages(
                article_text=article.get("text", ""),
                citation=citation,
                citation_format=citation_format,
                user_question=user_question,
            )

        client = get_client(model, api_key)
        result = client.complete(messages, temperature)
        return result + f" Reference: {citation}", citation

def spawn_remote(user_question: str, articles: list[dict], model: str, citation_format: str, temperature: float, api_key: str, messages_list: list[list[dict]] | None = None) -> tuple[list[str], list[str]]:
    papers, citations = [], []
    call_fun = inspect.stack()[-2].function
    sep = "<br>" if call_fun == "ask" else "\n"

    for i, article in enumerate(articles):
        custom_messages = messages_list[i] if messages_list else None
        try:
            summary_text, citation = summary_remote(
                article, user_question, model, citation_format, temperature, api_key,
                messages=custom_messages,
            )
            print("Process successful!")
            papers.append(summary_text)
            citations.append(citation + sep)
        except Exception as e:
            print(f"Task failed: {e}")

        if i < len(articles) - 1:
            time.sleep(10)

    return papers, citations