from openai import OpenAI
import os
from openai import OpenAI
from ollama import chat
import inspect

def generate_summary(papers,question,model,citation_format,citations,temperature,trait_info,api_key):
    # Function to generate a summary of summaries based on the provided papers and user input
    # Definte the context and example for the LLM
    Context =  f"""
    Assume you are a helpful researcher with no memory. You are tasked with answering questions based solely on the information provided to you. This includes article summaries, grades for how well the article answers the user’s question, and microbial trait information from the database metaTraits. The entries provided will be formatted in a Python list, with each entry in the list containing the following format:
    - An article summary
    - A grade indicating the relevance of the article to the question
    - The citation of the article in {citation_format}

    **Important Instructions:**
    1. **Do not use any information outside of what is provided in the current task.** You are not allowed to use any knowledge beyond the articles and metaTraits provided to you.
    2. **Do not invent or hallucinate citations.** Only use citations explicitly included in the provided articles. If the information comes from metaTraits, refer to it as "metaTraits" and do not cite any external sources or previous knowledge.
    3. **Cite all information correctly** using the {citation_format} provided. 
    - If information comes from an article, cite it in **in-text citation format** as specified in {citation_format} (i.e., using the article citation and not the grade).
    - If the citation is from metaTraits, refer to it as "metaTraits" and do not use an article citation.
    4. **Do not include the grade or a references section in your response**. The grade indicates the relevance of the article but should not appear in the final response. Only include the article citation or metaTraits in a n when referencing data.
    5. **If you cannot find a citation for a particular piece of information in the provided data, leave that piece out of your answer.** Do not attempt to fabricate or invent a citation.

    Your response should be a detailed paragraph of 5-10 sentences answering the user’s question, **using only the articles and metaTraits database**. You should prioritize the articles with the highest grade, but ensure that any data from metaTraits is also cited as requested. If you cannot find any relevant data to answer the question, clearly state that you do not have enough information to answer.

    Please ensure your response is fully based on the data presented to you in this session and **does not include any references that are not directly provided.**
    """

    all_messages = [
                {"role": "system", "content": Context},
                {"role": "user", "content": f"The information to answer the given question is: {papers}"},
                {"role": "user", "content": f"The citations to use are: {citations}"},
                {"role": "user", "content": f"The trait information to use is {trait_info}."},
                {"role": "user", "content": f"Please answer the following question{question}"}
            ]

    # Check the model type and call the appropriate LLM to create the summary of summaries
    if model == "ChatGPT-4o-mini":
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", api_key))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=all_messages,
            temperature=temperature,
        )
        final_response = response.choices[0].message.content

    elif model == "llama-3.3-70b-versatile":
        client = OpenAI(base_url = "https://api.groq.com/openai/v1",api_key=os.environ.get("GROQ_API_KEY", api_key))
        response = client.chat.completions.create(
            model=model, 
            messages=all_messages,
            temperature=temperature)
        final_response = response.choices[0].message.content
    elif model == "gpt-oss:120b" or model == "llama4:latest" or model == "gemma:latest":
        import requests
        import time
        data = None
        pause_time = 0
        url = "https://anvilgpt.rcac.purdue.edu/api/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": model,
            "messages": all_messages,
            "stream": False
        }
        
        token_total = 0
        for m in all_messages:
            words = len(m["content"].split())
            token_total += int(words * 1.3)
        print(token_total)
        #print(all_messages)
        time.sleep(10)
        while data is None:
            print("Attempting to run summary generation")
            response = requests.post(url, headers=headers, json=body)
            if response.status_code == 200:
                print("Positive API response")
                data = response.json()
                if data is None:
                    print("Failed to get response with positive API response")
                    pause_time += 10
                    if pause_time < 30:
                        print("Moving to next pause")
                        time.sleep(pause_time)
                        continue
                    else:
                        print("Still returned none type object")
                        final_response = "Could not generate an answer"
                        break
                else:
                    print("Got response with positive API response")
                    final_response = data['choices'][0]['message']['content']
            else:
                print("Failed to get response from API")
                pause_time += 10
                if pause_time > 30:
                    print("Breaking due to time constraint retries")
                    final_response = "Could not generate an answer"
                    break
                time.sleep(pause_time)
    else:
        response = chat(
            model=model,
            messages=all_messages,
            options = {'temperature': temperature})
        final_response = response.message.content
    # Append the citations to the final response
    stack = inspect.stack()
    call_fun = stack[-2].function
    if call_fun == "ask":
        final_response = final_response + f"<br><br>References: <br>{citations}"
    print(final_response)
    return final_response

def generate_supplement(final_response,model,temperature,api_key):
    # Function to generate a supplement from the final response
    # This is used to extract terms from the final response
    # Define the context and example for the LLM
    context = """Your job is to read the provided paragraph, and note the species, genes, or proteins referenced inside the paragraph.
            If they are, you are to list each within its own respective Python list. The following order for the lists should always be used: Taxonomy, then Protein, then Gene.
            The genus, species, and subspecies for one organism should constitute one term in the list.
            If only the genus is listed, exclude it from the list. The first term in each list must always be the term you are looking for, which in this case is Taxonomy, Gene, or Protein respectively.
            If you do not identify any organisms, genes, or proteins, instead return "None" for the category list. Never abbreviate the organism name, gene, or protein name.
            If an abbreviated species name is provided, omit it from the output list you are creating. Ignore any cases with a prime symbol in them, such as 'aph(3')-Ia'.
            """
    EX3 = """In Escherichia coli (E. coli), several genes are commonly associated with antibiotic resistance, 
    reflecting the bacterium's ability to evade the effects of various antibiotics. 
    Notably, the **blaCTX-M**, **blaTEM**, and **blaSHV** genes encode for extended-spectrum 
    beta-lactamases (ESBLs), which confer resistance to a wide range of beta-lactam antibiotics 
    (Ahmad, Joji, & Shahid, 2022). Additionally, the **mcr-1** gene is significant for providing 
    resistance to colistin, a last-resort antibiotic for treating multidrug-resistant infections 
    (Nasrollahian, Graham, & Halaji, 2024). Other important resistance genes include **aac(3)-Ib-cr**, 
    which is linked to aminoglycoside resistance, and **qnr** genes that protect against fluoroquinolones 
    by encoding proteins that shield target enzymes from antibiotic action (Nasrollahian et al., 2024). 
    Furthermore, the **sul1**, **sul2**, and **sul3** genes are associated with sulfonamide resistance, 
    while **tetA** and **tetB** are linked to tetracycline resistance (Ribeiro et al., 2023). The presence 
    of these genes highlights the genetic diversity and complexity of antibiotic resistance mechanisms in 
    E. coli, emphasizing the need for ongoing surveillance and management strategies to combat this public 
    health challenge (Silva et al., 2024).
    """
    A3 = """[["Taxonomy", "Escherichia coli"],["Gene", "blaCTX-M", "blaTEM", "blaSHV","mcr-1","aac(3)-Ib-cr","qnr","sul1", "sul2","sul3","tetA","tetB"]]"""
    # Check the model type and call the appropriate LLM to create the supplement
    if model == "ChatGPT-4o-mini":
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", api_key))
        supplement = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": context},
                {"role": "user","content": EX3},
                {"role": "assistant", "content": A3},
                {"role": "user", "content": f"The paragraph is: {final_response}"},
                    ],
            temperature=temperature,
            ) 
        termlist = supplement.choices[0].message.content
    elif model == "llama-3.3-70b-versatile":
        client = OpenAI(base_url = "https://api.groq.com/openai/v1",api_key=os.environ.get("GROQ_API_KEY", api_key))
        supplement = client.chat.completions.create(
            model=model, 
            messages=[
                {"role": "system", "content": context},
                {"role": "user","content": EX3},
                {"role": "assistant", "content": A3},
                {"role": "user", "content": f"The paragraph is: {final_response}"},
                    ],
            temperature=temperature)
        termlist = supplement.choices[0].message.content
    else:
        supplement = chat(
            model=model,
            messages=[
                {"role": "system", "content": context},
                {"role": "user","content": EX3},
                {"role": "assistant", "content": A3},
                {"role": "user", "content": f"The paragraph is: {final_response}"},
                    ],
            options = {'temperature': temperature})
        termlist = supplement.message.content
    return termlist

# Copyright Sep 2025 Glen Rogers. 
# Subject to MIT license.
