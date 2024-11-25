import models
import json
import re
import torch
import numpy as np
import torch.nn.functional as F
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
import requests

def create_prompt(question, validation_data):  
    prompt = {
        "question": question,
        "reference" : validation_data,
        "response" : "",
        "main_points" : []
    }
    return prompt

def search_string(original_string, tokens_found, substring):
    if substring !="" and substring.strip():
        #print(substring)
        for token in tokens_found:
            original_string = original_string.replace(token, "", 1)
        #print(original_string)
        if substring in original_string:
            return [True, substring]
        elif not any(char.isalpha() for char in original_string):
            return [True, ""]
    else:
        return [False, ""]
    return [False, substring]

def fetch_logits(main_points, tokens, logits):
    points_logits = []
    found_counter = 0
    to_find = "main_points"
    tokens_found = []
    logits_found = {}
    for i in range(0, len(tokens)):            
        token_text = models.tokenizer.decode(tokens[i], skip_special_tokens=True)
        search_data = search_string(to_find, tokens_found, token_text)
        if search_data[0] and search_data[1] == "":
            #print(f"{to_find} Found!")
            if found_counter >= 1:
                points_logits.append(logits_found)
            if found_counter >= len(main_points):
                break
            to_find = main_points[found_counter]
            found_counter += 1
            tokens_found = []
            logits_found = {}
        elif search_data[0]:
            tokens_found.append(search_data[1])
            #print(tokens_found)
            logits_found[tokens[i]] = logits[i][0]  
        elif search_data[1] != "":
            tokens_found = []
            logits_found = {}
    return points_logits

def calculate_proportion_below_threshold(point_logits, threshold=0.7):
    below_threshold_count = 0
    for point_token, point_logit in point_logits.items():
        probabilities = F.softmax(point_logit, dim=-1)
        token_prob = probabilities[point_token].item()
        if token_prob < threshold:
            below_threshold_count += 1
            
    proportion = below_threshold_count / len(point_logits) if len(point_logits) > 0 else 0.0
    return proportion

def internally_validated_response(main_question, references=[], val_threshold=0.1, visualize=True, initial_response=None):
    console = Console()
    main_response = ""
    no_hallucination = False

    #Initial response generation
    if initial_response is None:
        while main_response == "" or main_response["response"] == "":
            prompt = create_prompt(main_question, references)
            response, logits, tokens, prompt_index = models.LLAMA(prompt)
            main_response = models.process_response(response)
            initial_response=(response, logits, tokens, prompt_index)
    else:
        response, logits, tokens, prompt_index = initial_response
        main_response = models.process_response(response)

    #Validation loop
    while not no_hallucination:
        main_points = main_response["main_points"]
        points_logits = fetch_logits(main_points, tokens, logits)
        internal_data = []
        point_scores = []

        for i, point_logits in enumerate(points_logits):
            score = calculate_proportion_below_threshold(point_logits)
            point_scores.append(score)

            if score > val_threshold:
                claim = main_points[i]
                uncertain_text = Text(f"Uncertain Claim: {claim}", style="bold yellow")
                uncertain_score_text = Text(
                    f"Uncertain Proportion Score: {score:.2f}", style="italic yellow"
                )
                if visualize:
                    console.print(uncertain_text)
                    console.print(uncertain_score_text)
                    console.print("[bold green]Validating...[/bold green]")

                val_response = ""

                #Validate the claim
                while val_response == "" or val_response["response"] == "":
                    prompt = create_prompt(f"In the context of '{main_question}' is this claim accurate '{claim}'? Give short answer.", [])
                    response, logits, tokens, prompt_index = models.LLAMA(prompt)
                    val_response = models.process_response(response)

                val_data = val_response["response"]
                internal_data.append(f"{val_data}")
                if visualize:
                    console.print(Panel(f"Validation Result: [bold cyan]{val_data}[/bold cyan]", title="Validated Claim"))

        #Check maximum uncertainty and regenerate response if necessary
        if len(point_scores) > 0:
            mean_uncertain_proportion = np.max(point_scores)
            mean_text = Text(
                f"The Max Uncertainty For Current Response Is: {mean_uncertain_proportion:.2f}",
                style="bold blue",
            )
            if visualize:
                console.print(mean_text)

        if len(internal_data) > 0:
            main_response = ""
            no_hallucination = False
            if visualize:
                console.print("[bold green]Generating Response With Internally Validated Data...[/bold green]")

            while main_response == "" or main_response["response"] == "":
                prompt = create_prompt(main_question, internal_data)
                response, logits, tokens, prompt_index = models.LLAMA(prompt)
                main_response = models.process_response(response)
        else:
            no_hallucination = True
            if visualize:
                console.print(Panel(json.dumps(main_response, indent=1), title="Final Response", style="bold green"))
                console.print(Panel(main_response["response"], title="Internally Validated Answer", style="bold red"))

    return main_response["response"], initial_response


def get_external_data(query, subscription_key, endpoint="https://api.bing.microsoft.com/v7.0/search", mkt='en-US', num_results=1):
    params = {
        'q': query,
        'mkt': mkt,
        'responseFilter': 'Webpages'  #Filter to only return webpages
    }
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key
    }

    try:
        #Make API request
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        json_response = response.json()

        #Extract and clean snippets
        snippets = []
        if 'webPages' in json_response and 'value' in json_response['webPages']:
            results = json_response['webPages']['value']
            for result in results[:num_results]:  #Limit to specified number of results
                if 'snippet' in result:
                    #Remove special characters using regex
                    cleaned_snippet = re.sub(r'[^A-Za-z0-9\s]', '', result['snippet'])
                    snippets.append(cleaned_snippet)
        
        return snippets

    except Exception as ex:
        print(f"An error occurred: {ex}")
        return []

def externally_validated_response(API_KEY, main_question, references=[], val_threshold=0.1, visualize=True, initial_response=None):
    console = Console()
    main_response = ""
    no_hallucination = False

    #Initial response generation
    if initial_response is None:
        while main_response == "" or main_response["response"] == "":
            prompt = create_prompt(main_question, references)
            response, logits, tokens, prompt_index = models.LLAMA(prompt)
            main_response = models.process_response(response)
            initial_response=(response, logits, tokens, prompt_index)
    else:
        response, logits, tokens, prompt_index = initial_response
        main_response = models.process_response(response)

    #Validation loop
    while not no_hallucination:
        main_points = main_response["main_points"]
        points_logits = fetch_logits(main_points, tokens, logits)
        external_data = []
        point_scores = []

        for i, point_logits in enumerate(points_logits):
            score = calculate_proportion_below_threshold(point_logits)
            point_scores.append(score)

            if score > val_threshold:
                claim = main_points[i]
                if visualize:
                    console.print(Text(f"Uncertain Claim: {claim}", style="bold yellow"))
                    console.print(Text(f"Uncertain Proportion Score: {score:.2f}", style="italic yellow"))
                    console.print("[bold green]Validating with External Data...[/bold green]")

                ext_data = []
                #Fetch external data
                while not ext_data:
                    ext_data = get_external_data(main_question, API_KEY, num_results=2)
                for data in ext_data:
                    external_data.append(f"{data} is True")

                if visualize:
                    console.print(Panel(f"External Data Retrieved: [bold cyan]{ext_data}[/bold cyan]", title="External Validation"))
                break

        #Check maximum uncertainty and regenerate response if necessary
        if len(point_scores) > 0:
            mean_uncertain_proportion = np.max(point_scores)
            if visualize:
                console.print(Text(f"The Max Uncertainty For Current Response Is: {mean_uncertain_proportion:.2f}", style="bold blue"))

        if len(external_data) > 0:
            main_response = ""
            no_hallucination = False
            if visualize:
                console.print("[bold green]Generating Response With Externally Validated Data...[/bold green]")

            while main_response == "" or main_response["response"] == "":
                prompt = create_prompt(main_question, external_data)
                response, logits, tokens, prompt_index = models.LLAMA(prompt)
                main_response = models.process_response(response)

        else:
            no_hallucination = True
            if visualize:
                console.print(Panel(json.dumps(main_response, indent=1), title="Final Response", style="bold green"))
                console.print(Panel(main_response["response"], title="Externally Validated Answer", style="bold red"))

    return main_response["response"], initial_response

def agent_validated_response(agent_model, main_question, references=[], val_threshold=0.1, visualize=True, initial_response=None):
    console = Console()
    main_response = ""
    no_hallucination = False

    #Initial response generation
    if initial_response is None:
        while main_response == "" or main_response["response"] == "":
            prompt = create_prompt(main_question, references)
            response, logits, tokens, prompt_index = models.LLAMA(prompt)
            main_response = models.process_response(response)
            initial_response=(response, logits, tokens, prompt_index)
    else:
        response, logits, tokens, prompt_index = initial_response
        main_response = models.process_response(response)

    #Validation loop
    while not no_hallucination:
        main_points = main_response["main_points"]
        points_logits = fetch_logits(main_points, tokens, logits)
        agent_data = []
        point_scores = []

        for i, point_logits in enumerate(points_logits):
            pairs = []
            score = calculate_proportion_below_threshold(point_logits)
            point_scores.append(score)

            if score > val_threshold:
                claim = main_points[i]
                if visualize:
                    console.print(Text(f"Uncertain Claim: {claim}", style="bold yellow"))
                    console.print(Text(f"Uncertain Proportion Score: {score:.2f}", style="italic yellow"))
                    console.print("[bold green]Validating with Agent Prediction...[/bold green]")

                #Create pairs for agent validation
                pairs.append((main_question, claim))
                agent_predictions = agent_model.predict(pairs).cpu().item()
                
                #Evaluate the agent's prediction
                answer = "is True" if agent_predictions > 0.5 else "is not True"
                agent_data.append(f"{claim} {answer}")
                
                if visualize:
                    console.print(Panel(f"Agent Prediction: [bold cyan]{claim} {answer}[/bold cyan]", title="Agent Prediction"))

        #Check maximum uncertainty and regenerate response if necessary
        if len(point_scores) > 0:
            mean_uncertain_proportion = np.max(point_scores)
            if visualize:
                console.print(Text(f"The Max Uncertainty For Current Response Is: {mean_uncertain_proportion:.2f}", style="bold blue"))

        if len(agent_data) > 0:
            main_response = ""
            no_hallucination = False
            if visualize:
                console.print("[bold green]Generating Response With Agent Validated Data...[/bold green]")
                
            while main_response == "" or main_response["response"] == "":
                prompt = create_prompt(main_question, agent_data)
                response, logits, tokens, prompt_index = models.LLAMA(prompt)
                main_response = models.process_response(response)
        else:
            no_hallucination = True
            if visualize:
                console.print(Panel(json.dumps(main_response, indent=1), title="Final Response", style="bold green"))
                console.print(Panel(main_response["response"], title="Agent Validated Answer", style="bold red"))

    return main_response["response"], initial_response