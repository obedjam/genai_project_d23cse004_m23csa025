from transformers import PreTrainedTokenizerFast, LlamaForCausalLM, AutoModelForSequenceClassification
import torch
import numpy as np
from huggingface_hub import login
import warnings
import copy
import json

warnings.filterwarnings("ignore")
login(token="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")

model = None
tokenizer = None
agent_model = None
history = {
    "prompt_1": {
        "question": "Is Berlin the capital of France?",
        "reference" : ["the capital of France is Paris is True"],
        "response": "The capital of France is Paris. Berlin is the capital of Germany. Paris is one of the most iconic cities in France, known for its historical landmarks such as the Eiffel Tower.",
        "main_points": [
            "The capital of France is Paris.",
            "Berlin is the capital of Germany.",
            "Paris is known for its historical landmarks such as the Eiffel Tower."
        ]
    },
    "prompt_2": {
        "question": "Explain the process of photosynthesis.",
        "reference" : [],
        "response": "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to produce glucose and oxygen. The process primarily occurs in the chloroplasts of plant cells. It is essential for producing the oxygen that supports life on Earth.",
        "main_points": [
            "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
            "The process occurs in the chloroplasts of plant cells.",
            "Photosynthesis is essential for producing the oxygen that supports life on Earth."
        ]
    },
    "prompt_3": {
        "question": "Was 'Romeo and Juliet' written by J.K. Rowling?",
        "reference" : ["'Romeo and Juliet' was written by J.K. Rowling is not True"],
        "response": "'Romeo and Juliet' was not written by J.K. Rowling. It is one of the most famous tragedies in English literature by J.K. Rowling. J.K. Rowling is best known for the 'Harry Potter' series, written centuries after Shakespeare's works.",
        "main_points": [
            "'Romeo and Juliet' was written by William Shakespeare.",
            "It is one of the most famous tragedies in English literature.",
            "J.K. Rowling is known for the 'Harry Potter' series, written centuries after Shakespeare's works."
        ]
    },
    "prompt_4": {
        "question": "What is the boiling point of water at sea level?",
        "reference" : [],
        "response": "The boiling point of water at sea level is 100 degrees Celsius or 212 degrees Fahrenheit. This is a standard physical property under normal atmospheric pressure conditions.",
        "main_points": [
            "The boiling point of water at sea level is 100°C or 212°F.",
            "This is measured under normal atmospheric pressure."
        ]
    }
}

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_llama():
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = PreTrainedTokenizerFast.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.float16,
        use_auth_token=True,
    ).half().to("cuda")
    
    if torch.cuda.is_available():
        model.config.use_cache = True
    
    return torch.compile(model), tokenizer

def get_hhem():
    model = AutoModelForSequenceClassification.from_pretrained(
        'vectara/hallucination_evaluation_model', trust_remote_code=True).to("cuda")
    return model

def initialize_models():
    global model
    global tokenizer
    global agent_model
    model, tokenizer = get_llama()
    agent_model = get_hhem()
    model.eval()
    agent_model.eval()

def process_response(response):
    try:
        expected_keys = {"question", "reference", "response", "main_points"}
        extracted_json, found_keys = [], set()
        in_quotes = main_points_start = main_points_end = json_started = False

        for char in response:
            if not json_started and char == "{":
                json_started = True
            if json_started:
                extracted_json.append(char)
                if char == '"':
                    in_quotes = not in_quotes
                if not in_quotes and '"main_points"' in "".join(extracted_json) and not main_points_start:
                    main_points_start = True
                if main_points_start and not in_quotes and char == "]":
                    main_points_end = True
                if not in_quotes:
                    for key in expected_keys:
                        if f'"{key}"' in "".join(extracted_json) and key not in found_keys:
                            found_keys.add(key)
                if found_keys == expected_keys and main_points_end:
                    extracted_json.append("}}")
                    break

        try:
            parsed_json = json.loads("".join(extracted_json))
            if isinstance(parsed_json, dict):
                first_child_key = next(iter(parsed_json), None)
                if first_child_key:
                    child_json = parsed_json[first_child_key]
                    if all(key in child_json for key in expected_keys):
                        return child_json
            return ""
        except json.JSONDecodeError:
            return ""
    except:
        return ""
    
def LLAMA(prompt):
    prompt_index = f"prompt_{len(history) + 1}"
    template = copy.deepcopy(history)
    template[prompt_index] = prompt

    prompt_template = f"""
    You are an AI that must respond **only** in JSON format. Your responses should strictly follow the structure below:
    
    **Examples:**
    {json.dumps(template, indent=2)}
    
    **Instructions:**
    - Only generate one JSON for the most recent prompt as shown in the examples.
    - Use the provided "reference" field if available to validate if a claim is true or not .
    - Do not leave the "response" and "main_points" fields not empty.
    - Start your response with a valid JSON object.

    Answer to {prompt_index}:
    """
    inputs = tokenizer(prompt_template, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=300,
            output_scores=True, 
            return_dict_in_generate=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            return_legacy_cache=False
        )
    
    generated_tokens = outputs.sequences[:, inputs['input_ids'].shape[-1]:]
    response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    return response, outputs.scores, generated_tokens[0], prompt_index