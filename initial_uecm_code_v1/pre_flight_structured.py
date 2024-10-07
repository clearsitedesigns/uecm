import requests
import json
import pandas as pd
from rich.console import Console
from gliner import GLiNER  # Assuming GLiNER is available as an NER tool

# Initialize the console for pretty printing
console = Console()

# Configuration for OLLAMA LLM
API_URL = "http://localhost:11434/api/generate"  # Replace with actual API endpoint if different
MODEL = "llama3.1:latest"
SYSTEM_INSTRUCTIONS = "You are a helpful assistant specializing in data analysis and entity recognition. Provide concise and accurate responses in JSON format."

# Function to generate a response using OLLAMA LLM
def generate_response(prompt):
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    response = requests.post(API_URL, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()
    else:
        console.print(f"Error: {response.status_code}")
        return None

# Step 1: User input to define research objective
console.print("Please enter your research objective:")
user_objective = input("Research Objective: ")

# Load structured JSON data from a file
try:
    with open('structured_data.json', 'r') as file:
        structured_data = json.load(file)
except FileNotFoundError:
    console.print("Error: structured_data.json file not found. Please ensure the file exists in the current directory.")
    exit()
except json.JSONDecodeError:
    console.print("Error: Invalid JSON in structured_data.json. Please check the file contents.")
    exit()

# Convert JSON data to Pandas DataFrame
df = pd.DataFrame(structured_data)

# Discover schema using Pandas
console.print("Discovered Schema:")
schema_info = df.dtypes.apply(lambda x: x.name).to_dict()
console.print_json(data=json.dumps(schema_info, indent=4))

# Initialize GLiNER model with default labels for initial analysis
default_labels = ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "MISC"]
try:
    gliner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
except Exception as e:
    console.print(f"Error initializing GLiNER model: {str(e)}")
    exit()

# Function to perform initial entity recognition on DataFrame columns
def perform_initial_entity_recognition(df: pd.DataFrame, columns):
    entities = []
    for col in columns:
        text = ' '.join(df[col].dropna().astype(str))
        detected_entities = gliner_model.predict_entities(text, default_labels)
        entities.append({col: detected_entities})
    return entities

# Perform Initial Entity Recognition
initial_entity_recognition_results = perform_initial_entity_recognition(df, df.columns)

# Display Initial Entity Recognition Results
console.print("\nInitial Entity Recognition Results:")
console.print_json(data=json.dumps(initial_entity_recognition_results, indent=4))

# Use LLM to determine the most appropriate entity structure
llm_prompt = f"""
{SYSTEM_INSTRUCTIONS}

Given the following information:

1. User's Research Objective: {user_objective}

2. Structured Data Schema: {json.dumps(schema_info, indent=2)}

3. Initial Entity Recognition Results: {json.dumps(initial_entity_recognition_results, indent=2)}

Please analyze the user's objective and the structured data, and determine the most appropriate entity structure for this specific research goal. Return a JSON object with the following structure:

{{
    "entity_structure": [
        {{
            "entity_name": "string",
            "description": "string",
            "relevance_to_objective": "string"
        }},
        ...
    ],
    "explanation": "string"
}}

Ensure that the entity structure is tailored to the user's research objective and the content of the structured data.
"""

llm_response = generate_response(llm_prompt)

if llm_response:
    console.print("\nLLM Entity Structure Determination:")
    console.print_json(data=json.dumps(llm_response, indent=4))
    
    try:
        entity_structure = json.loads(llm_response.get('response', '{}')).get('entity_structure', [])
        explanation = json.loads(llm_response.get('response', '{}')).get('explanation', '')
        
        if entity_structure:
            console.print("\nDetermined Entity Structure:")
            console.print_json(data=json.dumps(entity_structure, indent=4))
            console.print(f"\nExplanation: {explanation}")
            
            # Use the determined entity structure for final entity recognition
            final_labels = [entity['entity_name'] for entity in entity_structure]
            
            # Perform final entity recognition with new labels
            final_entity_recognition_results = perform_initial_entity_recognition(df, df.columns)
            
            console.print("\nFinal Entity Recognition Results:")
            console.print_json(data=json.dumps(final_entity_recognition_results, indent=4))
            
            # Generate and save the final preflight schema
            preflight_schema = {
                "user_objective": user_objective,
                "entity_structure": entity_structure,
                "schema": schema_info,
                "entity_recognition_results": final_entity_recognition_results
            }
            
            with open('UECM_preflight_structured.json', 'w') as outfile:
                json.dump(preflight_schema, outfile, indent=4)
            
            console.print("\nFinal Preflight Schema saved to UECM_preflight_structured.json")
        else:
            console.print("Error: No entity structure provided by the LLM.")
    except json.JSONDecodeError:
        console.print("Error: Unable to parse the entity structure from LLM response.")
else:
    console.print("Failed to determine entity structure with the LLM.")