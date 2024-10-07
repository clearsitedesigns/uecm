import os
import json
import ollama
import tqdm
import loguru
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from gliner import GLiNER

# Initialize logging
loguru.logger.add("uecm_unstructured.log", rotation="10 MB")

# Initialize GLiNER model
gliner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
default_labels = ["DISEASE", "DRUG", "TREATMENT", "MECHANISM", "TRIAL_PHASE", "APPROVAL_STATUS", "SIDE_EFFECT"]

def extract_text_from_pdfs(pdf_folder):
    extracted_texts = []
    for filename in tqdm.tqdm(os.listdir(pdf_folder), desc="Processing files"):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_folder, filename)
            reader = PdfReader(filepath)
            text = ''
            for page in reader.pages:
                raw_text = page.extract_text()
                cleaned_text = clean_text(raw_text)
                text += cleaned_text
            extracted_texts.append(text)
    return extracted_texts

def clean_text(text):
    text = text.replace('\n', ' ').replace('\r', '')
    text = ' '.join(text.split())
    return text

def perform_ner(texts, threshold=0.5):
    entities = {}
    for text in tqdm.tqdm(texts, desc="Performing NER"):
        detected_entities = gliner_model.predict_entities(text, default_labels)
        for entity in detected_entities:
            if entity['score'] >= threshold:
                entity_name = f"{entity['label']}: {entity['text']}"
                if entity_name in entities:
                    entities[entity_name] += 1
                else:
                    entities[entity_name] = 1
    
    loguru.logger.info(f"Found {len(entities)} unique entities")
    return entities

def analyze_entities_with_objective(entities, objective):
    prompt = f"""SYSTEM INSTRUCTIONS:
    You are a helpful assistant specializing in data analysis and entity recognition. Your task is to analyze entities in the context of a research objective and provide a JSON formatted response. It is crucial that your entire response is valid JSON.

    Research Objective: {objective}

    Entities detected (format is "ENTITY_TYPE: entity_text"):
    {json.dumps(entities, indent=2)}

    Provide a JSON response with the following structure:
    {{
        "relevant_entities": [
            {{
                "entity": "string (entity from the list)",
                "relevance_score": number (0-1),
                "description": "string explaining relevance to objective"
            }},
            ...
        ],
        "key_concepts": ["list", "of", "key", "concepts"],
        "suggested_focus_areas": ["list", "of", "suggested", "research", "focus", "areas"],
        "data_quality_assessment": "string describing the quality and relevance of the detected entities"
    }}

    Remember, your entire response must be valid JSON. Do not include any text outside of the JSON structure.
    """
    
    response = ollama.generate(model='llama3.1:latest', prompt=prompt)
    try:
        return json.loads(response["response"])
    except json.JSONDecodeError:
        loguru.logger.error("Failed to parse LLM response as JSON. Using fallback analysis.")
        return {
            "relevant_entities": [],
            "key_concepts": [],
            "suggested_focus_areas": [],
            "data_quality_assessment": "Unable to assess"
        }

def generate_word_cloud(entities):
    if not entities:
        loguru.logger.error("No entities found or threshold too high.")
        return
    
    # Use only the entity text for the word cloud, not the label
    entity_frequencies = {}
    for entity, count in entities.items():
        entity_text = entity.split(": ", 1)[1] if ": " in entity else entity
        entity_frequencies[entity_text] = count
    
    try:
        wordcloud = WordCloud(width=800, height=400, 
                              background_color='white', 
                              min_font_size=10,
                              max_words=100).generate_from_frequencies(entity_frequencies)
    except ValueError as e:
        loguru.logger.error(f"Error generating word cloud: {e}")
        loguru.logger.info("Attempting to generate word cloud with default settings...")
        try:
            wordcloud = WordCloud(max_words=50).generate_from_frequencies(entity_frequencies)
        except Exception as e:
            loguru.logger.error(f"Failed to generate word cloud: {e}")
            return
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("UEMC: Preflight Entity Checks")
    plt.tight_layout(pad=0)
    
    try:
        plt.savefig('UECM_wordcloud.png')
        loguru.logger.info("Word cloud saved as 'UECM_wordcloud.png'")
    except Exception as e:
        loguru.logger.error(f"Failed to save word cloud: {e}")
    
    plt.close()

def save_uecm_schema(schema, filename='UECM_preflight_unstructured.json'):
    with open(filename, 'w') as f:
        json.dump(schema, f, indent=4)
    loguru.logger.info(f"UECM schema saved to '{filename}'")

def main():
    loguru.logger.info("Starting UECM Schema Generation for Unstructured Data")
    
    objective = input("Please enter your research objective: ")
    
    pdf_folder = input("Enter the path to the folder containing PDF files: ")
    texts = extract_text_from_pdfs(pdf_folder)
    
    threshold = float(input("Enter NER confidence threshold (0.1 to 1.0, default 0.5): ") or 0.5)
    
    loguru.logger.info(f"Performing NER with threshold {threshold}.")
    entities = perform_ner(texts, threshold)
    
    if not entities:
        loguru.logger.warning("No entities found. Consider lowering the threshold.")
    else:
        loguru.logger.info(f"Found {len(entities)} unique entities.")
    
    loguru.logger.info("Analyzing entities in relation to the research objective.")
    analysis = analyze_entities_with_objective(entities, objective)
    
    schema = {
        "research_objective": objective,
        "entity_analysis": analysis,
        "raw_entities": entities
    }
    
    loguru.logger.info("Saving UECM schema to file.")
    save_uecm_schema(schema)
    
    loguru.logger.info("Generating word cloud from entities.")
    generate_word_cloud(entities)

    loguru.logger.info("UECM Schema Generation completed successfully")
    print("UECM Schema has been generated and saved. Please check the output files for results.")

if __name__ == "__main__":
    main()