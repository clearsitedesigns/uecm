import os
import json
import ollama
import tqdm
import loguru
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from gliner import GLiNER
from bs4 import BeautifulSoup

# Initialize logging
loguru.logger.add("uecm_unstructured.log", rotation="10 MB")

# Initialize GLiNER model
gliner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
default_labels = ["DISEASE", "DRUG", "TREATMENT", "MECHANISM", "TRIAL_PHASE", "APPROVAL_STATUS", "SIDE_EFFECT"]

# Function to recursively ingest files from folders and subfolders
def ingest_files_from_folders(main_folder):
    extracted_texts = []
    for root, _, files in os.walk(main_folder):
        for filename in files:
            filepath = os.path.join(root, filename)
            if filename.endswith(".pdf"):
                extracted_texts.extend(extract_text_from_pdf(filepath))
            elif filename.endswith(".txt"):
                extracted_texts.extend(extract_text_from_text_file(filepath))
            elif filename.endswith(".md"):
                extracted_texts.extend(extract_text_from_markdown(filepath))
            elif filename.endswith(".html"):
                extracted_texts.extend(extract_text_from_html(filepath))
    return extracted_texts

# Function to extract text from PDFs
def extract_text_from_pdf(filepath):
    texts = []
    try:
        reader = PdfReader(filepath)
        text = ''
        for page in reader.pages:
            raw_text = page.extract_text()
            cleaned_text = clean_text(raw_text)
            text += cleaned_text
        texts.append(text)
    except Exception as e:
        loguru.logger.error(f"Failed to extract text from PDF {filepath}: {e}")
    return texts

# Function to extract text from plain text files
def extract_text_from_text_file(filepath):
    texts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
            cleaned_text = clean_text(text)
            texts.append(cleaned_text)
    except Exception as e:
        loguru.logger.error(f"Failed to extract text from text file {filepath}: {e}")
    return texts

# Function to extract text from markdown files
def extract_text_from_markdown(filepath):
    texts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
            cleaned_text = clean_text(text)
            texts.append(cleaned_text)
    except Exception as e:
        loguru.logger.error(f"Failed to extract text from markdown file {filepath}: {e}")
    return texts

# Function to extract text from HTML files
def extract_text_from_html(filepath):
    texts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            text = soup.get_text()
            cleaned_text = clean_text(text)
            texts.append(cleaned_text)
    except Exception as e:
        loguru.logger.error(f"Failed to extract text from HTML file {filepath}: {e}")
    return texts

# Function to clean the extracted text
def clean_text(text):
    text = text.replace('\n', ' ').replace('\r', '')
    text = ' '.join(text.split())
    return text

# Function to perform Named Entity Recognition (NER)
def perform_ner(texts, threshold=0.5):
    entities = {}
    for text in tqdm.tqdm(texts, desc="Performing NER"):
        try:
            detected_entities = gliner_model.predict_entities(text, default_labels)
            for entity in detected_entities:
                if entity['score'] >= threshold:
                    entity_name = f"{entity['label']}: {entity['text']}"
                    if entity_name in entities:
                        entities[entity_name] += 1
                    else:
                        entities[entity_name] = 1
        except Exception as e:
            loguru.logger.error(f"Error during NER: {e}")
    
    loguru.logger.info(f"Found {len(entities)} unique entities")
    return entities

# Function to analyze entities with respect to the research objective
def analyze_entities_with_objective(entities, objective):
    prompt = f"""SYSTEM INSTRUCTIONS:
    You are a helpful assistant specializing in data analysis and entity recognition. Your task is to analyze entities in the context of a research objective and provide a JSON formatted response. It is crucial that your entire response is valid JSON.

    Research Objective: {objective}

    Entities detected (format is "ENTITY_TYPE: entity_text"):
    {json.dumps(entities, indent=2)}

    Categorize the entities into the following fields:
    - "relevant_entities": List of entities with relevance scores and descriptions
    - "key_concepts": List of key concepts related to the research objective
    - "suggested_focus_areas": List of suggested focus areas for further research
    - "data_quality_assessment": Summary of the quality and relevance of the detected entities

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
    
    try:
        response = ollama.generate(model='llama3.1:latest', prompt=prompt)
        return json.loads(response["response"])
    except Exception as e:
        loguru.logger.error(f"Failed to analyze entities with objective: {e}")
        return {
            "relevant_entities": [],
            "key_concepts": [],
            "suggested_focus_areas": [],
            "data_quality_assessment": "Unable to assess"
        }

# Function to generate a word cloud from the entities
def generate_word_cloud(entities):
    if not entities:
        loguru.logger.error("No entities found or threshold too high.")
        return
    
    entity_frequencies = {entity.split(": ")[1]: count for entity, count in entities.items()}
    
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white', min_font_size=10, max_words=100).generate_from_frequencies(entity_frequencies)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("UEMC: Preflight Entity Checks")
        plt.tight_layout(pad=0)
        plt.savefig('UECM_wordcloud.png')
        loguru.logger.info("Word cloud saved as 'UECM_wordcloud.png'")
    except Exception as e:
        loguru.logger.error(f"Failed to generate or save word cloud: {e}")
    finally:
        plt.close()

# Function to save UECM schema
def save_uecm_schema(schema, filename='UECM_preflight_unstructured.json'):
    try:
        with open(filename, 'w') as f:
            json.dump(schema, f, indent=4)
        loguru.logger.info(f"UECM schema saved to '{filename}'")
    except Exception as e:
        loguru.logger.error(f"Failed to save UECM schema: {e}")

def main():
    loguru.logger.info("Starting UECM Schema Generation for Unstructured Data")
    
    # Get the research objective from the user
    objective = input("Please enter your research objective: ")
    
    # Get the path to the folder containing files to process
    pdf_folder = input("Enter the path to the folder containing your files (PDFs, text, markdown, HTML): ")
    texts = ingest_files_from_folders(pdf_folder)
    
    # Get the NER confidence threshold
    threshold = float(input("Enter NER confidence threshold (0.1 to 1.0, default 0.5): ") or 0.5)
    
    # Perform Named Entity Recognition (NER)
    loguru.logger.info(f"Performing NER with threshold {threshold}.")
    entities = perform_ner(texts, threshold)
    
    if not entities:
        loguru.logger.warning("No entities found. Consider lowering the threshold.")
    else:
        loguru.logger.info(f"Found {len(entities)} unique entities.")
    
    # Analyze the entities in relation to the research objective
    loguru.logger.info("Analyzing entities in relation to the research objective.")
    analysis = analyze_entities_with_objective(entities, objective)
    
    # Create the schema that includes the research objective, entity analysis, and raw entities
    schema = {
        "research_objective": objective,
        "entity_analysis": analysis,
        "raw_entities": entities
    }
    
    # Save the UECM schema to a JSON file
    loguru.logger.info("Saving UECM schema to file.")
    save_uecm_schema(schema)
    
    # Generate and save a word cloud based on the entities
    loguru.logger.info("Generating word cloud from entities.")
    generate_word_cloud(entities)

    loguru.logger.info("UECM Schema Generation completed successfully")
    print("UECM Schema has been generated and saved. Please check the output files for results.")

if __name__ == "__main__":
    main()
