import json
import loguru
from pyvis.network import Network
from rich.console import Console
import requests
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the console for pretty printing
console = Console()

# Ollama API Configuration
API_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:latest"
SYSTEM_INSTRUCTIONS = "You are a helpful assistant specializing in data analysis and entity recognition. Provide concise and accurate responses in JSON format."

class MemoryManager:
    def __init__(self, uecm_schema):
        self.schema = uecm_schema
        self.entities = {entity['name']: entity for entity in uecm_schema['entities']}
        self.nlp = spacy.load("en_core_web_sm")

    def process_query(self, query):
        doc = self.nlp(query)
        relevant_entities = self.find_relevant_entities(doc)
        return self.execute_query(doc, relevant_entities)

    def find_relevant_entities(self, doc):
        relevant_entities = []
        for token in doc:
            for entity_name, entity in self.entities.items():
                if token.text.lower() in entity_name.lower():
                    relevant_entities.append(entity)
        return relevant_entities

    def execute_query(self, doc, relevant_entities):
        if "mechanism of action" in doc.text.lower():
            return self.query_mechanism_of_action(relevant_entities)
        elif "clinical trial" in doc.text.lower():
            return self.query_clinical_trials(relevant_entities)
        else:
            return f"Query not recognized. Relevant entities: {[e['name'] for e in relevant_entities]}"

    def query_mechanism_of_action(self, relevant_entities):
        results = []
        for entity in relevant_entities:
            if entity['name'] == "Drug Name":
                moa = next((r for r in entity['relationships'] if r['target'] == "Mechanism of Action"), None)
                if moa:
                    results.append(f"The mechanism of action for {entity['name']} is: {moa['description']}")
        return "\n".join(results) if results else "No mechanism of action information found."

    def query_clinical_trials(self, relevant_entities):
        results = []
        for entity in relevant_entities:
            if entity['name'] == "Drug Name" or entity['name'] == "Disease Name":
                trial = next((r for r in entity['relationships'] if r['target'] == "Clinical Trial Phase"), None)
                if trial:
                    results.append(f"Clinical trial information for {entity['name']}: {trial['description']}")
        return "\n".join(results) if results else "No clinical trial information found."

    def visualize_query_results(self, query_result):
        net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
        entities = query_result.split("\n")
        for entity in entities:
            net.add_node(entity, label=entity, color="#00ff1e")
        net.show("query_result_visualization.html")

    def generate_heatmap(self, query_result):
        plt.figure(figsize=(10, 8))
        sns.heatmap([[1, 2, 3], [4, 5, 6], [7, 8, 9]], annot=True, cmap="YlGnBu")
        plt.title("Query Result Heatmap")
        plt.savefig("query_result_heatmap.png")
        plt.close()

def get_user_research_goal():
    console.print("Please review your initial research objective and provide any refinements or additional focus areas:")
    console.print("Initial Objective: To identify drugs that may work to treat new forms of cancer by analyzing the latest research techniques")
    refined_goal = input("Refined Research Goal: ")
    return refined_goal

def compare_and_integrate_schemas_with_llm(structured_schema, unstructured_schema, research_goal):
    # Update prompt with specific instructions for pediatric cancer research
    prompt = f"""
    SYSTEM INSTRUCTIONS:
    You are a helpful assistant specializing in data analysis and entity recognition. Please compare the following two JSON schemas and combine them into a unified schema. 
    Ensure that the final schema accurately represents all entities and their relationships, with a focus on pediatric cancer research.

    Research Goal: {research_goal}

    Structured Schema:
    {json.dumps(structured_schema, indent=4)}

    Unstructured Schema:
    {json.dumps(unstructured_schema, indent=4)}

    Your task:
    1. Merge the entities and relationships from both schemas.
    2. Prioritize entities and relationships that are most relevant to pediatric cancers.
    3. Assign a relevance score (0-1) to each entity based on its importance to the research goal.
    4. Exclude entities that do not directly relate to childhood cancers.
    5. Suggest new relationships or entities that might be valuable for pediatric cancer research.
    6. Provide a brief explanation for any significant merges or additions.

    Provide the final combined schema in valid JSON format with the following structure:
    {{
        "research_goal": "string",
        "entities": [
            {{
                "name": "string",
                "type": "string",
                "relevance_score": float,
                "description": "string",
                "relationships": [
                    {{
                        "target": "string",
                        "type": "string",
                        "description": "string"
                    }}
                ]
            }}
        ],
        "insights": [
            {{
                "description": "string",
                "relevance_to_goal": "string"
            }}
        ],
        "suggestions": [
            {{
                "title": "string",
                "description": "string"
            }}
        ]
    }}
    """
    # Same API call and response handling as before...


    headers = {'Content-Type': 'application/json'}
    data = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    response = requests.post(API_URL, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        try:
            combined_schema = json.loads(response.json().get("response", "{}"))
            return combined_schema
        except json.JSONDecodeError as e:
            console.print(f"JSON Decode Error: {e}")
            return None
    else:
        console.print(f"Error: {response.status_code}")
        return None

def visualize_knowledge_graph(schema):
    net = Network(height='750px', width='100%', directed=True, notebook=True)
    
    for entity in schema.get('entities', []):
        node_color = get_node_color(entity['relevance_score'])
        net.add_node(entity['name'], label=entity['name'], title=f"Type: {entity['type']}\nRelevance: {entity['relevance_score']:.2f}", color=node_color)
    
    for entity in schema.get('entities', []):
        for relationship in entity.get('relationships', []):
            if relationship['target'] in [node['id'] for node in net.nodes]:
                net.add_edge(entity['name'], relationship['target'], title=relationship['type'], arrows='to')
            else:
                loguru.logger.warning(f"Target node '{relationship['target']}' not found for relationship from '{entity['name']}'")
    
    net.show_buttons(filter_=['physics'])
    net.save_graph('UECM_knowledge_graph.html')
    console.print("Knowledge graph saved as 'UECM_knowledge_graph.html'")

def get_node_color(relevance_score):
    r = int(255 * (1 - relevance_score))
    g = int(255 * relevance_score)
    return f'rgb({r},{g},0)'

def generate_possible_queries(schema):
    prompt = f"""
    SYSTEM INSTRUCTIONS:
    Based on the following UECM schema, generate a list of 10 possible research queries that can now be conducted using the understood relationships. 
    Ensure that the queries are diverse and cover various aspects of the research goal.

    UECM Schema:
    {json.dumps(schema, indent=4)}

    Provide the list of queries in JSON format with the following structure:
    {{
        "possible_queries": [
            {{
                "query": "string",
                "relevance_to_goal": "string",
                "entities_involved": ["string"]
            }}
        ]
    }}
    """

    headers = {'Content-Type': 'application/json'}
    data = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    response = requests.post(API_URL, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        try:
            queries = json.loads(response.json().get("response", "{}"))
            return queries
        except json.JSONDecodeError as e:
            console.print(f"JSON Decode Error: {e}")
            return None
    else:
        console.print(f"Error: {response.status_code}")
        return None

def main():
    loguru.logger.info("Loading structured and unstructured schemas.")
    
    with open('UECM_preflight_structured.json', 'r') as file:
        structured_schema = json.load(file)
    with open('UECM_preflight_unstructured.json', 'r') as file:
        unstructured_schema = json.load(file)
    
    refined_research_goal = get_user_research_goal()
    
    loguru.logger.info("Comparing and integrating schemas with refined research goal.")
    combined_schema = compare_and_integrate_schemas_with_llm(structured_schema, unstructured_schema, refined_research_goal)

    if combined_schema:
        loguru.logger.info("Final UECM Schema generated successfully.")
        with open('UECM_final_schema.json', 'w') as outfile:
            json.dump(combined_schema, outfile, indent=4)
        console.print_json(data=json.dumps(combined_schema, indent=4))
        
        loguru.logger.info("Visualizing knowledge graph.")
        visualize_knowledge_graph(combined_schema)
        
        console.print("\nKey Insights:")
        for insight in combined_schema.get('insights', []):
            console.print(f"- {insight['description']}")
            console.print(f"  Relevance: {insight['relevance_to_goal']}")
        
        console.print("\nSuggestions:")
        for suggestion in combined_schema.get('suggestions', []):
            console.print(f"- {suggestion['title']}")
            console.print(f"  Description: {suggestion['description']}")
        
        loguru.logger.info("Generating possible research queries.")
        possible_queries = generate_possible_queries(combined_schema)
        
        if possible_queries:
            console.print("\nPossible Research Queries:")
            for query in possible_queries.get('possible_queries', []):
                console.print(f"- Query: {query['query']}")
                console.print(f"  Relevance: {query['relevance_to_goal']}")
                console.print(f"  Entities Involved: {', '.join(query['entities_involved'])}")
        
        loguru.logger.info("UECM analysis completed successfully.")

        # Initialize MemoryManager with the combined schema
        memory_manager = MemoryManager(combined_schema)

        # Enter query processing loop
        while True:
            query = input("\nEnter your query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break

            result = memory_manager.process_query(query)
            console.print("Query Result:")
            console.print(result)

            # Visualize the results
            memory_manager.visualize_query_results(result)
            memory_manager.generate_heatmap(result)

            console.print("Query result visualization saved as 'query_result_visualization.html'")
            console.print("Query result heatmap saved as 'query_result_heatmap.png'")

    else:
        loguru.logger.error("Failed to generate final UECM schema.")

if __name__ == "__main__":
    main()