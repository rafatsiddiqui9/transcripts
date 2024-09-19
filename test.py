from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv("NVIDIA_API_KEY")
)

completion = client.chat.completions.create(
  model="meta/llama-3.1-405b-instruct",
  messages=[{"role":"user","content":"Generate a text that describes the diet of a bird. Output the text and annotations only according to this json schema and provide only the json. { \"text\":\"\", \"plant food\":[], \"animal food\":[], \"group behavior\":[], \"group species\":[], \"eating time\":[], \"eating location\":[] }"}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")



from openai import OpenAI
import os
import dotenv
import json
import time
from typing import List, Dict
import pandas as pd
import re

dotenv.load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# Parse concepts
concepts = []

def clean_json_string(json_string: str) -> str:
    # Remove any leading/trailing whitespace
    json_string = json_string.strip()
    
    # Find the start and end of the JSON content
    json_start = json_string.find('{')
    json_end = json_string.rfind('}')
    
    if json_start != -1 and json_end != -1:
        json_string = json_string[json_start:json_end+1]
    
    # If the string starts and ends with triple backticks, remove them
    if json_string.startswith('```') and json_string.endswith('```'):
        json_string = json_string[3:-3].strip()
    
    # Remove any "json" or "JSON" language identifier after the opening backticks
    json_string = re.sub(r'^json\s*', '', json_string, flags=re.IGNORECASE)
    
    # Remove line breaks within string values
    json_string = re.sub(r'(?<=:)\s*\n\s*"([^"]*)"', r' "\1"', json_string)
    
    return json_string

def generate_concept_example(concept: str) -> Dict:
    prompt = f"""
    Generate a detailed explanation of the concept "{concept}" in qualitative market research, ethnography, or social anthropology. Then, extract and categorize relevant information according to the specified JSON schema. Ensure all fields are filled, using "N/A" if information is not directly applicable. Provide only the JSON output, strictly adhering to this format as mentioned below. Do not include any other text or comments.

    {{
        "concept_name": "{concept}",
        "definition": "A concise definition of the concept",
        "detailed_explanation": "A more in-depth explanation of the concept, its origins, and its significance. Ensure that the detailed explanation is very precise and thorough. If the concepts has types which in turn have subtypes, then ensure that the detailed explanation is very precise and thorough and explains types under the concepts and/or nuances of the concept as well as the subtypes and subcategories, till you cannot go deeper, in at least 2 sentences each`",
        # "key_theorists": ["List", "of", "key", "theorists", "associated", "with", "the", "concept"],
        # "related_concepts": ["List", "of", "related", "concepts", "in", "the", "field"],
        # "applications": ["List", "of", "practical", "applications", "or", "use", "cases"],
        # "research_methods": ["List", "of", "research", "methods", "associated", "with", "this", "concept"],
        # "challenges": ["List", "of", "challenges", "or", "limitations", "in", "applying", "this", "concept"],
        "example_scenario": ["List", "of", "10", "detailed", "high", "quality", "examples", "scenarios", "illustrating", "the", "concept", "in", "action.", "Each", "example", "MUST", "be", "at", "least", "3", "sentences", "long.", "Ensure", "that", "examples", "cover", "different", "subtypes", "or", "subcategories", "if", "applicable."]
    }}
    """

    try:
        completion = client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=2048
        )
        cleaned_content = clean_json_string(completion.choices[0].message.content)
        print(f"Cleaned JSON string: {cleaned_content}")  # Add this line
        return json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error for {concept}: {str(e)}")
        print("Raw response:", completion.choices[0].message.content)
        return {"concept_name": concept, "error": str(e)}
    except Exception as e:
        print(f"Error generating example for {concept}: {str(e)}")
        return {"concept_name": concept, "error": str(e)}

def generate_examples(concepts: List[str]) -> List[Dict]:
    all_examples = []
    for concept in concepts:
        example = generate_concept_example(concept)
        print(f"Generated example for: {concept}")
        print(f'Generated example: {example}')
        if example:
            all_examples.append(example)
        else:
            print(f"Failed to generate example for: {concept}")
        time.sleep(1)  # Rate limiting
    return all_examples

# Generate examples
examples = generate_examples(concepts)

# Convert to DataFrame
df = pd.DataFrame(examples)

# Save to CSV
df.to_csv("concept_examples.csv", index=False)

print(f"Generated {len(examples)} examples and saved to concept_examples.csv")

# Optional: Print a sample of the data
if examples:
    print("\nSample data for the first concept:")
    print(json.dumps(examples[0], indent=2))


from openai import OpenAI
import os
import dotenv
import json
import time
from typing import List, Dict
import pandas as pd
import re

dotenv.load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# Parse concepts
concepts = [
    # "Jungian Archetypes",
        "Phenomenology", "Grounded Theory", "Thick Description", "Symbolic Interactionism",
    "Ethnographic Interview", "Semiosphere", "Cultural Relativism", "Social Constructivism",
    "Intersectionality", "Cognitive Schemas", "Interpretative Phenomenological Analysis (IPA)",
    "Discourse Analysis", "Autoethnography", "Semiotics of Space", "Paraverbal Communication",
    "Cognitive Anthropology", "Cultural Models", "Multimodal Discourse Analysis",
    "Semiotic Landscape", "Polysemy", "Communities of Practice", "Chronemics", "Proxemics",
    "Kinesics", "Ethnolinguistics", "Culturomics", "Ethnocentrism", "Cultural Competence",
    "Pragmatics", "Metacommunication", "Ethnosemantics", "Ethnopsychology", "Cultural Consonance",
    "Semiotic Resources", "Intersubjectivity", "Ethnoecology", "Ethnohistory", "Ethnomethodology",
    "Semiotics of Fashion", "Social Constructionism", "Cultural Logic", "Psychogeography",
    "Thick Data", "Semiotic Square", "Ethnomethodological Indifference", "Symbolic Convergence Theory",
    "Cultural Affordances", "Ethnopragmatics", "Semiotic Mediation", "Affect Theory",
    "Cultural Semaphores", "Ethnosymbolism", "Social Proof", "Chronotope", "Cognitive Mapping",
    "Cultural Imperialism", "Heteroglossia", "Semiotic Chaining", "Cultural Intimacy",
    "Ethnometaphysics", "Semiotic Landscape", "Cultural Pluralism", "Semiotic Democracy",
    "Cultural Consonance", "Semiotic Remediation", "Cultural Cartography", "Semiotic Ideologies",
    "Cultural Entropy", "Ethnomimesis", "Semiotic Anthropology", "Cultural Humility",
    "Ethnosemiotics", "Semiotic Bundling", "Ethnolinguistic Vitality", "Semiotic Ecology",
    "Cultural Probes", "Ethnoaesthetics", "Semiotic Bricolage", "Cultural Metacognition",
    "Interpellation", "Liminality", "Representation", "Identity Construction", "Discourse Theory",
    "Symbolic Interaction", "Cultural Narrative", "Ritualized Behavior", "Social Identity",
    "Social Role", "Mental Models", "Social Influence", "Habitus", "Cultural Relativism",
    "Sociolinguistics", "Psychoanalytic Theory", "Poststructuralism", "Cultural Memory",
    "Collective Identity", "Fieldwork", "Narrative Inquiry", "Ethical Considerations",
    "Social Dynamics", "Meaning-Making", "Cultural Transmission", "Normative Behavior",
    "Social Capital", "Cognitive Bias", "Psycho-Social Development", "Social Construction of Reality",
    "Cultural Assimilation", "Semantic Memory", "Social Learning Theory", "Cultural Narratives",
    "Interaction Rituals", "Cultural Adaptation", "Behavioral Conditioning", "Experiential Learning",
    "Linguistic Relativity", "Performative Acts", "Critical Theory of Communication",
    "Emotional Intelligence", "Transitional Identity", "Normative Influence", "Cultural Encapsulation",
    "Intercultural Communication", "Self-Efficacy", "Social Identity Complexity", "Hermeneutic Circle",
    "Meta-Cognition", "Cultural Codes", "Themes, Triggers and Barriers", "Tensions and Aspirations",
    "Signifier and Signified", "Narrative Identity", "Cultural Scripts", "Ethnography",
    "Semiotic Analysis", "Heuristics", "Self-concept", "Role Theory", "Socialization",
    "Symbolic Capital", "Cognitive Dissonance", "Cultural Capital", "Mimesis", "Behavioral Economics",
    "Projective Techniques", "Metaphor Theory", "Critical Discourse Analysis", "Social Identity Theory",
    "Narrative Paradigm", "Agency", "Cultural Hybridity", "Ethnomethodology", "Social Cognition",
    "Cross-cultural Psychology", "Emic and Etic", "Hermeneutics", "Social Norms",
    "Social Exchange Theory", "Transcultural Psychology", "Postcolonial Theory", "Identity Formation",
    "Digital Semiotics", "Embodied Cognition", "Rituals and Rites of Passage",
    "Semantic Networks and Mental Models", "Collective Memory"
]

def clean_json_string(json_string: str) -> str:
    # Remove any leading/trailing whitespace
    json_string = json_string.strip()
    
    # Find the start and end of the JSON content
    json_start = json_string.find('{')
    json_end = json_string.rfind('}')
    
    if json_start != -1 and json_end != -1:
        json_string = json_string[json_start:json_end+1]
    
    # If the string starts and ends with triple backticks, remove them
    if json_string.startswith('```') and json_string.endswith('```'):
        json_string = json_string[3:-3].strip()
    
    # Remove any "json" or "JSON" language identifier after the opening backticks
    json_string = re.sub(r'^json\s*', '', json_string, flags=re.IGNORECASE)
    
    # Remove line breaks within string values
    json_string = re.sub(r'(?<=:)\s*\n\s*"([^"]*)"', r' "\1"', json_string)
    
    return json_string

def generate_concept_example(concept: str) -> Dict:
    prompt = f"""
    Generate a detailed explanation of the concept "{concept}" in qualitative market research, ethnography, or social anthropology. Then, extract and categorize relevant information according to the specified JSON schema. Ensure all fields are filled, using "N/A" if information is not directly applicable. Provide only the JSON output, strictly adhering to this format as mentioned below. Do not include any other text or comments.

    {{
        "concept_name": "{concept}",
        "detailed_explanation": "A more in-depth explanation of the concept, its origins, and its significance. Ensure that the detailed explanation is very precise and thorough. If the concepts has types which in turn have subtypes, then ensure that the detailed explanation is very precise and thorough and explains types under the concepts and/or nuances of the concept as well as the subtypes and subcategories, till you cannot go deeper, in at least 2 sentences each`",
        "example_scenario": ["List of 3 detailed high quality examples scenarios illustrating the concept in action. Each example MUST be at least 3 sentences long. Ensure that examples cover different subtypes or subcategories if applicable. Ensure that examples are not just limited to the concept, but also include types under the concept and/or nuances of the concept. Ensure that the examples are very precise and thorough and are not vague or ambiguous and are atleast 100 words each"]
    }}
    """

    try:
        completion = client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=2048
        )
        cleaned_content = clean_json_string(completion.choices[0].message.content)
        print(f"Cleaned JSON string: {cleaned_content}")  # Add this line
        return json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error for {concept}: {str(e)}")
        print("Raw response:", completion.choices[0].message.content)
        return {"concept_name": concept, "error": str(e)}
    except Exception as e:
        print(f"Error generating example for {concept}: {str(e)}")
        return {"concept_name": concept, "error": str(e)}

def generate_and_save_examples(concepts: List[str], output_file: str = "concept_examples.csv") -> None:
    df = pd.DataFrame(columns=["concept_name", "detailed_explanation", "example_scenario"])
    
    for concept in concepts:
        example = generate_concept_example(concept)
        print(f"Generated example for: {concept}")
        
        if example and "error" not in example:
            # Append the new example to the DataFrame
            new_row = pd.DataFrame([example])
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Save the updated DataFrame to CSV
            df.to_csv(output_file, index=False)
            print(f"Updated CSV with example for: {concept}")
        else:
            print(f"Failed to generate example for: {concept}")
        
        time.sleep(1)  # Rate limiting
    
    print(f"Generated {len(df)} examples and saved to {output_file}")
    
    # Optional: Print a sample of the data
    if not df.empty:
        print("\nSample data for the first concept:")
        print(json.dumps(df.iloc[0].to_dict(), indent=2))

# Generate examples and save incrementally
generate_and_save_examples(concepts)
