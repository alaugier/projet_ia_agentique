# app.py
from smolagents import DuckDuckGoSearchTool, tool
from smolagents.models import OpenAIServerModel
from smolagents.agents import CodeAgent
import datetime
import requests
import pytz
import yaml
import os
from dotenv import load_dotenv

from tools.final_answer import final_answer_tool
from tools.date_tools import get_today_date_french, get_current_time_in_timezone
from tools.quiz_generator_tool import generate_ai900_quiz_with_local_sources
from tools.ai900_search_tool import search_ai900_knowledge
from tools.llm_helper import set_global_llm_generator
from tools.source_adder_tool import add_sources_to_quiz_tool
from tools.prepare_json import prepare_json_for_final_answer
from tools.filter_questions import filter_questions_by_keyword
from tools.retrieve_sources import retrieve_sources_by_keywords
from Gradio_UI import GradioUI
from tools.final_answer_block import final_answer_block

from tools.llm_helper import test_topic_context_extraction

# Test avec un topic existant
result = test_topic_context_extraction("computer_vision")
print(result)

# Test avec un topic personnalisé
result = test_topic_context_extraction("machine_learning", "azure automated ml")
print(result)

# Chargement des variables d'environnement depuis .env
load_dotenv(".env")

# Vérification clé API
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("Erreur: la clé MISTRAL_API_KEY est manquante dans .env")
    exit(1)

# Configuration du modèle Mistral
model = OpenAIServerModel(
    api_key=MISTRAL_API_KEY,
    model_id="mistral-medium-latest",
    api_base="https://api.mistral.ai/v1",
    max_tokens=2048,
    temperature=0.5,
)

# Définition d'un outil personnalisé pour test
@tool
def my_custom_tool(arg1: str, arg2: int) -> str:
    """
    A custom test tool.
    Args:
        arg1: premier argument
        arg2: deuxième argument
    """
    return f"Tool reçu: {arg1} et {arg2}"

# Initialisation de l'agent
try:
    print(f"DEBUG: Type de 'model' avant la création de l'agent: {type(model)}")
    print(f"DEBUG: Objet 'model' avant la création de l'agent: {model}")

    agent = CodeAgent(
        model=model,
        tools=[
            final_answer_block,
            generate_ai900_quiz_with_local_sources,
            retrieve_sources_by_keywords,
            search_ai900_knowledge,
            set_global_llm_generator,
            add_sources_to_quiz_tool,
            my_custom_tool,
            get_current_time_in_timezone,
            DuckDuckGoSearchTool(),
            get_today_date_french,
            prepare_json_for_final_answer,
            filter_questions_by_keyword,
        ],
        max_steps=10,
        verbosity_level=2,
        additional_authorized_imports=["json"]
    )
    print("✅ Agent initialisé avec succès!")

except Exception as e:
    print(f"Erreur lors de l'initialisation de l'agent (sans templates): {e}")
    prompts_file = "prompts.yaml"
    if os.path.exists(prompts_file):
        try:
            with open(prompts_file, 'r', encoding='utf-8') as stream:
                prompt_templates = yaml.safe_load(stream)
                print("Templates de prompts chargés avec succès!")

            agent = CodeAgent(
                model=model,
                tools=[
                    final_answer_block,
                    generate_ai900_quiz_with_local_sources,
                    retrieve_sources_by_keywords,
                    search_ai900_knowledge,
                    set_global_llm_generator,
                    add_sources_to_quiz_tool,
                    my_custom_tool,
                    get_current_time_in_timezone,
                    DuckDuckGoSearchTool(),
                    get_today_date_french,
                    prepare_json_for_final_answer,
                    filter_questions_by_keyword
                ],
                max_steps=10,
                verbosity_level=2,
                additional_authorized_imports=["json"],
                prompt_templates=prompt_templates
            )
            print("✅ Agent initialisé avec succès avec templates!")

        except Exception as e2:
            print(f"Erreur lors de l'initialisation de l'agent (avec templates): {e2}")
            exit(1)
    else:
        print(f"Fichier {prompts_file} non trouvé, arrêt du programme")
        exit(1)

# Interface utilisateur Gradio
try:
    ui = GradioUI(agent=agent)
    ui.launch()
except Exception as e:
    print(f"Erreur lors du lancement de l'interface Gradio: {e}")
    exit(1)

if __name__ == "__main__":
    from tools.prepare_json import prepare_json_for_final_answer
    from tools.final_answer_block import final_answer_block

    # Appel correct avec tous les paramètres explicites
    result = generate_ai900_quiz_with_local_sources(
        topic="nlp", 
        num_questions=2,
        difficulty="intermediate",
        language="french",
        num_relevant_sources=3,
        output_format="json"
    )
    
    # Debug pour voir ce que retourne la fonction
    print(f"Type de result: {type(result)}")
    print(f"Contenu de result: {result}")
    
    # Si result est une string JSON, il faut la parser
    try:
        if isinstance(result, str):
            import json
            parsed_result = json.loads(result)
            json_data = prepare_json_for_final_answer(parsed_result.get("questions", []))
        else:
            # Si result est déjà un dict
            json_data = prepare_json_for_final_answer(result.get("questions", []))
            
        markdown = final_answer_block(json_data)
        print(markdown)
        
    except Exception as e:
        print(f"Erreur lors du traitement du résultat: {e}")
        print(f"Résultat brut: {result}")

# Correction 2: Test plus sûr en début de fichier
try:
    # Test avec un topic existant
    result = test_topic_context_extraction("computer_vision")
    print("✅ Test computer_vision réussi")
    print(result)
except Exception as e:
    print(f"❌ Erreur test computer_vision: {e}")

try:
    # Test avec un topic personnalisé
    result = test_topic_context_extraction("machine_learning", "azure automated ml")
    print("✅ Test machine_learning réussi")
    print(result)
except Exception as e:
    print(f"❌ Erreur test machine_learning: {e}")

