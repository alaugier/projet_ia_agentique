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
from tools.quiz_generator_tool import generate_ai900_quiz_with_local_sources, validate_quiz_quality, get_quiz_statistics
from tools.ai900_search_tool import search_ai900_knowledge
from tools.llm_helper import set_global_llm_generator
from tools.source_adder_tool import add_sources_to_quiz_tool, reload_ai900_database, check_ai900_database_status
from tools.prepare_json import prepare_json_for_final_answer
from tools.filter_questions import filter_questions_by_keyword
from tools.retrieve_sources import retrieve_sources_by_keywords
from Gradio_UI import GradioUI
from tools.source_adder_tool import reload_ai900_database

result = reload_ai900_database("~/projets/hugginface/tools/ai900_content.csv")
print(result)

print(search_ai900_knowledge("vision par ordinateur"))

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

# Vérification de la base AI-900
print("🔍 Vérification de la base de données AI-900...")
from tools.source_adder_tool import source_matcher
if source_matcher.is_loaded:
    print(f"✅ Base de données AI-900 prête: {len(source_matcher.content_df)} entrées")
else:
    print("⚠️  WARNING: Base de données AI-900 non chargée - les sources utiliseront les URLs de fallback")
    print("💡 Assurez-vous que le fichier 'ai900_content.csv' est présent dans le dossier 'tools/'")

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
            final_answer_tool,
            generate_ai900_quiz_with_local_sources,
            validate_quiz_quality,
            retrieve_sources_by_keywords,
            search_ai900_knowledge,
            set_global_llm_generator,
            add_sources_to_quiz_tool,
            reload_ai900_database,
            check_ai900_database_status,
            my_custom_tool,
            get_current_time_in_timezone,
            DuckDuckGoSearchTool(),
            get_today_date_french,
            prepare_json_for_final_answer,
            filter_questions_by_keyword
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
                    final_answer_tool,
                    generate_ai900_quiz_with_local_sources,
                    validate_quiz_quality,
                    retrieve_sources_by_keywords,
                    search_ai900_knowledge,
                    set_global_llm_generator,
                    add_sources_to_quiz_tool,
                    reload_ai900_database,
                    check_ai900_database_status,
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
    import json

    test_questions = [
        {
            "question": "Qu'est-ce que la vision par ordinateur ?",
            "options": ["A. Analyse d’images", "B. Traitement du son", "C. Base de données", "D. Robotique"],
            "correct_answer": "A. Analyse d’images",
            "explanation": "Elle permet aux machines de comprendre les images.",
            "source_url": "https://learn.microsoft.com/azure/cognitive-services/computer-vision"
        }
    ]

    json_str = prepare_json_for_final_answer(test_questions)
    print(final_answer_tool(json_str))

