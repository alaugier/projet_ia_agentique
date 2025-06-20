from smolagents import CodeAgent, DuckDuckGoSearchTool, tool
from smolagents.models import OpenAIServerModel
import datetime
import requests
import pytz
import yaml
import os
from tools.final_answer import final_answer_tool
from tools.date_tools import get_today_date_french, get_current_time_in_timezone
from tools.quiz_generator_tool import generate_ai900_quiz_with_local_sources, validate_quiz_answer, get_quiz_statistics
from tools.ai900_search_tool import search_ai900_knowledge
from tools.llm_helper import set_global_llm_generator
from tools.source_adder_tool import add_sources_to_quiz_tool, reload_ai900_database, check_ai900_database_status
from Gradio_UI import GradioUI
from dotenv import load_dotenv

# Charge les variables d'environnement depuis le fichier .env
load_dotenv('.env')

# Outil personnalisé exemple
@tool
def my_custom_tool(arg1: str, arg2: int) -> str:
    """A tool that does nothing yet
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"

# Clé API Mistral récupérée depuis les variables d'environnement
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    print("Erreur: La clé API Mistral n'est pas définie. Assurez-vous d'avoir un fichier .env avec MISTRAL_API_KEY=votre_cle.")
    exit(1)

# Configuration du modèle
model = OpenAIServerModel(
    api_key=MISTRAL_API_KEY,
    model_id="mistral-medium-latest",
    api_base="https://api.mistral.ai/v1",
    max_tokens=2096,
    temperature=0.5,
)

# Vérification du statut de la base de données AI-900 au démarrage
print("🔍 Vérification de la base de données AI-900...")
from tools.source_adder_tool import source_matcher
if source_matcher.is_loaded:
    print(f"✅ Base de données AI-900 prête: {len(source_matcher.content_df)} entrées")
else:
    print("⚠️  WARNING: Base de données AI-900 non chargée - les sources utiliseront les URLs de fallback")
    print("💡 Assurez-vous que le fichier 'ai900_content.csv' est présent dans le dossier 'tools/'")

# Essayons d'abord sans templates personnalisés
try:
    print(f"DEBUG: Type de 'model' avant la création de l'agent: {type(model)}")
    print(f"DEBUG: Objet 'model' avant la création de l'agent: {model}")

    # Création de l'agent avec tous les tools including the new diagnostic tool
    agent = CodeAgent(
        model=model,
        tools=[
            final_answer_tool, 
            generate_ai900_quiz_with_local_sources, 
            validate_quiz_answer, 
            search_ai900_knowledge, 
            set_global_llm_generator, 
            add_sources_to_quiz_tool,
            reload_ai900_database,
            check_ai900_database_status,  # Nouveau tool de diagnostic
            my_custom_tool, 
            get_current_time_in_timezone, 
            DuckDuckGoSearchTool(), 
            get_today_date_french
        ],
        max_steps=10,  # Augmenté pour permettre plus d'étapes de diagnostic
        verbosity_level=2,
    )
    print("✅ Agent initialisé avec succès!")
    
except Exception as e:
    print(f"Erreur lors de l'initialisation de l'agent (sans templates): {e}")
    
    # Si ça ne marche pas, essayons avec les templates
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
                    validate_quiz_answer, 
                    search_ai900_knowledge, 
                    set_global_llm_generator, 
                    add_sources_to_quiz_tool,
                    reload_ai900_database,
                    check_ai900_database_status,  # Nouveau tool de diagnostic
                    my_custom_tool, 
                    get_current_time_in_timezone, 
                    DuckDuckGoSearchTool(), 
                    get_today_date_french
                ],
                max_steps=10,  # Augmenté pour permettre plus d'étapes de diagnostic
                verbosity_level=2,
                prompt_templates=prompt_templates
            )
            print("✅ Agent initialisé avec succès avec templates!")
            
        except Exception as e2:
            print(f"Erreur lors de l'initialisation de l'agent (avec templates): {e2}")
            exit(1)
    else:
        print(f"Fichier {prompts_file} non trouvé, arrêt du programme")
        exit(1)

# Initialisation de l'interface Gradio
try:
    ui = GradioUI(agent=agent)
    ui.launch()
except Exception as e:
    print(f"Erreur lors du lancement de l'interface Gradio: {e}")
    exit(1)