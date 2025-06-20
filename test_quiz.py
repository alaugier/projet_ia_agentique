import os
from dotenv import load_dotenv
from smolagents.models import OpenAIServerModel
from tools.llm_helper import set_global_llm_generator, get_global_llm_generator
from tools.quiz_generator_tool import generate_ai900_quiz

# Charge les variables d'environnement
load_dotenv('.env')

# Initialise le modèle et le générateur global
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if MISTRAL_API_KEY:
    model = OpenAIServerModel(
        api_key=MISTRAL_API_KEY,
        model_id="mistral-medium-latest",
        api_base="https://api.mistral.ai/v1",
        max_tokens=2048,
        temperature=0.7,
    )
    set_global_llm_generator(model) # Définir le générateur global
else:
    print("MISTRAL_API_KEY non trouvée. Le générateur LLM ne sera probablement pas utilisable.")

# Tente de générer les questions
try:
    print("\n--- Tentative de génération de questions ---")
    questions_json = generate_ai900_quiz(topic="general", num_questions=5, difficulty="beginner", language="french")
    print("\n--- Résultat de generate_ai900_quiz ---")
    print(questions_json)
except Exception as e:
    print(f"\n--- Erreur lors de l'appel à generate_ai900_quiz ---")
    print(e)