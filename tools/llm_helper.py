# tools/llm_helper.py
import os
import random
import re
import json # Nécessaire pour la méthode generate_questions et le parsing JSON
from typing import Optional, List, Dict
from smolagents import tool
from smolagents.models import OpenAIServerModel

# La classe LLMQuestionGenerator est DÉFINIE ICI et UNIQUEMENT ICI.
class LLMQuestionGenerator:
    def __init__(self, model_instance=None):
        """
        Initialise le générateur avec une instance de modèle.
        Args:
            model_instance: Instance du modèle LLM (OpenAIServerModel).
        """
        self.model = model_instance
        if not self.model:
            # Fallback: créer une instance avec les credentials d'environnement
            api_key = os.getenv("MISTRAL_API_KEY")
            if api_key:
                print("DEBUG: Tentative de création d'OpenAIServerModel via MISTRAL_API_KEY.")
                self.model = OpenAIServerModel(
                    api_key=api_key,
                    model_id="mistral-medium-latest", # Utilisez votre modèle préféré
                    api_base="https://api.mistral.ai/v1",
                    max_tokens=2048,
                    temperature=0.7,
                )
            else:
                print("ATTENTION: MISTRAL_API_KEY non trouvée dans les variables d'environnement. Le LLM ne sera pas disponible pour la génération.")

    def generate_questions(self, topic: str, num_questions: int, difficulty: str, language: str) -> List[Dict]:
        """
        Génère une liste de questions de quiz sous forme de dictionnaires en utilisant le LLM.
        Le format de sortie attendu du LLM doit être une chaîne JSON représentant une liste de questions.
        """
        if self.model is None:
            print("ERROR: Le modèle LLM n'est pas initialisé dans LLMQuestionGenerator. Impossible de générer des questions.")
            return []

        # Nouveau prompt spécialisé AI-900
        prompt_template = """
        You are an expert assistant trained on the Microsoft AI-900 certification.
        Generate {num_questions} multiple-choice questions in {language} about the topic '{topic}'.
        The questions must strictly follow the official learning objectives of AI-900:
        - Azure Machine Learning
        - Azure Cognitive Services
        - Responsible AI
        - Azure AI capabilities and tools

        Each question must include:
        - A 'question' string
        - An 'options' list with four choices (format: "A. ...", "B. ...", etc.)
        - The 'correct_answer' (the full option string that is correct)
        - A short 'explanation' why this answer is correct

        Return a valid JSON array, without any extra commentary or formatting. Do not generate general knowledge or linguistic trivia.
            """

        full_prompt = prompt_template.format(
            num_questions=num_questions,
            topic=topic,
            difficulty=difficulty,
            language=language
        )

        messages = [{"role": "user", "content": full_prompt}]

        try:
            response = self.model.generate(messages=messages)

            # Extraction robuste du texte
            if hasattr(response, "content"):
                quiz_data_str = response.content
            elif isinstance(response, str):
                quiz_data_str = response
            else:
                raise ValueError(f"Type de réponse inattendu du LLM: {type(response)}. Contenu: {response}")

            # Tentative d'extraction du JSON
            json_match = re.search(r"```json\n(.*)\n```", quiz_data_str, re.DOTALL)
            if json_match:
                json_part = json_match.group(1).strip()
                print(f"DEBUG: JSON extrait du bloc de code: {json_part[:100]}...")
                questions = json.loads(json_part)
            else:
                print(f"DEBUG: Tentative de parsing direct du JSON: {quiz_data_str[:100]}...")
                questions = json.loads(quiz_data_str)

            if not isinstance(questions, list):
                if isinstance(questions, dict):
                    questions = [questions]
                else:
                    raise ValueError(f"Le LLM n'a pas généré une liste ou un objet JSON attendu. Type reçu: {type(questions)}")

            # Filtrage des questions valides
            filtered_questions = []
            for q in questions:
                if all(k in q for k in ['question', 'options', 'correct_answer', 'explanation']):
                    if isinstance(q['options'], str):
                        try:
                            q['options'] = json.loads(q['options'])
                        except:
                            continue
                    if isinstance(q['options'], list) and len(q['options']) == 4:
                        filtered_questions.append(q)

            if not filtered_questions:
                print("WARNING: Aucune question valide n'a été générée après filtrage.")
                return []

            return filtered_questions[:num_questions]

        except json.JSONDecodeError as e:
            print(f"ERROR: Erreur de format JSON du LLM: {e}. Output brut: {quiz_data_str[:500]}...")
            return []
        except Exception as e:
            print(f"ERROR: Erreur inattendue lors de la génération de questions LLM: {e}")
            return []

# Instance globale pour l'utilisation dans les tools
_global_generator: Optional[LLMQuestionGenerator] = None

@tool
def set_global_llm_generator(model_instance: OpenAIServerModel) -> None:
    """Définit l'instance du modèle LLM globalement.
    Args:
        model_instance: L'instance du modèle LLM à définir (e.g., OpenAIServerModel).
    """
    global _global_generator
    _global_generator = LLMQuestionGenerator(model_instance)
    print(f"DEBUG: LLMQuestionGenerator a été défini avec le modèle: {type(model_instance)}")

@tool
def get_global_llm_generator() -> 'LLMQuestionGenerator':
    """Récupère l'instance globale du générateur.
    Si non définie, tente de la créer en utilisant les variables d'environnement.
    Returns:
        Une instance de LLMQuestionGenerator.
    """
    global _global_generator
    if _global_generator is None:
        print("DEBUG: Le générateur global n'est pas défini, tentative de création par fallback.")
        _global_generator = LLMQuestionGenerator() # Le constructeur de LLMQuestionGenerator gère le fallback avec os.getenv
        if _global_generator.model:
            print("DEBUG: Le générateur global a été créé avec succès via fallback.")
        else:
            print("DEBUG: Le générateur global n'a PAS pu être créé via fallback (clé API manquante ou erreur).")
    return _global_generator