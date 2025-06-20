# tools/quiz_generator_tool.py
import json
import random
import csv
import os
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from smolagents import tool

# Importez get_global_llm_generator depuis llm_helper.py pour accéder à l'instance du générateur LLM
from tools.llm_helper import get_global_llm_generator

# La classe LLMQuestionGenerator N'EST PAS DÉFINIE ICI.

@tool
def generate_ai900_quiz(
    topic: str = "general",
    num_questions: int = 5,
    difficulty: str = "intermediate",
    language: str = "english"
) -> str:
    """
    Generates a multiple choice quiz for AI-900 certification using LLM and local knowledge base.

    Args:
        topic: The specific AI-900 topic (e.g., 'machine_learning', 'cognitive_services', 'azure_ai', 'responsible_ai', 'general')
        num_questions: Number of questions to generate (1-20)
        difficulty: Difficulty level ('beginner', 'intermediate', 'advanced')
        language: Language for questions ('english', 'french')
    """
    # Validation des paramètres
    if num_questions < 1 or num_questions > 20:
        return "Error: Number of questions must be between 1 and 20"

    valid_topics = ['machine_learning', 'cognitive_services', 'azure_ai', 'responsible_ai', 'general']
    valid_difficulties = ['beginner', 'intermediate', 'advanced']
    valid_languages = ['english', 'french']

    if topic not in valid_topics:
        return f"Error: Invalid topic '{topic}'. Valid topics are: {', '.join(valid_topics)}"
    if difficulty not in valid_difficulties:
        return f"Error: Invalid difficulty '{difficulty}'. Valid difficulties are: {', '.join(valid_difficulties)}"
    if language not in valid_languages:
        return f"Error: Invalid language '{language}'. Valid languages are: {', '.join(valid_languages)}"

    try:
        generator = get_global_llm_generator()
        if generator is None or generator.model is None:
            return "Erreur: Le générateur LLM n'est pas initialisé. Vérifiez votre clé API ou la configuration."

        # Appelle la méthode generate_questions du générateur LLM
        # Cette méthode est censée retourner une liste de dictionnaires
        questions_list = generator.generate_questions(topic, num_questions, difficulty, language)

        if not questions_list:
            return "Erreur: Aucune question n'a pu être générée par le LLM."
        
        # Convertit la liste de dictionnaires en une chaîne JSON
        return json.dumps(questions_list, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"Erreur lors de la génération du quiz: {e}"

@tool
def generate_ai900_quiz_with_local_sources(
    topic: str = "general",
    num_questions: int = 5,
    difficulty: str = "intermediate",
    language: str = "french"
) -> str:
    """
    Génère un quiz AI-900 avec des questions et ajoute des URLs de sources pertinentes
    à partir d'un fichier CSV local. Cet outil combine la génération de quiz par le LLM
    et la recherche de sources locales.

    Args:
        topic: Le sujet spécifique du quiz (par exemple, 'machine_learning', 'cognitive_services', 'azure_ai', 'responsible_ai', 'general').
        num_questions: Le nombre de questions à générer (1-20).
        difficulty: Le niveau de difficulté ('beginner', 'intermediate', 'advanced').
        language: La langue des questions ('english', 'french').
    Returns:
        Une chaîne JSON des questions du quiz avec la clé 'source_url' ajoutée à chaque question.
        Retourne une chaîne d'erreur si la génération ou l'ajout de sources échoue.
    """
    try:
        # Étape 1: Générer les questions avec le LLM (sans sources initiales)
        print("DEBUG: Génération des questions par le LLM...")
        generator = get_global_llm_generator()
        
        if generator is None or generator.model is None:
            return "Erreur: Le générateur LLM n'est pas initialisé. Impossible de générer des questions."

        # La méthode generate_questions de LLMQuestionGenerator doit retourner List[Dict]
        questions_raw_list = generator.generate_questions(topic, num_questions, difficulty, language)
        
        if not questions_raw_list:
            return "Erreur: Le LLM n'a pas pu générer de questions valides."

        # Convertir la liste de questions en une chaîne JSON pour le passage à add_sources_to_quiz_tool
        # C'est cette chaîne JSON qui est attendue par add_sources_to_quiz_tool
        questions_json = json.dumps(questions_raw_list, ensure_ascii=False, indent=2)

        # Étape 2: Ajouter les sources depuis le CSV local
        print("DEBUG: Appel de add_sources_to_quiz_tool pour ajouter les sources...")
        # L'outil add_sources_to_quiz_tool reçoit une chaîne JSON et retourne une chaîne JSON avec les sources
        from tools.source_adder_tool import add_sources_to_quiz_tool
        questions_with_sources = add_sources_to_quiz_tool(questions_json)
        
        # Vérifier si l'ajout de sources a fonctionné
        if questions_with_sources.startswith("Erreur"):
            print(f"ERROR: add_sources_to_quiz_tool a retourné une erreur: {questions_with_sources}")
            return f"Erreur lors de l'ajout des sources: {questions_with_sources}"
        
        print("DEBUG: Sources ajoutées avec succès depuis le CSV local.")
        return questions_with_sources
        
    except Exception as e:
        # Capture toutes les erreurs inattendues durant ce processus
        print(f"ERROR: Erreur générale dans generate_ai900_quiz_with_local_sources: {e}")
        return f"Erreur lors de la génération (generate_ai900_quiz_with_local_sources): {e}"

@tool
def validate_quiz_answer(quiz_json_string: str, user_answer_json_string: str) -> str:
    """
    Validates a user's answer against a quiz, providing feedback and a corrected answer.
    
    Args:
        quiz_json_string: A JSON string of the quiz question, as generated by generate_ai900_quiz.
        user_answer_json_string: A JSON string containing the user's answer (e.g., {"question_number": 1, "answer": "B"}).
    Returns:
        A JSON string with validation results, including whether the answer was correct,
        the correct answer, and an explanation.
    """
    try:
        quiz_data = json.loads(quiz_json_string)
        user_answer = json.loads(user_answer_json_string)

        question_number = user_answer.get("question_number")
        user_selected_answer = user_answer.get("answer") # e.g., "A", "B", "C", "D"

        if question_number is None or user_selected_answer is None:
            return json.dumps({"error": "Invalid user answer format. Requires 'question_number' and 'answer'."})

        # Assuming quiz_data is a list of questions, find the relevant question
        # Note: The quiz_json_string might contain a single question or multiple.
        # If it's always a single question for validation, adjust logic.
        # For now, let's assume quiz_data might be a list and we pick the first one
        # or find by question_number if it's implicitly part of the data.
        # A more robust solution would be to pass the specific question object.
        
        # Simple approach: if quiz_data is a list, take the first question (or find by index if questions are indexed)
        question_to_validate = None
        if isinstance(quiz_data, list) and question_number <= len(quiz_data):
            question_to_validate = quiz_data[question_number - 1] # -1 because list is 0-indexed
        elif isinstance(quiz_data, dict): # If only one question was passed directly as a dict
            question_to_validate = quiz_data
        
        if question_to_validate is None:
            return json.dumps({"error": "Question not found in the provided quiz data."})

        correct_answer_text = question_to_validate.get("correct_answer")
        explanation = question_to_validate.get("explanation")
        options = question_to_validate.get("options", [])

        # Map user's selected letter to the actual option text
        selected_option_text = None
        # Assuming options are like ["A. Option A", "B. Option B", ...]
        option_prefix_map = {chr(65 + i): opt for i, opt in enumerate(options)} # 'A': 'Option A', 'B': 'Option B'
        selected_option_text = option_prefix_map.get(user_selected_answer.upper())


        is_correct = (selected_option_text == correct_answer_text)

        result = {
            "question_number": question_number,
            "user_answer": user_selected_answer,
            "user_selected_text": selected_option_text,
            "is_correct": is_correct,
            "correct_answer": correct_answer_text,
            "explanation": explanation
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON format for quiz or user answer: {e}"})
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred during validation: {e}"})

# Fonction de test rapide pour le CSV local
@tool
def test_local_csv_integration() -> str:
    """
    Teste l'intégration avec le CSV local (ai900_content.csv) utilisé par source_adder_tool.
    Vérifie si la base de données est chargée et effectue un test de recherche simple.
    """
    try:
        from tools.source_adder_tool import source_matcher
        
        if source_matcher is None:
            return "❌ `source_matcher` n'est pas initialisé. Vérifiez `source_adder_tool.py` et le démarrage de l'application."

        if not source_matcher.is_loaded:
            # Tente de recharger au cas où il aurait échoué à l'initialisation
            source_matcher.load_data()
            if not source_matcher.is_loaded:
                return f"❌ CSV local non chargé après tentative de rechargement. Erreur: {source_matcher.csv_path} introuvable ou problème interne."
        
        if source_matcher.content_df is None or source_matcher.content_df.empty:
            return "❌ CSV local chargé mais vide ou DataFrame non valide. Vérifiez le contenu de `ai900_content.csv`."
        
        num_entries = len(source_matcher.content_df)
        
        # Test avec une question simple
        test_question = "Quel service Azure permet d'analyser des images ?"
        url, confidence = source_matcher.find_best_source(test_question)
        
        status_message = f"""
✅ CSV local opérationnel:
- {num_entries} entrées dans la base
- Test de recherche avec '{test_question}':
  - Confiance: {confidence:.3f}
  - URL trouvée: {url if url else 'Aucune'}
- Statut: Prêt à utiliser pour l'ajout de sources.
"""
        return status_message
        
    except ImportError:
        return "❌ Erreur: Le module `source_adder_tool` n'a pas pu être importé. Assurez-vous que le fichier existe et qu'il n'y a pas d'erreurs d'importation."
    except Exception as e:
        return f"❌ Erreur inattendue lors du test d'intégration du CSV local: {e}"

@tool
def get_quiz_statistics() -> str:
    """
    Récupère des statistiques sur les quizzes générés (à implémenter si nécessaire).
    """
    return "Statistiques des quizzes : Fonctionnalité à implémenter."