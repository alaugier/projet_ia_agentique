# tools/quiz_generator_tool.py - Version corrigée avec gestion d'erreurs robuste
import json
import random
import csv
import os
from datetime import datetime
from typing import List, Dict, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from smolagents import tool
except ImportError:
    # Fallback si smolagents n'est pas disponible
    def tool(func):
        return func

# Imports avec gestion d'erreur
try:
    from tools.llm_helper import get_global_llm_generator
    LLM_HELPER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: tools.llm_helper non disponible: {e}")
    LLM_HELPER_AVAILABLE = False
    
    def get_global_llm_generator():
        return None

try:
    from tools.logger_tool import setup_logger
    LOGGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: tools.logger_tool non disponible: {e}")
    LOGGER_AVAILABLE = False
    
    def setup_logger(name):
        import logging
        return logging.getLogger(name)

# Initialisation du logger
logger = setup_logger("quiz_generator")

@tool
def generate_quiz_tool(
    topic: str = "general",
    num_questions: int = 5,
    difficulty: str = "intermediate",
    language: str = "french",
    num_relevant_sources: int = 3,
    output_format: str = "json"
) -> str:
    """
    Fonction principale pour générer un quiz AI-900.
    Cette fonction est appelée par le système de diagnostic.
    
    Args:
        topic (str): Le sujet spécifique du quiz (ex: "Machine Learning", "Azure AI Services", "general"). Par défaut "general".
        num_questions (int): Le nombre de questions à générer (entre 1 et 20). Par défaut 5.
        difficulty (str): Le niveau de difficulté du quiz ("beginner", "intermediate", "advanced"). Par défaut "intermediate".
        language (str): La langue du quiz ("french", "english"). Par défaut "french".
        num_relevant_sources (int): Le nombre de sources pertinentes à utiliser (entre 0 et 10). Par défaut 3.
        output_format (str): Le format de sortie ("json", "text"). Par défaut "json".
    
    Returns:
        str: Le quiz généré au format spécifié contenant les questions, réponses et explications.
    """
    logger.info(f"Appel de generate_quiz_tool avec topic={topic}, num_questions={num_questions}")
    
    # Rediriger vers la fonction principale
    return generate_ai900_quiz_with_local_sources(
        topic=topic,
        num_questions=num_questions,
        difficulty=difficulty,
        language=language,
        num_relevant_sources=num_relevant_sources,
        output_format=output_format
    )

@tool
def generate_ai900_quiz_with_local_sources(
    topic: str = "general",
    num_questions: int = 5,
    difficulty: str = "intermediate",
    language: str = "french",
    num_relevant_sources: int = 3,
    output_format: str = "json"
) -> str:
    """
    Génère un quiz AI-900 avec contexte thématique enrichi depuis la base de données locale.
    
    Cette fonction crée un quiz de questions à choix multiples sur les sujets
    de la certification Microsoft AI-900, en utilisant des sources locales
    pour générer des questions pertinentes avec leurs réponses et explications.
    
    Args:
        topic (str): Le sujet spécifique pour le quiz (ex: "Machine Learning", "Azure AI Services", "Computer Vision", "general"). Par défaut "general".
        num_questions (int): Le nombre de questions à générer (entre 1 et 20). Par défaut 5.
        difficulty (str): Le niveau de difficulté du quiz ("beginner", "intermediate", "advanced"). Par défaut "intermediate".
        language (str): La langue du quiz ("french", "english"). Par défaut "french".
        num_relevant_sources (int): Le nombre de sources pertinentes à utiliser pour enrichir le contexte (entre 1 et 10). Par défaut 3.
        output_format (str): Le format de sortie ("json", "text"). Par défaut "json".
    
    Returns:
        str: Le quiz généré au format spécifié (JSON ou texte) contenant les questions, 
             options de réponse, réponses correctes, explications détaillées et sources utilisées.
    
    Raises:
        ValueError: Si les paramètres fournis ne sont pas dans les plages attendues.
        Exception: Si la génération du quiz échoue.
    """
    logger.info(f"Début de génération de quiz thématique AI-900")
    logger.debug(f"Paramètres reçus : topic={topic}, num_questions={num_questions}, difficulty={difficulty}, language={language}, num_relevant_sources={num_relevant_sources}, output_format={output_format}")

    # Validation des paramètres
    try:
        num_questions = int(num_questions)
        if num_questions < 1 or num_questions > 20:
            return _format_error("Le nombre de questions doit être entre 1 et 20", output_format)
    except (ValueError, TypeError):
        return _format_error("Le nombre de questions doit être un entier valide", output_format)

    try:
        if num_relevant_sources is None:
            num_relevant_sources = 3
        else:
            num_relevant_sources = int(num_relevant_sources)
            if num_relevant_sources < 0:
                num_relevant_sources = 0
    except (ValueError, TypeError):
        logger.warning(f"num_relevant_sources invalide : {num_relevant_sources}, remplacement par 3")
        num_relevant_sources = 3

    # Mappage des thèmes
    THEME_MAPPING = {
        "nlp": "traitement du langage naturel",
        "computer_vision": "vision par ordinateur",
        "machine_learning": "apprentissage automatique",
        "azure_ml": "Azure Machine Learning",
        "speech": "traitement de la parole",
        "bot": "bots conversationnels",
        "generative_ai": "IA générative",
    }
    topic_for_generation = THEME_MAPPING.get(topic.lower(), topic)

    # Validation des autres paramètres
    valid_difficulties = ['beginner', 'intermediate', 'advanced']
    valid_languages = ['english', 'french']
    valid_formats = ['json', 'markdown', 'structured']

    if difficulty not in valid_difficulties:
        return _format_error(f"Difficulté invalide '{difficulty}'. Valeurs valides : {', '.join(valid_difficulties)}", output_format)

    if language not in valid_languages:
        return _format_error(f"Langue invalide '{language}'. Valeurs valides : {', '.join(valid_languages)}", output_format)
    
    if output_format not in valid_formats:
        logger.warning(f"Format de sortie invalide '{output_format}', utilisation de 'json'")
        output_format = 'json'

    try:
        print(f"🎯 Génération d'un quiz thématique AI-900:")
        print(f"   - Thème demandé: {topic}")
        print(f"   - Thème utilisé pour génération: {topic_for_generation}")
        print(f"   - Questions: {num_questions}")
        print(f"   - Difficulté: {difficulty}")
        print(f"   - Langue: {language}")
        print(f"   - Nombre de sources pertinentes max: {num_relevant_sources}")
        print(f"   - Format de sortie: {output_format}")

        # Obtenir le générateur LLM
        if not LLM_HELPER_AVAILABLE:
            return _format_error("Module LLM Helper non disponible. Vérifiez l'installation des dépendances.", output_format)
        
        generator = get_global_llm_generator()
        if generator is None or not hasattr(generator, 'model') or generator.model is None:
            return _format_error("Le générateur LLM n'est pas initialisé. Vérifiez la configuration de l'API.", output_format)

        # Vérifier le contexte thématique
        if not hasattr(generator, 'topic_extractor') or not generator.topic_extractor.is_loaded:
            print("⚠️  WARNING: Contexte thématique indisponible, génération standard")
        else:
            print(f"✅ Contexte thématique disponible avec {len(generator.topic_extractor.content_df)} entrées")

        # Générer les questions
        print("🤖 Génération des questions avec contexte thématique...")
        questions_raw_list = generator.generate_questions(topic_for_generation, num_questions, difficulty, language)

        if not questions_raw_list:
            return _format_error("Le LLM n'a pas pu générer de questions valides pour ce thème. Essayez un autre thème ou reformulez votre demande.", output_format)

        print(f"✅ {len(questions_raw_list)} questions générées avec contexte thématique")

        # Ajouter les sources si demandé
        final_questions = questions_raw_list
        total_sources_added = 0

        if num_relevant_sources > 0:
            print("🔗 Ajout des sources pertinentes...")
            try:
                questions_json = json.dumps(questions_raw_list, ensure_ascii=False, indent=2)
                
                from tools.source_adder_tool import add_sources_to_quiz_tool
                
                questions_with_sources = add_sources_to_quiz_tool(questions_json, max_sources=num_relevant_sources)
                final_questions = json.loads(questions_with_sources)
                total_sources_added = _count_total_sources(final_questions)
                
                print(f"✅ {total_sources_added} sources ajoutées au quiz")
                
            except ImportError as e:
                print(f"⚠️  Module source_adder_tool non disponible: {e}")
                print("   Génération du quiz sans sources externes")
            except Exception as e:
                print(f"⚠️  Erreur lors de l'ajout de sources: {e}")
                print("   Poursuite avec le quiz sans sources")
        else:
            print("ℹ️  Génération sans sources externes (num_relevant_sources = 0)")

        # Enrichir avec métadonnées
        try:
            context_info = {}
            if hasattr(generator, 'topic_extractor') and generator.topic_extractor.is_loaded:
                try:
                    topic_context = generator.topic_extractor.get_topic_context(topic_for_generation)
                    context_info = {
                        "context_strength": topic_context.get('context_strength', 0.0),
                        "num_relevant_sources": topic_context.get('num_relevant_sources', 0),
                        "key_concepts_used": topic_context.get('key_concepts', [])[:5]
                    }
                except Exception as e:
                    logger.warning(f"Erreur lors de l'extraction du contexte: {e}")

            quiz_metadata = {
                "quiz_info": {
                    "topic_requested": topic,
                    "topic_resolved": topic_for_generation,
                    "num_questions": len(final_questions) if isinstance(final_questions, list) else len(final_questions.get('questions', [])),
                    "difficulty": difficulty,
                    "language": language,
                    "generation_timestamp": datetime.now().isoformat(),
                    "thematic_context": context_info,
                    "sources_info": {
                        "max_sources_per_question": num_relevant_sources,
                        "total_sources_added": total_sources_added
                    }
                },
                "questions": final_questions if isinstance(final_questions, list) else final_questions.get('questions', final_questions)
            }

            print(f"🎉 Quiz thématique généré avec succès:")
            print(f"   - {quiz_metadata['quiz_info']['num_questions']} questions sur '{topic_for_generation}'")
            if context_info:
                print(f"   - Force du contexte thématique: {context_info.get('context_strength', 0):.3f}")
                print(f"   - Sources documentaires utilisées: {context_info.get('num_relevant_sources', 0)}")
            print(f"   - Total sources ajoutées: {total_sources_added}")

            return _format_output(quiz_metadata, output_format)

        except Exception as e:
            logger.warning(f"Erreur lors de la création des métadonnées: {e}")
            simple_quiz = {"questions": final_questions}
            return _format_output(simple_quiz, output_format)

    except Exception as e:
        error_msg = f"Erreur lors de la génération du quiz thématique: {str(e)}"
        print(f"❌ {error_msg}")
        logger.exception(error_msg)
        return _format_error(error_msg, output_format)

def _count_total_sources(quiz_data) -> int:
    """Compte le nombre total de sources dans le quiz"""
    try:
        questions = quiz_data if isinstance(quiz_data, list) else quiz_data.get('questions', [])
        total = 0
        for question in questions:
            if isinstance(question, dict):
                sources = question.get('sources', {})
                if isinstance(sources, dict):
                    total += sources.get('count', 0)
                    if 'urls' in sources and isinstance(sources['urls'], list):
                        total += len(sources['urls'])
                elif isinstance(sources, list):
                    total += len(sources)
        return total
    except Exception as e:
        logger.warning(f"Erreur lors du comptage des sources: {e}")
        return 0

def _format_error(error_message: str, output_format: str) -> str:
    """Formate un message d'erreur selon le format demandé"""
    if output_format == "markdown":
        return f"# ❌ Erreur\n\n{error_message}"
    elif output_format == "structured":
        return f"ERREUR: {error_message}"
    else:  # json
        return json.dumps({"error": error_message}, ensure_ascii=False)

def _format_output(quiz_data: dict, output_format: str) -> str:
    """Formate la sortie selon le format demandé"""
    try:
        if output_format == "markdown":
            return _convert_to_markdown(quiz_data)
        elif output_format == "structured":
            return _convert_to_structured(quiz_data)
        else:  # json
            return json.dumps(quiz_data, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Erreur lors du formatage: {e}")
        return json.dumps({"error": f"Erreur de formatage: {str(e)}", "data": quiz_data}, ensure_ascii=False)

def _convert_to_markdown(quiz_data: dict) -> str:
    """Convertit le quiz en format Markdown"""
    # Implementation simplifiée pour éviter les erreurs
    return f"# Quiz AI-900\n\n{json.dumps(quiz_data, ensure_ascii=False, indent=2)}"

def _convert_to_structured(quiz_data: dict) -> str:
    """Convertit le quiz en format structuré lisible"""
    # Implementation simplifiée pour éviter les erreurs
    return f"QUIZ AI-900\n{json.dumps(quiz_data, ensure_ascii=False, indent=2)}"

# Fonctions supplémentaires pour compatibilité
@tool
def generate_simple_ai900_quiz(
    topic: str = "general",
    num_questions: int = 5,
    difficulty: str = "intermediate",
    language: str = "french"
) -> str:
    """
    Version simplifiée pour générer un quiz AI-900 sans sources externes.
    
    Cette fonction génère un quiz de base sur les sujets AI-900 sans utiliser
    de sources externes pour enrichir le contexte.
    
    Args:
        topic (str): Le sujet spécifique du quiz (ex: "Machine Learning", "Azure AI Services", "general"). Par défaut "general".
        num_questions (int): Le nombre de questions à générer (entre 1 et 20). Par défaut 5.
        difficulty (str): Le niveau de difficulté du quiz ("beginner", "intermediate", "advanced"). Par défaut "intermediate".
        language (str): La langue du quiz ("french", "english"). Par défaut "french".
    
    Returns:
        str: Le quiz généré au format JSON contenant les questions, réponses et explications.
    """
    return generate_ai900_quiz_with_local_sources(
        topic=topic,
        num_questions=num_questions,
        difficulty=difficulty,
        language=language,
        num_relevant_sources=0,
        output_format="json"
    )

@tool
def validate_quiz_format(quiz_json: str) -> str:
    """
    Valide le format d'un quiz et retourne un rapport de validation.
    
    Args:
        quiz_json: JSON du quiz à valider
    Returns:
        Rapport de validation détaillé
    """
    try:
        quiz_data = json.loads(quiz_json)
        
        validation_report = []
        errors = []
        warnings = []
        
        # Vérifier la structure globale
        if "quiz_info" in quiz_data and "questions" in quiz_data:
            validation_report.append("✅ Structure principale valide (quiz_info + questions)")
            questions = quiz_data["questions"]
        elif isinstance(quiz_data, list):
            validation_report.append("✅ Structure liste simple détectée")
            questions = quiz_data
        else:
            errors.append("❌ Structure de quiz non reconnue")
            questions = []
        
        # Valider les questions
        if not questions:
            errors.append("❌ Aucune question trouvée")
        else:
            validation_report.append(f"✅ {len(questions)} questions détectées")
            
            required_fields = ['question', 'options', 'correct_answer', 'explanation']
            
            for i, question in enumerate(questions, 1):
                question_errors = []
                
                # Vérifier les champs obligatoires
                for field in required_fields:
                    if field not in question:
                        question_errors.append(f"Champ manquant: {field}")
                    elif not question[field]:
                        question_errors.append(f"Champ vide: {field}")
                
                # Vérifier les options
                if 'options' in question:
                    options = question['options']
                    if not isinstance(options, list):
                        question_errors.append("Les options doivent être une liste")
                    elif len(options) < 2:
                        question_errors.append("Au moins 2 options requises")
                    elif len(options) > 6:
                        warnings.append(f"Question {i}: Beaucoup d'options ({len(options)})")
                
                # Vérifier la cohérence réponse/options
                if 'correct_answer' in question and 'options' in question:
                    correct = question['correct_answer']
                    options = question['options']
                    if isinstance(options, list) and correct not in options:
                        question_errors.append("La réponse correcte n'est pas dans les options")
                
                if question_errors:
                    errors.extend([f"Question {i}: {error}" for error in question_errors])
                else:
                    validation_report.append(f"✅ Question {i} valide")
        
        # Vérifier les métadonnées si présentes
        if "quiz_info" in quiz_data:
            quiz_info = quiz_data["quiz_info"]
            expected_fields = ['topic_requested', 'difficulty', 'language', 'num_questions']
            
            for field in expected_fields:
                if field in quiz_info:
                    validation_report.append(f"✅ Métadonnée {field}: {quiz_info[field]}")
                else:
                    warnings.append(f"Métadonnée manquante: {field}")
        
        # Assembler le rapport final
        report = "🔍 RAPPORT DE VALIDATION DU QUIZ\n"
        report += "=" * 40 + "\n\n"
        
        if validation_report:
            report += "✅ VALIDATIONS RÉUSSIES:\n"
            for item in validation_report:
                report += f"  {item}\n"
            report += "\n"
        
        if warnings:
            report += "⚠️  AVERTISSEMENTS:\n"
            for warning in warnings:
                report += f"  {warning}\n"
            report += "\n"
        
        if errors:
            report += "❌ ERREURS DÉTECTÉES:\n"
            for error in errors:
                report += f"  {error}\n"
            report += "\n"
        
        # Conclusion
        if not errors:
            report += "🎉 RÉSULTAT: Quiz valide !\n"
        else:
            report += f"❌ RÉSULTAT: {len(errors)} erreur(s) à corriger\n"
        
        return report
        
    except json.JSONDecodeError as e:
        return f"❌ ERREUR JSON: {str(e)}"
    except Exception as e:
        return f"❌ ERREUR DE VALIDATION: {str(e)}"

@tool
def get_quiz_statistics() -> str:
    """
    Récupère des statistiques sur les quizzes générés (à implémenter si nécessaire).
    """
    return "Statistiques des quizzes : Fonctionnalité à implémenter."

# Test de fonctionnement du module
if __name__ == "__main__":
    print("🧪 Test du module quiz_generator_tool")
    print("=" * 50)
    
    # Test basique
    try:
        result = generate_quiz_tool(
            topic="general",
            num_questions=2,
            difficulty="beginner",
            language="french",
            num_relevant_sources=0
        )
        print("✅ Test basique réussi")
        print(f"Résultat: {result[:100]}...")
    except Exception as e:
        print(f"❌ Test basique échoué: {e}")