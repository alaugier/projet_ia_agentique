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

@tool
def generate_ai900_quiz_with_local_sources(
    topic: str = "general",
    num_questions: int = 5,
    difficulty: str = "intermediate",
    language: str = "french"
) -> list:
    """
    Génère un quiz AI-900 avec contexte thématique enrichi depuis la base de données locale,
    puis ajoute des URLs de sources pertinentes. Cette version utilise la vectorisation
    TF-IDF pour une correspondance sémantique précise entre le thème demandé et le contenu.

    Args:
        topic: Le sujet spécifique du quiz. Peut être général ou spécialisé :
               - Thèmes principaux : 'computer_vision', 'nlp', 'speech', 'machine_learning', 
                 'responsible_ai', 'bot', 'cognitive_services', 'generative_ai'
               - Thèmes spécifiques : 'azure_ml', 'custom_vision', 'text_analytics', 'luis', etc.
               - 'general' pour un mélange de tous les thèmes
        num_questions: Le nombre de questions à générer (1-20).
        difficulty: Le niveau de difficulté ('beginner', 'intermediate', 'advanced').
        language: La langue des questions ('english', 'french').
    Returns:
        Une chaîne JSON des questions du quiz avec contexte thématique et sources.
        list: Une liste de dictionnaires représentant les questions du quiz.
        ⚠️ La sortie n'est pas un dictionnaire, mais directement une liste.
    """
    
    # Validation des paramètres
    if num_questions < 1 or num_questions > 20:
        return json.dumps({"error": "Le nombre de questions doit être entre 1 et 20"}, ensure_ascii=False)

    valid_difficulties = ['beginner', 'intermediate', 'advanced']
    valid_languages = ['english', 'french']

    if difficulty not in valid_difficulties:
        return json.dumps({
            "error": f"Difficulté invalide '{difficulty}'. Valeurs valides : {', '.join(valid_difficulties)}"
        }, ensure_ascii=False)
    
    if language not in valid_languages:
        return json.dumps({
            "error": f"Langue invalide '{language}'. Valeurs valides : {', '.join(valid_languages)}"
        }, ensure_ascii=False)

    try:
        print(f"🎯 Génération d'un quiz thématique AI-900:")
        print(f"   - Thème: {topic}")
        print(f"   - Questions: {num_questions}")
        print(f"   - Difficulté: {difficulty}")
        print(f"   - Langue: {language}")
        
        # Étape 1: Obtenir le générateur LLM avec contexte thématique
        generator = get_global_llm_generator()
        
        if generator is None or generator.model is None:
            return json.dumps({
                "error": "Le générateur LLM n'est pas initialisé. Vérifiez la configuration de l'API."
            }, ensure_ascii=False)

        # Vérifier si l'extracteur de contexte est disponible
        if not hasattr(generator, 'topic_extractor') or not generator.topic_extractor.is_loaded:
            print("⚠️  WARNING: Contexte thématique indisponible, génération standard")
        else:
            print(f"✅ Contexte thématique disponible avec {len(generator.topic_extractor.content_df)} entrées")

        # Étape 2: Générer les questions avec contexte thématique enrichi
        print("🤖 Génération des questions avec contexte thématique...")
        questions_raw_list = generator.generate_questions(topic, num_questions, difficulty, language)
        
        if not questions_raw_list:
            return json.dumps({
                "error": "Le LLM n'a pas pu générer de questions valides pour ce thème. Essayez un autre thème ou reformulez votre demande."
            }, ensure_ascii=False)

        print(f"✅ {len(questions_raw_list)} questions générées avec contexte thématique")

        # Étape 3: Convertir en JSON pour l'ajout de sources
        questions_json = json.dumps(questions_raw_list, ensure_ascii=False, indent=2)

        # Étape 4: Ajouter les sources depuis le CSV local
        print("🔗 Ajout des sources pertinentes...")
        from tools.source_adder_tool import add_sources_to_quiz_tool
        questions_with_sources = add_sources_to_quiz_tool(questions_json)
        
        # Vérifier si l'ajout de sources a fonctionné
        if isinstance(questions_with_sources, str) and questions_with_sources.startswith("❌"):
            print(f"WARNING: Problème lors de l'ajout des sources: {questions_with_sources}")
            # Retourner les questions sans sources plutôt que d'échouer complètement
            questions_with_sources = questions_json
        
        # Étape 5: Ajouter des métadonnées sur le contexte thématique utilisé
        try:
            final_questions = json.loads(questions_with_sources)
            
            # Obtenir les informations de contexte pour les métadonnées
            context_info = {}
            if hasattr(generator, 'topic_extractor') and generator.topic_extractor.is_loaded:
                topic_context = generator.topic_extractor.get_topic_context(topic)
                context_info = {
                    "context_strength": topic_context.get('context_strength', 0.0),
                    "num_relevant_sources": topic_context.get('num_relevant_sources', 0),
                    "key_concepts_used": topic_context.get('key_concepts', [])[:5]  # Limiter à 5 concepts
                }
            
            # Ajouter les métadonnées
            quiz_metadata = {
                "quiz_info": {
                    "topic_requested": topic,
                    "num_questions": len(final_questions),
                    "difficulty": difficulty,
                    "language": language,
                    "generation_timestamp": datetime.now().isoformat(),
                    "thematic_context": context_info
                },
                "questions": final_questions
            }
            
            print(f"🎉 Quiz thématique généré avec succès:")
            print(f"   - {len(final_questions)} questions sur '{topic}'")
            if context_info:
                print(f"   - Force du contexte thématique: {context_info['context_strength']:.3f}")
                print(f"   - Sources documentaires utilisées: {context_info['num_relevant_sources']}")
            
            return json.dumps(quiz_metadata, ensure_ascii=False, indent=2)
            
        except json.JSONDecodeError:
            # Si on ne peut pas parser les questions avec sources, retourner tel quel
            print("INFO: Retour du quiz sans métadonnées additionnelles")
            return questions_with_sources
        
    except Exception as e:
        error_msg = f"Erreur lors de la génération du quiz thématique: {str(e)}"
        print(f"❌ {error_msg}")
        return json.dumps({"error": error_msg}, ensure_ascii=False)

@tool
def generate_ai900_quiz(
    topic: str = "general",
    num_questions: int = 5,
    difficulty: str = "intermediate",
    language: str = "english"
) -> str:
    """
    Version simplifiée pour la rétrocompatibilité. Génère un quiz AI-900 de base.
    Pour une génération avec contexte thématique enrichi, utilisez generate_ai900_quiz_with_local_sources.
    
    Args:
        topic: Le sujet du quiz (ex: 'computer_vision', 'nlp', 'machine_learning', etc.)
        num_questions: Nombre de questions (1-20)
        difficulty: Niveau de difficulté ('beginner', 'intermediate', 'advanced')
        language: Langue ('english', 'french')
    Returns:
        JSON des questions générées
    """
    print("INFO: Utilisation de la version simplifiée. Pour un contexte thématique enrichi, utilisez generate_ai900_quiz_with_local_sources.")
    
    # Redirection vers la version enrichie
    return generate_ai900_quiz_with_local_sources(topic, num_questions, difficulty, language)

@tool 
def get_available_quiz_topics() -> str:
    """
    Retourne la liste des topics disponibles pour la génération de quiz avec leur description.
    Utile pour guider l'utilisateur dans le choix du thème.
    
    Returns:
        JSON des topics disponibles avec descriptions
    """
    topics = {
        "main_themes": {
            "computer_vision": {
                "description": "Vision par ordinateur - Reconnaissance d'images, OCR, détection d'objets",
                "keywords": ["Custom Vision", "Computer Vision API", "Form Recognizer", "Face API"]
            },
            "nlp": {
                "description": "Traitement du langage naturel - Analyse de texte, sentiment, traduction",
                "keywords": ["Text Analytics", "LUIS", "Language Understanding", "QnA Maker"]
            },
            "speech": {
                "description": "Technologies vocales - Reconnaissance et synthèse vocale",
                "keywords": ["Speech to Text", "Text to Speech", "Speaker Recognition"]
            },
            "machine_learning": {
                "description": "Apprentissage automatique - Algorithmes, Azure ML, AutoML",
                "keywords": ["Azure Machine Learning", "Automated ML", "Regression", "Classification"]
            },
            "responsible_ai": {
                "description": "IA responsable - Éthique, biais, fairness, transparence",
                "keywords": ["AI Ethics", "Bias Detection", "Fairness", "Transparency"]
            },
            "bot": {
                "description": "Agents conversationnels - Chatbots, Bot Framework",
                "keywords": ["Azure Bot Service", "Bot Framework", "Conversational AI"]
            },
            "cognitive_services": {
                "description": "Services cognitifs Azure - APIs prêtes à l'emploi",
                "keywords": ["Cognitive Services", "Personalizer", "Anomaly Detector"]
            },
            "generative_ai": {
                "description": "IA générative - OpenAI, GPT, génération de contenu",
                "keywords": ["Azure OpenAI", "GPT", "DALL-E", "Prompt Engineering"]
            }
        },
        "special_topics": {
            "general": "Mélange de tous les thèmes AI-900",
            "azure_fundamentals": "Concepts de base d'Azure pour l'IA",
            "ai_workloads": "Types de charges de travail IA",
            "ai_principles": "Principes fondamentaux de l'IA"
        },
        "difficulty_levels": {
            "beginner": "Questions de base, concepts fondamentaux",
            "intermediate": "Questions moyennes, applications pratiques", 
            "advanced": "Questions avancées, scénarios complexes"
        },
        "supported_languages": ["english", "french"]
    }
    
    return json.dumps(topics, ensure_ascii=False, indent=2)

@tool
def validate_quiz_quality(quiz_json: str) -> str:
    """
    Valide la qualité d'un quiz généré en vérifiant la structure, 
    la cohérence et la pertinence des questions.
    
    Args:
        quiz_json: Le JSON du quiz à valider
    Returns:
        Rapport de validation avec score de qualité
    """
    try:
        quiz_data = json.loads(quiz_json)
        
        # Extraire les questions selon le format
        if "questions" in quiz_data:
            questions = quiz_data["questions"]
        elif isinstance(quiz_data, list):
            questions = quiz_data
        else:
            return json.dumps({"error": "Format de quiz non reconnu"}, ensure_ascii=False)
        
        validation_report = {
            "total_questions": len(questions),
            "validation_timestamp": datetime.now().isoformat(),
            "quality_score": 0.0,
            "issues": [],
            "strengths": [],
            "detailed_analysis": []
        }
        
        points = 0
        max_points = 0
        
        for i, question in enumerate(questions, 1):
            question_analysis = {
                "question_number": i,
                "issues": [],
                "strengths": []
            }
            
            # Test 1: Structure de base (obligatoire)
            max_points += 10
            required_fields = ['question', 'options', 'correct_answer', 'explanation']
            missing_fields = [field for field in required_fields if field not in question]
            
            if missing_fields:
                question_analysis["issues"].append(f"Champs manquants: {missing_fields}")
                validation_report["issues"].append(f"Q{i}: Champs manquants {missing_fields}")
            else:
                points += 10
                question_analysis["strengths"].append("Structure complète")
            
            # Test 2: Format des options (4 options A, B, C, D)
            max_points += 5
            options = question.get('options', [])
            if len(options) == 4 and all(opt.startswith(('A.', 'B.', 'C.', 'D.')) for opt in options):
                points += 5
                question_analysis["strengths"].append("Format d'options correct")
            else:
                question_analysis["issues"].append("Format d'options incorrect")
            
            # Test 3: Réponse correcte valide
            max_points += 5
            correct_answer = question.get('correct_answer', '')
            if correct_answer in options:
                points += 5
                question_analysis["strengths"].append("Réponse correcte valide")
            else:
                question_analysis["issues"].append("Réponse correcte non trouvée dans les options")
            
            # Test 4: Longueur appropriée des textes
            max_points += 5
            question_text = question.get('question', '')
            explanation = question.get('explanation', '')
            
            if 10 <= len(question_text) <= 500 and 20 <= len(explanation) <= 1000:
                points += 5
                question_analysis["strengths"].append("Longueurs de texte appropriées")
            else:
                question_analysis["issues"].append("Longueurs de texte inappropriées")
            
            # Test 5: Diversité des options (pas de répétitions évidentes)
            max_points += 3
            options_text = [opt.split('.', 1)[1].strip().lower() for opt in options if '.' in opt]
            if len(set(options_text)) == len(options_text):
                points += 3
                question_analysis["strengths"].append("Options diverses")
            else:
                question_analysis["issues"].append("Options trop similaires")
            
            # Test 6: Présence de mots-clés AI-900
            max_points += 2
            ai900_keywords = [
                'azure', 'cognitive', 'machine learning', 'ai', 'artificial intelligence',
                'computer vision', 'nlp', 'speech', 'bot', 'ml', 'algorithm'
            ]
            combined_text = (question_text + ' ' + explanation).lower()
            keywords_found = [kw for kw in ai900_keywords if kw in combined_text]
            
            if keywords_found:
                points += 2
                question_analysis["strengths"].append(f"Mots-clés AI-900 trouvés: {keywords_found[:3]}")
            else:
                question_analysis["issues"].append("Peu de mots-clés AI-900 spécifiques")
            
            validation_report["detailed_analysis"].append(question_analysis)
        
        # Calcul du score final
        validation_report["quality_score"] = round((points / max_points) * 100, 2) if max_points > 0 else 0
        
        # Résumé des forces et faiblesses
        all_issues = [issue for qa in validation_report["detailed_analysis"] for issue in qa["issues"]]
        all_strengths = [strength for qa in validation_report["detailed_analysis"] for strength in qa["strengths"]]
        
        # Compter les problèmes les plus fréquents
        issue_counts = {}
        for issue in all_issues:
            issue_type = issue.split(':')[0] if ':' in issue else issue
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        validation_report["common_issues"] = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        validation_report["total_strengths"] = len(all_strengths)
        validation_report["total_issues"] = len(all_issues)
        
        # Recommandations
        recommendations = []
        if validation_report["quality_score"] < 70:
            recommendations.append("Score de qualité faible - révision recommandée")
        if validation_report["total_issues"] > validation_report["total_questions"] * 2:
            recommendations.append("Nombreux problèmes détectés - regénération suggérée")
        if validation_report["quality_score"] > 85:
            recommendations.append("Excellente qualité - quiz prêt à utiliser")
        
        validation_report["recommendations"] = recommendations
        
        return json.dumps(validation_report, ensure_ascii=False, indent=2)
        
    except json.JSONDecodeError:
        return json.dumps({
            "error": "Format JSON invalide",
            "quality_score": 0.0
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "error": f"Erreur lors de la validation: {str(e)}",
            "quality_score": 0.0
        }, ensure_ascii=False)

@tool
def export_quiz_to_formats(quiz_json: str, formats: List[str] = ["json", "csv"]) -> str:
    """
    Exporte un quiz vers différents formats pour utilisation externe.
    
    Args:
        quiz_json: Le JSON du quiz à exporter
        formats: Liste des formats souhaités ['json', 'csv', 'markdown', 'txt']
    Returns:
        Chemins des fichiers exportés
    """
    try:
        quiz_data = json.loads(quiz_json)
        
        # Extraire les questions
        if "questions" in quiz_data:
            questions = quiz_data["questions"]
            metadata = quiz_data.get("quiz_info", {})
        elif isinstance(quiz_data, list):
            questions = quiz_data
            metadata = {}
        else:
            return json.dumps({"error": "Format de quiz non reconnu"}, ensure_ascii=False)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic = metadata.get("topic_requested", "quiz")
        base_filename = f"quiz_{topic}_{timestamp}"
        
        exported_files = []
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        
        # Export JSON
        if "json" in formats:
            json_path = os.path.join(export_dir, f"{base_filename}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(quiz_data, f, ensure_ascii=False, indent=2)
            exported_files.append(json_path)
        
        # Export CSV
        if "csv" in formats:
            csv_path = os.path.join(export_dir, f"{base_filename}.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Question', 'Option_A', 'Option_B', 'Option_C', 'Option_D', 
                               'Correct_Answer', 'Explanation', 'Sources'])
                
                for q in questions:
                    options = q.get('options', ['', '', '', ''])
                    # Assurer 4 options
                    while len(options) < 4:
                        options.append('')
                    
                    sources = ', '.join(q.get('sources', {}).get('urls', []))
                    
                    writer.writerow([
                        q.get('question', ''),
                        options[0] if len(options) > 0 else '',
                        options[1] if len(options) > 1 else '',
                        options[2] if len(options) > 2 else '',
                        options[3] if len(options) > 3 else '',
                        q.get('correct_answer', ''),
                        q.get('explanation', ''),
                        sources
                    ])
            exported_files.append(csv_path)
        
        # Export Markdown
        if "markdown" in formats:
            md_path = os.path.join(export_dir, f"{base_filename}.md")
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# Quiz AI-900: {topic.title()}\n\n")
                if metadata:
                    f.write(f"**Difficulté:** {metadata.get('difficulty', 'N/A')}\n")
                    f.write(f"**Langue:** {metadata.get('language', 'N/A')}\n")
                    f.write(f"**Généré le:** {metadata.get('generation_timestamp', 'N/A')}\n\n")
                
                for i, q in enumerate(questions, 1):
                    f.write(f"## Question {i}\n\n")
                    f.write(f"{q.get('question', '')}\n\n")
                    
                    for option in q.get('options', []):
                        f.write(f"- {option}\n")
                    
                    f.write(f"\n**Réponse correcte:** {q.get('correct_answer', '')}\n\n")
                    f.write(f"**Explication:** {q.get('explanation', '')}\n\n")
                    
                    sources = q.get('sources', {}).get('urls', [])
                    if sources:
                        f.write("**Sources:**\n")
                        for source in sources:
                            f.write(f"- {source}\n")
                    f.write("\n---\n\n")
            
            exported_files.append(md_path)
        
        # Export texte simple
        if "txt" in formats:
            txt_path = os.path.join(export_dir, f"{base_filename}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"QUIZ AI-900: {topic.upper()}\n")
                f.write("=" * 50 + "\n\n")
                
                for i, q in enumerate(questions, 1):
                    f.write(f"QUESTION {i}:\n")
                    f.write(f"{q.get('question', '')}\n\n")
                    
                    for option in q.get('options', []):
                        f.write(f"{option}\n")
                    
                    f.write(f"\nRÉPONSE CORRECTE: {q.get('correct_answer', '')}\n")
                    f.write(f"EXPLICATION: {q.get('explanation', '')}\n")
                    f.write("\n" + "-" * 50 + "\n\n")
            
            exported_files.append(txt_path)
        
        return json.dumps({
            "success": True,
            "exported_files": exported_files,
            "total_questions": len(questions),
            "formats_generated": formats
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": f"Erreur lors de l'export: {str(e)}",
            "exported_files": []
        }, ensure_ascii=False)

@tool
def get_quiz_statistics() -> str:
    """
    Récupère des statistiques sur les quizzes générés (à implémenter si nécessaire).
    """
    return "Statistiques des quizzes : Fonctionnalité à implémenter."