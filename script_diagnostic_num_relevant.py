#!/usr/bin/env python3
"""
Script de diagnostic pour identifier l'erreur num_relevant_sources
"""

import os
import sys
import traceback
import json

def check_imports():
    """Vérifie que tous les modules peuvent être importés"""
    print("🔍 Vérification des imports...")
    
    try:
        from tools.source_adder_tool import SourceMatcher, add_sources_to_quiz_tool, test_source_matching
        print("✅ source_adder_tool importé avec succès")
    except Exception as e:
        print(f"❌ Erreur import source_adder_tool: {e}")
        traceback.print_exc()
        return False
    
    try:
        from tools.quiz_generator_tool import generate_quiz_tool
        print("✅ quiz_generator_tool importé avec succès")
    except Exception as e:
        print(f"❌ Erreur import quiz_generator_tool: {e}")
        traceback.print_exc()
        return False
    
    try:
        from tools.llm_helper import get_sources_for_topic
        print("✅ llm_helper importé avec succès")
    except Exception as e:
        print(f"❌ Erreur import llm_helper: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_source_matcher_basic():
    """Test de base du SourceMatcher"""
    print("\n🧪 Test de base du SourceMatcher...")
    
    try:
        from tools.source_adder_tool import SourceMatcher
        
        # Créer une instance
        matcher = SourceMatcher()
        
        if not matcher.is_loaded:
            print("❌ SourceMatcher n'a pas pu charger les données")
            return False
        
        # Test de recherche simple
        sources = matcher.find_relevant_sources("intelligence artificielle", 2)
        print(f"✅ Recherche réussie: {len(sources)} sources trouvées")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test SourceMatcher: {e}")
        traceback.print_exc()
        return False

def test_quiz_generation():
    """Test de génération de quiz simple"""
    print("\n🧪 Test de génération de quiz...")
    
    try:
        from tools.quiz_generator_tool import generate_quiz_tool
        
        # Paramètres de test
        result = generate_quiz_tool(
            topic="nlp",
            num_questions=1,
            difficulty="beginner",
            language="french",
            num_relevant_sources=0,  # Pas de sources pour ce test
            output_format="json"
        )
        
        print("✅ Génération de quiz réussie (sans sources)")
        
        # Test avec sources
        result_with_sources = generate_quiz_tool(
            topic="nlp",
            num_questions=1,
            difficulty="beginner",
            language="french",
            num_relevant_sources=2,  # Avec sources
            output_format="json"
        )
        
        print("✅ Génération de quiz réussie (avec sources)")
        return True
        
    except Exception as e:
        print(f"❌ Erreur génération quiz: {e}")
        traceback.print_exc()
        return False

def check_app_structure():
    """Vérifie la structure de app.py"""
    print("\n🔍 Vérification de app.py...")
    
    if not os.path.exists("app.py"):
        print("❌ app.py non trouvé")
        return False
    
    try:
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Chercher des références à num_relevant_sources
        if "num_relevant_sources" in content:
            print("⚠️  num_relevant_sources trouvé dans app.py")
            
            # Extraire les lignes contenant num_relevant_sources
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if "num_relevant_sources" in line:
                    print(f"   Ligne {i}: {line.strip()}")
        else:
            print("✅ Aucune référence directe à num_relevant_sources dans app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lecture app.py: {e}")
        return False

def test_full_integration():
    """Test d'intégration complète"""
    print("\n🧪 Test d'intégration complète...")
    
    try:
        # Simuler un appel complet comme dans app.py
        from tools.quiz_generator_tool import generate_quiz_tool
        
        # Test avec tous les paramètres
        result = generate_quiz_tool(
            topic="computer_vision",
            num_questions=2,
            difficulty="intermediate",
            language="french",
            num_relevant_sources=3,
            output_format="json"
        )
        
        # Vérifier le résultat
        if result and len(result) > 0:
            # Essayer de parser le JSON
            try:
                quiz_data = json.loads(result)
                print("✅ Quiz généré et parsé avec succès")
                
                # Vérifier la structure
                if "questions" in quiz_data:
                    questions = quiz_data["questions"]
                    print(f"✅ {len(questions)} questions générées")
                    
                    # Vérifier les sources
                    for i, q in enumerate(questions):
                        if "sources" in q:
                            print(f"   Question {i+1}: {q['sources']['count']} sources")
                        else:
                            print(f"   Question {i+1}: Pas de sources")
                
                return True
                
            except json.JSONDecodeError as e:
                print(f"❌ Erreur parsing JSON: {e}")
                print("Contenu reçu:", result[:200] + "..." if len(result) > 200 else result)
                return False
        else:
            print("❌ Résultat vide ou None")
            return False
        
    except Exception as e:
        print(f"❌ Erreur test intégration: {e}")
        traceback.print_exc()
        return False

def main():
    """Fonction principale de diagnostic"""
    print("🚀 Diagnostic complet du système de quiz AI-900")
    print("=" * 50)
    
    # Vérifier le répertoire de travail
    print(f"📂 Répertoire de travail: {os.getcwd()}")
    
    # Vérifier les fichiers essentiels
    essential_files = [
        "tools/ai900_content.csv",
        "tools/source_adder_tool.py",
        "tools/quiz_generator_tool.py",
        "tools/llm_helper.py",
        "app.py"
    ]
    
    missing_files = []
    for file in essential_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Fichiers manquants: {missing_files}")
        return False
    
    # Tests séquentiels
    tests = [
        ("Imports", check_imports),
        ("SourceMatcher", test_source_matcher_basic),
        ("Structure app.py", check_app_structure),
        ("Génération quiz", test_quiz_generation),
        ("Intégration complète", test_full_integration)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Erreur inattendue dans {test_name}: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    # Résumé final
    print(f"\n{'='*50}")
    print("📊 RÉSUMÉ DES TESTS")
    print(f"{'='*50}")
    
    for test_name, success in results.items():
        status = "✅ RÉUSSI" if success else "❌ ÉCHEC"
        print(f"{test_name:20} : {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 Tous les tests sont passés! Le système devrait fonctionner.")
    else:
        print("\n⚠️  Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
    
    return all_passed

if __name__ == "__main__":
    main()