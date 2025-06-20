# test_csv_existant.py
"""Test du fichier CSV existant et du source matcher"""

import sys
import os
import json
import pandas as pd

# Ajouter le répertoire tools au path
sys.path.append('tools')

def test_csv_content():
    """Teste le contenu du CSV existant"""
    print("=== TEST DU CSV EXISTANT ===\n")
    
    csv_path = "tools/ai900_content.csv"
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ CSV chargé avec succès")
        print(f"   - Nombre de lignes: {len(df)}")
        print(f"   - Colonnes: {list(df.columns)}")
        
        # Vérifier les colonnes requises
        required_cols = ['module_name', 'unit_name', 'content']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Colonnes manquantes: {missing_cols}")
            print("   Colonnes disponibles:", list(df.columns))
        else:
            print("✅ Toutes les colonnes requises sont présentes")
        
        # Afficher quelques exemples
        print(f"\n📊 Exemples de contenu:")
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            print(f"   Ligne {i+1}:")
            print(f"     Module: {str(row.get('module_name', 'N/A'))[:50]}...")
            print(f"     Unité: {str(row.get('unit_name', 'N/A'))[:50]}...")
            print(f"     Contenu: {str(row.get('content', 'N/A'))[:80]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du CSV: {e}")
        return False

def test_source_matcher():
    """Test du matcher de sources avec le CSV existant"""
    print("=== TEST DU SOURCE MATCHER ===\n")
    
    try:
        from source_adder_tool import source_matcher
        
        # Vérifier l'état du matcher
        if source_matcher.content_df is None:
            print("❌ Le matcher n'a pas pu charger les données")
            print("   Vérification du chemin CSV...")
            
            # Tenter différents chemins
            possible_paths = [
                "tools/ai900_content.csv",
                "ai900_content.csv",
                os.path.join(os.getcwd(), "tools", "ai900_content.csv")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"   ✅ CSV trouvé à: {path}")
                else:
                    print(f"   ❌ CSV non trouvé à: {path}")
            
            return False
        
        print(f"✅ Matcher initialisé avec {len(source_matcher.content_df)} entrées")
        
        # Test avec des questions de différents domaines
        test_questions = [
            "Quel service Azure permet d'analyser des images et détecter des objets ?",
            "Comment utiliser Azure Text Analytics pour analyser le sentiment ?", 
            "Quelle est la différence entre l'apprentissage supervisé et non supervisé ?",
            "Comment créer un bot avec Azure Bot Service ?",
            "Quels sont les principes de l'IA responsable ?"
        ]
        
        print(f"\n🔍 Test avec {len(test_questions)} questions:")
        
        for i, question in enumerate(test_questions):
            print(f"\n   Question {i+1}: {question[:60]}...")
            
            url, confidence = source_matcher.find_best_source(question)
            print(f"   URL trouvée: {url}")
            print(f"   Confiance: {confidence:.3f}")
            
            # Extraire les concepts clés
            concepts = source_matcher.extract_key_concepts(question)
            print(f"   Concepts détectés: {concepts}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test du matcher: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_add_sources_tool():
    """Test de l'outil d'ajout de sources"""
    print("=== TEST DE L'OUTIL D'AJOUT DE SOURCES ===\n")
    
    try:
        from source_adder_tool import add_sources_to_quiz_tool
        
        # Quiz test sans sources
        test_quiz = [
            {
                "question": "Quel service Azure permet d'analyser des images ?",
                "options": [
                    "A. Azure Computer Vision",
                    "B. Azure Text Analytics", 
                    "C. Azure Speech Services",
                    "D. Azure Translator"
                ],
                "correct_answer": "A. Azure Computer Vision",
                "explanation": "Azure Computer Vision est spécialisé dans l'analyse d'images."
            },
            {
                "question": "Quelle technique d'apprentissage automatique utilise des données étiquetées ?",
                "options": [
                    "A. Apprentissage non supervisé",
                    "B. Apprentissage supervisé",
                    "C. Apprentissage par renforcement", 
                    "D. Apprentissage profond"
                ],
                "correct_answer": "B. Apprentissage supervisé",
                "explanation": "L'apprentissage supervisé utilise des données d'entraînement étiquetées."
            }
        ]
        
        quiz_json = json.dumps(test_quiz, ensure_ascii=False)
        print(f"Quiz d'entrée (sans sources):")
        print(f"   {len(test_quiz)} questions")
        
        # Appliquer l'ajout de sources
        result = add_sources_to_quiz_tool(quiz_json)
        
        # Parser le résultat
        try:
            updated_quiz = json.loads(result)
            print(f"\n✅ Sources ajoutées avec succès!")
            print(f"   {len(updated_quiz)} questions traitées")
            
            for i, q in enumerate(updated_quiz):
                print(f"\n   Question {i+1}:")
                print(f"     URL: {q.get('source_url', 'N/A')}")
                print(f"     Confiance: {q.get('source_confidence', 'N/A')}")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ Erreur de parsing JSON: {e}")
            print(f"Résultat brut: {result[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors du test de l'outil: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_flow():
    """Test du flux d'intégration complet"""
    print("=== TEST DU FLUX D'INTÉGRATION ===\n")
    
    # Simuler ce que fait l'agent actuellement
    print("Simulation du comportement actuel de l'agent:")
    print("1. Génération de quiz avec generate_ai900_quiz")
    print("2. Les questions ont déjà des source_url")
    print("3. add_sources_to_quiz_tool n'est pas appelé")
    
    print("\n💡 Solution recommandée:")
    print("1. Modifier quiz_generator_tool pour générer SANS sources")  
    print("2. Puis appeler add_sources_to_quiz_tool automatiquement")
    print("3. Ou créer un nouvel outil generate_complete_ai900_quiz")

if __name__ == "__main__":
    print("🔍 TEST COMPLET DU SYSTÈME AVEC CSV EXISTANT\n")
    
    # 1. Tester le CSV
    csv_ok = test_csv_content()
    
    if csv_ok:
        # 2. Tester le matcher
        matcher_ok = test_source_matcher()
        
        if matcher_ok:
            # 3. Tester l'outil d'ajout
            tool_ok = test_add_sources_tool()
            
            # 4. Analyser l'intégration
            test_integration_flow()
        else:
            print("❌ Impossible de continuer sans matcher fonctionnel")
    else:
        print("❌ Impossible de continuer sans CSV valide")
    
    print("\n🎯 TEST TERMINÉ")