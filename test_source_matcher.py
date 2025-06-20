# test_source_matcher.py
"""
Script de test pour valider le fonctionnement du matcher de sources AI-900
"""

import pandas as pd
import json
from tools.source_adder_tool import add_sources_to_quiz_tool, source_matcher, AI900SourceMatcher # Import AI900SourceMatcher

# NOTE IMPORTANTE :
# L'instance `source_matcher` importée directement de `source_adder_tool`
# est initialisée SANS argument `csv_path` par défaut.
# Si vous exécutez `test_source_matcher.py` depuis `~/projets/hugginface/`,
# et que votre CSV est dans `~/projets/hugginface/tools/ai900_content.csv`,
# vous devez réinitialiser `source_matcher` AVANT d'appeler les fonctions de test.
# Sinon, `source_matcher` cherchera le CSV au mauvais endroit.

# CORRECTION CLÉ : Réinitialiser source_matcher avec le chemin correct
# Assurez-vous que le chemin "tools/ai900_content.csv" est correct par rapport
# à l'endroit d'où vous exécutez ce script (par exemple, si vous êtes dans
# ~/projets/hugginface/ et que le CSV est dans ~/projets/hugginface/tools/).
source_matcher = AI900SourceMatcher(csv_path="tools/ai900_content.csv")


def test_csv_loading():
    """Test le chargement du CSV"""
    print("=== Test de chargement du CSV ===")
    
    if source_matcher.content_df is not None:
        print(f"✅ CSV chargé avec succès!")
        print(f"📊 Nombre d'entrées: {len(source_matcher.content_df)}")
        print(f"📋 Colonnes: {list(source_matcher.content_df.columns)}")
        
        # Afficher quelques exemples
        print("\n🔍 Aperçu des données:")
        for i in range(min(3, len(source_matcher.content_df))):
            row = source_matcher.content_df.iloc[i]
            print(f"  Module: {row['module_name']}")
            print(f"  Unité: {row['unit_name']}")
            print(f"  URL générée: {row['url']}")
            print(f"  Contenu (extrait): {str(row['content'])[:100]}...")
            print("-" * 50)
    else:
        print("❌ Erreur: CSV non chargé")
        return False
    
    return True

def test_concept_extraction():
    """Test l'extraction de concepts"""
    print("\n=== Test d'extraction de concepts ===")
    
    test_questions = [
        "Qu'est-ce que le machine learning supervisé?",
        "Comment utiliser Computer Vision pour détecter des objets?",
        "Quels sont les services Azure pour le traitement du langage naturel?",
        "Comment créer un chatbot avec Bot Framework?",
        "Qu'est-ce que l'IA responsable?"
    ]
    
    for question in test_questions:
        concepts = source_matcher.extract_key_concepts(question)
        print(f"❓ Question: {question}")
        print(f"🔍 Concepts extraits: {concepts}")
        print()

def test_source_matching():
    """Test la recherche de sources"""
    print("\n=== Test de recherche de sources ===")
    
    test_questions = [
        "Quelle est la différence entre l'apprentissage supervisé et non supervisé?",
        "Comment utiliser Azure Computer Vision pour analyser des images?",
        "Quels sont les principes de l'IA responsable?",
        "Comment créer un bot conversationnel avec Azure Bot Service?"
    ]
    
    for question in test_questions:
        url, score = source_matcher.find_best_source(question)
        print(f"❓ Question: {question}")
        print(f"🔗 URL trouvée: {url}")
        print(f"📊 Score de confiance: {score:.3f}")
        print("-" * 80)

def test_full_tool():
    """Test complet de l'outil avec un quiz exemple"""
    print("\n=== Test complet de l'outil ===")
    
    # Quiz exemple
    sample_quiz = [
        {
            "question": "Qu'est-ce que l'apprentissage supervisé en machine learning?",
            "options": [
                "Un algorithme qui apprend sans données d'entraînement",
                "Un algorithme qui apprend à partir d'exemples étiquetés",
                "Un algorithme qui ne nécessite pas de supervision humaine",
                "Un algorithme qui fonctionne uniquement avec des images"
            ],
            "correct_answer": "Un algorithme qui apprend à partir d'exemples étiquetés",
            "explanation": "L'apprentissage supervisé utilise des données d'entraînement étiquetées."
        },
        {
            "question": "Quel service Azure permet d'analyser des images?",
            "options": [
                "Azure Text Analytics",
                "Azure Computer Vision",
                "Azure Speech Services",
                "Azure Bot Service"
            ],
            "correct_answer": "Azure Computer Vision",
            "explanation": "Azure Computer Vision est spécialement conçu pour l'analyse d'images."
        }
    ]
    
    quiz_json = json.dumps(sample_quiz, ensure_ascii=False)
    
    print("🔄 Traitement du quiz...")
    result = add_sources_to_quiz_tool(quiz_json)
    
    try:
        updated_quiz = json.loads(result)
        print("✅ Quiz traité avec succès!")
        
        for i, question in enumerate(updated_quiz):
            print(f"\n📝 Question {i+1}:")
            print(f"  Texte: {question['question']}")
            print(f"  Source: {question.get('source_url', 'Aucune')}")
            print(f"  Confiance: {question.get('source_confidence', 'N/A')}")
            
    except json.JSONDecodeError:
        print("❌ Erreur lors du traitement:")
        print(result)

def analyze_csv_content():
    """Analyse le contenu du CSV pour mieux comprendre les données"""
    print("\n=== Analyse du contenu CSV ===")
    
    if source_matcher.content_df is None:
        print("❌ CSV non chargé")
        return
    
    df = source_matcher.content_df
    
    print(f"📊 Statistiques générales:")
    print(f"  - Nombre total d'entrées: {len(df)}")
    print(f"  - Modules uniques: {df['module_name'].nunique()}")
    print(f"  - Unités uniques: {df['unit_name'].nunique()}")
    
    print(f"\n📋 Top 10 des modules les plus fréquents:")
    module_counts = df['module_name'].value_counts().head(10)
    for module, count in module_counts.items():
        print(f"  - {module}: {count} unités")
    
    print(f"\n🔍 Exemples de contenu par longueur:")
    df['content_length'] = df['content'].str.len()
    print(f"  - Contenu le plus court: {df['content_length'].min()} caractères")
    print(f"  - Contenu le plus long: {df['content_length'].max()} caractères")
    print(f"  - Longueur moyenne: {df['content_length'].mean():.0f} caractères")

if __name__ == "__main__":
    print("🚀 Démarrage des tests du matcher de sources AI-900")
    print("=" * 60)
    
    # Test 1: Chargement du CSV
    if not test_csv_loading():
        print("❌ Impossible de continuer sans CSV valide")
        exit(1)
    
    # Test 2: Analyse du contenu
    analyze_csv_content()
    
    # Test 3: Extraction de concepts
    test_concept_extraction()
    
    # Test 4: Recherche de sources
    test_source_matching()
    
    # Test 5: Test complet
    test_full_tool()
    
    print("\n✅ Tests terminés!")
    print("💡 Si vous voyez des erreurs, vérifiez que:")
    print("   1. Le fichier 'ai900_content.csv' est dans le bon répertoire")
    print("   2. Les colonnes 'module_name', 'unit_name', 'content' existent")
    print("   3. Les dépendances sont installées (pandas, scikit-learn)")