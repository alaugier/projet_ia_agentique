# tools/source_adder_tool.py
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import nltk
from nltk.corpus import stopwords
from smolagents import tool

# Assurez-vous que les stop words sont téléchargés
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SourceMatcher:
    """Classe pour matcher les questions avec les sources pertinentes du CSV"""
    
    def __init__(self, csv_path: str = "tools/ai900_content.csv"):
        self.csv_path = csv_path
        self.content_df = None
        self.vectorizer = None
        self.content_vectors = None
        self.is_loaded = False
        self.load_data()
    
    def load_data(self):
        """Charge et prépare les données du CSV"""
        try:
            # Essayer plusieurs chemins possibles
            possible_paths = [
                self.csv_path,
                os.path.join(os.path.dirname(__file__), "ai900_content.csv"),
                os.path.join(os.getcwd(), "tools", "ai900_content.csv"),
                "ai900_content.csv"
            ]
            
            csv_found = False
            for path in possible_paths:
                if os.path.exists(path):
                    self.csv_path = path
                    csv_found = True
                    print(f"✅ CSV trouvé à : {path}")
                    break
            
            if not csv_found:
                print(f"❌ CSV non trouvé dans les chemins : {possible_paths}")
                return
            
            # Charger le CSV
            self.content_df = pd.read_csv(self.csv_path)
            print(f"📊 CSV chargé avec {len(self.content_df)} lignes")
            
            # Vérifier les colonnes requises
            required_columns = ['module_name', 'unit_name', 'content', 'source_url']
            missing_columns = [col for col in required_columns if col not in self.content_df.columns]
            
            if missing_columns:
                print(f"❌ Colonnes manquantes : {missing_columns}")
                print(f"📋 Colonnes disponibles : {list(self.content_df.columns)}")
                return
            
            # Nettoyer les données - filtrer les lignes avec content null
            self.content_df = self.content_df.dropna(subset=['content'])
            print(f"🧹 Après nettoyage : {len(self.content_df)} lignes")
            
            if len(self.content_df) == 0:
                print("❌ Aucune donnée utilisable après nettoyage")
                return
            
            # Créer le texte combiné pour la vectorisation
            self.content_df['combined_text'] = (
                self.content_df['module_name'].astype(str) + " " + 
                self.content_df['unit_name'].astype(str) + " " + 
                self.content_df['content'].astype(str)
            ).str.lower()
            
            # Initialiser le vectorizer
            try:
                french_stopwords = set(stopwords.words('french'))
                english_stopwords = set(stopwords.words('english'))
                combined_stopwords = list(french_stopwords.union(english_stopwords))
            except:
                # Fallback si les stopwords ne sont pas disponibles
                combined_stopwords = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus', 'par', 'grand', 'comme', 'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with']
            
            self.vectorizer = TfidfVectorizer(
                max_features=3000,
                stop_words=combined_stopwords,
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.85,
                token_pattern=r'\b[a-zA-ZÀ-ÿ]{2,}\b'
            )
            
            # Vectoriser le contenu
            self.content_vectors = self.vectorizer.fit_transform(self.content_df['combined_text'])
            print(f"🔢 Vectorisation terminée : {self.content_vectors.shape}")
            
            self.is_loaded = True
            print("✅ SourceMatcher initialisé avec succès")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données : {e}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False
    
    def find_relevant_sources(self, question_text: str, max_sources: int = 3) -> List[Dict]:
        """Trouve les sources les plus pertinentes pour une question"""
        if not self.is_loaded:
            print("⚠️  SourceMatcher non chargé")
            return []
        
        try:
            # Préparer le texte de la question
            query_text = question_text.lower()
            
            # Vectoriser la question
            query_vector = self.vectorizer.transform([query_text])
            
            # Calculer les similarités
            similarities = cosine_similarity(query_vector, self.content_vectors).flatten()
            
            # Obtenir les indices des sources les plus pertinentes
            top_indices = np.argsort(similarities)[::-1][:max_sources * 2]  # Prendre plus pour filtrer
            
            relevant_sources = []
            seen_urls = set()
            
            for idx in top_indices:
                if len(relevant_sources) >= max_sources:
                    break
                
                similarity = similarities[idx]
                if similarity > 0.1:  # Seuil de pertinence
                    row = self.content_df.iloc[idx]
                    source_url = row['source_url']
                    
                    # Éviter les doublons d'URLs
                    if source_url not in seen_urls:
                        relevant_sources.append({
                            'url': source_url,
                            'title': f"{row['module_name']} - {row['unit_name']}",
                            'similarity': float(similarity),
                            'content_preview': row['content'][:200] + "..." if len(row['content']) > 200 else row['content']
                        })
                        seen_urls.add(source_url)
            
            print(f"🔍 Trouvé {len(relevant_sources)} sources pour la question")
            return relevant_sources
            
        except Exception as e:
            print(f"❌ Erreur lors de la recherche de sources : {e}")
            import traceback
            traceback.print_exc()
            return []

@tool
def add_sources_to_quiz_tool(quiz_json: str, max_sources: int = 3) -> str:
    """
    Ajoute des sources pertinentes à chaque question du quiz en utilisant 
    la base de données locale AI-900.
    
    Args:
        quiz_json: JSON du quiz généré
        max_sources: Nombre maximum de sources par question (défaut: 3)
    
    Returns:
        JSON du quiz enrichi avec sources
    """
    try:
        print(f"🔗 Début de l'ajout de sources (max {max_sources} par question)")
        
        # Parser le JSON d'entrée
        try:
            quiz_data = json.loads(quiz_json)
        except json.JSONDecodeError as e:
            print(f"❌ Erreur de parsing JSON : {e}")
            return quiz_json  # Retourner le JSON original si erreur
        
        # Extraire les questions selon le format
        if "questions" in quiz_data:
            questions = quiz_data["questions"]
            has_metadata = True
        elif isinstance(quiz_data, list):
            questions = quiz_data
            has_metadata = False
        else:
            print("❌ Format de quiz non reconnu")
            return quiz_json
        
        if not questions:
            print("⚠️  Aucune question trouvée dans le quiz")
            return quiz_json
        
        print(f"📝 Traitement de {len(questions)} questions")
        
        # Initialiser le matcher de sources
        source_matcher = SourceMatcher()
        
        if not source_matcher.is_loaded:
            print("⚠️  Impossible de charger les sources, retour du quiz original")
            return quiz_json
        
        # Traiter chaque question
        questions_with_sources = []
        total_sources_added = 0
        
        for i, question in enumerate(questions):
            question_copy = question.copy()
            
            # Créer le texte de recherche (question + explication pour plus de contexte)
            search_text = question.get('question', '')
            explanation = question.get('explanation', '')
            if explanation:
                search_text += " " + explanation
            
            # Trouver les sources pertinentes
            relevant_sources = source_matcher.find_relevant_sources(search_text, max_sources)
            
            if relevant_sources:
                # Ajouter les sources à la question
                question_copy['sources'] = {
                    'urls': [source['url'] for source in relevant_sources],
                    'details': relevant_sources,
                    'count': len(relevant_sources)
                }
                total_sources_added += len(relevant_sources)
                print(f"  ✅ Question {i+1}: {len(relevant_sources)} sources ajoutées")
            else:
                # Même sans sources, ajouter une structure vide pour la cohérence
                question_copy['sources'] = {
                    'urls': [],
                    'details': [],
                    'count': 0
                }
                print(f"  ⚠️  Question {i+1}: Aucune source pertinente trouvée")
            
            questions_with_sources.append(question_copy)
        
        # Reconstruire le JSON final
        if has_metadata:
            quiz_data["questions"] = questions_with_sources
            # Ajouter des métadonnées sur les sources
            if "quiz_info" not in quiz_data:
                quiz_data["quiz_info"] = {}
            quiz_data["quiz_info"]["sources_added"] = total_sources_added
            quiz_data["quiz_info"]["avg_sources_per_question"] = round(total_sources_added / len(questions), 2) if len(questions) > 0 else 0
            final_result = quiz_data
        else:
            final_result = questions_with_sources
        
        print(f"🎉 Ajout de sources terminé : {total_sources_added} sources au total")
        
        return json.dumps(final_result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        print(f"❌ Erreur lors de l'ajout de sources : {e}")
        import traceback
        traceback.print_exc()
        # En cas d'erreur, retourner le quiz original plutôt que d'échouer
        return quiz_json

@tool
def test_source_matching(question_text: str, max_sources: int = 5) -> str:
    """
    Teste la fonction de matching de sources pour une question donnée.
    Utile pour déboguer et optimiser la pertinence des sources.
    
    Args:
        question_text: Texte de la question à tester
        max_sources: Nombre maximum de sources à retourner (défaut: 5)
    
    Returns:
        Résultats du matching formatés
    """
    try:
        print(f"🧪 Test de matching pour : {question_text[:100]}...")
        
        source_matcher = SourceMatcher()
        
        if not source_matcher.is_loaded:
            return "❌ SourceMatcher non chargé - vérifiez le fichier CSV"
        
        sources = source_matcher.find_relevant_sources(question_text, max_sources)
        
        result = f"🔍 Résultats du matching de sources :\n\n"
        result += f"📋 Question testée : {question_text}\n\n"
        result += f"📊 Sources trouvées : {len(sources)}\n\n"
        
        if sources:
            for i, source in enumerate(sources, 1):
                result += f"{i}. **{source['title']}**\n"
                result += f"   - URL : {source['url']}\n"
                result += f"   - Similarité : {source['similarity']:.3f}\n"
                result += f"   - Aperçu : {source['content_preview'][:150]}...\n\n"
        else:
            result += "⚠️  Aucune source pertinente trouvée\n"
        
        return result
        
    except Exception as e:
        print(f"❌ Erreur lors du test : {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Erreur lors du test : {e}"

# Fonction utilitaire pour vérifier le CSV
def check_csv_structure(csv_path: str = "tools/ai900_content.csv") -> str:
    """
    Vérifie la structure du CSV et affiche des informations utiles pour le debug
    """
    try:
        # Essayer plusieurs chemins possibles
        possible_paths = [
            csv_path,
            os.path.join(os.path.dirname(__file__), "ai900_content.csv"),
            os.path.join(os.getcwd(), "tools", "ai900_content.csv"),
            "ai900_content.csv"
        ]
        
        csv_found = False
        actual_path = ""
        for path in possible_paths:
            if os.path.exists(path):
                actual_path = path
                csv_found = True
                break
        
        if not csv_found:
            return f"❌ CSV non trouvé dans les chemins : {possible_paths}"
        
        # Charger et analyser le CSV
        df = pd.read_csv(actual_path)
        
        info = f"✅ CSV trouvé à : {actual_path}\n"
        info += f"📊 Nombre de lignes : {len(df)}\n"
        info += f"📋 Colonnes : {list(df.columns)}\n"
        info += f"🔍 Aperçu des première lignes :\n"
        info += str(df.head(2).to_string()) + "\n"
        
        # Vérifier les colonnes requises
        required_columns = ['module_name', 'unit_name', 'content', 'source_url']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            info += f"❌ Colonnes manquantes : {missing_columns}\n"
        else:
            info += f"✅ Toutes les colonnes requises sont présentes\n"
            
        # Vérifier les valeurs nulles
        null_counts = df[required_columns].isnull().sum()
        info += f"🔍 Valeurs nulles par colonne :\n{null_counts}\n"
        
        # Statistiques sur le contenu
        df_clean = df.dropna(subset=['content'])
        info += f"📊 Lignes avec contenu valide : {len(df_clean)}\n"
        
        return info
        
    except Exception as e:
        return f"❌ Erreur lors de la vérification du CSV : {e}"

if __name__ == "__main__":
    # Test rapide du module
    print("🧪 Test du module source_adder_tool")
    print(check_csv_structure())
    
    # Test de base
    test_question = "Qu'est-ce que l'intelligence artificielle?"
    print(f"\n🔍 Test avec la question : {test_question}")
    print(test_source_matching(test_question, 3))