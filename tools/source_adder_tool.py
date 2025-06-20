# tools/source_adder_tool.py
import json
import pandas as pd
import re
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
import os
import nltk
from nltk.corpus import stopwords
from smolagents import tool

# Assurez-vous que les stop words sont téléchargés
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

class AI900SourceMatcher:
    def __init__(self, csv_path: str = "ai900_content.csv"):
        """
        Initialise le matcher avec la base de données CSV
        
        Args:
            csv_path: Chemin vers le fichier CSV contenant le contenu AI-900 scrapé
        """
        self.csv_path = csv_path
        self.content_df = None
        self.vectorizer = None
        self.content_vectors = None
        self.is_loaded = False
        self.load_data()
    
    def load_data(self):
        """Charge et prépare les données du CSV avec gestion d'erreurs améliorée"""
        try:
            # Essai de plusieurs chemins possibles
            possible_paths = [
                self.csv_path,
                os.path.join(os.path.dirname(__file__), self.csv_path),
                os.path.join(os.path.dirname(__file__), "ai900_content.csv"),
                os.path.join(os.getcwd(), "tools", "ai900_content.csv"),
                os.path.join(os.getcwd(), "ai900_content.csv")
            ]
            
            csv_found = False
            for path in possible_paths:
                if os.path.exists(path):
                    self.csv_path = path
                    csv_found = True
                    print(f"✅ CSV trouvé: {path}")
                    break
            
            if not csv_found:
                print(f"❌ ERREUR: Fichier CSV non trouvé dans les chemins suivants:")
                for path in possible_paths:
                    print(f"   - {path}")
                self.content_df = None
                self.is_loaded = False
                return
            
            self.content_df = pd.read_csv(self.csv_path)
            
            # Vérifier les colonnes requises
            required_columns = ['module_name', 'unit_name', 'content', 'source_url']
            missing_columns = [col for col in required_columns if col not in self.content_df.columns]
            
            if missing_columns:
                print(f"❌ ERREUR: Colonnes manquantes dans le CSV: {missing_columns}")
                print(f"Colonnes disponibles: {list(self.content_df.columns)}")
                self.content_df = None
                self.is_loaded = False
                return
            
            # Nettoyer les données
            self.content_df = self.content_df.dropna(subset=['module_name', 'unit_name', 'content', 'source_url'])
            
            # Générer les URLs à partir des noms de modules et unités
            self.content_df['url'] = self.content_df.apply(self._generate_url_from_names, axis=1)
            
            # Créer un texte combiné pour la recherche sémantique
            self.content_df['search_text'] = (
                self.content_df['module_name'].astype(str) + " " + 
                self.content_df['unit_name'].astype(str) + " " + 
                self.content_df['content'].astype(str) + " " + 
                self.content_df['source_url'].astype(str)
            ).str.lower()
            
            # Initialiser le vectorizer et les vecteurs
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words=stopwords.words('french'),
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
            self.content_vectors = self.vectorizer.fit_transform(self.content_df['search_text'])
            self.is_loaded = True
            
            print(f"✅ Base de données AI-900 chargée avec succès: {len(self.content_df)} entrées")
            
        except Exception as e:
            print(f"❌ ERREUR lors du chargement des données CSV: {e}")
            self.content_df = None
            self.vectorizer = None
            self.content_vectors = None
            self.is_loaded = False
    
    def _generate_url_from_names(self, row) -> str:
        """Génère une URL Microsoft Learn à partir du nom du module et de l'unité"""
        module_name_raw = str(row['module_name'])
        unit_name_raw = str(row['unit_name'])
        
        module_name_lower = module_name_raw.lower()
        unit_name_lower = unit_name_raw.lower()

        # Mapping amélioré avec URLs plus précises
        precise_url_mapping = {
            # AI Fundamentals Overview
            "vue d'ensemble de l'ia": ('get-started-ai-fundamentals', 'module'),
            "introduction à l'ia": ('get-started-ai-fundamentals', 'module'),
            'comprendre le machine learning': ('fundamentals-machine-learning', 'module'),
            'apprentissage supervisé': ('fundamentals-machine-learning', 'module'),
            'apprentissage non supervisé': ('fundamentals-machine-learning', 'module'),
            'apprentissage par renforcement': ('fundamentals-machine-learning', 'module'),

            # Computer Vision
            'vision par ordinateur': ('explore-computer-vision-microsoft-azure', 'path'),
            'analyser des images': ('analyze-images-computer-vision', 'module'),
            'classifier des images': ('classify-images-custom-vision', 'module'),
            'détecter des objets': ('detect-objects-images', 'module'),
            'reconnaissance optique': ('read-text-computer-vision', 'module'),
            'form recognizer': ('analyze-receipts-form-recognizer', 'module'),
            
            # Natural Language Processing
            'traitement en langage naturel': ('explore-natural-language-processing', 'path'),
            'analyse de texte': ('analyze-text-with-text-analytics-service', 'module'),
            'traduction': ('translate-text-with-translation-service', 'module'),
            'compréhension du langage': ('create-language-understanding-model', 'module'),
            
            # Bots
            'bot': ('build-chat-bot-with-azure-bot-service', 'module'),
            'chatbot': ('build-chat-bot-with-azure-bot-service', 'module'),
            'azure bot service': ('build-chat-bot-with-azure-bot-service', 'module'),

            # Speech
            'reconnaissance vocale': ('recognize-synthesize-speech', 'module'),
            'synthèse vocale': ('recognize-synthesize-speech', 'module'),
            'traduction de la parole': ('translate-speech-with-speech-service', 'module'),
            
            # Responsible AI
            'ia responsable': ('responsible-ai-principles', 'module'),
            'responsible ai': ('responsible-ai-principles', 'module'),
            'principes': ('responsible-ai-principles', 'module'),
            
            # Generative AI
            'ia générative': ('fundamentals-generative-ai', 'module'),
            'generative ai': ('fundamentals-generative-ai', 'module'),
            'azure openai': ('explore-azure-openai', 'module'),
            
            # Document Intelligence
            'intelligence des documents': ('explore-document-intelligence', 'path'),
            'exploration des connaissances': ('explore-knowledge-mining-azure', 'path'),
            'document intelligence': ('explore-document-intelligence', 'path'),
        }
        
        best_match_slug = None
        best_match_type = None
        best_score = 0

        # Recherche dans le nom du module et de l'unité
        combined_text = f"{module_name_lower} {unit_name_lower}"
        
        for key_phrase, (slug, item_type) in precise_url_mapping.items():
            if key_phrase in combined_text:
                current_score = len(key_phrase)
                # Bonus si trouvé dans l'unité (plus spécifique)
                if key_phrase in unit_name_lower:
                    current_score += 10
                
                if current_score > best_score:
                    best_score = current_score
                    best_match_slug = slug
                    best_match_type = item_type

        if best_match_slug:
            if best_match_type == 'module':
                return f"https://learn.microsoft.com/fr-fr/training/modules/{best_match_slug}/"
            elif best_match_type == 'path':
                return f"https://learn.microsoft.com/fr-fr/training/paths/{best_match_slug}/"
        
        # Fallback
        return "https://learn.microsoft.com/fr-fr/credentials/certifications/azure-ai-fundamentals/"
    
    def extract_key_concepts(self, question_text: str) -> List[str]:
        """Extrait les concepts clés d'une question AI-900"""
        text_lower = question_text.lower()
        
        ai900_concepts = {
            'machine_learning': ['machine learning', 'apprentissage automatique', 'ml', 'algorithme', 'supervisé', 'non supervisé', 'renforcement'],
            'computer_vision': ['computer vision', 'vision par ordinateur', 'image', 'photo', 'détection', 'reconnaissance', 'classification', 'objets', 'texte', 'ocr'],
            'nlp': ['natural language processing', 'nlp', 'traitement du langage', 'texte', 'sentiment', 'langue', 'traduction', 'compréhension du langage'],
            'speech': ['speech', 'parole', 'voix', 'audio', 'reconnaissance vocale', 'synthèse vocale'],
            'bot': ['bot', 'chatbot', 'conversation', 'bot framework', 'azure bot service'],
            'cognitive_services': ['cognitive services', 'services cognitifs', 'api', 'azure'],
            'responsible_ai': ['responsible ai', 'ia responsable', 'éthique', 'biais', 'fairness', 'principes'],
            'azure': ['azure', 'microsoft azure', 'cloud', 'service'],
            'generative_ai': ['ia générative', 'generative ai', 'openai service', 'gpt'],
            'document_intelligence': ['intelligence des documents', 'exploration des connaissances', 'form recognizer'],
        }
        
        found_concepts = []
        for concept_key, synonyms in ai900_concepts.items():
            for synonym in synonyms:
                if synonym in text_lower:
                    found_concepts.append(concept_key)
                    break
        
        return list(set(found_concepts))
    
    def find_best_sources(self, question_text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Trouve les meilleures sources pour une question donnée, avec scores.

        Args:
            question_text: La question à rechercher.
            top_k: Nombre maximal de sources à retourner.

        Returns:
            Liste de tuples (url, score) ordonnée du meilleur au moins bon.
        """
        if not self.is_loaded or self.content_df is None or self.vectorizer is None:
            print(f"⚠️  WARNING: Base de données non chargée, utilisation du fallback")
            return [(self.get_fallback_url(question_text), 0.0)]
        
        try:
            question_vector = self.vectorizer.transform([question_text.lower()])
            similarities = cosine_similarity(question_vector, self.content_vectors).flatten()
            top_indices = np.argsort(similarities)[::-1]
            
            concepts = self.extract_key_concepts(question_text)
            
            scored_sources = []
            
            for rank, idx in enumerate(top_indices):
                if rank >= top_k:
                    break
                    
                base_score = similarities[idx]
                row = self.content_df.iloc[idx]
                url = row['url']
                module_name = row['module_name'].lower()
                unit_name = row['unit_name'].lower()
                content = row['content'].lower()
                
                final_score = base_score + (top_k - rank) * 0.02
                
                for concept in concepts:
                    concept_text = concept.replace('_', ' ')
                    if concept_text in module_name:
                        final_score += 0.5
                    if concept_text in unit_name:
                        final_score += 0.7
                    elif concept_text in content:
                        final_score += 0.2
                
                if '/training/modules/' in url:
                    final_score += 0.1
                elif '/training/paths/' in url:
                    final_score += 0.05
                elif '/credentials/certifications/' in url:
                    final_score -= 0.1
                
                final_score = max(0.0, final_score)
                
                if final_score > 0.05:
                    scored_sources.append((url, final_score))
            
            if not scored_sources:
                # Aucun bon score, fallback unique
                scored_sources = [(self.get_fallback_url(question_text), 0.0)]
            
            return scored_sources
        
        except Exception as e:
            print(f"❌ ERREUR lors de la recherche des meilleures sources: {e}")
            return [(self.get_fallback_url(question_text), 0.0)]

    
    def get_fallback_url(self, question_text: str = "") -> str:
        """Retourne une URL de fallback basée sur l'analyse de la question"""
        fallback_urls = {
            'computer_vision': "https://learn.microsoft.com/fr-fr/training/paths/explore-computer-vision-microsoft-azure/",
            'nlp': "https://learn.microsoft.com/fr-fr/training/paths/explore-natural-language-processing/",
            'speech': "https://learn.microsoft.com/fr-fr/training/modules/recognize-synthesize-speech/",
            'bot': "https://learn.microsoft.com/fr-fr/training/modules/build-chat-bot-with-azure-bot-service/",
            'machine_learning': "https://learn.microsoft.com/fr-fr/training/modules/get-started-ai-fundamentals/",
            'responsible_ai': "https://learn.microsoft.com/fr-fr/training/modules/responsible-ai-principles/",
            'generative_ai': "https://learn.microsoft.com/fr-fr/training/modules/fundamentals-generative-ai/",
            'document_intelligence': "https://learn.microsoft.com/fr-fr/training/paths/explore-document-intelligence/",
            'default': "https://learn.microsoft.com/fr-fr/credentials/certifications/azure-ai-fundamentals/"
        }
        
        if question_text:
            concepts = self.extract_key_concepts(question_text)
            for concept in concepts:
                if concept in fallback_urls:
                    return fallback_urls[concept]
        
        return fallback_urls['default']

# Instance globale avec gestion d'erreurs
def get_source_matcher():
    """Factory function pour obtenir une instance du matcher"""
    # Essai de plusieurs chemins possibles pour le CSV
    possible_csv_paths = [
        "tools/ai900_content.csv",
        "ai900_content.csv",
        os.path.join(os.path.dirname(__file__), "ai900_content.csv")
    ]
    
    for csv_path in possible_csv_paths:
        if os.path.exists(csv_path):
            return AI900SourceMatcher(csv_path)
    
    # Si aucun CSV trouvé, créer quand même l'instance (elle utilisera les fallbacks)
    print("⚠️  WARNING: Aucun fichier CSV AI-900 trouvé, utilisation des URLs de fallback")
    return AI900SourceMatcher()

# Instance globale
source_matcher = get_source_matcher()

@tool
def add_sources_to_quiz_tool(quiz_questions_json_string: str, max_sources: int = 3) -> str:
    """
    Prend une chaîne JSON de questions de quiz et ajoute des sources pertinentes à chaque question
    en utilisant la base de données locale AI-900 pour une recherche sémantique.

    Args:
        quiz_questions_json_string: Une chaîne JSON représentant la liste des questions du quiz.
        max_sources: Nombre maximal de sources à ajouter par question (par défaut 3).

    Returns:
        Une chaîne JSON des questions du quiz mises à jour, incluant les URL de source.
    """
    global source_matcher
    
    try:
        questions = json.loads(quiz_questions_json_string)
        
        if not source_matcher.is_loaded:
            print("⚠️  WARNING: Matcher non chargé, rechargement...")
            source_matcher = get_source_matcher()
        
        updated_questions = []
        for i, q in enumerate(questions):
            question_text = q.get('question', '')
            print(f"🔍 Traitement question {i+1}/{len(questions)}: '{question_text[:50]}...'")
            
            # Trouver les meilleures sources (liste de tuples)
            best_sources = source_matcher.find_best_sources(question_text, top_k=max_sources)
            
            # Extraire uniquement les URLs
            source_urls = [url for url, score in best_sources]
            confidence_scores = [round(score, 3) for url, score in best_sources]
            
            print(f"✅ Sources trouvées (scores): {list(zip(source_urls, confidence_scores))}")
            
            # Ajouter les informations de source sous forme de liste
            q['source_urls'] = source_urls
            q['source_confidences'] = confidence_scores
            
            updated_questions.append(q)
        
        print(f"🎉 Traitement terminé: {len(updated_questions)} questions avec sources")
        return json.dumps(updated_questions, indent=2, ensure_ascii=False)
    
    except json.JSONDecodeError as e:
        error_msg = f"❌ ERREUR JSON: {e}. Entrée: {quiz_questions_json_string[:200]}..."
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ ERREUR générale: {e}. Entrée: {quiz_questions_json_string[:200]}..."
        print(error_msg)
        return error_msg


@tool
def reload_ai900_database(csv_path: str = "ai900_content.csv") -> str:
    """
    Recharge la base de données AI-900 depuis un nouveau fichier CSV
    
    Args:
        csv_path: Chemin vers le nouveau fichier CSV
    Returns:
        Message de confirmation
    """
    global source_matcher
    try:
        source_matcher = AI900SourceMatcher(csv_path)
        if source_matcher.is_loaded:
            return f"✅ Base de données AI-900 rechargée depuis {csv_path} ({len(source_matcher.content_df)} entrées)"
        else:
            return f"❌ Échec du rechargement depuis {csv_path}"
    except Exception as e:
        return f"❌ Erreur lors du rechargement: {e}"

@tool
def check_ai900_database_status() -> str:
    """
    Vérifie le statut de la base de données AI-900
    
    Returns:
        Statut de la base de données
    """
    global source_matcher
    
    if source_matcher.is_loaded:
        return f"✅ Base de données chargée: {len(source_matcher.content_df)} entrées depuis {source_matcher.csv_path}"
    else:
        return f"❌ Base de données non chargée. Fichier recherché: {source_matcher.csv_path}"