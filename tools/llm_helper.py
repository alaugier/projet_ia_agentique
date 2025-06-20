# tools/llm_helper.py
import os
import random
import re
import json
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from smolagents import tool
from smolagents.models import OpenAIServerModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Assurez-vous que les stop words sont téléchargés
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

class TopicContextExtractor:
    """Extrait le contexte thématique depuis la base de données CSV pour guider la génération LLM"""
    
    def __init__(self, csv_path: str = "tools/ai900_content.csv"):
        self.csv_path = csv_path
        self.content_df = None
        self.vectorizer = None
        self.content_vectors = None
        self.topic_contexts = {}
        self.is_loaded = False
        self.load_and_process_data()
    
    def load_and_process_data(self):
        """Charge et traite les données pour l'extraction de contexte thématique"""
        try:
            # Essai de plusieurs chemins possibles
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
                    break
            
            if not csv_found:
                print(f"⚠️  TopicContextExtractor: CSV non trouvé, contexte thématique indisponible")
                return
            
            self.content_df = pd.read_csv(self.csv_path)
            
            # Vérifier les colonnes requises
            required_columns = ['module_name', 'unit_name', 'content', 'source_url']
            if not all(col in self.content_df.columns for col in required_columns):
                print(f"⚠️  TopicContextExtractor: Colonnes manquantes dans le CSV")
                return
            
            # Nettoyer les données
            self.content_df = self.content_df.dropna(subset=required_columns)
            
            # Créer un texte combiné pour chaque entrée
            self.content_df['combined_text'] = (
                self.content_df['module_name'].astype(str) + " " + 
                self.content_df['unit_name'].astype(str) + " " + 
                self.content_df['content'].astype(str) + " " + 
                self.content_df['source_url'].astype(str)
            ).str.lower()
            
            # Initialiser le vectorizer avec stop words français et anglais
            french_stopwords = set(stopwords.words('french'))
            english_stopwords = set(stopwords.words('english'))
            combined_stopwords = list(french_stopwords.union(english_stopwords))
            
            self.vectorizer = TfidfVectorizer(
                max_features=3000,
                stop_words=combined_stopwords,
                ngram_range=(1, 3),  # Utiliser des n-grammes pour capturer des concepts multi-mots
                min_df=1,
                max_df=0.85,
                token_pattern=r'\b[a-zA-ZÀ-ÿ]{2,}\b'  # Support des caractères accentués
            )
            
            self.content_vectors = self.vectorizer.fit_transform(self.content_df['combined_text'])
            
            # Pré-calculer les contextes pour les thèmes principaux
            self._build_topic_contexts()
            
            self.is_loaded = True
            print(f"✅ TopicContextExtractor initialisé avec {len(self.content_df)} entrées")
            
        except Exception as e:
            print(f"❌ Erreur TopicContextExtractor: {e}")
            self.is_loaded = False
    
    def _build_topic_contexts(self):
        """Construit des contextes thématiques pré-calculés"""
        # Définir les thèmes principaux et leurs mots-clés
        theme_keywords = {
            'computer_vision': ['computer vision', 'vision par ordinateur', 'image', 'photo', 'détection', 
                               'reconnaissance', 'classification', 'objets', 'texte', 'ocr', 'form recognizer',
                               'custom vision', 'face api', 'video indexer'],
            'nlp': ['natural language', 'nlp', 'traitement du langage', 'texte', 'sentiment', 
                   'langue', 'traduction', 'compréhension du langage', 'luis', 'text analytics',
                   'language understanding', 'cognitive search'],
            'speech': ['speech', 'parole', 'voix', 'audio', 'reconnaissance vocale', 'synthèse vocale',
                      'speech to text', 'text to speech', 'speaker recognition'],
            'machine_learning': ['machine learning', 'apprentissage automatique', 'ml', 'algorithme', 
                                'supervisé', 'non supervisé', 'renforcement', 'azure ml', 'automated ml',
                                'regression', 'classification', 'clustering'],
            'responsible_ai': ['responsible ai', 'ia responsable', 'éthique', 'biais', 'fairness', 
                              'transparence', 'explicabilité', 'accountability', 'inclusivité'],
            'bot': ['bot', 'chatbot', 'conversation', 'bot framework', 'azure bot service', 
                   'dialogue', 'conversational ai'],
            'cognitive_services': ['cognitive services', 'services cognitifs', 'api', 'azure cognitive',
                                  'personalizer', 'anomaly detector', 'content moderator'],
            'generative_ai': ['generative ai', 'ia générative', 'openai', 'gpt', 'dall-e', 
                             'azure openai service', 'prompt engineering']
        }
        
        for theme, keywords in theme_keywords.items():
            self.topic_contexts[theme] = self._extract_context_for_keywords(keywords)
    
    def _extract_context_for_keywords(self, keywords: List[str], top_k: int = 10) -> Dict:
        """Extrait le contexte pour une liste de mots-clés"""
        if not self.is_loaded:
            return {'examples': [], 'key_concepts': keywords, 'context_strength': 0.0}
        
        query = ' '.join(keywords)
        query_vector = self.vectorizer.transform([query.lower()])
        similarities = cosine_similarity(query_vector, self.content_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        examples = []
        total_similarity = 0.0
        
        for idx in top_indices:
            if similarities[idx] > 0.05:
                row = self.content_df.iloc[idx]
                examples.append({
                    'module': row['module_name'],
                    'unit': row['unit_name'],
                    'content_snippet': row['content'][:200] + "..." if len(row['content']) > 200 else row['content'],
                    'source_url': row.get('source_url', 'N/A'),
                    'similarity': float(similarities[idx])
                })
                total_similarity += similarities[idx]
        
        return {
            'examples': examples,
            'key_concepts': keywords,
            'context_strength': float(total_similarity / len(examples)) if examples else 0.0,
            'num_relevant_sources': len(examples)
        }
    
    def get_topic_context(self, topic: str, user_query: str = "") -> Dict:
        """Obtient le contexte pour un topic donné, avec recherche dynamique si nécessaire"""
        # Normaliser le topic
        topic_normalized = topic.lower().replace(' ', '_').replace('-', '_')
        
        # Si on a un contexte pré-calculé, l'utiliser
        if topic_normalized in self.topic_contexts:
            context = self.topic_contexts[topic_normalized].copy()
        else:
            # Recherche dynamique basée sur le topic et la requête utilisateur
            search_terms = [topic]
            if user_query:
                search_terms.append(user_query)
            
            # Extraire des mots-clés de la requête utilisateur
            query_keywords = self._extract_keywords_from_query(user_query) if user_query else []
            search_terms.extend(query_keywords)
            
            context = self._extract_context_for_keywords(search_terms)
        
        # Enrichir avec une recherche spécifique si on a une requête utilisateur
        if user_query and self.is_loaded:
            user_context = self._extract_context_for_keywords([user_query], top_k=5)
            # Fusionner les contextes
            context['examples'].extend(user_context['examples'][:3])  # Ajouter les 3 meilleurs
            context['context_strength'] = max(context['context_strength'], user_context['context_strength'])
        
        return context
    
    def _extract_keywords_from_query(self, query: str) -> List[str]:
        """Extrait des mots-clés pertinents d'une requête utilisateur"""
        if not self.is_loaded:
            return []
        
        # Utiliser le vocabulaire du vectorizer pour identifier les termes pertinents
        query_words = query.lower().split()
        vocabulary = set(self.vectorizer.vocabulary_.keys())
        
        relevant_keywords = []
        for word in query_words:
            if word in vocabulary and len(word) > 2:
                relevant_keywords.append(word)
        
        return relevant_keywords

class LLMQuestionGenerator:
    def __init__(self, model_instance=None):
        """
        Initialise le générateur avec une instance de modèle et un extracteur de contexte.
        Args:
            model_instance: Instance du modèle LLM (OpenAIServerModel).
        """
        self.model = model_instance
        self.topic_extractor = TopicContextExtractor()
        
        if not self.model:
            # Fallback: créer une instance avec les credentials d'environnement
            api_key = os.getenv("MISTRAL_API_KEY")
            if api_key:
                print("DEBUG: Tentative de création d'OpenAIServerModel via MISTRAL_API_KEY.")
                self.model = OpenAIServerModel(
                    api_key=api_key,
                    model_id="mistral-medium-latest",
                    api_base="https://api.mistral.ai/v1",
                    max_tokens=2048,
                    temperature=0.7,
                )
            else:
                print("ATTENTION: MISTRAL_API_KEY non trouvée dans les variables d'environnement.")

    def generate_questions(self, topic: str, num_questions: int, difficulty: str, language: str) -> List[Dict]:
        """
        Génère une liste de questions de quiz en utilisant le contexte thématique de la base de données.
        """
        if self.model is None:
            print("ERROR: Le modèle LLM n'est pas initialisé dans LLMQuestionGenerator.")
            return []

        # Obtenir le contexte thématique depuis la base de données
        topic_context = self.topic_extractor.get_topic_context(topic)
        
        print(f"🎯 Contexte thématique pour '{topic}':")
        print(f"   - Force du contexte: {topic_context['context_strength']:.3f}")
        print(f"   - Sources pertinentes: {topic_context['num_relevant_sources']}")
        
        # Construire le prompt enrichi avec le contexte
        context_examples = ""
        if topic_context['examples']:
            context_examples = "\n\nContexte spécialisé basé sur la documentation officielle Microsoft AI-900 :\n"
            for i, example in enumerate(topic_context['examples'][:3], 1):  # Limiter à 3 exemples
                context_examples += f"{i}. Module: {example['module']}\n"
                context_examples += f"   Unité: {example['unit']}\n"
                context_examples += f"   Contenu: {example['content_snippet']}\n\n"
        
        # Créer des concepts clés spécifiques au contexte
        key_concepts = ", ".join(topic_context['key_concepts'][:10])  # Limiter à 10 concepts
        
        prompt_template = """
Tu es un expert certifié Microsoft AI-900 qui génère des questions d'examen officielles.
Génère {num_questions} questions à choix multiples en {language} sur le thème '{topic}'.

CONSIGNES STRICTES :
- Les questions doivent être basées EXCLUSIVEMENT sur le programme officiel Microsoft AI-900
- Utilise le contexte documentaire fourni ci-dessous pour créer des questions précises et techniques
- Niveau de difficulté : {difficulty}
- Focus sur ces concepts clés : {key_concepts}

{context_examples}

Format de réponse requis (JSON valide uniquement) :
[
  {{
    "question": "Question précise et technique...",
    "options": ["A. Première option", "B. Deuxième option", "C. Troisième option", "D. Quatrième option"],
    "correct_answer": "A. Première option",
    "explanation": "Explication détaillée basée sur la documentation Microsoft..."
  }}
]

IMPORTANT : 
- Une seule réponse correcte par question
- Les distracteurs (mauvaises réponses) doivent être plausibles mais clairement incorrects
- Les explications doivent référencer les concepts AI-900 officiels
- Pas de commentaires, uniquement le JSON valide
        """

        full_prompt = prompt_template.format(
            num_questions=num_questions,
            topic=topic,
            difficulty=difficulty,
            language=language,
            key_concepts=key_concepts,
            context_examples=context_examples
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
                raise ValueError(f"Type de réponse inattendu du LLM: {type(response)}")

            print(f"🤖 Réponse LLM reçue (premiers 200 chars): {quiz_data_str[:200]}...")

            # Tentative d'extraction du JSON
            json_match = re.search(r"```json\n(.*)\n```", quiz_data_str, re.DOTALL)
            if json_match:
                json_part = json_match.group(1).strip()
                questions = json.loads(json_part)
            else:
                # Tentative de trouver un array JSON dans la réponse
                json_array_match = re.search(r'\[.*\]', quiz_data_str, re.DOTALL)
                if json_array_match:
                    json_part = json_array_match.group(0)
                    questions = json.loads(json_part)
                else:
                    print(f"DEBUG: Tentative de parsing direct du JSON complet")
                    questions = json.loads(quiz_data_str)

            if not isinstance(questions, list):
                if isinstance(questions, dict):
                    questions = [questions]
                else:
                    raise ValueError(f"Format inattendu: {type(questions)}")

            # Validation et filtrage des questions
            validated_questions = []
            for i, q in enumerate(questions):
                if self._validate_question(q, i+1):
                    validated_questions.append(q)

            if not validated_questions:
                print("WARNING: Aucune question valide après validation")
                return []

            print(f"✅ {len(validated_questions)} questions valides générées pour le thème '{topic}'")
            return validated_questions[:num_questions]

        except json.JSONDecodeError as e:
            print(f"ERROR: Erreur JSON: {e}")
            print(f"Réponse brute: {quiz_data_str[:500]}...")
            return []
        except Exception as e:
            print(f"ERROR: Erreur lors de la génération: {e}")
            return []
    
    def _validate_question(self, question: Dict, question_num: int) -> bool:
        """Valide une question générée"""
        required_fields = ['question', 'options', 'correct_answer', 'explanation']
        
        # Vérifier la présence des champs obligatoires
        for field in required_fields:
            if field not in question:
                print(f"⚠️  Question {question_num}: Champ manquant '{field}'")
                return False
        
        # Vérifier le format des options
        options = question['options']
        if not isinstance(options, list) or len(options) != 4:
            print(f"⚠️  Question {question_num}: Format d'options invalide")
            return False
        
        # Vérifier que la réponse correcte est dans les options
        correct_answer = question['correct_answer']
        if correct_answer not in options:
            print(f"⚠️  Question {question_num}: Réponse correcte non trouvée dans les options")
            return False
        
        # Vérifier que les textes ne sont pas vides
        if not all([question['question'].strip(), question['explanation'].strip()]):
            print(f"⚠️  Question {question_num}: Textes vides détectés")
            return False
        
        return True

# Instance globale pour l'utilisation dans les tools
_global_generator: Optional[LLMQuestionGenerator] = None

@tool
def set_global_llm_generator(model_instance: OpenAIServerModel) -> None:
    """Définit l'instance du modèle LLM globalement avec support du contexte thématique.
    Args:
        model_instance: L'instance du modèle LLM à définir (e.g., OpenAIServerModel).
    """
    global _global_generator
    _global_generator = LLMQuestionGenerator(model_instance)
    print(f"DEBUG: LLMQuestionGenerator avec contexte thématique défini: {type(model_instance)}")

@tool
def get_global_llm_generator() -> 'LLMQuestionGenerator':
    """Récupère l'instance globale du générateur avec support thématique.
    Returns:
        Une instance de LLMQuestionGenerator avec TopicContextExtractor.
    """
    global _global_generator
    if _global_generator is None:
        print("DEBUG: Création du générateur global avec contexte thématique via fallback.")
        _global_generator = LLMQuestionGenerator()
        if _global_generator.model:
            print("DEBUG: Générateur global avec contexte créé avec succès.")
        else:
            print("DEBUG: Générateur créé mais modèle LLM indisponible.")
    return _global_generator

@tool
def test_topic_context_extraction(topic: str = "computer_vision", user_query: str = "") -> str:
    """Teste l'extraction de contexte thématique pour un topic donné.
    Args:
        topic: Le thème à tester (ex: 'computer_vision', 'nlp', 'machine_learning')
        user_query: Requête utilisateur optionnelle pour enrichir le contexte
    Returns:
        Résultats de l'extraction de contexte formatés
    """
    try:
        extractor = TopicContextExtractor()
        
        if not extractor.is_loaded:
            return "❌ TopicContextExtractor non chargé - vérifiez le fichier CSV"
        
        context = extractor.get_topic_context(topic, user_query)
        
        result = f"🎯 Contexte thématique pour '{topic}' :\n\n"
        result += f"📊 Statistiques :\n"
        result += f"   - Force du contexte : {context['context_strength']:.3f}\n"
        result += f"   - Sources pertinentes : {context['num_relevant_sources']}\n"
        result += f"   - Concepts clés : {', '.join(context['key_concepts'][:5])}...\n\n"
        
        if context['examples']:
            result += f"📚 Exemples de sources pertinentes :\n"
            for i, example in enumerate(context['examples'][:3], 1):
                result += f"{i}. {example['module']} > {example['unit']}\n"
                result += f"   Similarité : {example['similarity']:.3f}\n"
                result += f"   Extrait : {example['content_snippet'][:100]}...\n\n"
        else:
            result += "⚠️  Aucun exemple trouvé pour ce thème\n"
        
        return result
        
    except Exception as e:
        return f"❌ Erreur lors du test : {e}"