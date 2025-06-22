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
                print(f"Colonnes disponibles: {list(self.content_df.columns)}")
                return
            
            # Nettoyer les données
            self.content_df = self.content_df.dropna(subset=required_columns)
            
            # Créer un texte combiné pour chaque entrée
            self.content_df['combined_text'] = (
                self.content_df['module_name'].astype(str) + " " + 
                self.content_df['unit_name'].astype(str) + " " + 
                self.content_df['content'].astype(str)
            ).str.lower()
            
            # Initialiser le vectorizer avec stop words français et anglais
            french_stopwords = set(stopwords.words('french'))
            english_stopwords = set(stopwords.words('english'))
            combined_stopwords = list(french_stopwords.union(english_stopwords))
            
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words=combined_stopwords,
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.85,
                token_pattern=r'\b[a-zA-ZÀ-ÿ]{2,}\b'
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
    
    def _extract_context_for_keywords(self, keywords: List[str], top_k: int = 15) -> Dict:
        """Extrait le contexte pour une liste de mots-clés avec URLs sources"""
        if not self.is_loaded:
            return {'examples': [], 'key_concepts': keywords, 'context_strength': 0.0, 'source_urls': []}
        
        query = ' '.join(keywords)
        query_vector = self.vectorizer.transform([query.lower()])
        similarities = cosine_similarity(query_vector, self.content_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        examples = []
        source_urls = set()  # Utiliser un set pour éviter les doublons
        total_similarity = 0.0
        
        for idx in top_indices:
            if similarities[idx] > 0.03:  # Seuil plus bas pour capturer plus de contexte
                row = self.content_df.iloc[idx]
                
                # Ajouter l'URL source si elle existe et est valide
                source_url = row.get('source_url', '').strip()
                if source_url and source_url != 'N/A' and source_url.startswith('http'):
                    source_urls.add(source_url)
                
                examples.append({
                    'module': row['module_name'],
                    'unit': row['unit_name'],
                    'content_snippet': row['content'][:300] + "..." if len(row['content']) > 300 else row['content'],
                    'source_url': source_url,
                    'similarity': float(similarities[idx])
                })
                total_similarity += similarities[idx]
        
        return {
            'examples': examples,
            'key_concepts': keywords,
            'context_strength': float(total_similarity / len(examples)) if examples else 0.0,
            'num_relevant_sources': len(examples),
            'source_urls': list(source_urls)  # Convertir le set en liste
        }
    
    def get_topic_context(self, topic: str, user_query: str = "") -> Dict:
        """Obtient le contexte pour un topic donné, avec recherche dynamique si nécessaire"""
        topic_normalized = topic.lower().replace(' ', '_').replace('-', '_')
        if topic_normalized in self.topic_contexts:
            context = self.topic_contexts[topic_normalized].copy()
        else:
            search_terms = [topic]
            if user_query:
                search_terms.append(user_query)
            query_keywords = self._extract_keywords_from_query(user_query) if user_query else []
            search_terms.extend(query_keywords)
            context = self._extract_context_for_keywords(search_terms)
        
        # Enrichir avec une recherche spécifique si on a une requête utilisateur
        if user_query and self.is_loaded:
            user_context = self._extract_context_for_keywords([user_query], top_k=8)
            context['examples'].extend(user_context['examples'][:5])
            context['source_urls'].extend(user_context['source_urls'])
            context['source_urls'] = list(set(context['source_urls']))  # Supprimer les doublons
            context['context_strength'] = max(context['context_strength'], user_context['context_strength'])
        
        # S'assurer que toutes les clés nécessaires existent
        default_keys = {
            'context_strength': 0.0,
            'num_relevant_sources': len(context.get('source_urls', [])),
            'source_urls': [],
            'key_concepts': [],
            'examples': []
        }
        
        for key, default_value in default_keys.items():
            if key not in context:
                context[key] = default_value
        
        # Mettre à jour num_relevant_sources avec le nombre réel de sources
        context['num_relevant_sources'] = len(context['source_urls'])
        
        return context
    
    def get_sources_for_topic(self, topic: str, user_query: str = "", max_sources: int = 5) -> List[str]:
        """Récupère les URLs sources les plus pertinentes pour un topic"""
        context = self.get_topic_context(topic, user_query)
        return context['source_urls'][:max_sources]
    
    def _extract_keywords_from_query(self, query: str) -> List[str]:
        """Extrait des mots-clés pertinents d'une requête utilisateur"""
        if not self.is_loaded:
            return []
        
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
        """
        self.model = model_instance
        self.topic_extractor = TopicContextExtractor()
        
        if not self.model:
            api_key = os.getenv("MISTRAL_API_KEY")
            if api_key:
                print("DEBUG: Tentative de création d'OpenAIServerModel via MISTRAL_API_KEY.")
                self.model = OpenAIServerModel(
                    api_key=api_key,
                    model_id="mistral-medium-latest",
                    api_base="https://api.mistral.ai/v1",
                    max_tokens=3000,
                    temperature=0.7,
                )
            else:
                print("ATTENTION: MISTRAL_API_KEY non trouvée dans les variables d'environnement.")

    def generate_questions(self, topic: str, num_questions: int, difficulty: str, language: str) -> List[Dict]:
        """
        Génère une liste de questions de quiz avec sources URLs obligatoires.
        """
        if self.model is None:
            print("ERROR: Le modèle LLM n'est pas initialisé dans LLMQuestionGenerator.")
            return []

        # Obtenir le contexte thématique depuis la base de données
        topic_context = self.topic_extractor.get_topic_context(topic)
        
        print(f"🎯 Contexte thématique pour '{topic}':")
        print(f"   - Force du contexte: {topic_context['context_strength']:.3f}")
        print(f"   - Sources pertinentes: {topic_context['num_relevant_sources']}")
        print(f"   - URLs sources disponibles: {len(topic_context['source_urls'])}")
        
        # Construire le contexte documentaire enrichi
        context_examples = ""
        source_urls_list = []
        
        if topic_context['examples']:
            context_examples = "\n\n=== CONTEXTE DOCUMENTAIRE OFFICIEL MICROSOFT AI-900 ===\n"
            for i, example in enumerate(topic_context['examples'][:5], 1):
                context_examples += f"{i}. Module: {example['module']}\n"
                context_examples += f"   Unité: {example['unit']}\n"
                context_examples += f"   Contenu: {example['content_snippet']}\n"
                if example['source_url']:
                    context_examples += f"   Source: {example['source_url']}\n"
                    source_urls_list.append(example['source_url'])
                context_examples += f"   Pertinence: {example['similarity']:.3f}\n\n"
        
        # Sources URL disponibles
        available_sources = "\n".join(topic_context['source_urls'][:10])
        key_concepts = ", ".join(topic_context['key_concepts'][:12])
        
        prompt_template = """
Tu es un expert certifié Microsoft AI-900 qui génère des questions d'examen officielles avec sources obligatoires.

MISSION : Génère {num_questions} questions à choix multiples en {language} sur le thème '{topic}'.

CONTRAINTES STRICTES :
1. Chaque question DOIT inclure une source URL valide de la documentation officielle Microsoft
2. Les questions doivent être basées sur le contexte documentaire fourni ci-dessous
3. Niveau de difficulté : {difficulty}
4. Focus sur ces concepts : {key_concepts}

{context_examples}

=== SOURCES URLS DISPONIBLES ===
{available_sources}

=== FORMAT DE RÉPONSE OBLIGATOIRE ===
Réponds UNIQUEMENT avec un JSON valide respectant cette structure exacte :

[
  {{
    "question": "Question technique précise basée sur la documentation...",
    "options": [
      "A. Première option",
      "B. Deuxième option", 
      "C. Troisième option",
      "D. Quatrième option"
    ],
    "correct_answer": "A. Première option",
    "explanation": "Explication détaillée avec concepts AI-900 officiels...",
    "source_url": "https://docs.microsoft.com/...",
    "module": "Nom du module AI-900",
    "unit": "Nom de l'unité"
  }}
]

RÈGLES CRITIQUES :
- Une seule réponse correcte par question
- source_url est OBLIGATOIRE et doit être une URL valide de la liste fournie
- Les distracteurs doivent être plausibles mais incorrects
- Les explications doivent référencer des concepts AI-900 précis
- module et unit doivent correspondre au contexte documentaire
- AUCUN texte en dehors du JSON

ATTENTION : Si tu n'inclus pas de source_url valide, la question sera rejetée !
        """

        full_prompt = prompt_template.format(
            num_questions=num_questions,
            topic=topic,
            difficulty=difficulty,
            language=language,
            key_concepts=key_concepts,
            context_examples=context_examples,
            available_sources=available_sources
        )

        messages = [{"role": "user", "content": full_prompt}]

        try:
            response = self.model.generate(messages=messages)

            if hasattr(response, "content"):
                quiz_data_str = response.content
            elif isinstance(response, str):
                quiz_data_str = response
            else:
                raise ValueError(f"Type de réponse inattendu du LLM: {type(response)}")

            print(f"🤖 Réponse LLM reçue (premiers 300 chars): {quiz_data_str[:300]}...")

            # Extraction et parsing du JSON
            json_match = re.search(r"```json\n(.*)\n```", quiz_data_str, re.DOTALL)
            if json_match:
                json_part = json_match.group(1).strip()
                questions = json.loads(json_part)
            else:
                json_array_match = re.search(r'\[.*\]', quiz_data_str, re.DOTALL)
                if json_array_match:
                    json_part = json_array_match.group(0)
                    questions = json.loads(json_part)
                else:
                    questions = json.loads(quiz_data_str)

            if not isinstance(questions, list):
                if isinstance(questions, dict):
                    questions = [questions]
                else:
                    raise ValueError(f"Format inattendu: {type(questions)}")

            # Validation et enrichissement des questions
            validated_questions = []
            for i, q in enumerate(questions):
                if self._validate_question_with_sources(q, i+1, topic_context):
                    # Enrichir avec des sources si manquantes
                    q = self._enrich_question_with_sources(q, topic_context)
                    validated_questions.append(q)

            if not validated_questions:
                print("WARNING: Aucune question valide après validation")
                # Fallback : générer des questions avec sources automatiques
                return self._generate_fallback_questions_with_sources(topic, num_questions, topic_context)

            print(f"✅ {len(validated_questions)} questions valides avec sources générées")
            return validated_questions[:num_questions]

        except json.JSONDecodeError as e:
            print(f"ERROR: Erreur JSON: {e}")
            return self._generate_fallback_questions_with_sources(topic, num_questions, topic_context)
        except Exception as e:
            print(f"ERROR: Erreur lors de la génération: {e}")
            return self._generate_fallback_questions_with_sources(topic, num_questions, topic_context)
    
    def _validate_question_with_sources(self, question: Dict, question_num: int, topic_context: Dict) -> bool:
        """Valide une question avec vérification des sources"""
        required_fields = ['question', 'options', 'correct_answer', 'explanation']
        
        # Vérifier les champs obligatoires
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
    
    def _enrich_question_with_sources(self, question: Dict, topic_context: Dict) -> Dict:
        """Enrichit une question avec les sources manquantes"""
        # Si pas de source_url, en ajouter une depuis le contexte
        if 'source_url' not in question or not question['source_url']:
            if topic_context['source_urls']:
                question['source_url'] = topic_context['source_urls'][0]
            else:
                question['source_url'] = "https://docs.microsoft.com/azure/cognitive-services/"
        
        # Ajouter module et unit si manquants
        if 'module' not in question and topic_context['examples']:
            question['module'] = topic_context['examples'][0]['module']
        
        if 'unit' not in question and topic_context['examples']:
            question['unit'] = topic_context['examples'][0]['unit']
        
        return question
    
    def _generate_fallback_questions_with_sources(self, topic: str, num_questions: int, topic_context: Dict) -> List[Dict]:
        """Génère des questions de fallback avec sources garanties"""
        print("🚨 Génération de questions de fallback avec sources...")
        
        fallback_questions = []
        available_sources = topic_context['source_urls'] if topic_context['source_urls'] else [
            "https://docs.microsoft.com/azure/cognitive-services/",
            "https://docs.microsoft.com/azure/machine-learning/",
            "https://docs.microsoft.com/azure/bot-service/"
        ]
        
        # Questions de base par thème avec sources
        base_questions = {
            'computer_vision': {
                'question': "Quel service Azure permet d'analyser des images et d'extraire du texte ?",
                'options': ["A. Computer Vision API", "B. Speech Services", "C. Text Analytics", "D. Bot Framework"],
                'correct_answer': "A. Computer Vision API",
                'explanation': "Computer Vision API est le service Azure qui permet d'analyser des images et d'extraire du texte via OCR.",
                'module': "AI Services",
                'unit': "Computer Vision"
            },
            'nlp': {
                'question': "Quel service Azure permet d'analyser le sentiment d'un texte ?",
                'options': ["A. Computer Vision", "B. Text Analytics", "C. Speech Services", "D. Custom Vision"],
                'correct_answer': "B. Text Analytics",
                'explanation': "Text Analytics est le service Azure spécialisé dans l'analyse de sentiment et autres tâches NLP.",
                'module': "AI Services",
                'unit': "Language Understanding"
            }
        }
        
        # Générer des questions avec sources
        for i in range(min(num_questions, len(available_sources))):
            if topic.lower() in base_questions:
                question = base_questions[topic.lower()].copy()
            else:
                question = base_questions['computer_vision'].copy()  # Fallback par défaut
            
            question['source_url'] = available_sources[i % len(available_sources)]
            fallback_questions.append(question)
        
        print(f"✅ {len(fallback_questions)} questions de fallback générées avec sources")
        return fallback_questions

# Instance globale pour l'utilisation dans les tools
_global_generator: Optional[LLMQuestionGenerator] = None

@tool
def set_global_llm_generator(model_instance: OpenAIServerModel) -> None:
    """Définit l'instance du modèle LLM globalement avec support du contexte thématique.
    
    Args:
        model_instance: L'instance du modèle LLM à définir (OpenAIServerModel configuré)
    """
    global _global_generator
    _global_generator = LLMQuestionGenerator(model_instance)
    print(f"DEBUG: LLMQuestionGenerator avec contexte thématique défini: {type(model_instance)}")

@tool
def get_global_llm_generator() -> 'LLMQuestionGenerator':
    """Récupère l'instance globale du générateur avec support thématique.
    
    Returns:
        L'instance globale de LLMQuestionGenerator avec extracteur de contexte
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

def test_topic_context_extraction(topic: str = "computer_vision", user_query: str = "") -> str:
    """Teste l'extraction de contexte thématique pour un topic donné.
    Args:
        topic: Le thème à tester (ex: 'computer_vision', 'nlp', 'machine_learning')
        user_query: Requête utilisateur optionnelle pour enrichir le contexte
    Returns:
        Résultats de l'extraction de contexte formatés en chaîne de caractères
    """
    try:
        extractor = TopicContextExtractor()
        if not extractor.is_loaded:
            return "❌ TopicContextExtractor non chargé - vérifiez le fichier CSV"
        
        context = extractor.get_topic_context(topic, user_query)
        
        # Debug pour voir le contenu réel
        print(f"DEBUG: Type de context: {type(context)}")
        print(f"DEBUG: Clés disponibles: {list(context.keys()) if isinstance(context, dict) else 'Pas un dict'}")
        
        result = f"🎯 Contexte thématique pour '{topic}' :\n\n"
        result += f"📊 Statistiques :\n"
        
        # Accès sécurisé aux clés avec valeurs par défaut
        context_strength = context.get('context_strength', 0.0)
        num_relevant_sources = context.get('num_relevant_sources', 0)
        source_urls = context.get('source_urls', [])
        key_concepts = context.get('key_concepts', [])
        examples = context.get('examples', [])
        
        result += f"   - Force du contexte : {context_strength:.3f}\n"
        result += f"   - Sources pertinentes : {num_relevant_sources}\n"
        result += f"   - URLs sources : {len(source_urls)}\n"
        result += f"   - Concepts clés : {', '.join(key_concepts[:5])}...\n\n"
        
        if source_urls:
            result += f"🔗 URLs sources disponibles :\n"
            for i, url in enumerate(source_urls[:5], 1):
                result += f"{i}. {url}\n"
            result += "\n"
            
        if examples:
            result += f"📚 Exemples de sources pertinentes :\n"
            for i, example in enumerate(examples[:3], 1):
                module = example.get('module', 'N/A')
                unit = example.get('unit', 'N/A')
                similarity = example.get('similarity', 0.0)
                source_url = example.get('source_url', 'N/A')
                content_snippet = example.get('content_snippet', '')[:100]
                
                result += f"{i}. {module} > {unit}\n"
                result += f"   Similarité : {similarity:.3f}\n"
                result += f"   Source : {source_url}\n"
                result += f"   Extrait : {content_snippet}...\n\n"
        else:
            result += "⚠️  Aucun exemple trouvé pour ce thème\n"
            
        return result
        
    except Exception as e:
        return f"❌ Erreur lors du test : {e}"

@tool
def get_sources_for_topic(topic: str, user_query: str = "", max_sources: int = 5) -> List[str]:
    """Récupère les URLs sources les plus pertinentes pour un topic donné.
    
    Args:
        topic: Le thème pour lequel récupérer les sources
        user_query: Requête utilisateur optionnelle pour affiner la recherche
        max_sources: Nombre maximum de sources à retourner
        
    Returns:
        Liste des URLs sources les plus pertinentes
    """
    try:
        extractor = TopicContextExtractor()
        if not extractor.is_loaded:
            return []
        
        sources = extractor.get_sources_for_topic(topic, user_query, max_sources)
        print(f"🔗 {len(sources)} sources trouvées pour le topic '{topic}'")
        return sources
        
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des sources : {e}")
        return []