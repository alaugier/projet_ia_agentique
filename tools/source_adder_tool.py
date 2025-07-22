# tools/source_adder_tool.py - VERSION CORRIGÉE POUR DIVERSIFIER LES SOURCES
import os
import time
import csv
import json
import pandas as pd
from typing import List, Dict, Optional, Tuple
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from transformers import AutoTokenizer
from smolagents import tool
from dotenv import load_dotenv
import re
from collections import defaultdict
import hashlib

load_dotenv(".env")

# Token HuggingFace
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN is None:
    raise ValueError("La variable d'environnement HUGGINGFACE_TOKEN n'est pas définie.")

# Tokenizer Mistral
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    use_auth_token=os.getenv("HF_TOKEN")
)

class PreciseSourceMatcher:
    def __init__(self, driver: webdriver.Chrome, tokenizer, max_tokens: int = 512, csv_path: str = "azure_learning_chunks.csv"):
        self.driver = driver
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.csv_path = csv_path
        self.chunks_data = self._load_chunks_data()
        self.used_urls = set()  # CORRIGÉ: Tracker des URLs utilisées (pas des hashes)
        self.used_modules = set()  # NOUVEAU: Tracker des modules utilisés
        self.used_units = set()
        print(f"🔍 PreciseSourceMatcher initialisé avec {len(self.chunks_data)} chunks")

    def _load_chunks_data(self) -> List[Dict]:
        """Charge tous les chunks en mémoire avec leurs URLs spécifiques"""
        chunks = []
        try:
            possible_paths = [
                self.csv_path,
                os.path.join(os.path.dirname(__file__), "azure_learning_chunks.csv"),
                os.path.join(os.getcwd(), "azure_learning_chunks.csv")
            ]
            
            actual_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    actual_path = path
                    break
            
            if not actual_path:
                print(f"❌ CSV file not found in paths: {possible_paths}")
                return []
            
            with open(actual_path, newline='', encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    chunk = {
                        'module_name': row['module_name'],
                        'unit_name': row['unit_name'],
                        'unit_url': row['unit_url'],  # CRITIQUE: Cette URL doit être différente pour chaque chunk
                        'chunk_id': row['chunk_id'],
                        'text_chunk': row['text_chunk'],
                        'token_count': int(row.get('token_count', 0))
                    }
                    
                    # Ajouter un hash unique pour référencement précis
                    chunk['content_hash'] = hashlib.md5(chunk['text_chunk'].encode()).hexdigest()[:8]
                    
                    # Créer un titre de section basé sur le contenu
                    chunk['section_title'] = self._generate_section_title(chunk['text_chunk'])
                    
                    chunks.append(chunk)
            
            print(f"📚 Chargé {len(chunks)} chunks depuis {actual_path}")
            
            # DIAGNOSTIC: Vérifier la diversité des URLs
            unique_urls = set(chunk['unit_url'] for chunk in chunks)
            unique_modules = set(chunk['module_name'] for chunk in chunks)
            print(f"🔍 DIAGNOSTIC: {len(unique_urls)} URLs uniques sur {len(chunks)} chunks")
            print(f"🔍 DIAGNOSTIC: {len(unique_modules)} modules uniques")
            
            if len(unique_urls) < 5:
                print("⚠️  PROBLÈME DÉTECTÉ: Très peu d'URLs distinctes dans les données!")
                print("   URLs trouvées:")
                for url in list(unique_urls)[:10]:
                    print(f"     - {url}")
            
            return chunks
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des chunks : {e}")
            return []

    def _generate_section_title(self, text: str, max_length: int = 60) -> str:
        """Génère un titre de section représentatif du contenu du chunk"""
        lines = text.split('\n')
        
        # Chercher une ligne qui ressemble à un titre
        for line in lines[:5]:
            line = line.strip()
            if line and (
                line.isupper() or 
                line.startswith('#') or 
                line.startswith('##') or
                line.startswith('**') or
                (len(line) < 80 and len(line) > 10)
            ):
                clean_title = re.sub(r'^#+\s*', '', line)
                clean_title = re.sub(r'^\*\*|\*\*$', '', clean_title)
                return clean_title[:max_length].strip()
        
        # Chercher des patterns de titres dans le texte
        title_patterns = [
            r'^\s*([A-Z][^.!?]*(?:Azure|Microsoft|Learning|AI|Machine|Data)[^.!?]*)',
            r'^\s*(\d+\.?\s*[A-Z][^.!?]{10,60})',
            r'^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*:)',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                return match.group(1)[:max_length].strip()
        
        # Sinon, prendre la première phrase substantielle
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if 15 <= len(sentence) <= 100:
                return sentence[:max_length]
        
        # Fallback : premières mots du texte
        words = text.split()[:10]
        return ' '.join(words)[:max_length]

    def _calculate_relevance_score(self, chunk_text: str, keywords: List[str], chunk_data: Dict = None, exclude_used_sources: bool = True) -> Tuple[float, Dict]:
        """Calcule un score de pertinence avec pénalité pour sources déjà utilisées"""
        text_lower = chunk_text.lower()
        score_details = {
            'exact_matches': 0,
            'partial_matches': 0,
            'keyword_density': 0,
            'position_bonus': 0,
            'semantic_bonus': 0,
            'diversity_bonus': 0,
            'matched_keywords': [],
            'diversity_penalty': False
        }
        score = 0.0
        
        # CORRIGÉ: Pénalité basée sur l'URL ET le module
        if exclude_used_sources and chunk_data:
            chunk_url = chunk_data.get('unit_url', '')
            chunk_module = chunk_data.get('unit_name', '')  # ✅ unité pédagogique plus fine
            
            # Pénalité forte si URL déjà utilisée
            if chunk_url in self.used_urls:
                score -= 15.0  # Pénalité très forte pour URL dupliquée
                score_details['diversity_penalty'] = True
                print(f"   ❌ URL déjà utilisée: {chunk_url[:50]}...")
            
            # Pénalité modérée si même module (mais URL différente)
            elif chunk_module in self.used_units:
                score -= 5.0  # Pénalité modérée pour module déjà utilisé
                score_details['module_reuse'] = True
                print(f"   ⚠️  Module déjà utilisé: {chunk_module}")
            
            # Bonus pour nouvelle URL et nouveau module
            else:
                score_details['diversity_bonus'] = 3.0
                score += 3.0
        
        # 1. Compter les occurrences exactes avec position
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if len(keyword_lower) < 3:
                continue
                
            exact_count = text_lower.count(keyword_lower)
            if exact_count > 0:
                score_details['exact_matches'] += exact_count
                score_details['matched_keywords'].append(keyword)
                
                # Score variable selon l'importance du mot-clé
                if len(keyword_lower) > 8:  # Mots techniques longs
                    score += exact_count * 5.0
                elif keyword_lower in ['azure', 'microsoft', 'learning', 'machine', 'ai', 'data']:
                    score += exact_count * 4.0
                else:
                    score += exact_count * 3.0
                
                # Bonus si le mot-clé apparaît tôt dans le texte
                first_occurrence = text_lower.find(keyword_lower)
                if first_occurrence < len(text_lower) * 0.2:
                    score_details['position_bonus'] += 1
                    score += 2.0
        
        # 2. Correspondances partielles et variantes
        words = re.findall(r'\b\w+\b', text_lower)
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if len(keyword_lower) < 4:
                continue
            
            partial_matches = sum(1 for word in words 
                                if keyword_lower in word and word != keyword_lower and len(word) > len(keyword_lower))
            if partial_matches > 0:
                score_details['partial_matches'] += partial_matches
                score += partial_matches * 1.5
        
        # 3. Bonus sémantique pour co-occurrence de termes liés
        semantic_groups = [
            ['machine', 'learning', 'apprentissage', 'modèle', 'algorithme'],
            ['azure', 'microsoft', 'cloud', 'service'],
            ['data', 'données', 'dataset', 'base'],
            ['neural', 'réseau', 'neurone', 'deep'],
            ['nlp', 'natural', 'language', 'processing', 'text', 'linguistique'],
            ['computer', 'vision', 'image', 'reconnaissance', 'visuelle'],
            ['speech', 'voice', 'audio', 'parole', 'reconnaissance']
        ]
        
        for group in semantic_groups:
            group_matches = sum(1 for term in group if term in text_lower)
            if group_matches >= 2:
                score_details['semantic_bonus'] += group_matches
                score += group_matches * 1.0
        
        # 4. Densité de mots-clés
        total_words = len(words)
        if total_words > 0:
            matched_keywords_count = len(score_details['matched_keywords'])
            density = matched_keywords_count / len(keywords) if keywords else 0
            score_details['keyword_density'] = round(density, 3)
            score += density * 5.0
        
        # 5. Bonus pour les chunks courts mais denses
        if total_words < 200 and score > 5:
            score += 2.0
        
        # 6. Pénalité pour chunks très longs avec peu de matches
        if total_words > 500 and score < 3:
            score *= 0.5
        
        return round(score, 2), score_details

    def _create_precise_reference(self, chunk_data: Dict) -> Dict:
        """Crée une référence précise utilisant l'URL SPÉCIFIQUE du chunk"""
        specific_url = chunk_data['unit_url']
        chunk_id = chunk_data['chunk_id']
        content_hash = chunk_data['content_hash']
        section_title = chunk_data['section_title']
        
        # CORRIGÉ: Nettoyer l'URL et créer les variantes
        if '#' in specific_url:
            base_url = specific_url.split('#')[0]
            primary_url = specific_url  # Garder l'URL complète avec ancre
        else:
            base_url = specific_url
            primary_url = f"{specific_url}#chunk-{chunk_id}"
        
        references = {
            'primary_url': primary_url,  # URL avec ancre spécifique
            'base_url': base_url,        # URL de base propre
            'chunk_url': f"{base_url}#chunk-{chunk_id}",
            'hash_url': f"{base_url}#content-{content_hash}",
            'section_url': f"{base_url}#{self._url_safe_string(section_title)}",
            'chunk_identifier': f"{chunk_data['module_name']}/chunk-{chunk_id}",
            'section_title': section_title,
            'content_snippet': chunk_data['text_chunk'][:200] + "..." if len(chunk_data['text_chunk']) > 200 else chunk_data['text_chunk'],
            'is_specific_chunk': True
        }
        
        return references

    def _url_safe_string(self, text: str) -> str:
        """Convertit un texte en format URL-safe pour les ancres"""
        safe = re.sub(r'[^\w\-_]', '-', text.lower())
        safe = re.sub(r'-+', '-', safe)
        return safe.strip('-')[:50]

    def _extract_question_specific_keywords(self, question_text: str) -> List[str]:
        """
        Extrait des mots-clés spécifiques à la question posée
        """
        keywords = []
        text_lower = question_text.lower()
        
        # 1. Termes techniques spécialisés dans la question
        technical_patterns = [
            # NLP et Language
            r'\b(?:nlp|natural language processing|linguistique|text|texte|sentiment|entity|entities|entités|tokenization|lemmatization)\b',
            # Computer Vision
            r'\b(?:computer vision|vision|image|images|reconnaissance|detection|classification|opencv|cnn|convolution)\b',
            # Speech/Audio
            r'\b(?:speech|voice|audio|parole|reconnaissance vocale|text-to-speech|speech-to-text|phonème)\b',
            # Azure Services
            r'\b(?:cognitive services|bot framework|luis|qna maker|custom vision|face api|speech service|text analytics)\b',
            # ML Général
            r'\b(?:machine learning|deep learning|neural network|algorithm|model|training|supervised|unsupervised|clustering|regression|classification)\b',
            # Azure ML spécifique
            r'\b(?:azure ml|ml studio|automl|pipeline|compute|workspace|datastore|experiment)\b'
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text_lower)
            keywords.extend(matches)
        
        # 2. Concepts spécifiques mentionnés
        concept_keywords = {
            'analyse sentiment': ['sentiment', 'analysis', 'emotion', 'opinion'],
            'reconnaissance entités': ['entity', 'entities', 'ner', 'named entity'],
            'classification': ['classification', 'classifier', 'category', 'classe'],
            'détection objets': ['object detection', 'detection', 'bounding box', 'yolo'],
            'reconnaissance faciale': ['face recognition', 'facial', 'biometric', 'visage'],
            'traitement image': ['image processing', 'opencv', 'filter', 'transformation'],
            'bot': ['bot', 'chatbot', 'conversation', 'dialog', 'assistant'],
            'apis': ['api', 'rest', 'endpoint', 'service', 'integration']
        }
        
        for concept, related_terms in concept_keywords.items():
            if concept in text_lower:
                keywords.extend(related_terms)
        
        # 3. Extraire les mots significatifs uniques de la question
        question_words = re.findall(r'\b[a-zA-ZàâäéèêëïîôöùûüÿæœÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÆŒ]{4,}\b', question_text.lower())
        
        # Filtrer les mots courants
        stop_words = {
            'question', 'réponse', 'suivant', 'suivante', 'quelle', 'quelles', 'comment', 'pourquoi',
            'permet', 'utilise', 'faire', 'créer', 'service', 'azure', 'microsoft'
        }
        
        significant_words = [word for word in question_words if word not in stop_words and len(word) > 3]
        keywords.extend(significant_words)
        
        # Dédupliquer et retourner
        return list(set([kw.lower() for kw in keywords if kw and len(kw) > 2]))

    def retrieve_diverse_sources(self, question_text: str, max_results: int = 3, force_diversity: bool = True) -> List[Dict]:
        """
        Recherche avec diversification forcée des sources
        """
        if not self.chunks_data:
            print("❌ Aucune donnée de chunk disponible")
            return []
        
        print(f"🎯 Recherche DIVERSE pour: {question_text[:60]}...")
        
        # Extraire les mots-clés spécifiques à cette question
        enriched_question = self._enrich_question(question_text)
        question_keywords = self._extract_question_specific_keywords(enriched_question)
        print(f"   🔍 Mots-clés: {question_keywords[:6]}")
        
        # Calculer les scores pour chaque chunk
        scored_chunks = []
        for chunk in self.chunks_data:
            score, score_details = self._calculate_relevance_score(
                chunk['text_chunk'], 
                question_keywords, 
                chunk_data=chunk,  # CORRIGÉ: Passer les données du chunk
                exclude_used_sources=force_diversity
            )
            
            if score > 0.5:  # Seuil plus bas pour permettre plus de diversité
                scored_chunks.append((score, chunk, score_details))
        
        # Trier par score décroissant
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        print(f"📊 {len(scored_chunks)} chunks pertinents trouvés")
        
        # Algorithme de diversification AMÉLIORÉ
        selected_sources = []
        
        for score, chunk, score_details in scored_chunks:
            if len(selected_sources) >= max_results:
                break
            
            # Vérifier la diversité stricte
            chunk_url = chunk['unit_url']
            chunk_module = chunk['module_name']
            
            # CORRIGÉ: Diversité stricte par URL
            if force_diversity and chunk_url in self.used_urls:
                print(f"   ⏭️  Ignoré (URL déjà utilisée): {chunk_url[:50]}...")
                continue
            
            # Sélectionner cette source
            references = self._create_precise_reference(chunk)
            
            result = {
                "chunk_id": chunk['chunk_id'],
                "content_hash": chunk['content_hash'],
                "module_name": chunk['module_name'],
                "unit_name": chunk['unit_name'],
                "section_title": chunk['section_title'],
                "full_text": chunk['text_chunk'],
                "content_snippet": references['content_snippet'],
                "token_count": chunk['token_count'],
                "references": references,
                "relevance_score": score,
                "score_details": score_details,
                "matched_keywords": score_details['matched_keywords'],
                "search_metadata": {
                    "chunk_size": len(chunk['text_chunk']),
                    "keyword_coverage": len(score_details['matched_keywords']) / len(question_keywords) if question_keywords else 0,
                    "uses_specific_url": True,
                    "diversity_enforced": force_diversity,
                    "url_is_unique": chunk_url not in self.used_urls,
                    "module_is_unique": chunk_module not in self.used_modules
                }
            }
            
            selected_sources.append(result)
            
            # CORRIGÉ: Marquer les sources comme utilisées
            if force_diversity:
                self.used_urls.add(chunk_url)
                self.used_modules.add(chunk_module)
                self.used_units.add(chunk['unit_name'])  # ✅ Nouvelle règle
            
            print(f"  ✅ Sélectionné: {chunk_module} - Score {score}")
            print(f"     🔗 URL: {chunk_url}")
        
        print(f"🎉 {len(selected_sources)} sources diverses sélectionnées")
        return selected_sources

    def get_best_sources_for_quiz_question(self, quiz_question: str, options: List[str] = None, explanation: str = None, max_sources: int = 3) -> List[Dict]:
        """
        Recherche basée sur la question SPÉCIFIQUE du quiz avec diversification
        """
        print(f"🎯 Recherche pour question: {quiz_question[:80]}...")
        
        # Utiliser la nouvelle méthode de recherche diverse
        return self.retrieve_diverse_sources(
            question_text=quiz_question,
            max_results=max_sources,
            force_diversity=True
        )

    def reset_used_sources(self):
        """Remet à zéro les trackers de sources utilisées"""
        self.used_urls.clear()
        self.used_modules.clear()
        print("🔄 Sources utilisées remises à zéro")

    def _enrich_question(self, raw_question: str) -> str:
        """Reformule la question de l'utilisateur pour améliorer la recherche"""
        return f"Quels sont les concepts ou services clés évoqués dans cette question : « {raw_question.strip()} » ?"

import json
import pandas as pd
import re
from smolagents import tool
from difflib import get_close_matches

# Chargement du CSV des unités locales (adapté à ton format actuel)
df_sources = pd.read_csv("azure_learning_chunks.csv")  # colonnes : module_name, unit_name, unit_url

@tool
def add_precise_sources_to_quiz_tool(quiz_string: str, num_relevant_sources: int = 3) -> str:
    """
    Enrichit chaque question du quiz avec des sources pertinentes (unités d'apprentissage).

    Args:
        quiz_string (str): Chaîne JSON contenant les questions.
        num_relevant_sources (int): Nombre maximal de sources à ajouter par question.

    Returns:
        str: Chaîne JSON enrichie avec les sources pertinentes.
    """
    questions = json.loads(quiz_string)

    for question in questions:
        texte_question = question["question"]
        mots_cles = extraire_mots_cles(texte_question)

        # Recherche dans le DataFrame des unités
        scores = []
        for _, row in df_sources.iterrows():
            titre_complet = f"{row['module_name']} - {row['unit_name']}".lower()
            score = sum(1 for mot in mots_cles if mot in titre_complet)
            if score > 0:
                scores.append((score, row['module_name'], row['unit_name'], row['unit_url']))

        # Tri des meilleures correspondances
        scores = sorted(scores, reverse=True)[:num_relevant_sources]

        question["sources"] = [
            {
                "module": module,
                "title": unit,
                "url": url
            }
            for _, module, unit, url in scores
        ]

    return json.dumps(questions, indent=2, ensure_ascii=False)


def extraire_mots_cles(texte: str):
    # Enlève les mots trop courts et les stopwords simples
    stopwords = {"les", "des", "une", "avec", "dans", "pour", "quoi", "que", "comment", "est", "ce", "qui"}
    mots = re.findall(r"\b\w+\b", texte.lower())
    return [mot for mot in mots if mot not in stopwords and len(mot) > 3]


def create_driver() -> webdriver.Chrome:
    """Crée un driver Chrome headless optimisé"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-logging")
    chrome_options.add_argument("--silent")
    return webdriver.Chrome(options=chrome_options)

# Test de diversification
# Test de diversification des sources - Version complète
import sys
from collections import Counter
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def setup_chrome_driver():
    """Configure et initialise le driver Chrome"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def analyze_source_diversity(chunks_data):
    """Analyse la diversité des sources dans les données"""
    if not chunks_data:
        print("❌ Aucune donnée à analyser")
        return None
    
    urls = [chunk.get('unit_url', 'Unknown') for chunk in chunks_data]
    modules = [chunk.get('module_name', 'Unknown') for chunk in chunks_data]
    
    # Compter les occurrences
    url_counts = Counter(urls)
    module_counts = Counter(modules)
    
    # Calculer les métriques de diversité
    total_chunks = len(chunks_data)
    unique_urls = len(set(urls))
    unique_modules = len(set(modules))
    
    # Indice de diversité (Shannon)
    from math import log
    def shannon_diversity(counts):
        total = sum(counts.values())
        if total == 0:
            return 0
        return -sum((count/total) * log(count/total) for count in counts.values() if count > 0)
    
    url_diversity = shannon_diversity(url_counts)
    module_diversity = shannon_diversity(module_counts)
    
    return {
        'total_chunks': total_chunks,
        'unique_urls': unique_urls,
        'unique_modules': unique_modules,
        'url_counts': url_counts,
        'module_counts': module_counts,
        'url_diversity': url_diversity,
        'module_diversity': module_diversity,
        'diversity_ratio_urls': unique_urls / total_chunks if total_chunks > 0 else 0,
        'diversity_ratio_modules': unique_modules / total_chunks if total_chunks > 0 else 0
    }

def print_diversity_report(analysis):
    """Affiche un rapport détaillé de la diversité"""
    if not analysis:
        return
    
    print(f"📊 RAPPORT DE DIVERSITÉ DES SOURCES")
    print("=" * 60)
    
    # Métriques générales
    print(f"📈 MÉTRIQUES GÉNÉRALES:")
    print(f"   Total chunks analysés: {analysis['total_chunks']:,}")
    print(f"   URLs uniques: {analysis['unique_urls']:,}")
    print(f"   Modules uniques: {analysis['unique_modules']:,}")
    print(f"   Ratio diversité URLs: {analysis['diversity_ratio_urls']:.2%}")
    print(f"   Ratio diversité Modules: {analysis['diversity_ratio_modules']:.2%}")
    
    # Indices de diversité Shannon
    print(f"\n🔬 INDICES DE DIVERSITÉ (Shannon):")
    print(f"   Diversité URLs: {analysis['url_diversity']:.3f}")
    print(f"   Diversité Modules: {analysis['module_diversity']:.3f}")
    
    # Top URLs
    print(f"\n🏆 TOP 10 URLS LES PLUS UTILISÉES:")
    for i, (url, count) in enumerate(analysis['url_counts'].most_common(10), 1):
        percentage = (count / analysis['total_chunks']) * 100
        print(f"   {i:2d}. {url[:60]}... ({count:,} chunks, {percentage:.1f}%)")
    
    # Top Modules
    print(f"\n📚 TOP 10 MODULES LES PLUS UTILISÉS:")
    for i, (module, count) in enumerate(analysis['module_counts'].most_common(10), 1):
        percentage = (count / analysis['total_chunks']) * 100
        print(f"   {i:2d}. {module} ({count:,} chunks, {percentage:.1f}%)")
    
    # Évaluation de la diversité
    print(f"\n💡 ÉVALUATION DE LA DIVERSITÉ:")
    if analysis['diversity_ratio_urls'] > 0.7:
        print("   ✅ Excellente diversité des URLs")
    elif analysis['diversity_ratio_urls'] > 0.5:
        print("   ⚠️  Diversité des URLs modérée")
    else:
        print("   ❌ Faible diversité des URLs - risque de sur-représentation")
    
    if analysis['diversity_ratio_modules'] > 0.3:
        print("   ✅ Bonne diversité des modules")
    else:
        print("   ⚠️  Diversité des modules limitée")

def test_source_matching_quality(matcher, sample_queries):
    """Test la qualité du matching des sources"""
    print(f"\n🧪 TEST DE QUALITÉ DU MATCHING")
    print("-" * 40)
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n📝 Test {i}: '{query}'")
        try:
            # Simuler une recherche
            results = matcher.find_relevant_sources(query, top_k=5)
            if results:
                print(f"   ✅ {len(results)} sources trouvées")
                # Analyser la diversité des résultats
                result_urls = [r.get('url', 'Unknown') for r in results]
                unique_result_urls = len(set(result_urls))
                print(f"   🎯 Diversité: {unique_result_urls}/{len(results)} URLs uniques")
            else:
                print("   ❌ Aucune source trouvée")
        except Exception as e:
            print(f"   ❌ Erreur: {str(e)}")

if __name__ == "__main__":
    print("🎯 TEST DE DIVERSIFICATION DES SOURCES - VERSION CORRIGÉE")
    print("=" * 80)
    
    # Configuration
    driver = None
    matcher = None
    
    try:
        # Initialiser le driver
        print("🔧 Initialisation du driver Chrome...")
        driver = setup_chrome_driver()
        
        # Créer le matcher (assuming PreciseSourceMatcher exists)
        print("🔧 Initialisation du matcher...")
        # matcher = PreciseSourceMatcher(driver=driver, tokenizer=tokenizer)
        
        # Pour la démonstration, créer des données factices
        print("📊 Génération de données de test...")
        sample_chunks_data = [
            {'unit_url': 'https://example1.com/page1', 'module_name': 'module_a', 'content': 'Sample content 1'},
            {'unit_url': 'https://example1.com/page2', 'module_name': 'module_a', 'content': 'Sample content 2'},
            {'unit_url': 'https://example2.com/page1', 'module_name': 'module_b', 'content': 'Sample content 3'},
            {'unit_url': 'https://example3.com/page1', 'module_name': 'module_c', 'content': 'Sample content 4'},
            {'unit_url': 'https://example1.com/page3', 'module_name': 'module_a', 'content': 'Sample content 5'},
        ] * 100  # Multiplier pour avoir plus de données
        
        # DIAGNOSTIC: Analyser les données
        print(f"\n🔍 ANALYSE DES DONNÉES:")
        analysis = analyze_source_diversity(sample_chunks_data)
        
        if analysis:
            print_diversity_report(analysis)
            
            # Test de qualité du matching
            sample_queries = [
                "recherche exemple 1",
                "module spécifique",
                "contenu technique",
                "information générale"
            ]
            
            # Décommenter si PreciseSourceMatcher est disponible
            # test_source_matching_quality(matcher, sample_queries)
            
        print(f"\n✅ Test terminé avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Nettoyage
        if driver:
            print("🧹 Fermeture du driver...")
            driver.quit()
        
        print("🏁 Fin du test de diversification")
