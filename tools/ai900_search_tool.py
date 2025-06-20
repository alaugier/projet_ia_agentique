from smolagents import tool
import json
from typing import Dict, List

@tool
def search_ai900_knowledge(query: str, category: str = "all") -> str:
    """
    Searches for AI-900 related information based on query.
    
    Args:
        query: The search query related to AI-900 topics
        category: Category to search in ('machine_learning', 'cognitive_services', 'azure_ai', 'responsible_ai', 'all')
    """
    
    # Base de connaissances structurée
    knowledge_base = {
        'machine_learning': {
            'supervised_learning': "Supervised learning uses labeled data to train models. Examples include classification and regression.",
            'unsupervised_learning': "Unsupervised learning finds patterns in data without labels. Examples include clustering and dimensionality reduction.",
            'azure_ml': "Azure Machine Learning is a cloud service for building, training, and deploying ML models at scale."
        },
        'cognitive_services': {
            'computer_vision': "Azure Computer Vision analyzes images and videos to extract information like text, objects, and faces.",
            'speech_services': "Azure Speech Services include speech-to-text, text-to-speech, and speech translation capabilities.",
            'language_understanding': "LUIS (Language Understanding) helps build natural language understanding into apps.",
            'text_analytics': "Text Analytics API provides sentiment analysis, key phrase extraction, and language detection."
        },
        'responsible_ai': {
            'fairness': "AI systems should treat all people fairly and avoid discriminatory impacts.",
            'reliability': "AI systems should perform reliably and safely under normal and unexpected conditions.",
            'safety': "AI systems should be safe and not cause harm to people or society.",
            'privacy': "AI systems should respect privacy and be designed with appropriate privacy protections.",
            'inclusiveness': "AI systems should be inclusive and accessible to all users.",
            'transparency': "AI systems should be understandable and provide clear explanations of their decisions.",
            'accountability': "People should be accountable for AI systems and their outcomes."
        }
    }
    
    try:
        query_lower = query.lower()
        results = []
        
        # Fonction de recherche dans une catégorie
        def search_in_category(cat_name: str, cat_data: Dict) -> List[Dict]:
            matches = []
            for key, value in cat_data.items():
                if query_lower in key.lower() or query_lower in value.lower():
                    matches.append({
                        'category': cat_name,
                        'topic': key,
                        'description': value,
                        'relevance_score': _calculate_relevance(query_lower, key, value)
                    })
            return matches
        
        # Recherche dans les catégories
        if category == "all":
            for cat_name, cat_data in knowledge_base.items():
                results.extend(search_in_category(cat_name, cat_data))
        elif category in knowledge_base:
            results.extend(search_in_category(category, knowledge_base[category]))
        else:
            return f"Error: Invalid category '{category}'. Valid categories: {list(knowledge_base.keys()) + ['all']}"
        
        # Tri par pertinence
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        if not results:
            return f"No relevant information found for query: '{query}'"
        
        # Limitation à 5 résultats les plus pertinents
        top_results = results[:5]
        
        search_response = {
            'query': query,
            'category': category,
            'results_count': len(top_results),
            'results': top_results
        }
        
        return json.dumps(search_response, indent=2, ensure_ascii=False)
        
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

def _calculate_relevance(query: str, key: str, value: str) -> float:
    """Calcule un score de pertinence simple"""
    score = 0.0
    
    # Correspondance exacte dans le titre
    if query in key.lower():
        score += 10
    
    # Correspondance exacte dans la description
    if query in value.lower():
        score += 5
    
    # Correspondance partielle
    query_words = query.split()
    for word in query_words:
        if word in key.lower():
            score += 3
        if word in value.lower():
            score += 1
    
    return score