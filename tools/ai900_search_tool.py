from smolagents import tool
from tools.llm_helper import TopicContextExtractor

extractor = TopicContextExtractor()

@tool
def search_ai900_knowledge(query: str, category: str = "all") -> str:
    """
    Recherche des sources pertinentes sur un thème donné en s'appuyant sur le contenu AI-900 vectorisé.

    Args:
        query (str): La requête utilisateur (ex. "vision par ordinateur", "sécurité dans le cloud", etc.)
        category (str): Catégorie à filtrer (non utilisée pour l'instant, valeur par défaut "all")

    Returns:
        str: Résumé des sources pertinentes trouvées, avec lien et extrait de contenu.
    """
    if not extractor.is_loaded:
        return "❌ Erreur : base documentaire non chargée."

    keywords = query.lower().split()
    results = extractor._extract_context_for_keywords(keywords)

    if not results["examples"]:
        return f"❌ Aucun résultat pertinent pour la requête : '{query}'"

    output = f"🔍 Résultats pour : '{query}'\n"
    for example in results["examples"]:
        output += f"\n- **{example['module']} > {example['unit']}**\n"
        output += f"  🔗 Source : {example.get('source_url', 'non disponible')}\n"
        output += f"  🧠 Extrait : {example['content_snippet']}\n"
    return output
