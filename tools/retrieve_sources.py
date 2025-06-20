# tools/retrieve_sources.py
from smolagents import tool
from tools.llm_helper import TopicContextExtractor

extractor = TopicContextExtractor()

@tool
def retrieve_sources_by_keywords(keywords: list[str]) -> list[dict]:
    """
    Recherche les sources les plus pertinentes en fonction d'une liste de mots-clés.

    Args:
        keywords (list[str]): Mots-clés pour la recherche.

    Returns:
        list[dict]: Liste des sources pertinentes avec content et source_url.
    """
    print(f"🔍 Appel de retrieve_sources_by_keywords avec mots-clés : {keywords}")
    if not extractor.is_loaded:
        return [{"error": "TopicContextExtractor non chargé"}]

    query = " ".join(keywords).lower()
    results = extractor.search(query)
    return results
