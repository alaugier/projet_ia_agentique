from smolagents import tool

@tool
def filter_questions_by_keyword(questions: list, keyword: str) -> list:
    """
    Filtre les questions contenant un mot-clé donné dans l'intitulé.

    Args:
        questions (list): Liste des questions.
        keyword (str): Mot-clé à rechercher dans les intitulés de question.

    Returns:
        list: Sous-ensemble des questions contenant le mot-clé.
    """
    return [q for q in questions if keyword.lower() in q.get("question", "").lower()]
