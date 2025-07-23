# tools/prepare_json.py
from smolagents import tool
import json

@tool
def prepare_json_for_final_answer(questions: list) -> str:
    """
    Prépare les données brutes en les encapsulant dans un objet JSON {"questions": [...]}

    Args:
        questions (list): Liste de questions sous forme de dictionnaires

    Returns:
        str: Chaîne JSON propre avec clé "questions"
    """
    try:
        return json.dumps({"questions": questions}, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"❌ Erreur lors de la sérialisation JSON : {e}"
