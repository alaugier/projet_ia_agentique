# tools/prepare_json.py
from smolagents import tool
import json

@tool
def prepare_json_for_final_answer(questions: list) -> str:
    """
    Prépare les données brutes en les transformant proprement en JSON sérialisé pour final_answer_tool.

    Args:
        questions (list): Liste de questions sous forme de dictionnaires

    Returns:
        str: Chaîne JSON propre
    """
    try:
        return json.dumps(questions, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"❌ Erreur lors de la sérialisation JSON : {e}"
