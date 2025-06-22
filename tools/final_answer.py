import json
from smolagents import tool

@tool
def final_answer_tool(questions_json_string: str) -> str:
    """
    Formate les questions AI-900 pour un affichage clair en Markdown, avec liens cliquables vers les sources.

    Args:
        questions_json_string (str): Une chaîne JSON représentant une liste de questions,
            chaque question étant un dictionnaire avec les clés:
            "question", "options", "correct_answer", "explanation", "source_url".

    Returns:
        str: Une chaîne Markdown lisible contenant le QCM formaté avec explication et source.
    """
    try:
        data = json.loads(questions_json_string)
        questions = data.get("questions", [])

        if not isinstance(questions, list):
            return "❌ Erreur : le champ 'questions' est manquant ou invalide."

        output = "## 📘 Quiz AI-900 généré\n\n"
        for i, q in enumerate(questions, 1):
            if not isinstance(q, dict):
                return f"❌ Erreur : un élément n'est pas un dictionnaire: {q}"
            output += f"### ❓ Question {i} : {q.get('question', 'N/A')}\n\n"
            for option in q.get('options', []):
                output += f"- {option}\n"
            output += f"\n✅ **Réponse correcte :** {q.get('correct_answer', 'N/A')}\n"
            output += f"💡 **Explication :** {q.get('explanation', 'N/A')}\n"

            source_url = q.get('source_url', '').strip()
            if source_url:
                output += f"🔗 **Source :** [Documentation Microsoft Azure AI]({source_url})\n"
            else:
                output += f"🔗 **Source :** Aucune source disponible.\n"

            output += "\n---\n\n"

        return output

    except Exception as e:
        return f"❌ Erreur inattendue : {e}"
