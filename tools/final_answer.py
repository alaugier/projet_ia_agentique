# tools/final_answer.py
from smolagents import tool
import json

@tool
def final_answer_tool(questions_json_string: str) -> str:
    """
    Formate les questions AI-900 pour un affichage clair en Markdown, avec liens cliquables vers les sources.

    Args:
        questions_json_string (str): Chaîne JSON représentant une liste de questions au format :
            [
                {
                    "question": "Texte de la question",
                    "options": ["A. ...", "B. ...", ...],
                    "correct_answer": "B. ...",
                    "explanation": "Explication de la réponse",
                    "source_url": "https://..."
                },
                ...
            ]

    Returns:
        str: Une chaîne Markdown lisible contenant le QCM formaté avec explication et source.
    """
    try:
        questions = json.loads(questions_json_string)

        output = "## 📘 Quiz AI-900 généré\n\n"
        for i, q in enumerate(questions, 1):
            output += f"### ❓ Question {i} : {q.get('question', 'N/A')}\n\n"
            for option in q.get('options', []):
                output += f"- {option}\n"
            output += f"\n✅ **Réponse correcte :** {q.get('correct_answer', 'N/A')}\n"
            output += f"💡 **Explication :** {q.get('explanation', 'N/A')}\n"

            source_url = q.get('source_url', '').strip()
            if source_url:
                output += f"🔗 **Source :** [Documentation Microsoft Azure AI]({source_url})\n"
            else:
                output += "🔗 **Source :** Aucune source disponible.\n"

            output += "\n---\n\n"

        return output

    except json.JSONDecodeError:
        return f"❌ Erreur : le LLM n’a pas renvoyé un JSON valide. Voici la sortie brute :\n\n{questions_json_string}"
    except Exception as e:
        return f"❌ Erreur inattendue : {e}\n\nSortie brute :\n{questions_json_string}"
