from smolagents import tool
from tools.final_answer import final_answer_tool

@tool
def final_answer_block(questions_json_string: str) -> str:
    """
    Transforme une chaîne JSON représentant un quiz en markdown interprété.

    Args:
        questions_json_string (str): Chaîne JSON contenant le quiz à formater.

    Returns:
        str: Markdown formaté du quiz, sans bloc de code.
    """
    markdown_quiz = final_answer_tool(questions_json_string)
    markdown_quiz = markdown_quiz.strip()
    
    # Nettoyage optionnel si jamais du code était resté encodé en ```py
    markdown_quiz = markdown_quiz.replace("```py", "").replace("```python", "").replace("```", "")
    
    return markdown_quiz
