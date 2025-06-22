#!/usr/bin/env python
# coding=utf-8
# Gradio_UI.py - VERSION CORRIGÉE

import mimetypes
import os
import re
import shutil
from typing import Optional

from smolagents.agent_types import AgentAudio, AgentImage, AgentText, handle_agent_output_types
from smolagents.agents import ActionStep, MultiStepAgent
from smolagents.memory import MemoryStep
from smolagents.utils import _is_package_available

import json
from tools.final_answer import final_answer_tool
import gradio as gr

def pull_messages_from_step(step_log: MemoryStep):
    """Extract ChatMessage objects from agent steps with proper nesting"""
    if isinstance(step_log, ActionStep):
        step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else ""
        yield ("chatmessage", f"**{step_number}**")

        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            model_output = step_log.model_output.strip()
            model_output = re.sub(r"```\s*<end_code>", "```", model_output)
            model_output = re.sub(r"<end_code>\s*```", "```", model_output)
            model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)
            model_output = model_output.strip()

            # CORRECTION: Détection améliorée du contenu Markdown
            is_markdown = (
                model_output.strip().startswith("```py") or 
                model_output.strip().startswith("```python") or
                re.search(r"^#{1,6} ", model_output, re.MULTILINE) or 
                re.search(r"^- ", model_output, re.MULTILINE) or
                "📘 Quiz" in model_output or
                "❓ Question" in model_output or
                "✅ **Réponse correcte" in model_output or
                "💡 **Explication" in model_output or
                "🔗 **Source" in model_output
            )
            
            if is_markdown:
                yield ("markdown", model_output)
            else:
                yield ("chatmessage", model_output)

        # NOUVEAU: Vérifier les tool_calls pour final_answer_block
        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
            for tool_call in step_log.tool_calls:
                if tool_call.name == "final_answer_block":
                    # C'est un appel à final_answer_block, récupérer le résultat
                    if hasattr(step_log, "observations") and step_log.observations:
                        markdown_content = step_log.observations.strip()
                        # Nettoyer les logs d'exécution
                        markdown_content = re.sub(r"^Execution logs:\s*", "", markdown_content)
                        if markdown_content and ("📘 Quiz" in markdown_content or "❓ Question" in markdown_content):
                            yield ("quiz_markdown", markdown_content)
                            continue
                
                # Pour les autres tools
                first_tool_call = step_log.tool_calls[0]
                used_code = first_tool_call.name == "python_interpreter"
                
                args = first_tool_call.arguments
                if isinstance(args, dict):
                    content = str(args.get("answer", str(args)))
                else:
                    content = str(args).strip()

                if used_code:
                    content = re.sub(r"```.*?\n", "", content)
                    content = re.sub(r"\s*<end_code>\s*", "", content)
                    content = content.strip()
                    if not content.startswith("```python"):
                        content = f"```python\n{content}\n```"

                yield ("chatmessage", content)

            # Vérifier les observations pour du contenu quiz
            if hasattr(step_log, "observations") and step_log.observations and step_log.observations.strip():
                log_content = step_log.observations.strip()
                if log_content:
                    log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                    
                    # CORRECTION: Détection du contenu quiz dans les observations
                    is_quiz_markdown = (
                        "📘 Quiz" in log_content or
                        "❓ Question" in log_content or
                        "✅ **Réponse correcte" in log_content or
                        (log_content.startswith("##") and "Quiz" in log_content)
                    )
                    
                    if is_quiz_markdown:
                        yield ("quiz_markdown", log_content)
                    else:
                        yield ("chatmessage", log_content)

            if hasattr(step_log, "error") and step_log.error is not None:
                yield ("chatmessage", str(step_log.error))

        elif hasattr(step_log, "error") and step_log.error is not None:
            yield ("chatmessage", str(step_log.error))

        step_footnote = f"{step_number}"
        if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
            token_str = f" | Input-tokens:{step_log.input_token_count:,} | Output-tokens:{step_log.output_token_count:,}"
            step_footnote += token_str
        if hasattr(step_log, "duration"):
            step_duration = f" | Duration: {round(float(step_log.duration), 2)}" if step_log.duration else None
            step_footnote += step_duration
        step_footnote = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
        yield ("chatmessage", step_footnote)
        yield ("chatmessage", "-----")

def stream_to_gradio(agent, task: str, reset_agent_memory: bool = False, additional_args: Optional[dict] = None):
    """NOUVEAU: Version qui retourne aussi le contenu markdown"""
    import gradio as gr
    quiz_markdown_content = ""
    
    for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
        for msg_type, content in pull_messages_from_step(step_log):
            if msg_type == "quiz_markdown":
                quiz_markdown_content = content
                # Ne pas afficher dans le chat, juste stocker
                continue
            elif msg_type == "markdown":
                # Vérifier si c'est du contenu quiz
                if "📘 Quiz" in content or "❓ Question" in content:
                    quiz_markdown_content = content
                    continue
                else:
                    yield gr.ChatMessage(role="assistant", content=content), quiz_markdown_content
            else:
                yield gr.ChatMessage(role="assistant", content=content), quiz_markdown_content

    # Traitement final
    final_answer = step_log
    final_answer = handle_agent_output_types(final_answer)

    if isinstance(final_answer, AgentText):
        content = final_answer.to_string().strip()
        if content.startswith("## 📘 Quiz") or "❓ Question" in content:
            quiz_markdown_content = content
        else:
            yield gr.ChatMessage(
                role="assistant",
                content=f"**Final answer:**\n{content}\n",
            ), quiz_markdown_content
    elif isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
        ), quiz_markdown_content
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
        ), quiz_markdown_content
    else:
        yield gr.ChatMessage(role="assistant", content=f"**Final answer:** {str(final_answer)}"), quiz_markdown_content

class GradioUI:
    """A one-line interface to launch your agent in Gradio"""

    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None):
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
            )
        self.agent = agent
        self.file_upload_folder = file_upload_folder
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)
    
    @staticmethod
    def clean_markdown(md_text: str) -> str:
        """
        Enlève les backticks triples entourant un bloc de code Markdown 
        pour permettre un rendu visuel interprété par gr.Markdown().
        """
        if md_text.startswith("```") and md_text.endswith("```"):
            lines = md_text.splitlines()
            if len(lines) > 2:
                return "\n".join(lines[1:-1]).strip()
        return md_text

    def interact_with_agent_with_quiz_markdown(self, prompt, messages, quiz_md_component):
        """CORRIGÉ: Gestion séparée du chat et du markdown"""
        messages.append(gr.ChatMessage(role="user", content=prompt))
        yield messages, quiz_md_component

        current_quiz_markdown = ""

        try:
            for msg, quiz_content in stream_to_gradio(self.agent, task=prompt, reset_agent_memory=False):
                if msg:  # Si on a un message pour le chat
                    messages.append(msg)
                
                if quiz_content and quiz_content != current_quiz_markdown:
                    current_quiz_markdown = self.clean_markdown(quiz_content)
                
                yield messages, current_quiz_markdown
                
        except Exception as e:
            print(f"Erreur dans interact_with_agent_with_quiz_markdown: {e}")
            messages.append(gr.ChatMessage(role="assistant", content=f"Erreur: {e}"))
            yield messages, current_quiz_markdown

        yield messages, current_quiz_markdown

    # Méthodes upload_file et log_user_message restent identiques...
    def upload_file(self, file, file_uploads_log, allowed_file_types=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
            "text/plain",
        ]):
        import gradio as gr
        if file is None:
            return gr.Textbox("No file uploaded", visible=True), file_uploads_log
        try:
            mime_type, _ = mimetypes.guess_type(file.name)
        except Exception as e:
            return gr.Textbox(f"Error: {e}", visible=True), file_uploads_log
        if mime_type not in allowed_file_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(r"[^\w\-.]", "_", original_name)
        type_to_ext = {}
        for ext, t in mimetypes.types_map.items():
            if t not in type_to_ext:
                type_to_ext[t] = ext
        sanitized_name = sanitized_name.split(".")[:-1]
        sanitized_name.append("" + type_to_ext[mime_type])
        sanitized_name = "".join(sanitized_name)
        file_path = os.path.join(self.file_upload_folder, os.path.basename(sanitized_name))
        shutil.copy(file.name, file_path)
        return gr.Textbox(f"File uploaded: {file_path}", visible=True), file_uploads_log + [file_path]

    def log_user_message(self, text_input, file_uploads_log):
        return (
            text_input + (
                f"\nYou have been provided with these files, which might be helpful or not: {file_uploads_log}"
                if len(file_uploads_log) > 0 else ""
            ),
            "",
        )

    def launch(self, **kwargs):
        import gradio as gr

        with gr.Blocks(fill_height=True) as demo:
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])
            
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Agent",
                        type="messages",
                        avatar_images=(
                            None,
                            "https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/Alfred.png",
                        ),
                        height=600,
                    )
                with gr.Column(scale=1):
                    quiz_markdown = gr.Markdown(
                        label="Quiz AI-900 (Markdown formaté)",
                        value="*Le quiz s'affichera ici une fois généré...*",
                        height=600
                    )
            
            # Upload file section
            if self.file_upload_folder is not None:
                upload_file = gr.File(label="Upload a file")
                upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                upload_file.change(
                    self.upload_file,
                    [upload_file, file_uploads_log],
                    [upload_status, file_uploads_log],
                )
            
            text_input = gr.Textbox(lines=1, label="Chat Message", placeholder="Tapez votre message...")
            
            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input],
            ).then(
                self.interact_with_agent_with_quiz_markdown,
                [stored_messages, chatbot, quiz_markdown],
                [chatbot, quiz_markdown],
            )

        demo.launch(debug=True, share=True, **kwargs)

__all__ = ["stream_to_gradio", "GradioUI"]