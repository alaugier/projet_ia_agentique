#!/usr/bin/env python
# coding=utf-8
# Gradio_UI.py - Version améliorée et refactorisée

import mimetypes
import os
import re
import shutil
from typing import Optional, Generator, Tuple
import gradio as gr

from smolagents.agent_types import AgentAudio, AgentImage, AgentText, handle_agent_output_types
from smolagents.agents import ActionStep, MultiStepAgent
from smolagents.memory import MemoryStep
from smolagents.utils import _is_package_available

import json
from tools.final_answer import final_answer_tool

class MessageProcessor:
    """Classe pour traiter les messages et étapes de l'agent"""
    
    @staticmethod
    def clean_model_output(model_output: str) -> str:
        """Nettoie la sortie du modèle en supprimant les balises de code indésirables"""
        model_output = model_output.strip()
        model_output = re.sub(r"```\s*<end_code>", "```", model_output)
        model_output = re.sub(r"<end_code>\s*```", "```", model_output)
        model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)
        return model_output.strip()
    
    @staticmethod
    def is_markdown_content(content: str) -> bool:
        """Détermine si le contenu doit être traité comme du Markdown"""
        return (
            content.strip().startswith("```py") or 
            content.strip().startswith("```python") or
            re.search(r"^#{1,6} ", content, re.MULTILINE) or 
            re.search(r"^- ", content, re.MULTILINE) or
            "📘 Quiz" in content or
            "❓ Question" in content or
            "✅ **Réponse correcte" in content or
            "💡 **Explication" in content or
            "🔗 **Source" in content
        )
    
    @staticmethod
    def is_quiz_content(content: str) -> bool:
        """Détermine si le contenu est un quiz AI-900"""
        return (
            "📘 Quiz" in content or
            "❓ Question" in content or
            "✅ **Réponse correcte" in content or
            (content.startswith("##") and "Quiz" in content)
        )
    
    @staticmethod
    def clean_execution_logs(content: str) -> str:
        """Supprime les logs d'exécution du contenu"""
        return re.sub(r"^Execution logs:\s*", "", content)

def pull_messages_from_step(step_log: MemoryStep) -> Generator[Tuple[str, str], None, None]:
    """Extrait les messages de ChatMessage depuis les étapes de l'agent avec une structuration appropriée"""
    processor = MessageProcessor()
    
    if not isinstance(step_log, ActionStep):
        return
    
    # En-tête de l'étape
    step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else ""
    yield ("chatmessage", f"**{step_number}**")

    # Traitement de la sortie du modèle
    if hasattr(step_log, "model_output") and step_log.model_output is not None:
        model_output = processor.clean_model_output(step_log.model_output)
        
        if processor.is_markdown_content(model_output):
            yield ("markdown", model_output)
        else:
            yield ("chatmessage", model_output)

    # Traitement des appels d'outils
    if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
        for tool_call in step_log.tool_calls:
            # Gestion spéciale pour final_answer_block
            if tool_call.name == "final_answer_block":
                if hasattr(step_log, "observations") and step_log.observations:
                    markdown_content = processor.clean_execution_logs(step_log.observations.strip())
                    if markdown_content and processor.is_quiz_content(markdown_content):
                        yield ("quiz_markdown", markdown_content)
                        continue
            
            # Traitement général des outils
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

        # Traitement des observations
        if hasattr(step_log, "observations") and step_log.observations and step_log.observations.strip():
            log_content = processor.clean_execution_logs(step_log.observations.strip())
            
            if log_content:
                if processor.is_quiz_content(log_content):
                    yield ("quiz_markdown", log_content)
                else:
                    yield ("chatmessage", log_content)

        # Traitement des erreurs
        if hasattr(step_log, "error") and step_log.error is not None:
            yield ("chatmessage", f"❌ Erreur: {str(step_log.error)}")

    elif hasattr(step_log, "error") and step_log.error is not None:
        yield ("chatmessage", f"❌ Erreur: {str(step_log.error)}")

    # Pied de page de l'étape avec métadonnées
    step_footnote = f"{step_number}"
    if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
        token_str = f" | Input-tokens:{step_log.input_token_count:,} | Output-tokens:{step_log.output_token_count:,}"
        step_footnote += token_str
    if hasattr(step_log, "duration"):
        step_duration = f" | Duration: {round(float(step_log.duration), 2)}s" if step_log.duration else ""
        step_footnote += step_duration
    
    step_footnote = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span>"""
    yield ("chatmessage", step_footnote)
    yield ("chatmessage", "-----")

def stream_to_gradio(agent, task: str, reset_agent_memory: bool = False, additional_args: Optional[dict] = None) -> Generator[Tuple[gr.ChatMessage, str], None, None]:
    """
    Streaming vers Gradio avec gestion séparée du contenu quiz
    
    Returns:
        Generator yielding (ChatMessage, quiz_markdown_content) tuples
    """
    quiz_markdown_content = ""
    
    try:
        for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
            for msg_type, content in pull_messages_from_step(step_log):
                if msg_type == "quiz_markdown":
                    quiz_markdown_content = content
                    # Ne pas afficher dans le chat, juste stocker
                    continue
                elif msg_type == "markdown":
                    # Vérifier si c'est du contenu quiz
                    if MessageProcessor.is_quiz_content(content):
                        quiz_markdown_content = content
                        continue
                    else:
                        yield gr.ChatMessage(role="assistant", content=content), quiz_markdown_content
                else:
                    yield gr.ChatMessage(role="assistant", content=content), quiz_markdown_content

        # Traitement de la réponse finale
        final_answer = handle_agent_output_types(step_log)

        if isinstance(final_answer, AgentText):
            content = final_answer.to_string().strip()
            if content.startswith("## 📘 Quiz") or MessageProcessor.is_quiz_content(content):
                quiz_markdown_content = content
            else:
                yield gr.ChatMessage(
                    role="assistant",
                    content=f"**Réponse finale:**\n{content}\n",
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
            yield gr.ChatMessage(
                role="assistant", 
                content=f"**Réponse finale:** {str(final_answer)}"
            ), quiz_markdown_content

    except Exception as e:
        yield gr.ChatMessage(
            role="assistant",
            content=f"❌ Erreur lors de l'exécution: {str(e)}"
        ), quiz_markdown_content

class GradioUI:
    """Interface utilisateur Gradio pour l'agent de génération de quiz"""

    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None):
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Veuillez installer 'gradio' pour utiliser GradioUI: `pip install 'smolagents[gradio]'`"
            )
        self.agent = agent
        self.file_upload_folder = file_upload_folder
        if self.file_upload_folder is not None:
            os.makedirs(file_upload_folder, exist_ok=True)
    
    @staticmethod
    def clean_markdown(md_text: str) -> str:
        """
        Supprime les backticks triples entourant un bloc de code Markdown 
        pour permettre un rendu visuel interprété par gr.Markdown().
        """
        if md_text.startswith("```") and md_text.endswith("```"):
            lines = md_text.splitlines()
            if len(lines) > 2:
                return "\n".join(lines[1:-1]).strip()
        return md_text

    def interact_with_agent_with_quiz_markdown(self, prompt: str, messages: list, quiz_md_component: str):
        """Gestion de l'interaction avec l'agent avec affichage séparé du quiz"""
        messages.append(gr.ChatMessage(role="user", content=prompt))
        yield messages, quiz_md_component

        current_quiz_markdown = quiz_md_component

        try:
            for msg, quiz_content in stream_to_gradio(self.agent, task=prompt, reset_agent_memory=False):
                if msg:  # Si on a un message pour le chat
                    messages.append(msg)
                
                if quiz_content and quiz_content != current_quiz_markdown:
                    current_quiz_markdown = self.clean_markdown(quiz_content)
                
                yield messages, current_quiz_markdown
                
        except Exception as e:
            print(f"❌ Erreur dans interact_with_agent_with_quiz_markdown: {e}")
            messages.append(gr.ChatMessage(
                role="assistant", 
                content=f"❌ Erreur lors de l'interaction: {str(e)}"
            ))
            yield messages, current_quiz_markdown

    def upload_file(self, file, file_uploads_log: list, allowed_file_types: list = None):
        """Gestion de l'upload de fichiers avec validation"""
        if allowed_file_types is None:
            allowed_file_types = [
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                "text/plain",
            ]
        
        if file is None:
            return gr.Textbox("Aucun fichier uploadé", visible=True), file_uploads_log
        
        try:
            mime_type, _ = mimetypes.guess_type(file.name)
        except Exception as e:
            return gr.Textbox(f"Erreur: {e}", visible=True), file_uploads_log
        
        if mime_type not in allowed_file_types:
            return gr.Textbox(
                f"Type de fichier non autorisé. Types acceptés: {', '.join(allowed_file_types)}", 
                visible=True
            ), file_uploads_log
        
        # Nettoyage et sauvegarde du fichier
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(r"[^\w\-.]", "_", original_name)
        
        # Gestion de l'extension
        type_to_ext = {t: ext for ext, t in mimetypes.types_map.items()}
        if mime_type in type_to_ext:
            name_parts = sanitized_name.split(".")[:-1]
            name_parts.append(type_to_ext[mime_type].lstrip('.'))
            sanitized_name = ".".join(name_parts)
        
        file_path = os.path.join(self.file_upload_folder, sanitized_name)
        shutil.copy(file.name, file_path)
        
        return gr.Textbox(f"✅ Fichier uploadé: {file_path}", visible=True), file_uploads_log + [file_path]

    def log_user_message(self, text_input: str, file_uploads_log: list):
        """Préparation du message utilisateur avec les fichiers uploadés"""
        final_message = text_input
        if file_uploads_log:
            final_message += f"\n\n📎 Fichiers fournis: {', '.join(file_uploads_log)}"
        
        return final_message, ""

    def launch(self, **kwargs):
        """Lancement de l'interface Gradio"""
        with gr.Blocks(
            fill_height=True, 
            title="Agent Quiz AI-900",
            theme=gr.themes.Soft()
        ) as demo:
            
            # État de l'application
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])
            
            # En-tête
            gr.Markdown("""
            # 🤖 Agent Quiz AI-900
            Générez des quiz personnalisés pour la certification Microsoft AI-900 avec des sources documentées.
            """)
            
            # Layout principal
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="💬 Conversation avec l'Agent",
                        type="messages",
                        avatar_images=(
                            None,
                            "https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/Alfred.png",
                        ),
                        height=600,
                        show_copy_button=True,
                    )
                
                with gr.Column(scale=1):
                    quiz_markdown = gr.Markdown(
                        label="📝 Quiz AI-900 (Formaté)",
                        value="*Le quiz s'affichera ici une fois généré...*",
                        height=600
                    )
            
            # Section d'upload de fichiers
            if self.file_upload_folder is not None:
                with gr.Group():
                    gr.Markdown("### 📎 Upload de fichiers")
                    upload_file = gr.File(
                        label="Sélectionnez un fichier (PDF, DOCX, TXT)",
                        file_count="single"
                    )
                    upload_status = gr.Textbox(
                        label="Statut de l'upload", 
                        interactive=False, 
                        visible=False
                    )
                    
                    upload_file.change(
                        self.upload_file,
                        [upload_file, file_uploads_log],
                        [upload_status, file_uploads_log],
                    )
            
            # Zone de saisie
            with gr.Group():
                text_input = gr.Textbox(
                    lines=2, 
                    label="💬 Votre message", 
                    placeholder="Tapez votre demande ici... (ex: 'Génère un quiz sur le NLP avec 5 questions')",
                    show_copy_button=True
                )
                
                with gr.Row():
                    submit_btn = gr.Button("📤 Envoyer", variant="primary")
                    clear_btn = gr.Button("🗑️ Effacer", variant="secondary")
            
            # Exemples de prompts
            with gr.Group():
                gr.Markdown("### 💡 Exemples de demandes")
                example_prompts = [
                    "Génère un quiz sur Computer Vision avec 3 questions niveau débutant",
                    "Crée 5 questions sur Azure Cognitive Services niveau intermédiaire",
                    "Quiz sur Machine Learning avec sources documentées",
                    "Questions sur les services IA Azure pour la certification AI-900"
                ]
                
                for example in example_prompts:
                    gr.Button(
                        example, 
                        size="sm"
                    ).click(
                        lambda x=example: x,
                        outputs=text_input
                    )
            
            # Gestion des événements
            def clear_conversation():
                return [], "*Le quiz s'affichera ici une fois généré...*", ""
            
            submit_event = text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input],
            ).then(
                self.interact_with_agent_with_quiz_markdown,
                [stored_messages, chatbot, quiz_markdown],
                [chatbot, quiz_markdown],
            )
            
            submit_btn.click(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input],
            ).then(
                self.interact_with_agent_with_quiz_markdown,
                [stored_messages, chatbot, quiz_markdown],
                [chatbot, quiz_markdown],
            )
            
            clear_btn.click(
                clear_conversation,
                outputs=[chatbot, quiz_markdown, text_input]
            )

        # Configuration du lancement
        launch_kwargs = {
            "debug": True,
            "share": True,
            "server_name": "0.0.0.0",
            "show_error": True,
            **kwargs
        }
        
        demo.launch(**launch_kwargs)

__all__ = ["stream_to_gradio", "GradioUI", "MessageProcessor"]