# app.py - Version améliorée
from smolagents import DuckDuckGoSearchTool, tool
from smolagents.models import OpenAIServerModel
from smolagents.agents import CodeAgent
import datetime
import requests
import pytz
import yaml
import os
import json
from dotenv import load_dotenv

# Import des outils personnalisés
from tools.final_answer import final_answer_tool
from tools.date_tools import get_today_date_french, get_current_time_in_timezone
from tools.quiz_generator_tool import generate_ai900_quiz_with_local_sources
from tools.ai900_search_tool import search_ai900_knowledge
from tools.llm_helper import set_global_llm_generator, test_topic_context_extraction
from tools.source_adder_tool import add_precise_sources_to_quiz_tool
from tools.prepare_json import prepare_json_for_final_answer
from tools.filter_questions import filter_questions_by_keyword
from tools.retrieve_sources import retrieve_sources_by_keywords
from tools.final_answer_block import final_answer_block
from Gradio_UI_v2 import GradioUI

class QuizAgentApp:
    """Classe principale pour l'application de génération de quiz AI-900"""
    
    def __init__(self):
        self.agent = None
        self.model = None
        self.ui = None
        
    def load_environment(self):
        """Chargement des variables d'environnement"""
        load_dotenv(".env")
        
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            raise ValueError("Erreur: la clé MISTRAL_API_KEY est manquante dans .env")
        
        return mistral_api_key
    
    def setup_model(self, api_key: str):
        """Configuration du modèle Mistral"""
        self.model = OpenAIServerModel(
            api_key=api_key,
            model_id="mistral-medium-latest",
            api_base="https://api.mistral.ai/v1",
            max_tokens=2048,
            temperature=0.5,
        )
        print(f"✅ Modèle configuré: {type(self.model)}")
    
    def setup_tools(self):
        """Configuration des outils pour l'agent"""
        @tool
        def custom_test_tool(arg1: str, arg2: int) -> str:
            """
            Outil de test personnalisé.
            Args:
                arg1: premier argument (string)
                arg2: deuxième argument (entier)
            """
            return f"Outil reçu: {arg1} et {arg2}"
        
        return [
            final_answer_block,
            generate_ai900_quiz_with_local_sources,
            retrieve_sources_by_keywords,
            search_ai900_knowledge,
            set_global_llm_generator,
            add_precise_sources_to_quiz_tool,
            custom_test_tool,
            get_current_time_in_timezone,
            DuckDuckGoSearchTool(),
            get_today_date_french,
            prepare_json_for_final_answer,
            filter_questions_by_keyword,
        ]
    
    def setup_agent(self, tools: list):
        """Initialisation de l'agent avec gestion d'erreurs améliorée"""
        try:
            print(f"DEBUG: Configuration de l'agent avec le modèle: {type(self.model)}")
            
            # Tentative sans templates d'abord
            self.agent = CodeAgent(
                model=self.model,
                tools=tools,
                max_steps=10,
                verbosity_level=2,
                additional_authorized_imports=["json", "yaml", "datetime"]
            )
            print("✅ Agent initialisé avec succès (sans templates)!")
            
        except Exception as e:
            print(f"⚠️ Erreur sans templates: {e}")
            
            # Tentative avec templates
            prompts_file = "prompts.yaml"
            if not os.path.exists(prompts_file):
                raise FileNotFoundError(f"Fichier {prompts_file} non trouvé")
            
            try:
                with open(prompts_file, 'r', encoding='utf-8') as stream:
                    prompt_templates = yaml.safe_load(stream)
                    print("✅ Templates de prompts chargés!")

                self.agent = CodeAgent(
                    model=self.model,
                    tools=tools,
                    max_steps=10,
                    verbosity_level=2,
                    additional_authorized_imports=["json", "yaml", "datetime"],
                    prompt_templates=prompt_templates
                )
                print("✅ Agent initialisé avec succès (avec templates)!")
                
            except Exception as e2:
                raise RuntimeError(f"Impossible d'initialiser l'agent: {e2}")
    
    def run_tests(self):
        """Exécution des tests de fonctionnalité"""
        test_results = []
        
        # Test 1: Extraction de contexte - topic existant
        try:
            result = test_topic_context_extraction("computer_vision")
            test_results.append(("computer_vision", "✅ RÉUSSI", result))
        except Exception as e:
            test_results.append(("computer_vision", f"❌ ÉCHEC: {e}", None))
        
        # Test 2: Extraction de contexte - topic personnalisé
        try:
            result = test_topic_context_extraction("machine_learning", "azure automated ml")
            test_results.append(("machine_learning", "✅ RÉUSSI", result))
        except Exception as e:
            test_results.append(("machine_learning", f"❌ ÉCHEC: {e}", None))
        
        # Affichage des résultats
        print("\n" + "="*50)
        print("RÉSULTATS DES TESTS")
        print("="*50)
        for test_name, status, result in test_results:
            print(f"{test_name}: {status}")
            if result and len(str(result)) < 200:
                print(f"  Résultat: {result}")
        print("="*50 + "\n")
        
        return test_results
    
    def test_quiz_generation(self):
        """Test de génération de quiz"""
        try:
            print("🧪 Test de génération de quiz...")
            
            result = generate_ai900_quiz_with_local_sources(
                topic="nlp", 
                num_questions=2,
                difficulty="intermediate",
                language="french",
                num_relevant_sources=3,
                output_format="json"
            )
            
            print(f"Type de résultat: {type(result)}")
            
            # Traitement du résultat
            if isinstance(result, str):
                parsed_result = json.loads(result)
                json_data = prepare_json_for_final_answer(parsed_result.get("questions", []))
            else:
                json_data = prepare_json_for_final_answer(result.get("questions", []))
            
            markdown = final_answer_block(json_data)
            print("✅ Test de génération de quiz réussi!")
            print("Aperçu du markdown généré:")
            print(markdown[:200] + "..." if len(markdown) > 200 else markdown)
            
        except Exception as e:
            print(f"❌ Erreur lors du test de génération: {e}")
    
    def launch_ui(self):
        """Lancement de l'interface utilisateur Gradio"""
        try:
            self.ui = GradioUI(agent=self.agent)
            print("🚀 Lancement de l'interface Gradio...")
            self.ui.launch()
        except Exception as e:
            raise RuntimeError(f"Erreur lors du lancement de l'interface Gradio: {e}")
    
    def run(self):
        """Méthode principale pour exécuter l'application"""
        try:
            # 1. Chargement de l'environnement
            print("🔧 Chargement de l'environnement...")
            api_key = self.load_environment()
            
            # 2. Configuration du modèle
            print("🤖 Configuration du modèle...")
            self.setup_model(api_key)
            
            # 3. Configuration des outils
            print("🛠️ Configuration des outils...")
            tools = self.setup_tools()
            
            # 4. Initialisation de l'agent
            print("🧠 Initialisation de l'agent...")
            self.setup_agent(tools)
            
            # 5. Tests de fonctionnalité
            print("🧪 Exécution des tests...")
            self.run_tests()
            self.test_quiz_generation()
            
            # 6. Lancement de l'interface
            print("🎮 Lancement de l'interface utilisateur...")
            self.launch_ui()
            
        except Exception as e:
            print(f"💥 Erreur fatale: {e}")
            return False
        
        return True

def main():
    """Point d'entrée principal"""
    app = QuizAgentApp()
    success = app.run()
    
    if not success:
        print("❌ L'application n'a pas pu démarrer correctement.")
        exit(1)
    
    print("✅ Application terminée avec succès.")

if __name__ == "__main__":
    main()