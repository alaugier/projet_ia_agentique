# 🤖 Agent IA Générateur de QCM AI-900

Un agent IA intelligent qui génère automatiquement des questionnaires à choix multiples (QCM) pour la certification Microsoft AI-900, avec des réponses détaillées, des explications et des sources contextuelles.

## 🎯 Fonctionnalités

- **Génération automatique de QCM** sur les sujets AI-900 (Machine Learning, Azure AI Services, Computer Vision, NLP, etc.)
- **Contexte thématique enrichi** grâce à une base de données locale de 111 sources
- **Personnalisation avancée** : nombre de questions, difficulté, langue, sources
- **Interface web intuitive** avec Gradio
- **Sources et explications détaillées** pour chaque question
- **Recherche intelligente** dans la base de connaissances AI-900

## 🛠️ Architecture

Le projet utilise l'architecture **Smolagents** avec les composants suivants :

### Agent Principal
- **CodeAgent** : Agent principal orchestrant tous les outils
- **Modèle LLM** : Mistral Medium Latest via API

### Outils Spécialisés
- `generate_ai900_quiz_with_local_sources` : Génération de quiz avec contexte
- `search_ai900_knowledge` : Recherche dans la base de connaissances
- `retrieve_sources_by_keywords` : Récupération de sources pertinentes
- `filter_questions_by_keyword` : Filtrage thématique des questions
- `add_sources_to_quiz_tool` : Enrichissement avec sources externes

### Base de Données
- **TopicContextExtractor** : Extraction de contexte thématique depuis CSV
- **111 sources documentaires** couvrant tous les sujets AI-900

## 📋 Prérequis

- Python 3.12+
- Clé API Mistral AI
- Environnement virtuel recommandé

## 🚀 Installation

1. **Cloner le repository**
```bash
git clone https://github.com/alaugier/projet_ia_agentique.git
cd projet_ia_agentique
```

2. **Créer l'environnement virtuel**
```bash
python -m venv env_hf
source env_hf/bin/activate  # Linux/Mac
# ou
env_hf\Scripts\activate     # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Configuration des variables d'environnement**
```bash
# Créer le fichier .env
cp .env.example .env

# Éditer .env et ajouter votre clé API Mistral
MISTRAL_API_KEY=votre_cle_api_mistral
```

## 💡 Utilisation

### Lancement de l'interface web
```bash
python app.py
```

L'interface Gradio sera accessible à l'adresse : `http://127.0.0.1:7860`

### Utilisation en ligne de commande
```python
from tools.quiz_generator_tool import generate_ai900_quiz_with_local_sources

# Générer un quiz sur le Machine Learning
quiz = generate_ai900_quiz_with_local_sources(
    topic="machine_learning",
    num_questions=5,
    difficulty="intermediate",
    language="french",
    num_relevant_sources=3
)
```

### Paramètres disponibles

| Paramètre | Type | Description | Valeurs |
|-----------|------|-------------|---------|
| `topic` | str | Sujet du quiz | "machine_learning", "computer_vision", "nlp", "azure_ai_services", "general" |
| `num_questions` | int | Nombre de questions | 1-20 |
| `difficulty` | str | Niveau de difficulté | "beginner", "intermediate", "advanced" |
| `language` | str | Langue du quiz | "french", "english" |
| `num_relevant_sources` | int | Nombre de sources contextuelles | 0-10 |
| `output_format` | str | Format de sortie | "json", "text" |

## 🔧 Scripts de diagnostic

Le projet inclut plusieurs scripts de test et diagnostic :

```bash
# Test des imports
python diagnostic_imports.py

# Test du paramètre num_relevant_sources
python script_diagnostic_num_relevant.py
```

## 📁 Structure du projet

```
projet_ia_agentique/
├── app.py                          # Application principale
├── Gradio_UI.py                    # Interface utilisateur Gradio
├── requirements.txt                # Dépendances Python
├── .env.example                    # Template variables environnement
├── data/
│   └── ai900_knowledge_base.csv    # Base de données des sources
├── tools/                          # Outils de l'agent
│   ├── quiz_generator_tool.py      # Générateur de quiz principal
│   ├── llm_helper.py              # Utilitaires LLM et contexte
│   ├── ai900_search_tool.py       # Recherche dans la base
│   ├── source_adder_tool.py       # Ajout de sources
│   ├── filter_questions.py        # Filtrage des questions
│   ├── retrieve_sources.py        # Récupération de sources
│   ├── prepare_json.py            # Formatage JSON
│   ├── final_answer.py            # Formatage réponse finale
│   ├── final_answer_block.py      # Bloc de réponse Markdown
│   ├── date_tools.py              # Outils de date/heure
│   └── logger_tool.py             # Système de logging
└── diagnostics/
    ├── diagnostic_imports.py       # Test des imports
    └── script_diagnostic_num_relevant.py  # Test paramètres
```

## 🧪 Tests et validation

### Test des fonctionnalités principales
```bash
# Test de génération de quiz
python -c "
from tools.quiz_generator_tool import generate_ai900_quiz_with_local_sources
result = generate_ai900_quiz_with_local_sources('computer_vision', 3)
print(result)
"

# Test de recherche
python -c "
from tools.ai900_search_tool import search_ai900_knowledge
results = search_ai900_knowledge('azure cognitive services')
print(results)
"
```

## 📊 Base de données

La base de connaissances contient **111 sources documentaires** couvrant :

- **Machine Learning** : Algorithmes, AutoML, entraînement de modèles
- **Computer Vision** : Reconnaissance d'images, OCR, détection d'objets  
- **NLP** : Traitement du langage, analyse de sentiments, traduction
- **Azure AI Services** : Cognitive Services, Speech, Language Understanding
- **IA Responsable** : Éthique, biais, transparence
- **Concepts fondamentaux** : Types d'IA, cas d'usage, bonnes pratiques

## 🔍 Fonctionnalités avancées

### Extraction de contexte thématique
Le système utilise un **TopicContextExtractor** qui :
- Analyse sémantiquement le contenu des sources
- Extrait les concepts clés par sujet
- Calcule la pertinence contextuelle
- Sélectionne automatiquement les meilleures sources

### Génération intelligente
- **Prompts adaptatifs** selon le niveau de difficulté
- **Validation automatique** des questions générées
- **Évitement des doublons** et questions trop similaires
- **Équilibrage des sujets** pour une couverture complète

## 🚧 Limitations connues

- Nécessite une connexion internet pour l'API Mistral
- Base de données limitée à 111 sources (extensible)
- Génération en français et anglais uniquement
- Dépendant de la qualité des prompts pour la cohérence

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le projet
2. Créez une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Committez vos changements (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Ouvrez une Pull Request

## 📝 Améliorations futures

- [ ] Support de nouveaux modèles LLM (GPT-4, Claude, etc.)
- [ ] Export des quiz en PDF/Word
- [ ] Historique des quiz générés
- [ ] Métriques de performance et analytics
- [ ] API REST pour intégration externe
- [ ] Support multilingue étendu
- [ ] Base de données vectorielle pour améliorer la recherche
- [ ] Interface mobile responsive

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👤 Auteur

**Alexandre Laugier**
- GitHub: [@alaugier](https://github.com/alaugier)
- Email: [votre-email@domain.com]

## 🙏 Remerciements

- **Smolagents** pour le framework d'agents IA
- **Mistral AI** pour l'API de génération de texte
- **Gradio** pour l'interface utilisateur web
- **Microsoft** pour la documentation AI-900

---

*Générez des QCM de qualité professionnelle pour réussir votre certification AI-900 ! 🎓*
