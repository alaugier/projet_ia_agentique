# 🤖 Agent IA Générateur de QCM AI-900

Un agent IA intelligent qui génère automatiquement des **questionnaires à choix multiples (QCM)** pour la certification **Microsoft AI-900**, avec :
- des **réponses détaillées**,
- des **explications pédagogiques**, 
- et des **sources fiables issues de la documentation Microsoft Learn**.

---

## 🎯 Objectifs

- Faciliter la préparation à l’examen AI-900.
- Générer des QCM ciblés selon un sujet, un niveau et une langue.
- Fournir des **sources vérifiables** avec **explication contextuelle**.
- S'appuyer sur une **base de connaissances locale vectorisée** pour plus de pertinence.

---

## 🛠️ Architecture

Ce projet repose sur **SmolAgents**, une architecture orientée agents LLM spécialisés.

### 🧠 Agent Principal
- `CodeAgent` : Orchestration globale des outils via Mistral Medium (API).
- LLM utilisé : `Mistral Medium Latest`.

### 🧰 Outils Spécialisés

| Nom de l'outil | Rôle |
|----------------|------|
| `generate_ai900_quiz_with_local_sources` | Génération des QCM avec contexte |
| `search_ai900_knowledge` | Recherche dans la base vectorisée |
| `retrieve_sources_by_keywords` | Récupération de passages pertinents |
| `filter_questions_by_keyword` | Filtrage de QCM existants par thème |
| `add_precise_sources_to_quiz_tool` | Ajout automatique de sources fiables |

---

## 🧠 Base de Données

- `TopicContextExtractor` : moteur d’analyse et de recherche contextuelle.
- **274 sources documentaires** issues de Microsoft Learn couvrant tous les thèmes AI-900.
- Fichiers CSV générés :
  - `ai900_units_list.csv` : liste des unités de formation
  - `azure_learning_chunks.csv` : corpus découpé et vectorisé

---

## 🖥️ Interface utilisateur

L’interface est lancée avec **Gradio** :
```bash
python app.py
```

## 💡 Exemples d’usage en Python

from tools.quiz_generator_tool import generate_ai900_quiz_with_local_sources

quiz = generate_ai900_quiz_with_local_sources(
    topic="azure_ai_services",
    num_questions=5,
    difficulty="intermediate",
    language="french",
    num_relevant_sources=3
)
print(quiz)

## ⚙️ Paramètres

| Paramètre              | Type  | Description                               | Exemples                                           |
| ---------------------- | ----- | ----------------------------------------- | -------------------------------------------------- |
| `topic`                | `str` | Sujet ciblé                               | `"nlp"`, `"machine_learning"`, `"computer_vision"` |
| `num_questions`        | `int` | Nombre de questions générées              | `5`, `10`, `15`                                    |
| `difficulty`           | `str` | Niveau de difficulté                      | `"beginner"`, `"intermediate"`, `"advanced"`       |
| `language`             | `str` | Langue de sortie                          | `"french"` ou `"english"`                          |
| `num_relevant_sources` | `int` | Nombre de sources documentaires associées | `0-10`                                             |

## ⚙️ Installation

### 1. Cloner le dépôt

```git clone https://github.com/alaugier/projet_ia_agentique.git
cd projet_ia_agentique
```

### 2. Créer l’environnement virtuel

```python -m venv env_hf
source env_hf/bin/activate  # ou env_hf\Scripts\activate pour Windows
```

### 3. Installer les dépendances

pip install -r requirements.txt

### 4. Configurer les variables d’environnement

```cp .env.example .env```
# Puis ajouter votre clé API Mistral dans le fichier .env

## 📁 Arborescence du projet

```
projet_ia_agentique/
├── app.py
├── Gradio_UI.py
├── requirements.txt
├── .env.example
├── data/
│   ├── ai900_units_list.csv
│   └── azure_learning_chunks.csv
├── tools/
│   ├── quiz_generator_tool.py
│   ├── llm_helper.py
│   ├── ai900_search_tool.py
│   ├── retrieve_sources.py
│   ├── filter_questions.py
│   ├── source_adder_tool.py
│   ├── final_answer.py
│   ├── final_answer_block.py
│   ├── date_tools.py
│   └── logger_tool.py
├── diagnostics/
│   ├── diagnostic_imports.py
│   └── script_diagnostic_num_relevant.py
└── scripts/
    ├── collect_learning_paths.py
    ├── collect_unit_urls_selenium.py
    ├── scrape_units_and_chunk.py
    └── vectorize_chunks.py
```

## 🔍 Fonctionnalités avancées
### 🔬 Extraction contextuelle

    - TopicContextExtractor analyse sémantiquement les contenus.

    - Vectorisation avec TF-IDF sur les textes pré-nettoyés.

    - Recherche par similarité pour retrouver les passages les plus pertinents (avec URL source propre).

### 🧠 Génération intelligente

    - Questions adaptées au niveau demandé.

    - Équilibrage thématique automatique.

    - Éviction des doublons.

    - Références incluses dans chaque QCM.

## 🧪 Tests rapides

### Générer un quiz simple
python -c "
from tools.quiz_generator_tool import generate_ai900_quiz_with_local_sources
print(generate_ai900_quiz_with_local_sources('computer_vision', 3))

## ⚠️ Limitations

    - API Mistral requise (connexion internet nécessaire).

    - Base locale encore limitée à 274 chunks (extensible).

    - Non optimisé pour des exécutions massives en parallèle.

## 📝 Prochaines améliorations

- Vectorisation sémantique (OpenAI, Mistral embeddings ou HuggingFace)

- Export PDF/Word des QCM

- Historique de sessions

- API REST pour intégration externe

- Version mobile responsive

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
