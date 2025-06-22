#!/usr/bin/env python3
# diagnostic_imports.py - Script pour diagnostiquer les problèmes d'imports

import sys
import os
import traceback

print("🔍 Diagnostic des imports du module quiz_generator_tool")
print("=" * 60)

# Test 1: Import du module complet
print("\n1. Test d'import du module complet...")
try:
    import tools.quiz_generator_tool
    print("✅ Module tools.quiz_generator_tool importé avec succès")
except Exception as e:
    print(f"❌ Erreur import module: {e}")
    print("Traceback complet:")
    traceback.print_exc()
    print("\n" + "="*60)

# Test 2: Import des dépendances une par une
print("\n2. Test des imports individuels...")

imports_to_test = [
    ("json", "json"),
    ("random", "random"),
    ("csv", "csv"),
    ("os", "os"),
    ("datetime", "datetime"),
    ("typing", "typing"),
    ("pandas", "pandas as pd"),
    ("smolagents", "smolagents"),
]

for module_name, import_statement in imports_to_test:
    try:
        exec(f"import {import_statement}")
        print(f"✅ {module_name}")
    except Exception as e:
        print(f"❌ {module_name}: {e}")

# Test 3: Import des modules locaux
print("\n3. Test des imports locaux...")

local_imports = [
    "tools.llm_helper",
    "tools.logger_tool",
]

for module in local_imports:
    try:
        exec(f"import {module}")
        print(f"✅ {module}")
    except Exception as e:
        print(f"❌ {module}: {e}")
        traceback.print_exc()

# Test 4: Test spécifique des fonctions dans llm_helper
print("\n4. Test spécifique de llm_helper...")
try:
    from tools.llm_helper import get_global_llm_generator
    print("✅ get_global_llm_generator importé")
    
    # Test de l'appel
    generator = get_global_llm_generator()
    if generator is None:
        print("⚠️  get_global_llm_generator() retourne None")
    else:
        print("✅ get_global_llm_generator() retourne un objet")
        
except Exception as e:
    print(f"❌ get_global_llm_generator: {e}")
    traceback.print_exc()

# Test 5: Test spécifique du logger
print("\n5. Test spécifique du logger...")
try:
    from tools.logger_tool import setup_logger
    print("✅ setup_logger importé")
    
    logger = setup_logger("test")
    print("✅ Logger créé avec succès")
    
except Exception as e:
    print(f"❌ setup_logger: {e}")
    traceback.print_exc()

# Test 6: Import avec gestion d'erreur comme dans le fichier original
print("\n6. Test d'import avec try/except...")
try:
    sys.path.append('.')
    from tools.quiz_generator_tool import generate_ai900_quiz_with_local_sources
    print("✅ generate_ai900_quiz_with_local_sources importé")
except Exception as e:
    print(f"❌ generate_ai900_quiz_with_local_sources: {e}")
    traceback.print_exc()

# Test 7: Vérification du contenu du module
print("\n7. Inspection du module...")
try:
    import tools.quiz_generator_tool as qgt
    functions = [name for name in dir(qgt) if not name.startswith('_')]
    print(f"✅ Fonctions disponibles dans le module: {functions}")
    
    # Chercher spécifiquement les fonctions tool
    tool_functions = [name for name in functions if hasattr(getattr(qgt, name), '__call__')]
    print(f"✅ Fonctions callable: {tool_functions}")
    
except Exception as e:
    print(f"❌ Inspection du module: {e}")

print("\n" + "="*60)
print("🏁 Fin du diagnostic")