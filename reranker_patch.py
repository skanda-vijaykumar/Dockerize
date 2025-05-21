import os
import re

# Target file patterns
patterns = [
    re.compile(r'from\s+llama_index\.postprocessor\.flag_embedding_reranker\s+import\s+FlagEmbeddingReranker'),
    re.compile(r'base_reranker\s*=\s*FlagEmbeddingReranker\(.*\)'),
    re.compile(r'reranker\s*=\s*RankedNodesLogger\(base_reranker\)')
]

# Replacements
replacements = [
    'from llama_index.core.postprocessor import SimilarityPostprocessor',
    'base_reranker = SimilarityPostprocessor(similarity_cutoff=0.7)',
    'reranker = base_reranker # Disabled RankedNodesLogger'
]

def patch_file(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    for pattern, replacement in zip(patterns, replacements):
        content = pattern.sub(replacement, content)
    
    if content != original:
        print(f"Patching file: {filepath}")
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

# Find route18.py and patch it
for root, _, files in os.walk('/app'):
    if 'route18.py' in files:
        filepath = os.path.join(root, 'route18.py')
        if patch_file(filepath):
            print(f"Successfully patched: {filepath}")
            break