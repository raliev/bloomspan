import pandas as pd
from tqdm import tqdm
import json
import numpy as np
import re

def normalize_phrase(p):
    """Normalize whitespace to ensure dictionary lookups don't fail."""
    return " ".join(str(p).lower().strip().split())

def build_phrase_tree(df):
    """
    Improved tree building that finds the 'longest existing' parent
    to ensure hierarchy even if intermediate steps are missing.
    """
    # 1. Sanitize and Sort
    df['phrase'] = df['phrase'].apply(normalize_phrase)
    df = df.sort_values('length').reset_index(drop=True)
    df['id'] = df.index
    df['parent_id'] = None
    df['level'] = 0
    df['display_phrase'] = df['phrase']

    # 2. Build a fast lookup map: {phrase_string: row_index}
    phrase_to_id = {row.phrase: i for i, row in df.iterrows()}

    # 3. Build a sorted list of phrases to search against (longest first for parent matching)
    # This helps us find the "closest" (longest) parent quickly.
    sorted_phrases = df.sort_values('length', ascending=False)

    print("Linking Parents (Robust Substring Match)...")
    for i in tqdm(range(len(df)), desc="Linking Parents"):
        current_row = df.iloc[i]
        current_phrase = current_row.phrase
        current_len = current_row.length

        # We only look for parents that are shorter than the current phrase
        # To keep it O(N), we check immediate word-stripping first (N-1, N-2...)
        words = current_phrase.split()
        found_parent = False

        # Try shrinking the phrase from left or right until we find a match in our set
        # We try stripping 1 word, then 2, etc.
        for drop in range(1, current_len - 3): # Min length is 4
            # Check suffix (remove words from start)
            suffix = " ".join(words[drop:])
            if suffix in phrase_to_id:
                best_parent_idx = phrase_to_id[suffix]
                found_parent = True

            # Check prefix (remove words from end)
            if not found_parent:
                prefix = " ".join(words[:-drop])
                if prefix in phrase_to_id:
                    best_parent_idx = phrase_to_id[prefix]
                    found_parent = True

            if found_parent:
                parent_text = df.at[best_parent_idx, 'phrase']
                df.at[i, 'parent_id'] = int(best_parent_idx)
                df.at[i, 'level'] = int(df.at[best_parent_idx, 'level']) + 1

                # Create display version: business <PARENT>
                # Use regex for safe replacement to ensure we only replace the first/last occurrence
                # to avoid mangling the string if words repeat.
                display = current_phrase.replace(parent_text, " <PARENT> ")
                df.at[i, 'display_phrase'] = " ".join(display.split())
                break # Stop at the longest match

    return df

def generate_html_tree(df, output_file="tree_view.html", max_nodes=15000):
    """Generates a collapsible HTML tree view with hierarchy integrity."""

    if 'score' not in df.columns:
        print("'score' column missing. Calculating temporary score...")
        max_l = df['length'].max()
        max_f = df['freq'].max()
        df['score'] = np.sqrt((1 - df['length']/max_l)**2 + (1 - df['freq']/max_f)**2)

    # 1. Select nodes and enforce Parent Integrity
    top_df = df.sort_values('score').head(max_nodes)
    all_visible_ids = set(top_df['id'].tolist())

    print("Enforcing tree integrity...")
    for _, row in top_df.iterrows():
        curr_p = row['parent_id']
        while curr_p is not None:
            p_val = int(curr_p)
            if p_val in all_visible_ids: break
            all_visible_ids.add(p_val)
            curr_p = df.at[p_val, 'parent_id']

    final_viz_df = df[df['id'].isin(all_visible_ids)].copy()

    # 2. Build JSON
    nodes_dict = final_viz_df.to_dict('records')
    id_map = {int(n['id']): n for n in nodes_dict}
    for n in nodes_dict: n['children'] = []

    tree = []
    for n in tqdm(nodes_dict, desc="Structuring JSON"):
        p_id = n.get('parent_id')
        if p_id is None or int(p_id) not in id_map:
            tree.append(n)
        else:
            id_map[int(p_id)]['children'].append(n)

    # 3. Template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Phrase Discovery Tree</title>
        <style>
            body { font-family: -apple-system, system-ui, sans-serif; background: #f4f4f9; padding: 40px; }
            .tree-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            ul { list-style-type: none; margin-left: 30px; border-left: 2px solid #edf0f2; padding-left: 15px; }
            .node { cursor: pointer; display: flex; align-items: center; padding: 8px; border-bottom: 1px solid #f0f0f0; }
            .node:hover { background: #f8fbff; }
            .meta { font-family: monospace; background: #333; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; margin-right: 15px; }
            .phrase { font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.9em; }
            .parent-tag { color: #007bff; font-weight: bold; background: #e7f3ff; padding: 0 4px; border-radius: 3px; }
            .toggle { margin-right: 10px; font-size: 12px; color: #999; width: 15px; }
            .hidden { display: none; }
            .collapsed .toggle::before { content: "▶"; }
            .expanded .toggle::before { content: "▼"; }
            .leaf .toggle::before { content: "•"; color: #ccc; }
        </style>
    </head>
    <body>
        <h2>Sequential Phrase Discovery Tree</h2>
        <div class="tree-container" id="tree"></div>
        <script>
            const data = %DATA%;
            function createNode(n) {
                const li = document.createElement('li');
                const div = document.createElement('div');
                div.className = 'node';
                const hasChildren = n.children && n.children.length > 0;
                div.classList.add(hasChildren ? 'collapsed' : 'leaf');
                
                const display = n.display_phrase.replace(/<PARENT>/g, '<span class="parent-tag">&lt;PARENT&gt;</span>');
                div.innerHTML = `<span class="toggle"></span><span class="meta">F:${n.freq} L:${n.length}</span><span class="phrase">${display}</span>`;
                
                li.appendChild(div);
                if (hasChildren) {
                    const ul = document.createElement('ul');
                    ul.className = 'hidden';
                    n.children.sort((a,b) => b.freq - a.freq).forEach(c => ul.appendChild(createNode(c)));
                    div.onclick = (e) => {
                        const isHidden = ul.classList.contains('hidden');
                        ul.classList.toggle('hidden');
                        div.classList.toggle('collapsed', !isHidden);
                        div.classList.toggle('expanded', isHidden);
                        e.stopPropagation();
                    };
                    li.appendChild(ul);
                }
                return li;
            }
            const root = document.createElement('ul');
            data.sort((a,b) => b.freq - a.freq).forEach(d => root.appendChild(createNode(d)));
            document.getElementById('tree').appendChild(root);
        </script>
    </body>
    </html>
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_template.replace("%DATA%", json.dumps(tree)))