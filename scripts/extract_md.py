import json

def extract(files, out_file):
    with open(out_file, 'w', encoding='utf-8') as f_out:
        for f in files:
            f_out.write(f"=== {f} ===\n")
            try:
                with open(f, 'r', encoding='utf-8') as f_in:
                    data = json.load(f_in)
                    for c in data.get('cells', []):
                        if c.get('cell_type') == 'markdown':
                            f_out.write(''.join(c.get('source', [])) + '\n\n')
            except Exception as e:
                f_out.write(f"Error: {e}\n")

extract(['alphaearth_workshop.ipynb', 'challenge.ipynb'], 'notebooks_summary.txt')
