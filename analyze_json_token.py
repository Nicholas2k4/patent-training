import json
import os
from transformers import AutoTokenizer
import pandas as pd

# CONFIGURATION 
JSON_DIR = './patent_jsons'
OUTPUT_CSV = 'json_token.csv'

MODEL_IDS = {
    'llama':   'meta-llama/Llama-3.1-8B-Instruct',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.3',
    'qwen':    'Qwen/Qwen3-8B',
}


def load_tokenizers():
    tokenizers = {}
    for name, model_id in MODEL_IDS.items():
        print(f"Loading tokenizer: {model_id}")
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tokenizers[name] = tok
    return tokenizers


def get_json_parts(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    res = {}
    for k, v in data.items():
        res[k] = v if k != 'claims' else "\n".join(v)
    return res


def get_tokens(json_parts, tokenizers):
    res = {}
    num_models = len(tokenizers)
    model_sums = {name: 0 for name in tokenizers}

    for section_name, section_text in json_parts.items():
        part_total = 0
        for model_name, tokenizer in tokenizers.items():
            count = len(tokenizer.encode(section_text))
            res[f"{section_name}_{model_name}"] = count
            part_total += count
            model_sums[model_name] += count
        res[f"{section_name}_avg"] = round(part_total / num_models, 2)

    # Sum across all sections per model
    total_all = 0
    for model_name, total in model_sums.items():
        res[f"sum_{model_name}"] = total
        total_all += total
    res["sum_avg"] = round(total_all / num_models, 2)

    return res


def main():
    print("\nLoading tokenizers...")
    tokenizers = load_tokenizers()

    print(f"\nScanning: {JSON_DIR}")
    files = sorted([f for f in os.listdir(JSON_DIR) if f.endswith('.json')])
    print(f"Found {len(files)} JSON files\n")

    rows = []
    for i, filename in enumerate(files):
        filepath = os.path.join(JSON_DIR, filename)
        json_parts = get_json_parts(filepath)
        token_data = get_tokens(json_parts, tokenizers)
        token_data['filename'] = filename
        rows.append(token_data)

        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            print(f"  Processed {i + 1}/{len(files)}")

    # Build DataFrame with filename as first column
    df = pd.DataFrame(rows)
    cols = ['filename'] + [c for c in df.columns if c != 'filename']
    df = df[cols]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
