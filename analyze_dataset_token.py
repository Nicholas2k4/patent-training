import json
import os
import numpy as np
from transformers import AutoTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# CONFIGURATION 
MODEL_IDS = {
    'Llama-3.1-8B':  'meta-llama/Llama-3.1-8B-Instruct',
    'Mistral-7B':    'mistralai/Mistral-7B-Instruct-v0.3',
    'Qwen3-8B':      'Qwen/Qwen3-8B',
}

DATASETS = {
    'task1':              './patent_datasets/task_1_multilingual.json',
    'task2':              './patent_datasets/task_2_multilingual.json',
    'task2_truncated':    './patent_datasets/task_2_multilingual_truncated.json',
}

MAX_SEQ_LENGTHS = {
    'task1': 4096,     
    'task2': 16384,    
    'task2_truncated': 8192, 
}

OUTPUT_DIR = '.'  


def build_messages(model_id, entry):
    instruction = entry['instruction']
    input_text = entry['input']
    output_text = entry['output']

    if 'mistral' in model_id.lower():
        return [
            {"role": "user", "content": instruction + "\n\n" + input_text},
            {"role": "assistant", "content": output_text},
        ]
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]


def count_tokens(tokenizer, model_id, entry):
    messages = build_messages(model_id, entry)
    token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    return len(token_ids)


def analyze_dataset(dataset_path, model_id, tokenizer, max_seq):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    token_counts = np.array([
        count_tokens(tokenizer, model_id, entry) for entry in data
    ])

    fit = int((token_counts <= max_seq).sum())
    over = int((token_counts > max_seq).sum())

    return {
        'samples': len(token_counts),
        'mean': float(np.mean(token_counts)),
        'median': float(np.median(token_counts)),
        'min': int(np.min(token_counts)),
        'max': int(np.max(token_counts)),
        'p75': float(np.percentile(token_counts, 75)),
        'p90': float(np.percentile(token_counts, 90)),
        'p95': float(np.percentile(token_counts, 95)),
        'p99': float(np.percentile(token_counts, 99)),
        'fit': fit,
        'over': over,
        'fit_pct': fit / len(token_counts) * 100,
        'max_seq': max_seq,
        'raw': token_counts,
    }


def plot_dataset_distributions(ds_name, results, max_seq):
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, r) in zip(axes, results.items()):
        ax.hist(r['raw'], bins=50, color='skyblue', edgecolor='black', alpha=0.8)
        ax.axvline(max_seq, color='red', linestyle='dashed', linewidth=2,
                   label=f'MAX_SEQ={max_seq}')
        ax.set_title(f"{model_name}\n"
                     f"mean={r['mean']:.0f}  max={r['max']}  "
                     f"p95={r['p95']:.0f}", fontsize=9)
        ax.set_xlabel('Tokens', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle(f"Token Distribution — {ds_name} (chat template)", fontsize=12, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f'token_distribution_{ds_name}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def main():
    # Load tokenizers
    print("Loading tokenizers...")
    tokenizers = {}
    for model_name, model_id in MODEL_IDS.items():
        print(f"  {model_id}")
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tokenizers[model_name] = (model_id, tok)

    all_stats = []  # For summary CSV

    for ds_name, ds_path in DATASETS.items():
        if not os.path.exists(ds_path):
            print(f"\n[SKIP] {ds_path} not found")
            continue

        max_seq = MAX_SEQ_LENGTHS.get(ds_name, 8192)

        print(f"\n{'=' * 70}")
        print(f"Dataset: {ds_name}")
        print(f"Path   : {ds_path}")
        print(f"Max Seq: {max_seq}")
        print(f"{'=' * 70}")

        ds_results = {}

        for model_name, (model_id, tokenizer) in tokenizers.items():
            print(f"\nAnalyzing {model_name}...")
            r = analyze_dataset(ds_path, model_id, tokenizer, max_seq)
            ds_results[model_name] = r

            print(f"Samples : {r['samples']}")
            print(f"Mean    : {r['mean']:.0f}")
            print(f"Median  : {r['median']:.0f}")
            print(f"Min     : {r['min']}")
            print(f"Max     : {r['max']}")
            print(f"P95     : {r['p95']:.0f}")
            print(f"P99     : {r['p99']:.0f}")
            print(f"≤{max_seq}   : {r['fit']}/{r['samples']} "
                  f"({r['fit_pct']:.1f}%)")

            if r['over'] > 0:
                exceeding = r['raw'][r['raw'] > max_seq]
                print(f"{r['over']} samples exceed, "
                      f"max excess = {int(np.max(exceeding) - max_seq)}")

            # Collect for CSV
            all_stats.append({
                'dataset': ds_name,
                'model': model_name,
                'model_id': model_id,
                'max_seq_length': max_seq,
                'samples': r['samples'],
                'mean': round(r['mean'], 1),
                'median': round(r['median'], 1),
                'min': r['min'],
                'max': r['max'],
                'p75': round(r['p75'], 1),
                'p90': round(r['p90'], 1),
                'p95': round(r['p95'], 1),
                'p99': round(r['p99'], 1),
                'fit_count': r['fit'],
                'over_count': r['over'],
                'fit_pct': round(r['fit_pct'], 2),
            })

        # Plot distribution for this dataset
        plot_path = plot_dataset_distributions(ds_name, ds_results, max_seq)
        print(f"\nPlot saved: {plot_path}")

    # Save summary CSV
    csv_path = os.path.join(OUTPUT_DIR, 'dataset_token_stats.csv')
    df = pd.DataFrame(all_stats)
    df.to_csv(csv_path, index=False)
    print(f"\n{'=' * 70}")
    print(f"SUMMARY CSV saved: {csv_path}")
    print(f"{'=' * 70}")

    # Print summary table
    print(f"\n{'Dataset':<20} {'Model':<15} {'Samples':<8} {'Mean':<8} "
          f"{'P95':<8} {'Max':<8} {'Fit %':<10}")
    print("-" * 80)
    for s in all_stats:
        print(f"{s['dataset']:<20} {s['model']:<15} {s['samples']:<8} "
              f"{s['mean']:<8.0f} {s['p95']:<8.0f} {s['max']:<8} "
              f"{s['fit_pct']:.1f}%")


if __name__ == "__main__":
    main()
