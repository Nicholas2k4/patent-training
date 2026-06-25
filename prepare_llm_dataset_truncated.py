import json
from transformers import AutoTokenizer

# CONFIGURATION
MAX_SEQ_LENGTH = 8192

MODEL_IDS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen3-8B",
]

INPUT_DATASET = './patent_datasets/task_2_multilingual.json'
OUTPUT_DATASET = './patent_datasets/task_2_multilingual_truncated.json'
SAFETY_MARGIN = 5


def build_messages(model_id, instruction, input_text, output_text):
    if "mistral" in model_id.lower():
        return [
            {"role": "user", "content": instruction + "\n\n" + input_text},
            {"role": "assistant", "content": output_text},
        ]
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]


def count_chat_tokens(tokenizer, messages):
    """Count exact tokens after applying the model's chat template."""
    token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    return len(token_ids)


def find_worst_case(tokenizers_info, instruction, input_text, output_text):
    """
    Find which model produces the most tokens for this sample.
    Returns (worst_token_count, worst_model_id, worst_tokenizer).
    """
    worst = (0, None, None)
    for model_id, tokenizer in tokenizers_info:
        messages = build_messages(model_id, instruction, input_text, output_text)
        n_tokens = count_chat_tokens(tokenizer, messages)
        if n_tokens > worst[0]:
            worst = (n_tokens, model_id, tokenizer)
    return worst


def truncate_sample(tokenizers_info, instruction, input_text, output_json_str, max_seq_length):
    """
    Truncate detailed_description so the chat-templated sequence 
    fits within max_seq_length for ALL models.
    """
    orig_worst, worst_model, worst_tok = find_worst_case(
        tokenizers_info, instruction, input_text, output_json_str
    )

    if orig_worst <= max_seq_length:
        return output_json_str, False, orig_worst, orig_worst

    # Parse JSON to access detailed_description
    sections = json.loads(output_json_str)
    dd_text = sections.get("detailed_description", "")

    if not dd_text:
        return None, True, orig_worst, orig_worst

    # Calculate how many tokens to remove from detailed_description
    excess = orig_worst - max_seq_length + SAFETY_MARGIN
    dd_tokens = worst_tok.encode(dd_text)
    target_dd_len = len(dd_tokens) - excess

    if target_dd_len <= 0:
        return None, True, orig_worst, orig_worst

    # Truncate at token level and decode back to text
    truncated_dd = worst_tok.decode(
        dd_tokens[:target_dd_len], skip_special_tokens=True
    ).rstrip()

    # Try to end at sentence boundary
    if truncated_dd and not truncated_dd.endswith("."):
        last_period = truncated_dd.rfind(". ")
        if last_period > len(truncated_dd) * 0.8:
            truncated_dd = truncated_dd[:last_period + 1]

    # Rebuild JSON output
    sections["detailed_description"] = truncated_dd
    new_output = json.dumps(sections, indent=2, ensure_ascii=False)

    # Verify all models fit
    final_worst, _, _ = find_worst_case(
        tokenizers_info, instruction, input_text, new_output
    )

    # If still over, trim one more token at a time until it fits
    while final_worst > max_seq_length and target_dd_len > 0:
        target_dd_len -= 1
        truncated_dd = worst_tok.decode(
            dd_tokens[:target_dd_len], skip_special_tokens=True
        ).rstrip()
        sections["detailed_description"] = truncated_dd
        new_output = json.dumps(sections, indent=2, ensure_ascii=False)
        final_worst, _, _ = find_worst_case(
            tokenizers_info, instruction, input_text, new_output
        )

    if final_worst > max_seq_length:
        return None, True, orig_worst, final_worst

    return new_output, True, orig_worst, final_worst


def main():
    # Load all tokenizers
    print("Loading tokenizers...")
    tokenizers_info = []
    for model_id in MODEL_IDS:
        print(f"  {model_id}")
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tokenizers_info.append((model_id, tok))

    print(f"\nReading: {INPUT_DATASET}")
    with open(INPUT_DATASET, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Samples: {len(data)}, MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")

    output_data = []
    truncated_count = 0
    skipped_count = 0

    for i, entry in enumerate(data):
        instruction = entry["instruction"]
        input_text = entry["input"]
        output_text = entry["output"]

        new_output, was_truncated, orig_tokens, final_tokens = truncate_sample(
            tokenizers_info, instruction, input_text, output_text, MAX_SEQ_LENGTH
        )

        if new_output is None:
            skipped_count += 1
            print(f"  SKIP [{i}]: {orig_tokens} tokens cannot fit")
            continue

        if was_truncated:
            truncated_count += 1
            print(f"  TRUNC [{i}]: {orig_tokens} -> {final_tokens}")

        output_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": new_output,
        })

    with open(OUTPUT_DATASET, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"{'='*60}")
    print(f"Input    : {INPUT_DATASET} ({len(data)} samples)")
    print(f"Output   : {OUTPUT_DATASET} ({len(output_data)} samples)")
    print(f"Truncated: {truncated_count}")
    print(f"Skipped  : {skipped_count}")


if __name__ == "__main__":
    main()
