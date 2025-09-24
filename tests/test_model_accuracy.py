import json

jsonl_file = "results/results_grpo_4.jsonl"

is_same_true_count = 0
is_same_false_count = 0
illegal_rows = []

with open(jsonl_file, "r") as f:
    for idx, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)

        model_card = row.get("model_card")
        legal_cards = row.get("legal_cards", [])
        is_same = row.get("is_same", False)

        if is_same:
            is_same_true_count += 1
        else:
            is_same_false_count += 1

        if model_card not in legal_cards:
            illegal_rows.append((idx, model_card, legal_cards))

print(f"Total rows is_same=True: {is_same_true_count}")
print(f"Total rows is_same=False: {is_same_false_count}")


if illegal_rows:
    print("\nRows where model_card is NOT in legal_cards:")
    for idx, model_card, legal_cards in illegal_rows:
        print(f"Row {idx}: model_card='{model_card}' not in legal_cards")
else:
    print("\nAll model_card values were inside legal_cards.")


print("Total illegal rows:", len(illegal_rows))
