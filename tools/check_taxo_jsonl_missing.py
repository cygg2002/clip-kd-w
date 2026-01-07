import json
import argparse

def parse_taxo_path(taxo_path: str):
    parts = [p.strip() for p in taxo_path.split(">")]
    out = {}
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        out[k.strip()] = v.strip()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, default="/home/111_wcy/work/clip-kd/Hierarchical-KD/CLIP-KD-main/datasets/out_json.jsonl")
    ap.add_argument("--show", type=int, default=5, help="show N examples for each missing type")
    args = ap.parse_args()

    required = ["class", "order", "family", "genus", "species"]
    miss_counts = {k: 0 for k in required}
    total = 0

    examples = {k: [] for k in required}
    examples_prompts = {k: [] for k in required}

    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            taxo = parse_taxo_path(obj.get("taxo_path", ""))

            for k in required:
                if k not in taxo:
                    miss_counts[k] += 1
                    if len(examples[k]) < args.show:
                        examples[k].append(obj.get("taxo_path", ""))
                        examples_prompts[k].append(obj.get("prompts", {}))

    print(f"File: {args.jsonl}")
    print(f"Total lines: {total}")

    def pct(x):
        return (100.0 * x / total) if total > 0 else 0.0

    print("\nMissing taxo fields:")
    for k in required:
        print(f"  missing {k:7s}: {miss_counts[k]:8d}  ({pct(miss_counts[k]):6.2f}%)")

    print("\nExamples (up to show=N):")
    for k in required:
        if miss_counts[k] == 0:
            continue
        print(f"\n--- Missing {k} examples ---")
        for i, tp in enumerate(examples[k]):
            print(f"[{i+1}] taxo_path: {tp}")
            pr = examples_prompts[k][i]
            # Ö»´òÓ¡ prompt keys£¬±ÜÃâÌ«³¤
            print(f"    prompts keys: {sorted(list(pr.keys()))}")

if __name__ == "__main__":
    main()
