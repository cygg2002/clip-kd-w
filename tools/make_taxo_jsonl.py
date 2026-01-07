import os
import json
import argparse
from multiprocessing import Pool


def parse_taxo_path(taxo_path: str):
    """½âÎö·ÖÀàÂ·¾¶×Ö·û´®"""
    if not taxo_path:
        return {}
    parts = [p.strip() for p in taxo_path.split(">")]
    out = {}
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def build_unified_prompts(taxo: dict):
    """
    Í³Ò» prompts ·ç¸ñ£º
    - species ±ØÐëÓë train.csv / val.csv ÍêÈ«Ò»ÖÂ£º'a photo of a {species}'
    - ÆäËû²ã¼¶Í¬ÑùÓÃÐ¡Ð´¡¢ÎÞ¾äºÅ£¬¾¡Á¿ÉÙ±äÌå
    """
    cls_ = taxo["class"]
    ord_ = taxo["order"]
    fam_ = taxo["family"]
    gen_ = taxo["genus"]
    sp_  = taxo["species"]

    prompts = {
        "species": f"a photo of a {sp_}",
        "genus":   f"a photo of a {gen_}",
        "family":  f"a photo of a {fam_}",
        "order":   f"a photo of a {ord_}",
        "class":   f"a photo of a {cls_}",
    }
    return prompts


def process_file_task(task_args):
    """
    µ¥¸öÎÄ¼þ´¦Àíº¯Êý£¬ÓÉ×Ó½ø³ÌÖ´ÐÐ
    ·µ»ØÖµ: (json_line_string, status_code)
    """
    file_path, target_split, required_taxo, required_prompts, prompt_style = task_args

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        # 1) ¹ýÂË Split
        file_split = obj.get("split", None)
        if target_split != "all" and file_split != target_split:
            return None, "skip_split"

        # 2) Ð£Ñé²¢½âÎö Taxonomy
        taxo_path = obj.get("taxo_path", "")
        taxo = parse_taxo_path(taxo_path)
        if not all(k in taxo for k in required_taxo):
            return None, "skip_taxo"

        # 3) Prompts ´¦Àí£¨¹Ø¼ü¸Ä¶¯µã£©
        if prompt_style == "unified":
            # Í³Ò»Éú³É prompts£¨ÍÆ¼ö£ºÓë csv Ä£°åÒ»ÖÂ£©
            prompts = build_unified_prompts(taxo)
        else:
            # ±£ÁôÔ­Ê¼ prompts£¨²»ÍÆ¼ö£¬»áÓë csv Ä£°å²»Ò»ÖÂ£©
            prompts = obj.get("prompts", {})

        # 4) Ð£Ñé prompts ÊÇ·ñÍêÕû£¨unified Ò»¶¨ÍêÕû£»original ¿ÉÄÜ²»ÍêÕû£©
        if not all(k in prompts for k in required_prompts):
            return None, "skip_prompts"

        out_data = {
            "taxo_path": taxo_path,
            "prompts": prompts,
        }
        return json.dumps(out_data, ensure_ascii=False), "success"

    except Exception:
        return None, "error"


def main():
    ap = argparse.ArgumentParser(description="Fast JSON to JSONL Converter (taxonomy prompts unified)")
    ap.add_argument("--json_dir", type=str,
                    default="/home/111_wcy/work/clip-kd/CLIP-KD-main/data/mammal_images/train/json",
                    help="ÊäÈëJSONÄ¿Â¼")
    ap.add_argument("--out_jsonl", type=str,
                    default="/home/111_wcy/work/clip-kd/Hierarchical-KD/CLIP-KD-wcy/datasets/out_train_json.jsonl",
                    help="Êä³öJSONLÂ·¾¶")
    ap.add_argument("--split", type=str, default="train", help="¹ýÂË±êÇ© (train/val/test/all)")
    ap.add_argument("--workers", type=int, default=8, help="½ø³ÌÊý")
    ap.add_argument("--prompt_style", type=str, default="unified",
                    choices=["unified", "original"],
                    help="prompts ·ç¸ñ£ºunified=ÓëcsvÒ»ÖÂ£»original=Ê¹ÓÃÔ­Ê¼jsonÀïµÄprompts")
    args = ap.parse_args()

    print(f"[*] Scanning directory: {args.json_dir}")
    all_files = [
        os.path.join(args.json_dir, f)
        for f in os.listdir(args.json_dir)
        if f.endswith(".json")
    ]
    all_files.sort()
    total_files = len(all_files)
    print(f"[*] Found {total_files} files.")

    required_taxo = ["class", "order", "family", "genus", "species"]
    required_prompts = ["species", "genus", "family", "order", "class"]

    tasks = [
        (f, args.split, required_taxo, required_prompts, args.prompt_style)
        for f in all_files
    ]

    counts = {"success": 0, "skip_split": 0, "skip_taxo": 0, "skip_prompts": 0, "error": 0}

    print(f"[*] Processing with {args.workers} workers... (prompt_style={args.prompt_style})")

    with open(args.out_jsonl, "w", encoding="utf-8") as w:
        with Pool(processes=args.workers) as pool:
            for result_str, status in pool.imap(process_file_task, tasks, chunksize=200):
                counts[status] += 1
                if result_str:
                    w.write(result_str + "\n")

                processed = sum(counts.values())
                if processed % max(1, (total_files // 10)) == 0:
                    print(f"    Progress: {processed}/{total_files} ({(processed / total_files) * 100:.1f}%)")

    print("\n" + "=" * 30)
    print(f"DONE! Saved to: {args.out_jsonl}")
    print(f"Total processed:  {total_files}")
    print(f"Successfully wrote: {counts['success']}")
    print(f"Skipped (Split):    {counts['skip_split']}")
    print(f"Skipped (Taxo):     {counts['skip_taxo']}")
    print(f"Skipped (Prompts):  {counts['skip_prompts']}")
    if counts['error'] > 0:
        print(f"Errors encountered: {counts['error']}")
    print("=" * 30)


if __name__ == "__main__":
    main()
