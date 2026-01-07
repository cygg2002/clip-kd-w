import os, json
import torch

def parse_taxo_path(taxo_path: str):
    parts = [p.strip() for p in taxo_path.split(">")]
    out = {}
    for p in parts:
        k, v = p.split(":", 1)
        out[k.strip()] = v.strip()
    return out  # class/order/family/genus/species

def load_all_json(json_dir):
    items = []
    for fn in os.listdir(json_dir):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(json_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        items.append(obj)
    return items

def build_assets(items):
    # ÊÕ¼¯ name->prompt
    prompts = {lvl: {} for lvl in ["species","genus","family","order","class"]}
    sp2gen, sp2fam, sp2ord, sp2cls = {}, {}, {}, {}

    for obj in items:
        taxo = parse_taxo_path(obj["taxo_path"])
        pr = obj["prompts"]  # ÄãÒÑ¾­ÓÐÁË

        sp = taxo["species"]; ge = taxo["genus"]; fa = taxo["family"]; orr = taxo["order"]; cl = taxo["class"]
        sp2gen[sp] = ge; sp2fam[sp] = fa; sp2ord[sp] = orr; sp2cls[sp] = cl

        for lvl in prompts.keys():
            name = taxo[lvl]
            if name not in prompts[lvl]:
                prompts[lvl][name] = pr[lvl]

    # ¹Ì¶¨´Ê±íË³Ðò
    level_list = {lvl: sorted(prompts[lvl].keys()) for lvl in prompts.keys()}
    level_prompts = {lvl: [prompts[lvl][name] for name in level_list[lvl]] for lvl in prompts.keys()}

    # ½¨ index
    sp_list = level_list["species"]
    gen_to_id = {g:i for i,g in enumerate(level_list["genus"])}
    fam_to_id = {g:i for i,g in enumerate(level_list["family"])}
    ord_to_id = {g:i for i,g in enumerate(level_list["order"])}
    cls_to_id = {g:i for i,g in enumerate(level_list["class"])}

    sp_to_gen = torch.tensor([gen_to_id[sp2gen[sp]] for sp in sp_list], dtype=torch.long)
    sp_to_fam = torch.tensor([fam_to_id[sp2fam[sp]] for sp in sp_list], dtype=torch.long)
    sp_to_ord = torch.tensor([ord_to_id[sp2ord[sp]] for sp in sp_list], dtype=torch.long)
    sp_to_cls = torch.tensor([cls_to_id[sp2cls[sp]] for sp in sp_list], dtype=torch.long)

    def group_size(sp_to_group, n_group):
        cnt = torch.zeros(n_group, dtype=torch.float32)
        ones = torch.ones_like(sp_to_group, dtype=torch.float32)
        cnt.scatter_add_(0, sp_to_group, ones)
        return cnt.clamp_min(1.0)

    gen_size = group_size(sp_to_gen, len(level_list["genus"]))
    fam_size = group_size(sp_to_fam, len(level_list["family"]))
    ord_size = group_size(sp_to_ord, len(level_list["order"]))
    cls_size = group_size(sp_to_cls, len(level_list["class"]))

    return level_list, level_prompts, sp_to_gen, sp_to_fam, sp_to_ord, sp_to_cls, gen_size, fam_size, ord_size, cls_size

def save_assets(out_dir, level_list, level_prompts, sp_to_gen, sp_to_fam, sp_to_ord, sp_to_cls, gen_size, fam_size, ord_size, cls_size):
    os.makedirs(out_dir, exist_ok=True)
    # ´Ê±íÓë prompt
    with open(os.path.join(out_dir, "level_list.json"), "w", encoding="utf-8") as f:
        json.dump(level_list, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "level_prompts.json"), "w", encoding="utf-8") as f:
        json.dump(level_prompts, f, ensure_ascii=False)

    # Ó³ÉäÓë size
    torch.save({
        "sp_to_gen": sp_to_gen,
        "sp_to_fam": sp_to_fam,
        "sp_to_ord": sp_to_ord,
        "sp_to_cls": sp_to_cls,
        "gen_size": gen_size,
        "fam_size": fam_size,
        "ord_size": ord_size,
        "cls_size": cls_size,
    }, os.path.join(out_dir, "taxo_maps.pt"))

if __name__ == "__main__":
    json_dir = "/path/to/json_dir"
    out_dir = "/path/to/taxo_assets"

    items = load_all_json(json_dir)
    assets = build_assets(items)
    save_assets(out_dir, *assets)
    print("Done. Saved to", out_dir)
