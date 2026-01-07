import os
import torch
import numpy as np
from PIL import Image

# ===== µ¼Èë CLIP-KD µÄ open_clip Ä£¿é =====
from src.open_clip import create_model_and_transforms, tokenize

# ===== Éè±¸ÉèÖÃ =====
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
print(f"? Using device: {device}")

# ===== Ä£ÐÍÓë checkpoint Â·¾¶ =====
model_name = "ViT-B-32"  # ÓëÑµÁ·Ê±±£³ÖÒ»ÖÂ
pretrained = None
checkpoint_path = "/home/111_wcy/work/clip-kd/distillation/CLIP-KD-main/logs/bioclip2_to_clipb32_fd_icl/checkpoints/epoch_12.pt"

# ===== Àà±ð¶¨Òå =====
classes = [
    "a photo of a white tailed deer", "a photo of a moose", "a photo of a reindeer", #"a photo of a bison",
    "a photo of a black bear", "a photo of a grey wolf", "a photo of a goat", "a photo of a wild boar",
    "a photo of a person"#, "a photo of a Papio", "a photo of a Papio ursinus", "a photo of a Marmota caligata"
]

# ===== Òª²âÊÔµÄµ¥ÕÅÍ¼Æ¬Â·¾¶ =====
#image_path = "/home/111_wcy/work/clip-kd/CLIP-KD-main/data/mammal_images/test/images/ff063728c83368aa0397ddb1bae8e9f4.jpg"  # ¡û ¸Ä³ÉÄãµÄÍ¼Æ¬Â·¾¶
image_path = "/home/111_wcy/work/clip/CLIP/datasets1/animals/images/goat_147.jpg"
# ===== ´´½¨Ä£ÐÍÓëÔ¤´¦Àí =====
print("?? Loading CLIP backbone...")
model, preprocess_train, preprocess_val = create_model_and_transforms(
    model_name, pretrained=pretrained, device=device
)
model = model.to(device)

# ===== ¼ÓÔØÕôÁóºóµÄ checkpoint =====
try:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"? Loaded checkpoint: {checkpoint_path}")
    if missing:
        print(f"?? Missing keys: {len(missing)} (showing first 5): {missing[:5]}")
    if unexpected:
        print(f"?? Unexpected keys: {len(unexpected)} (showing first 5): {unexpected[:5]}")
except Exception as e:
    raise RuntimeError(f"? Failed to load checkpoint: {e}")

model.eval()

# ===== Ô¤¼ÆËãÎÄ±¾ÌØÕ÷ =====
with torch.no_grad():
    text_inputs = tokenize(classes).to(device)
    text_features = model.encode_text(text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# ===== ¶ÁÈ¡²¢ÍÆÀíµ¥ÕÅÍ¼Æ¬ =====
if not os.path.isfile(image_path):
    raise FileNotFoundError(f"?? Image not found: {image_path}")

try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    raise RuntimeError(f"?? Failed to open image: {e}")

# Õï¶ÏA£ºÀà¼äÎÄ±¾ÓàÏÒÏàËÆ¶È
with torch.no_grad():
    toks = tokenize(classes).to(device)
    txt = model.encode_text(toks)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    S = (txt @ txt.T).cpu().numpy()  # [C, C]

import numpy as np
mask = ~np.eye(len(classes), dtype=bool)
off = S[mask]
print(f"[Text cosine] off-diagonal min/mean/max = {off.min():.4f}/{off.mean():.4f}/{off.max():.4f}")
# Õý³£Ó¦µ± max < 0.9£¬Æ½¾ùÒ»°ãÔÚ 0.2~0.5£»Èç¹û max ½Ó½ü 1 »òÕûÌåºÜ¸ß£¬ÎÄ±¾ÌØÕ÷ÔÚ¡°¼·¡±¡ª¡ªÒªÃ´tokenize¶ÔËùÓÐÀàÉúÐ§Ïà½ü£¬ÒªÃ´ÎÄ±¾Ëþ»µÁË

# Õï¶ÏB£ºÁ½ÕÅ²»Í¬Í¼Æ¬µÄÍ¼ÏñÌØÕ÷ÓàÏÒ
img2_path = "/home/111_wcy/work/CLIP-KD/datasets/animals_class/bison/images/bison_2_roi1.jpg"  # »»Ò»ÕÅÍêÈ«²»Í¬µÄÍ¼
from PIL import Image
with torch.no_grad():
    im1 = preprocess_val(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    im2 = preprocess_val(Image.open(img2_path).convert("RGB")).unsqueeze(0).to(device)
    f1 = model.encode_image(im1); f1 = f1 / f1.norm(dim=-1, keepdim=True)
    f2 = model.encode_image(im2); f2 = f2 / f2.norm(dim=-1, keepdim=True)
    cos = (f1 @ f2.T).item()
print(f"[Image cosine between two very different images] = {cos:.4f}")


with torch.no_grad():
    image_tensor = preprocess_val(image).unsqueeze(0).to(device)
    image_features = model.encode_image(image_tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # CLIP-style logits: scale * cosine_similarity
    with torch.no_grad():
        scale = getattr(model, "logit_scale", None)
        if scale is None:
            # ¶µµ×£º¾­µä CLIP ÎÂ¶È 1/0.07
            scale = torch.tensor(np.log(1 / 0.07), device=image_features.device)
        # exp() µÃµ½ÎÂ¶ÈµÄµ¹Êý£¨·Å´óÏµÊý£©
        scale = scale.detach().exp()

        logits = (scale * (image_features @ text_features.T)).squeeze(0)  # [num_classes]
        probs = logits.softmax(dim=-1).cpu().numpy()

# ===== ´òÓ¡½á¹û£¨Top-1 ºÍÍêÕû¸ÅÂÊ±í£©=====
max_idx = int(np.argmax(probs))
max_prob = float(probs[max_idx])
pred_label = classes[max_idx]

print("\n========== Single Image Prediction ==========")
print(f"??? Image: {image_path}")
print(f"??? Predicted: {pred_label} (prob={max_prob:.4f})")

# °´¸ÅÂÊ´Ó¸ßµ½µÍ´òÓ¡ËùÓÐÀà±ð
sorted_indices = np.argsort(-probs)
print("\n?? Class probabilities (desc):")
for i in sorted_indices:
    print(f"{classes[i]:<22s} : {probs[i]:.4f}")

# ÕÒ³ö¼«Ïà½üµÄÎÄ±¾¶Ô
pairs = []
for i in range(len(classes)):
    for j in range(i+1, len(classes)):
        pairs.append(((i, j), S[i, j]))
pairs.sort(key=lambda x: -x[1])
print("Top-10 most similar text pairs:")
for (i, j), v in pairs[:10]:
    print(f"{i}:{classes[i]}  <->  {j}:{classes[j]}   cos={v:.4f}")
