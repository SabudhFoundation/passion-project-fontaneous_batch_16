import cv2
import numpy as np
import torch
from PIL import Image
from transformers import ViTModel, AutoImageProcessor



class attention_mapping:
    def __init__(self, img_rgb,MODEL_NAME, DISCARD_RATIO, HEAD_FUSION, TARGET_SIZE, PATCH_SIZE):
        self.img_rgb = img_rgb
        self.MODEL_NAME = MODEL_NAME
        self.DISCARD_RATIO=DISCARD_RATIO
        self.HEAD_FUSION=HEAD_FUSION
        self.TARGET_SIZE=TARGET_SIZE
        self.PATCH_SIZE=PATCH_SIZE

    def load_vit_model(self):    
        # ── 1. Load model with output_attentions=True ────────────────────────────────
        feature_extractor = AutoImageProcessor.from_pretrained(self.MODEL_NAME)
        model = ViTModel.from_pretrained(self.MODEL_NAME, output_attentions=True)
        model.eval()

        # ── 2. Preprocess image ───────────────────────────────────────────────────────
        pil_img   = Image.fromarray(self.img_rgb)                          # img_rgb = your HxWx3 uint8
        inputs    = feature_extractor(images=pil_img, return_tensors="pt")

        H_orig, W_orig = self.img_rgb.shape[:2]

        # ── 3. Forward pass – collect all layer attentions ───────────────────────────
        with torch.no_grad():
            outputs = model(**inputs)

        # outputs.attentions: tuple of (1, num_heads, num_tokens, num_tokens) per layer
        # num_tokens = 1 (CLS) + (224/16)^2 = 197
        attentions = outputs.attentions   # 12 layers for ViT-Base

        # ── 4. attention rollout ─────────────────────────────────────────
        rollout = self.attention_rollout(attentions, discard_ratio=self.DISCARD_RATIO,
                                    head_fusion=self.HEAD_FUSION)   # (197, 197)

        # ── 5. Extract CLS → patch attention ─────────────────────────────────────────
        cls_attention = rollout[0, 1:]                  # (196,)  — CLS row, skip CLS col
        ph = pw = self.TARGET_SIZE // self.PATCH_SIZE             # 14 × 14
        patch_scores = cls_attention.reshape(ph, pw).numpy()   # (14, 14)

        # Normalise to [0, 1]
        patch_scores = (patch_scores - patch_scores.min()) / \
                    (patch_scores.max() - patch_scores.min() + 1e-8)
        
        return patch_scores,attentions,pw,ph
     
    def attention_rollout(self,attentions, discard_ratio=0.9, head_fusion="mean"):
            
        #Implements Attention Rollout (Abnar & Zuidema 2020).
        #Returns a (197, 197) attention matrix after propagating through all layers.
        result = torch.eye(attentions[0].size(-1))          # identity

        for attn in attentions:
            attn = attn.squeeze(0)                          # (heads, tokens, tokens)

            if head_fusion == "mean":
                attn_fused = attn.mean(dim=0)
            elif head_fusion == "max":
                attn_fused = attn.max(dim=0).values
            elif isinstance(head_fusion, int):
                attn_fused = attn[head_fusion]
            else:
                raise ValueError(f"Unknown head_fusion: {head_fusion}")

            # Zero out the lowest-attention connections
            flat      = attn_fused.view(-1)
            threshold = flat.quantile(discard_ratio)
            attn_fused[attn_fused < threshold] = 0

            # Add residual & normalize
            attn_fused = attn_fused + torch.eye(attn_fused.size(-1))
            attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)

            result = torch.matmul(attn_fused, result)

        return result


        