import torch
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv
import os 

load_dotenv()

def build_backbone(cfg):
    """
    Loads a backbone from a local torch.hub checkpoint.
    Wraps with LoRA if cfg.model.backbone.lora is defined.
    """
    LOCAL_PATH = os.environ["DINOV3_PATH"]
    WEIGHTS = os.environ[cfg.model.backbone.name]
    
    backbone = torch.hub.load(
        LOCAL_PATH,
        cfg.model.backbone.name,
        source="local",
        weights=WEIGHTS,
    )

    if cfg.model.backbone.get("lora"):
        lora_cfg = LoraConfig(
            r=cfg.model.backbone.lora.r,
            lora_alpha=cfg.model.backbone.lora.alpha,
            lora_dropout=cfg.model.backbone.lora.dropout,
            target_modules=cfg.model.backbone.lora.target_modules,
            bias="none",
        )
        backbone = get_peft_model(backbone, lora_cfg)
        backbone.print_trainable_parameters()

    return backbone