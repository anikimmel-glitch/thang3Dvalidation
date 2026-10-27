"""Pinned pretrained inference. No source titles, captions, or GT programs."""
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

LOCK = json.loads((Path(__file__).resolve().parents[1]/"models.lock.json").read_text())


def model_args(key):
    entry = LOCK[key]
    return dict(pretrained_model_name_or_path=entry["model_id"],
                revision=entry["revision"], torch_dtype=torch.bfloat16,
                attn_implementation="sdpa", device_map="cuda:0")


def tokenizer_args(key):
    entry = LOCK[key]
    return dict(pretrained_model_name_or_path=entry["model_id"], revision=entry["revision"])


class Adapter:
    def __init__(self, name):
        self.name = name
        if name == "cad_recode_pc":
            from vendor.cad_recode.model import CADRecode
            self.model = CADRecode.from_pretrained(**model_args("cad_recode")).eval()
            self.processor = AutoTokenizer.from_pretrained(
                **tokenizer_args("Qwen_Qwen2-1.5B"),
                pad_token="<|im_end|>", padding_side="left")
        elif name in ("cadrille_pc", "cadrille_rgb"):
            from vendor.cadrille.cadrille import Cadrille
            self.model = Cadrille.from_pretrained(**model_args("cadrille")).eval()
            self.processor = AutoProcessor.from_pretrained(
                **tokenizer_args("Qwen_Qwen2-VL-2B-Instruct"),
                min_pixels=256*28*28, max_pixels=1280*28*28, padding_side="left")
        elif name == "cadreasoner_native":
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                **model_args("cadreasoner")).eval()
            self.processor = AutoProcessor.from_pretrained(
                **tokenizer_args("cadreasoner"), resized_width=14*17*2,
                resized_height=14*17*4, padding_side="left", use_fast=True)
        else:
            raise ValueError(name)

    @torch.inference_mode()
    def generate(self, prepared):
        proc, device = self.processor, self.model.device
        if self.name == "cad_recode_pc":
            points = np.load(prepared/"reference.npz")["pc256"].astype(np.float32)
            ids = [proc.pad_token_id]*len(points) + [proc("<|im_start|>")["input_ids"][0]]
            inputs = dict(
                input_ids=torch.tensor([ids], device=device),
                attention_mask=torch.tensor([[-1]*len(points)+[1]], device=device),
                point_cloud=torch.tensor(points[None], device=device))
            output = self.model.generate(**inputs, max_new_tokens=768, do_sample=False,
                                         pad_token_id=proc.pad_token_id)
            text = proc.batch_decode(output[:,len(ids):], skip_special_tokens=True)[0]
        elif self.name.startswith("cadrille"):
            from vendor.cadrille.cadrille import collate
            item = dict(file_name=prepared.name, description="Generate cadquery code")
            if self.name.endswith("_pc"):
                item["point_cloud"] = np.load(prepared/"reference.npz")["pc256"]
            else:
                item["video"] = [Image.open(prepared/"cadrille.png").convert("RGB")]
            inputs = collate([item], proc, n_points=256, eval=True)
            inputs.pop("file_name")
            inputs = {k:v.to(device) for k,v in inputs.items()}
            output = self.model.generate(**inputs, max_new_tokens=768, do_sample=False)
            text = proc.batch_decode(output[:,inputs["input_ids"].shape[1]:],
                skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        else:
            from qwen_vl_utils import process_vision_info
            messages = [[{"role":"user","content":[
                {"type":"image","image":Image.open(prepared/"cadreasoner.png").convert("RGB")},
                {"type":"text","text":"import cadquery as cq\n"}]}]]
            texts = [proc.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
                     for x in messages]
            images, videos = process_vision_info(messages)
            inputs = proc(text=texts, images=images, videos=videos, padding=True,
                          return_tensors="pt").to(device)
            output = self.model.generate(**inputs, max_new_tokens=1500, do_sample=False)
            text = proc.batch_decode(output[:,inputs["input_ids"].shape[1]:],
                skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return text
