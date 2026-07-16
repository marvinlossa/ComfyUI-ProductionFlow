import json
import os
import re

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
from comfy.cli_args import args


LORA_EXTENSIONS = (".safetensors", ".pt", ".ckpt", ".bin")
PROMPT_EXTENSIONS = (".txt", ".text", ".md")
PROMPT_FOLDER_TEXT_CHARS = 20
MAX_RESOLUTION = 16384

LATENT_RESOLUTION_PRESETS = {
    "custom": None,
    "1024 x 1024 (1:1 1K)": (1024, 1024),
    "1080 x 1920 (9:16 portrait)": (1080, 1920),
    "1920 x 1080 (16:9 landscape)": (1920, 1080),
    "2048 x 2048 (1:1 2K)": (2048, 2048),
    "1440 x 2560 (9:16 2K portrait)": (1440, 2560),
    "2560 x 1440 (16:9 2K landscape)": (2560, 1440),
    "1536 x 2048 (3:4 2K portrait)": (1536, 2048),
    "2048 x 1536 (4:3 2K landscape)": (2048, 1536),
    "1152 x 1536 (3:4 portrait)": (1152, 1536),
    "1536 x 1152 (4:3 landscape)": (1536, 1152),
    "832 x 1216 (SDXL portrait)": (832, 1216),
    "1216 x 832 (SDXL landscape)": (1216, 832),
}


def sanitize_path_part(value, fallback="untitled"):
    value = os.path.splitext(os.path.basename(str(value).strip()))[0]
    value = re.sub(r"[^A-Za-z0-9._ -]+", "_", value)
    value = re.sub(r"\s+", "_", value).strip("._- ")
    return value or fallback


def sanitize_text_part(value, fallback="untitled"):
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)[:PROMPT_FOLDER_TEXT_CHARS]
    value = re.sub(r"[^A-Za-z0-9._ -]+", "_", value)
    value = re.sub(r"\s+", "_", value).strip("._- ")
    return value or fallback


def prompt_text_snippet(prompt_text):
    return sanitize_text_part(prompt_text, "prompt")


def normalize_folder(value):
    return (value or ".").replace("\\", "/").strip("/") or "."


def lora_files():
    files = []
    for name in folder_paths.get_filename_list("loras"):
        normalized = name.replace("\\", "/")
        if normalized.lower().endswith(LORA_EXTENSIONS):
            files.append(normalized)
    return sorted(files, key=lambda x: x.lower())


def lora_folders():
    folders = set()
    for name in lora_files():
        folder = os.path.dirname(name).replace("\\", "/") or "."
        folders.add(folder)
    return sorted(folders, key=lambda x: (x == ".", x.lower())) or ["."]


def scan_loras(lora_folder, filter_text="", recursive=False):
    selected = normalize_folder(lora_folder)
    filter_text = (filter_text or "").strip().lower()
    out = []

    for name in lora_files():
        folder = os.path.dirname(name).replace("\\", "/") or "."
        in_folder = folder == selected
        if recursive and selected != ".":
            in_folder = in_folder or folder.startswith(selected + "/")
        elif recursive and selected == ".":
            in_folder = True

        if not in_folder:
            continue
        if filter_text and filter_text not in name.lower():
            continue
        out.append(name)

    return out


def folder_output_name(lora_folder):
    folder = normalize_folder(lora_folder)
    if not folder or folder == ".":
        return "loras_root"
    return sanitize_path_part(folder.split("/")[-1], "lora_test")


def prompt_root_dir():
    return folder_paths.get_input_directory()


def prompt_files():
    root = prompt_root_dir()
    files = []
    if not os.path.isdir(root):
        return files

    for current_root, _, names in os.walk(root):
        for name in names:
            if not name.lower().endswith(PROMPT_EXTENSIONS):
                continue
            path = os.path.join(current_root, name)
            relpath = os.path.relpath(path, root).replace("\\", "/")
            files.append(relpath)
    return sorted(files, key=lambda x: x.lower())


def prompt_folders():
    root = prompt_root_dir()
    folders = {"none", "."}
    if not os.path.isdir(root):
        return ["none", "."]

    for current_root, dirnames, _ in os.walk(root):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        for dirname in dirnames:
            path = os.path.join(current_root, dirname)
            relpath = os.path.relpath(path, root).replace("\\", "/")
            folders.add(relpath)
    return sorted(folders, key=lambda x: (x != "none", x == ".", x.lower()))


def scan_prompts(prompt_folder, filter_text="", recursive=False):
    selected = normalize_folder(prompt_folder)
    if selected == "none":
        return []

    filter_text = (filter_text or "").strip().lower()
    out = []
    for name in prompt_files():
        folder = os.path.dirname(name).replace("\\", "/") or "."
        in_folder = folder == selected
        if recursive and selected != ".":
            in_folder = in_folder or folder.startswith(selected + "/")
        elif recursive and selected == ".":
            in_folder = True

        if not in_folder:
            continue
        if filter_text and filter_text not in name.lower():
            continue
        out.append(name)

    return out


def prompt_output_name(prompt_name, prompt_text=None, prompt_index=None):
    if not prompt_name or prompt_name == "none":
        if prompt_text:
            return sanitize_text_part(prompt_text, "single_prompt")
        return "single_prompt"
    if prompt_text:
        suffix = f"_{prompt_index + 1:03d}" if prompt_index is not None else ""
        return sanitize_text_part(prompt_text, "prompt") + suffix
    return sanitize_path_part(prompt_name, "prompt")


def read_prompt_file(prompt_name):
    root = prompt_root_dir()
    normalized = prompt_name.replace("\\", "/").strip("/")
    path = os.path.abspath(os.path.join(root, normalized))
    root_abs = os.path.abspath(root)
    if not path.startswith(root_abs + os.sep) and path != root_abs:
        raise ValueError("ProductionFlow: prompt path escapes the prompt root.")
    with open(path, "r", encoding="utf-8") as file:
        return file.read().strip()


class ProductionFlowPromptFolderSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_folder": (prompt_folders(), {"tooltip": "Folder under ComfyUI/input containing prompt files. Choose none to use a standard connected prompt instead."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "Prompt index. The Queue All LoRAs/Prompts button sets this automatically per queued job."}),
                "recursive": ("BOOLEAN", {"default": False, "tooltip": "Include prompts in subfolders of the selected folder."}),
            },
            "optional": {
                "fallback_prompt": ("STRING", {"forceInput": True, "tooltip": "Prompt text used when prompt_folder is none."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("prompt", "prompt_text", "prompt_name", "prompt_index", "prompt_count")
    FUNCTION = "select_prompt"
    CATEGORY = "ProductionFlow/LoRA Testing"
    DESCRIPTION = "Selects prompts from a folder for outer-loop LoRA testing. Connect prompt to CLIP Text Encode, and connect prompt_text or prompt_name to the ProductionFlow save node when you want custom folder names."

    def select_prompt(self, prompt_folder, index, recursive=False, fallback_prompt=""):
        if normalize_folder(prompt_folder) == "none":
            prompt_text = prompt_text_snippet(fallback_prompt)
            return (fallback_prompt or "", prompt_text, "none", 0, 1)

        prompts = scan_prompts(prompt_folder, "", recursive)
        if not prompts:
            raise ValueError(f"ProductionFlow: no prompt files found in folder '{prompt_folder}'.")

        if index >= len(prompts):
            raise ValueError(f"ProductionFlow: prompt index {index} is out of range for {len(prompts)} prompts in '{prompt_folder}'.")

        prompt_name = prompts[index]
        prompt_text = read_prompt_file(prompt_name)
        return (prompt_text, prompt_output_name(prompt_name, prompt_text, index), sanitize_path_part(prompt_name, "prompt"), index, len(prompts))


class ProductionFlowLoraFolderLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_folder": (lora_folders(), {"tooltip": "Folder under ComfyUI/models/loras containing the LoRAs to test."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "LoRA index. The Queue All LoRAs button sets this automatically per queued job."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "recursive": ("BOOLEAN", {"default": False, "tooltip": "Include LoRAs in subfolders of the selected folder."}),
                "prompt_folder": (prompt_folders(), {"tooltip": "Folder under ComfyUI/input containing prompt files. Choose none to use a standard connected prompt instead."}),
                "prompt_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "Prompt index. The Queue All LoRAs/Prompts button sets this automatically per queued job."}),
                "prompt_recursive": ("BOOLEAN", {"default": False, "tooltip": "Include prompts in subfolders of the selected prompt folder."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING", "STRING", "STRING", "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("model", "clip", "lora_name", "lora_folder_name", "prompt", "prompt_text", "prompt_name", "lora_index", "lora_count", "prompt_index", "prompt_count")
    FUNCTION = "load_lora"
    CATEGORY = "ProductionFlow/LoRA Testing"
    DESCRIPTION = "Replaces the standard LoRA Loader for folder-based LoRA testing. Optionally loops prompt files as the outer loop and LoRAs as the inner loop."

    def load_lora(self, model, clip, lora_folder, index, strength_model, strength_clip, recursive=False, prompt_folder="none", prompt_index=0, prompt_recursive=False):
        loras = scan_loras(lora_folder, "", recursive)
        if not loras:
            raise ValueError(f"ProductionFlow: no LoRAs found in folder '{lora_folder}'.")

        if index >= len(loras):
            raise ValueError(f"ProductionFlow: LoRA index {index} is out of range for {len(loras)} LoRAs in '{lora_folder}'.")

        lora_name = loras[index]
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        prompt_text = ""
        prompt_name = "none"
        prompt_count = 1

        if normalize_folder(prompt_folder) != "none":
            prompts = scan_prompts(prompt_folder, "", prompt_recursive)
            if not prompts:
                raise ValueError(f"ProductionFlow: no prompt files found in folder '{prompt_folder}'.")
            if prompt_index >= len(prompts):
                raise ValueError(f"ProductionFlow: prompt index {prompt_index} is out of range for {len(prompts)} prompts in '{prompt_folder}'.")
            prompt_name = prompts[prompt_index]
            prompt_text = read_prompt_file(prompt_name)
            prompt_text_short = prompt_output_name(prompt_name, prompt_text, prompt_index)
            prompt_name = sanitize_path_part(prompt_name, "prompt")
            prompt_count = len(prompts)
        else:
            prompt_text_short = ""

        output_values = (lora_name, folder_output_name(lora_folder), prompt_text, prompt_text_short, prompt_name, index, len(loras), prompt_index, prompt_count)
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, *output_values)

        lora = None
        lora_metadata = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
                lora_metadata = self.loaded_lora[2]
            else:
                self.loaded_lora = None

        if lora is None:
            lora, lora_metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
            self.loaded_lora = (lora_path, lora, lora_metadata)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip, lora_metadata=lora_metadata)
        return (model_lora, clip_lora, *output_values)


class ProductionFlowLoraTestSaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Images to save."}),
                "lora_name": ("STRING", {"forceInput": True, "tooltip": "Connect lora_name from ProductionFlow LoRA Folder Loader."}),
                "lora_folder_name": ("STRING", {"forceInput": True, "tooltip": "Connect lora_folder_name from ProductionFlow LoRA Folder Loader."}),
            },
            "optional": {
                "filename_suffix": ("STRING", {"default": "", "tooltip": "Optional text appended after the LoRA name."}),
                "folder_name": ("STRING", {"forceInput": True, "tooltip": "Folder name source. Connect prompt_text for prompt-snippet folders or prompt_name for filename folders."}),
                "output_folder": ("STRING", {"default": "ProductionFlow", "tooltip": "Base output folder under ComfyUI/output."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "ProductionFlow/LoRA Testing"
    DESCRIPTION = "Saves LoRA test images into output/ProductionFlow/<prompt>/ using the LoRA filename as the image filename."

    def save_images(self, images, lora_name, lora_folder_name, filename_suffix="", folder_name="single_prompt", output_folder="ProductionFlow", prompt=None, extra_pnginfo=None):
        root = sanitize_path_part(output_folder, "ProductionFlow")
        folder = sanitize_path_part(folder_name, "single_prompt")
        stem = sanitize_path_part(lora_name, "lora")
        suffix = sanitize_path_part(filename_suffix, "") if filename_suffix else ""
        filename_prefix = f"{root}/{folder}/{stem}{'_' + suffix if suffix else ''}"

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            images[0].shape[1],
            images[0].shape[0],
        )

        results = []
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for key, value in extra_pnginfo.items():
                        metadata.add_text(key, json.dumps(value))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1

        return {"ui": {"images": results}, "result": (images,)}


class ProductionFlowNoisyLatentImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": (list(LATENT_RESOLUTION_PRESETS.keys()), {"tooltip": "Common pixel-size presets. Choose custom to use width and height."}),
                "width": ("INT", {"default": 1080, "min": 16, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Custom latent image width in pixels."}),
                "height": ("INT", {"default": 1920, "min": 16, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Custom latent image height in pixels."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1, "tooltip": "The number of latent images in the batch."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for deterministic gaussian noise."}),
                "std": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Standard deviation of the gaussian noise before tanh compression. Higher values push more samples toward the bounds."}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Final multiplier after tanh compression. This defines the output range: 1.0 = -1..1, 2.0 = -2..2."}),
                "latent_type": (("krea2/wan image (16ch)", "standard image (4ch)"), {"tooltip": "Krea2 uses Wan-style 16-channel image latents. Standard SD/SDXL models usually use 4-channel latents."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "ProductionFlow/Latent"
    DESCRIPTION = "Create an empty latent image filled with gaussian noise at a user-defined standard deviation, then tanh-compress and scale it to limit local minima and maxima."

    def generate(self, resolution, width, height, batch_size, seed, std, scale, latent_type="krea2/wan image (16ch)"):
        preset = LATENT_RESOLUTION_PRESETS.get(resolution)
        if preset is not None:
            width, height = preset

        latent_width = width // 8
        latent_height = height // 8
        channels = 16 if latent_type == "krea2/wan image (16ch)" else 4
        device = comfy.model_management.intermediate_device()
        dtype = comfy.model_management.intermediate_dtype()
        generator = torch.manual_seed(seed)

        noise = torch.randn(
            [batch_size, channels, latent_height, latent_width],
            dtype=torch.float32,
            generator=generator,
            device="cpu",
        ) * std

        noise = torch.tanh(noise) * scale

        if latent_type == "krea2/wan image (16ch)":
            noise = noise.unsqueeze(2)

        latent = noise.to(device=device, dtype=dtype)
        return ({"samples": latent, "downscale_ratio_spacial": 8},)


NODE_CLASS_MAPPINGS = {
    "ProductionFlowPromptFolderSelector": ProductionFlowPromptFolderSelector,
    "ProductionFlowLoraFolderLoader": ProductionFlowLoraFolderLoader,
    "ProductionFlowLoraTestSaveImage": ProductionFlowLoraTestSaveImage,
    "ProductionFlowNoisyLatentImage": ProductionFlowNoisyLatentImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProductionFlowPromptFolderSelector": "ProductionFlow Prompt Folder Selector",
    "ProductionFlowLoraFolderLoader": "ProductionFlow LoRA Folder Loader",
    "ProductionFlowLoraTestSaveImage": "ProductionFlow LoRA Test Save Image",
    "ProductionFlowNoisyLatentImage": "ProductionFlow Noisy Latent Image",
}
