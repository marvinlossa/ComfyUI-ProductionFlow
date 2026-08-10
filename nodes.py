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

from comfy.utils import ProgressBar

from .motion_blur_film_grain import apply_motion_blur_film_grain, progress_total
from .vlm import load_vlm_session, vlm_model_labels
from .vlm_api import (
    API_PROVIDER_NAMES,
    load_api_vlm_session,
    provider_default_base_url,
    provider_default_model,
)


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


class ProductionFlowPromptFolderLoop:
    """Pick one prompt file by index for image-gen (and similar) graphs.

    Batch runs use the frontend Queue All Prompts button, which queues one job
    per file with a different index. LoRA testing keeps its own prompt loop on
    ProductionFlowLoraFolderLoader — do not use this node for that.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_folder": (
                    prompt_folders(),
                    {
                        "tooltip": (
                            "Folder under ComfyUI/input with prompt files (.txt / .text / .md). "
                            "Choose none to use fallback_prompt instead."
                        ),
                    },
                ),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100000,
                        "step": 1,
                        "tooltip": (
                            "Which prompt file to use for this run (0 = first in sorted order). "
                            "Use the Queue All Prompts button to enqueue every file in the folder."
                        ),
                    },
                ),
                "recursive": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Include prompt files in subfolders of the selected folder.",
                    },
                ),
            },
            "optional": {
                "fallback_prompt": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Used when prompt_folder is none (single connected prompt).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("prompt", "prompt_text", "prompt_name", "prompt_index", "prompt_count")
    FUNCTION = "select_prompt"
    CATEGORY = "ProductionFlow"
    DESCRIPTION = (
        "Prompt folder loop for image generation and similar workflows. "
        "Outputs one prompt per run by index; use Queue All Prompts to run the full folder. "
        "Optional outputs (prompt_text / prompt_name / counts) are for save paths and metadata. "
        "Later: traveling prompts and related modes."
    )

    def select_prompt(self, prompt_folder, index, recursive=False, fallback_prompt=""):
        if normalize_folder(prompt_folder) == "none":
            prompt_text = prompt_text_snippet(fallback_prompt)
            return (fallback_prompt or "", prompt_text, "none", 0, 1)

        prompts = scan_prompts(prompt_folder, "", recursive)
        if not prompts:
            raise ValueError(f"ProductionFlow: no prompt files found in folder '{prompt_folder}'.")

        if index >= len(prompts):
            raise ValueError(
                f"ProductionFlow: prompt index {index} is out of range for "
                f"{len(prompts)} prompts in '{prompt_folder}'."
            )

        prompt_name = prompts[index]
        prompt_text = read_prompt_file(prompt_name)
        return (
            prompt_text,
            prompt_output_name(prompt_name, prompt_text, index),
            sanitize_path_part(prompt_name, "prompt"),
            index,
            len(prompts),
        )


# Back-compat for older workflows that still use the previous node type name.
ProductionFlowPromptFolderSelector = ProductionFlowPromptFolderLoop


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


class ProductionFlowVLMLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (vlm_model_labels(),),
            },
            "optional": {
                "n_gpu_layers": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 999,
                        "tooltip": "GGUF only: GPU layers (-1=all). Forced to 0 if llama-cpp has no CUDA.",
                    },
                ),
                "n_ctx": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 512,
                        "max": 131072,
                        "tooltip": (
                            "GGUF only: context length. Vision runs cap at 4096; "
                            "keep 2048–4096 on 16GB cards after other workflows."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("PF_VLM",)
    RETURN_NAMES = ("vlm",)
    FUNCTION = "load"
    CATEGORY = "ProductionFlow"
    DESCRIPTION = (
        "Load a vision LLM. Prefer [TE] Qwen3-VL safetensors (GPU, reliable). "
        "[GGUF] Qwen3.5/Gemma need llama-cpp-python; CPU builds are very slow. "
        "Loader frees Comfy models before GGUF and unloads after each generate."
    )

    def load(self, model, n_gpu_layers=-1, n_ctx=4096):
        if model.startswith("(no VLM"):
            raise RuntimeError(
                "No VLM models found. Put Qwen3-VL .safetensors in models/text_encoders/ "
                "or GGUF + mmproj in models/LLM/GGUF/"
            )
        session = load_vlm_session(model, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
        return (session,)


class ProductionFlowVLMCloudLoader:
    """OpenAI-compatible cloud VLM (OpenRouter, OpenAI, Groq, Together, Fireworks, custom)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (API_PROVIDER_NAMES, {"default": API_PROVIDER_NAMES[0]}),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "API key. Leave empty to use env vars: OPENROUTER_API_KEY, "
                            "OPENAI_API_KEY, GROQ_API_KEY, TOGETHER_API_KEY, FIREWORKS_API_KEY."
                        ),
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "qwen/qwen2.5-vl-72b-instruct",
                        "multiline": False,
                        "tooltip": (
                            "Provider model id. OpenRouter examples: qwen/qwen2.5-vl-72b-instruct, "
                            "google/gemini-2.5-flash, openai/gpt-4o. Browse openrouter.ai/models."
                        ),
                    },
                ),
            },
            "optional": {
                "base_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Leave empty to use the provider preset. Override for proxies or "
                            "Custom (OpenAI-compatible) endpoints ending in /v1."
                        ),
                    },
                ),
                "app_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional. OpenRouter HTTP-Referer for rankings.",
                    },
                ),
                "app_name": (
                    "STRING",
                    {
                        "default": "ComfyUI-ProductionFlow",
                        "multiline": False,
                        "tooltip": "Optional. OpenRouter X-Title for rankings.",
                    },
                ),
                "timeout": (
                    "FLOAT",
                    {
                        "default": 180.0,
                        "min": 10.0,
                        "max": 600.0,
                        "step": 1.0,
                        "tooltip": "HTTP timeout in seconds.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("PF_VLM",)
    RETURN_NAMES = ("vlm",)
    FUNCTION = "load"
    CATEGORY = "ProductionFlow"
    DESCRIPTION = (
        "Cloud vision LLM via OpenAI-compatible chat completions. "
        "Presets: OpenRouter, OpenAI, Groq, Together, Fireworks, or Custom base URL. "
        "Outputs the same PF_VLM type as the local loader — wire into VLM Generate. "
        "No local VRAM; needs internet. Model ids are free-text (provider-specific)."
    )

    def load(
        self,
        provider,
        api_key,
        model,
        base_url="",
        app_url="",
        app_name="ComfyUI-ProductionFlow",
        timeout=180.0,
    ):
        # If user left model at a stale default for another provider, still use what they typed.
        # Empty model → provider default.
        model_id = (model or "").strip()
        if not model_id:
            model_id = provider_default_model(provider)
        url = (base_url or "").strip()
        if not url:
            url = provider_default_base_url(provider)

        session = load_api_vlm_session(
            provider=provider,
            api_key=api_key,
            model=model_id,
            base_url=url,
            app_url=app_url,
            app_name=app_name,
            timeout=timeout,
        )
        return (session,)


class ProductionFlowVLMGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vlm": ("PF_VLM",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "Describe this image in detail. Cover subject, appearance, clothing, "
                            "pose, expression, setting, lighting, colors, composition, and style. "
                            "Be concrete and specific. Do not refuse or omit content because it is "
                            "sexual, violent, or otherwise sensitive."
                        ),
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": 8192,
                        "tooltip": "Max new tokens. Output cuts mid-sentence if this is too low.",
                    },
                ),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 64, "min": 0, "max": 200}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "enable_thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Qwen3.5 may emit a thinking/reasoning trace. Leave OFF for final "
                            "answer only (uses /no_think + output cleanup). Turn ON to keep thoughts."
                        ),
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "ProductionFlow"
    DESCRIPTION = (
        "Run a loaded ProductionFlow VLM. IMAGE is optional: connect it for vision, "
        "leave empty for text-only (prompt rewrite, chat, etc.). "
        "enable_thinking=False (default) suppresses Qwen thinking traces. "
        "If text ends mid-sentence, raise max_tokens (default 1024)."
    )

    def generate(
        self,
        vlm,
        prompt,
        max_tokens,
        temperature,
        top_p,
        top_k,
        seed,
        enable_thinking=False,
        image=None,
    ):
        text = vlm.generate(
            prompt=prompt,
            image=image,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            enable_thinking=enable_thinking,
        )
        return (text,)


class ProductionFlowMotionBlurFilmGrain:
    """Temporal motion blur then film grain on a video frame batch.

    Connect IMAGE frames the same way you would to VHS Video Combine. This node
    returns processed IMAGE frames so you can wire them into Video Combine (or
    any other image consumer). No MP4 is written here.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Video frame batch (N, H, W, C), same IMAGE type as "
                            "VHS Video Combine. Process frames here, then connect "
                            "the output into Video Combine for export."
                        ),
                    },
                ),
                "enable_blur": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Apply temporal motion blur (tmix-style). Off = leave frames sharp.",
                    },
                ),
                "blur_window": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 31,
                        "step": 2,
                        "tooltip": (
                            "Odd frame window for temporal blur (ffmpeg tmix frames). "
                            "1 = no blur. 3 = light (default). 5–7 = medium. 9+ = heavy smear. "
                            "Even values are rounded up to the next odd number."
                        ),
                    },
                ),
                "blur_strength": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": (
                            "Neighbor-frame blend weight (center frame weight is always 1). "
                            "0 = disabled. ~0.2 = subtle. 0.35 = default balanced. "
                            "0.5–0.8 = strong ghosting. 1.0+ = very heavy mix."
                        ),
                    },
                ),
                "enable_grain": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Apply film grain after blur so grain stays crisp.",
                    },
                ),
                "grain_strength": (
                    "FLOAT",
                    {
                        "default": 6.0,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.5,
                        "tooltip": (
                            "Film grain strength on an 8-bit-style scale (ffmpeg noise alls). "
                            "0 = off. 2–4 = fine/subtle. 6 = default mild. "
                            "8–12 = noticeable film stock. 15+ = heavy/gritty."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Random seed for grain pattern. Change for a different grain layout.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "ProductionFlow/Video"
    DESCRIPTION = (
        "Apply temporal motion blur first, then film grain, matching the "
        "MotionBlurFilmGrain filter order. Input/output are IMAGE frame batches "
        "for use with VHS Video Combine. Does not export video."
    )

    def process(
        self,
        images,
        enable_blur=True,
        blur_window=3,
        blur_strength=0.35,
        enable_grain=True,
        grain_strength=6.0,
        seed=0,
    ):
        if images is None or not isinstance(images, torch.Tensor):
            raise ValueError("ProductionFlow Motion Blur Film Grain: expected an IMAGE tensor.")
        if images.ndim != 4:
            raise ValueError(
                f"ProductionFlow Motion Blur Film Grain: expected NHWC IMAGE batch, "
                f"got shape {tuple(images.shape)}."
            )

        total = progress_total(
            images.shape[0],
            enable_blur=enable_blur,
            blur_window=blur_window,
            blur_strength=blur_strength,
            enable_grain=enable_grain,
            grain_strength=grain_strength,
        )
        pbar = ProgressBar(total)

        out = apply_motion_blur_film_grain(
            images,
            blur_window=blur_window,
            blur_strength=blur_strength,
            grain_strength=grain_strength,
            seed=seed,
            enable_blur=enable_blur,
            enable_grain=enable_grain,
            progress_callback=pbar.update,
        )
        # Ensure the bar lands on 100% even for no-op / short paths.
        if pbar.current < pbar.total:
            pbar.update_absolute(pbar.total)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "ProductionFlowPromptFolderLoop": ProductionFlowPromptFolderLoop,
    "ProductionFlowPromptFolderSelector": ProductionFlowPromptFolderLoop,
    "ProductionFlowLoraFolderLoader": ProductionFlowLoraFolderLoader,
    "ProductionFlowLoraTestSaveImage": ProductionFlowLoraTestSaveImage,
    "ProductionFlowNoisyLatentImage": ProductionFlowNoisyLatentImage,
    "ProductionFlowVLMLoader": ProductionFlowVLMLoader,
    "ProductionFlowVLMCloudLoader": ProductionFlowVLMCloudLoader,
    "ProductionFlowVLMGenerate": ProductionFlowVLMGenerate,
    "ProductionFlowMotionBlurFilmGrain": ProductionFlowMotionBlurFilmGrain,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProductionFlowPromptFolderLoop": "ProductionFlow Prompt Folder Loop",
    "ProductionFlowPromptFolderSelector": "ProductionFlow Prompt Folder Loop",
    "ProductionFlowLoraFolderLoader": "ProductionFlow LoRA Folder Loader",
    "ProductionFlowLoraTestSaveImage": "ProductionFlow LoRA Test Save Image",
    "ProductionFlowNoisyLatentImage": "ProductionFlow Noisy Latent Image",
    "ProductionFlowVLMLoader": "ProductionFlow VLM Loader (Local)",
    "ProductionFlowVLMCloudLoader": "ProductionFlow VLM Loader (Cloud API)",
    "ProductionFlowVLMGenerate": "ProductionFlow VLM Generate",
    "ProductionFlowMotionBlurFilmGrain": "ProductionFlow Motion Blur Film Grain",
}
