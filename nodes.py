import json
import os
import re

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import comfy.model_management
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
from comfy.cli_args import args

from comfy.utils import ProgressBar

from .face_square import _retinaface_class, crop_and_region, paste_mask
from .image_filters import apply_image_filters
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
TEXT_ROOTS = ("input", "output")
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


def text_root_dir(root="input"):
    name = (root or "input").strip().lower()
    if name == "output":
        return folder_paths.get_output_directory()
    if name == "input":
        return folder_paths.get_input_directory()
    raise ValueError("ProductionFlow: unknown text root. Use input or output.")


def prompt_root_dir():
    return text_root_dir("input")


def list_text_files(root="input"):
    root_dir = text_root_dir(root)
    files = []
    if not os.path.isdir(root_dir):
        return files

    for current_root, _, names in os.walk(root_dir):
        for name in names:
            if not name.lower().endswith(PROMPT_EXTENSIONS):
                continue
            path = os.path.join(current_root, name)
            relpath = os.path.relpath(path, root_dir).replace("\\", "/")
            files.append(relpath)
    return sorted(files, key=lambda x: x.lower())


def _walk_subfolders(root_dir):
    folders = {"."}
    if not os.path.isdir(root_dir):
        return folders
    for current_root, dirnames, _ in os.walk(root_dir):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        for dirname in dirnames:
            path = os.path.join(current_root, dirname)
            folders.add(os.path.relpath(path, root_dir).replace("\\", "/"))
    return folders


def list_text_folders(root="input"):
    folders = _walk_subfolders(text_root_dir(root))
    return sorted(folders, key=lambda x: (x == ".", x.lower()))


def all_text_folders():
    folders = set()
    for root in TEXT_ROOTS:
        folders.update(list_text_folders(root))
    return sorted(folders, key=lambda x: (x == ".", x.lower())) or ["."]


def scan_files_in_folder(files, folder, recursive=False, filter_text=""):
    selected = normalize_folder(folder)
    if selected == "none":
        return []

    filter_text = (filter_text or "").strip().lower()
    out = []
    for name in files:
        file_folder = os.path.dirname(name).replace("\\", "/") or "."
        in_folder = file_folder == selected
        if recursive and selected != ".":
            in_folder = in_folder or file_folder.startswith(selected + "/")
        elif recursive and selected == ".":
            in_folder = True

        if not in_folder:
            continue
        if filter_text and filter_text not in name.lower():
            continue
        out.append(name)
    return out


def scan_text_files(root, folder, recursive=False, filter_text=""):
    return scan_files_in_folder(list_text_files(root), folder, recursive, filter_text)


def list_image_files(root="input"):
    root_dir = text_root_dir(root)
    files = []
    if not os.path.isdir(root_dir):
        return files

    for current_root, _, names in os.walk(root_dir):
        for name in folder_paths.filter_files_content_types(names, ["image"]):
            path = os.path.join(current_root, name)
            files.append(os.path.relpath(path, root_dir).replace("\\", "/"))
    return sorted(files, key=lambda x: x.lower())


def scan_image_files(root, folder, recursive=False, filter_text=""):
    return scan_files_in_folder(list_image_files(root), folder, recursive, filter_text)


def file_view_info(root, relpath):
    normalized = relpath.replace("\\", "/").strip("/")
    subfolder = os.path.dirname(normalized)
    if subfolder in ("", "."):
        subfolder = ""
    return {
        "filename": os.path.basename(normalized),
        "subfolder": subfolder,
        "type": (root or "input").strip().lower(),
    }


def read_text_file(root, relpath, strip=True):
    root_dir = text_root_dir(root)
    normalized = relpath.replace("\\", "/").strip("/")
    path = os.path.abspath(os.path.join(root_dir, normalized))
    root_abs = os.path.abspath(root_dir)
    if not path.startswith(root_abs + os.sep) and path != root_abs:
        raise ValueError("ProductionFlow: text path escapes the selected root.")
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    return text.strip() if strip else text


def prompt_files():
    return list_text_files("input")


def prompt_folders():
    folders = _walk_subfolders(prompt_root_dir())
    folders.add("none")
    return sorted(folders, key=lambda x: (x != "none", x == ".", x.lower()))


def scan_prompts(prompt_folder, filter_text="", recursive=False):
    return scan_text_files("input", prompt_folder, recursive, filter_text)


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
    return read_text_file("input", prompt_name)


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


class ProductionFlowTextFolderLoad:
    """Load every text file in a folder for browsing after one run."""

    @classmethod
    def INPUT_TYPES(cls):
        folders = all_text_folders()
        default_folder = "Prompts" if "Prompts" in folders else folders[0]
        return {
            "required": {
                "root": (
                    list(TEXT_ROOTS),
                    {
                        "default": "input",
                        "tooltip": "ComfyUI/input or ComfyUI/output. Folder list is relative to this root.",
                    },
                ),
                "folder": (
                    folders,
                    {
                        "default": default_folder,
                        "tooltip": (
                            "Folder under the selected root. "
                            "The list updates when you change root or press refresh."
                        ),
                    },
                ),
                "recursive": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Include text files in subfolders of the selected folder.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("PF_TEXT_BATCH",)
    RETURN_NAMES = ("texts",)
    FUNCTION = "load"
    CATEGORY = "ProductionFlow"
    DESCRIPTION = (
        "Load all text files from one folder under input or output. "
        "Connect to ProductionFlow Show Texts to flip through them after a single run."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, root, folder):
        try:
            selected = normalize_folder(folder)
            folders = list_text_folders(root)
        except ValueError as exc:
            return str(exc)
        if selected not in folders:
            return f"ProductionFlow: folder '{folder}' not found under {root}."
        return True

    @classmethod
    def IS_CHANGED(cls, root, folder, recursive=False):
        files = scan_text_files(root, folder, recursive)
        root_dir = text_root_dir(root)
        parts = []
        for name in files:
            path = os.path.join(root_dir, name)
            try:
                stat = os.stat(path)
                parts.append(f"{name}:{stat.st_mtime_ns}:{stat.st_size}")
            except OSError:
                parts.append(f"{name}:missing")
        return "|".join(parts)

    def load(self, root, folder, recursive=False):
        files = scan_text_files(root, folder, recursive)
        if not files:
            raise ValueError(
                f"ProductionFlow: no text files found in {root}/{folder}."
            )

        batch = []
        for name in files:
            batch.append({"name": name, "text": read_text_file(root, name, strip=False)})
        return (batch,)


class ProductionFlowShowTexts:
    """Browse a loaded text-file batch without queueing again."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texts": (
                    "PF_TEXT_BATCH",
                    {
                        "tooltip": "Connect texts from ProductionFlow Text Folder Load.",
                    },
                ),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "show"
    OUTPUT_NODE = True
    CATEGORY = "ProductionFlow"
    DESCRIPTION = (
        "Show text files loaded by ProductionFlow Text Folder Load. "
        "After one run, use Previous / Next (or the file list) to flip files without queueing again."
    )

    def show(self, texts):
        if not texts:
            raise ValueError("ProductionFlow: no text files to display.")

        names = []
        bodies = []
        for item in texts:
            names.append(str(item.get("name", "")))
            bodies.append(str(item.get("text", "")))
        first = bodies[0] if bodies else ""
        return {"ui": {"names": names, "texts": bodies, "text": (first,)}}


class ProductionFlowImageFolderLoad:
    """Load every image in a folder for browsing after one run."""

    @classmethod
    def INPUT_TYPES(cls):
        folders = all_text_folders()
        return {
            "required": {
                "root": (
                    list(TEXT_ROOTS),
                    {
                        "default": "input",
                        "tooltip": "ComfyUI/input or ComfyUI/output. Folder list is relative to this root.",
                    },
                ),
                "folder": (
                    folders,
                    {
                        "default": ".",
                        "tooltip": (
                            "Folder under the selected root. "
                            "The list updates when you change root or press refresh."
                        ),
                    },
                ),
                "recursive": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Include images in subfolders of the selected folder.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("PF_IMAGE_BATCH",)
    RETURN_NAMES = ("images",)
    FUNCTION = "load"
    CATEGORY = "ProductionFlow"
    DESCRIPTION = (
        "Load all images from one folder under input or output. "
        "Connect to ProductionFlow Show Images to flip through them after a single run."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, root, folder):
        try:
            selected = normalize_folder(folder)
            folders = list_text_folders(root)
        except ValueError as exc:
            return str(exc)
        if selected not in folders:
            return f"ProductionFlow: folder '{folder}' not found under {root}."
        return True

    @classmethod
    def IS_CHANGED(cls, root, folder, recursive=False):
        files = scan_image_files(root, folder, recursive)
        root_dir = text_root_dir(root)
        parts = []
        for name in files:
            path = os.path.join(root_dir, name)
            try:
                stat = os.stat(path)
                parts.append(f"{name}:{stat.st_mtime_ns}:{stat.st_size}")
            except OSError:
                parts.append(f"{name}:missing")
        return "|".join(parts)

    def load(self, root, folder, recursive=False):
        files = scan_image_files(root, folder, recursive)
        if not files:
            raise ValueError(
                f"ProductionFlow: no image files found in {root}/{folder}."
            )

        batch = []
        for name in files:
            batch.append({"name": name, "root": root, "info": file_view_info(root, name)})
        return (batch,)


def _prompt_from_folder_loop(node):
    inputs = node.get("inputs") or {}
    folder = inputs.get("prompt_folder") or "none"
    if normalize_folder(folder) == "none":
        return ""
    try:
        index = int(inputs.get("index", 0))
    except (TypeError, ValueError):
        return ""
    prompts = scan_prompts(folder, "", bool(inputs.get("recursive", False)))
    if index < 0 or index >= len(prompts):
        return ""
    return read_prompt_file(prompts[index])


def prompt_from_image(path):
    """Positive prompt that was actually executed for this image.

    ComfyUI's workflow chunk keeps the CLIP widget from the open graph, so a
    queued prompt-folder loop stamps the same sentence into every PNG. The
    executed index lives in the API prompt chunk. Resolve that when the text
    encoder is linked to ProductionFlow Prompt Folder Loop. Otherwise use a
    literal CLIP text, then the workflow widget.
    """
    try:
        info = Image.open(path).info or {}
    except OSError:
        return ""

    api = None
    raw_api = info.get("prompt")
    if isinstance(raw_api, str):
        try:
            parsed = json.loads(raw_api)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            api = parsed

    literals = []
    if api:
        for node in api.values():
            if not isinstance(node, dict) or node.get("class_type") != "CLIPTextEncode":
                continue
            text = (node.get("inputs") or {}).get("text")
            if isinstance(text, str) and text.strip():
                literals.append(text.strip())
                continue
            if isinstance(text, (list, tuple)) and text:
                source = api.get(str(text[0]))
                if not isinstance(source, dict):
                    continue
                if source.get("class_type") in (
                    "ProductionFlowPromptFolderLoop",
                    "ProductionFlowPromptFolderSelector",
                ):
                    resolved = _prompt_from_folder_loop(source)
                    if resolved:
                        return resolved

    if literals:
        return max(literals, key=len)

    workflow = info.get("workflow")
    if isinstance(workflow, str):
        try:
            graph = json.loads(workflow)
        except json.JSONDecodeError:
            graph = None
        widget_texts = []
        if isinstance(graph, dict):
            for node in graph.get("nodes") or []:
                if node.get("type") != "CLIPTextEncode":
                    continue
                values = node.get("widgets_values") or []
                if values and isinstance(values[0], str) and values[0].strip():
                    widget_texts.append(values[0].strip())
        if widget_texts:
            return max(widget_texts, key=len)

    parameters = info.get("parameters")
    if isinstance(parameters, str) and parameters.strip():
        return parameters.split("Negative prompt:", 1)[0].strip()
    return ""


def load_image_file(root, relpath):
    """Load one image the way a single Load Image node would: RGB float image plus mask."""
    root_dir = text_root_dir(root)
    normalized = relpath.replace("\\", "/").strip("/")
    path = os.path.abspath(os.path.join(root_dir, normalized))
    root_abs = os.path.abspath(root_dir)
    if not path.startswith(root_abs + os.sep) and path != root_abs:
        raise ValueError("ProductionFlow: image path escapes the selected root.")

    from PIL import ImageOps, ImageSequence

    img = Image.open(path)
    output_images = []
    output_masks = []
    width = height = None
    for frame in ImageSequence.Iterator(img):
        frame = ImageOps.exif_transpose(frame)
        rgb = frame.convert("RGB")
        if width is None:
            width, height = rgb.size
        if rgb.size != (width, height):
            continue
        array = np.array(rgb).astype(np.float32) / 255.0
        output_images.append(torch.from_numpy(array)[None,])
        if "A" in frame.getbands():
            mask = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
            output_masks.append(1.0 - torch.from_numpy(mask))
        else:
            output_masks.append(torch.zeros((64, 64), dtype=torch.float32))

    if len(output_images) > 1:
        return torch.cat(output_images, dim=0), torch.stack(output_masks, dim=0)
    return output_images[0], output_masks[0].unsqueeze(0)


class ProductionFlowImageFolderLoop:
    """Pick one image by index, the same way Prompt Folder Loop picks one prompt.

    Batch runs use the Queue All Images button, which queues one job per file
    with a different index. ProductionFlow Image Folder Load stays the browser.
    """

    @classmethod
    def INPUT_TYPES(cls):
        folders = all_text_folders()
        return {
            "required": {
                "root": (
                    list(TEXT_ROOTS),
                    {
                        "default": "input",
                        "tooltip": "ComfyUI/input or ComfyUI/output. Folder list is relative to this root.",
                    },
                ),
                "folder": (
                    folders,
                    {
                        "default": ".",
                        "tooltip": "Folder under the selected root. The list updates when you change root or press refresh.",
                    },
                ),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100000,
                        "step": 1,
                        "tooltip": "Which image to load for this run (0 = first in sorted order). Queue All Images sets this per job.",
                    },
                ),
                "recursive": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Include images in subfolders of the selected folder.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "prompt", "image_name", "image_index", "image_count")
    FUNCTION = "select_image"
    CATEGORY = "ProductionFlow"
    DESCRIPTION = (
        "Image folder loop. Outputs one image per run by index, plus the positive "
        "prompt stored in that image's ComfyUI metadata when it has one. "
        "Use Queue All Images to enqueue every file in the folder."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, root, folder, index):
        try:
            selected = normalize_folder(folder)
            folders = list_text_folders(root)
        except ValueError as exc:
            return str(exc)
        if selected not in folders:
            return f"ProductionFlow: folder '{folder}' not found under {root}."
        return True

    @classmethod
    def IS_CHANGED(cls, root, folder, index, recursive=False):
        files = scan_image_files(root, folder, recursive)
        if not files or index < 0 or index >= len(files):
            return f"{root}|{folder}|{index}|missing|{len(files)}"
        name = files[index]
        path = os.path.join(text_root_dir(root), name)
        try:
            stat = os.stat(path)
            stamp = f"{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            stamp = "missing"
        return f"{root}|{name}|{index}|{len(files)}|{stamp}"

    def select_image(self, root, folder, index, recursive=False):
        files = scan_image_files(root, folder, recursive)
        if not files:
            raise ValueError(f"ProductionFlow: no image files found in {root}/{folder}.")
        if index >= len(files):
            raise ValueError(
                f"ProductionFlow: image index {index} is out of range for "
                f"{len(files)} images in {root}/{folder}."
            )
        name = files[index]
        image, mask = load_image_file(root, name)
        path = os.path.join(text_root_dir(root), name)
        return (image, mask, prompt_from_image(path), sanitize_path_part(name, "image"), index, len(files))


class ProductionFlowShowImages:
    """Browse a loaded image-file batch without queueing again."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "PF_IMAGE_BATCH",
                    {
                        "tooltip": "Connect images from ProductionFlow Image Folder Load.",
                    },
                ),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "show"
    OUTPUT_NODE = True
    CATEGORY = "ProductionFlow"
    DESCRIPTION = (
        "Show images loaded by ProductionFlow Image Folder Load. "
        "After one run, use Previous / Next (or the file list) to flip files without queueing again."
    )

    def show(self, images):
        if not images:
            raise ValueError("ProductionFlow: no images to display.")

        names = []
        infos = []
        for item in images:
            names.append(str(item.get("name", "")))
            info = item.get("info")
            if not info:
                info = file_view_info(item.get("root", "input"), item.get("name", ""))
            infos.append(info)
        return {"ui": {"names": names, "pf_images": infos}}


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


class ProductionFlowImageFilters:
    """Saturation, contrast, warmth, vignette, then grain on an IMAGE batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Still or frame batch (N, H, W, C)."}),
                "saturation": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "1 = unchanged. 0 = grayscale. >1 = punchier color.",
                    },
                ),
                "contrast": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "1 = unchanged. <1 = flatter. >1 = harder lights and darks.",
                    },
                ),
                "warmth": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "0 = unchanged. Positive = warmer (golden). Negative = cooler.",
                    },
                ),
                "vignette": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "0 = off. Darkens the frame edges. 0.35 = mild. 0.7+ = heavy.",
                    },
                ),
                "grain": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.5,
                        "tooltip": (
                            "Film grain on the same 8-bit-style scale as Motion Blur Film Grain. "
                            "0 = off. 2–4 = fine. 6 = mild. 8–12 = noticeable stock."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Random seed for grain. Unused when grain is 0.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "ProductionFlow"
    DESCRIPTION = (
        "Common photo filters on an image or frame batch: saturation, contrast, "
        "warmth, vignette, then grain. Defaults leave the image unchanged. "
        "Grain matches ProductionFlow Motion Blur Film Grain."
    )

    def process(
        self,
        images,
        saturation=1.0,
        contrast=1.0,
        warmth=0.0,
        vignette=0.0,
        grain=0.0,
        seed=0,
    ):
        if images is None or not isinstance(images, torch.Tensor):
            raise ValueError("ProductionFlow Image Filters: expected an IMAGE tensor.")
        if images.ndim != 4:
            raise ValueError(
                f"ProductionFlow Image Filters: expected NHWC IMAGE batch, "
                f"got shape {tuple(images.shape)}."
            )

        n = images.shape[0]
        pbar = ProgressBar(max(1, n))
        out = apply_image_filters(
            images,
            saturation=saturation,
            contrast=contrast,
            warmth=warmth,
            vignette=vignette,
            grain=grain,
            seed=seed,
            progress_callback=pbar.update,
        )
        if pbar.current < pbar.total:
            pbar.update_absolute(pbar.total)
        return (out,)


class ProductionFlowFaceSquare:
    """Detect a face and crop a padded square so Face Segment sees head, not torso."""

    def __init__(self):
        self.detector = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Full frame. Use the same image that goes into VAE Encode."}),
                "scale": (
                    "FLOAT",
                    {
                        "default": 1.8,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "Square size relative to the detected face. 1.5 tight. 1.8 default. 2.2 includes hair/neck.",
                    },
                ),
                "shift": (
                    "FLOAT",
                    {
                        "default": 0.42,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "0.5 = face centered. Lower includes more forehead/hair, less chest.",
                    },
                ),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16,
                        "tooltip": "Which detected face to use, largest first.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("crop", "region", "x", "y")
    FUNCTION = "crop"
    CATEGORY = "ProductionFlow"
    DESCRIPTION = (
        "RetinaFace crop of a padded square around the face. Wire crop into Face Segment, "
        "then ProductionFlow Paste Mask with x/y to put the mask back on the full frame."
    )

    def crop(self, images, scale=1.8, shift=0.42, index=0):
        if images is None or not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("ProductionFlow Face Square: expected an IMAGE tensor.")
        if self.detector is None:
            self.detector = _retinaface_class()(
                device=comfy.model_management.get_torch_device(),
                vis_thres=0.6,
                keep_top_k=20,
            )
        crop, region, x, y = crop_and_region(
            images, scale=scale, shift=shift, start_index=index, detector=self.detector
        )
        return (crop, region, x, y)


class ProductionFlowEulerAncestral:
    """KSampler locked to euler_ancestral, with the ancestral eta that stock KSampler hides.

    Krea 2 is a rectified-flow model, so this reaches sample_euler_ancestral_RF.
    eta 1 and s_noise 1 match the stock sampler. Lower eta fades existing texture
    without switching the scheduler, which is what keeps freckles in place.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "eta": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Ancestral noise amount. 1 is stock euler_ancestral. Lower keeps freckle positions and adds fewer new marks. 0 is plain Euler on this same schedule.",
                }),
                "s_noise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Scale on the ancestral noise sample. 1 matches stock. Leave this alone and use eta.",
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "ProductionFlow"
    DESCRIPTION = "euler_ancestral KSampler with an eta control. Use the same scheduler as the first pass."

    def sample(self, model, seed, steps, cfg, scheduler, positive, negative, latent_image, denoise, eta, s_noise):
        import comfy.sample
        import latent_preview

        image = latent_image["samples"]
        image = comfy.sample.fix_empty_latent_channels(
            model, image, latent_image.get("downscale_ratio_spacial", None), latent_image.get("downscale_ratio_temporal", None)
        )
        batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
        noise = comfy.sample.prepare_noise(image, seed, batch_inds)
        noise_mask = latent_image.get("noise_mask")

        schedule = comfy.samplers.KSampler(
            model,
            steps=steps,
            device=model.load_device,
            sampler="euler_ancestral",
            scheduler=scheduler,
            denoise=denoise,
            model_options=model.model_options,
        )
        sampler = comfy.samplers.ksampler("euler_ancestral", {"eta": float(eta), "s_noise": float(s_noise)})
        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample_custom(
            model,
            noise,
            cfg,
            sampler,
            schedule.sigmas,
            positive,
            negative,
            image,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
        out = latent_image.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples
        return (out,)


class ProductionFlowPasteMask:
    """Paste a crop-sized MASK back onto the full frame at x, y."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Mask from Face Segment run on the face crop."}),
                "image": ("IMAGE", {"tooltip": "Full frame used only for canvas size."}),
                "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "tooltip": "Connect x from Face Square."}),
                "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "tooltip": "Connect y from Face Square."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "paste"
    CATEGORY = "ProductionFlow"
    DESCRIPTION = (
        "Place a Face Segment crop mask back onto the full image using Face Square x/y. "
        "Connect the result to Set Latent Noise Mask."
    )

    def paste(self, mask, image, x, y):
        if image is None or not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("ProductionFlow Paste Mask: expected an IMAGE tensor for canvas size.")
        return (paste_mask(mask, image, x, y),)


NODE_CLASS_MAPPINGS = {
    "ProductionFlowPromptFolderLoop": ProductionFlowPromptFolderLoop,
    "ProductionFlowPromptFolderSelector": ProductionFlowPromptFolderLoop,
    "ProductionFlowTextFolderLoad": ProductionFlowTextFolderLoad,
    "ProductionFlowShowTexts": ProductionFlowShowTexts,
    "ProductionFlowImageFolderLoad": ProductionFlowImageFolderLoad,
    "ProductionFlowImageFolderLoop": ProductionFlowImageFolderLoop,
    "ProductionFlowShowImages": ProductionFlowShowImages,
    "ProductionFlowLoraFolderLoader": ProductionFlowLoraFolderLoader,
    "ProductionFlowLoraTestSaveImage": ProductionFlowLoraTestSaveImage,
    "ProductionFlowNoisyLatentImage": ProductionFlowNoisyLatentImage,
    "ProductionFlowVLMLoader": ProductionFlowVLMLoader,
    "ProductionFlowVLMCloudLoader": ProductionFlowVLMCloudLoader,
    "ProductionFlowVLMGenerate": ProductionFlowVLMGenerate,
    "ProductionFlowMotionBlurFilmGrain": ProductionFlowMotionBlurFilmGrain,
    "ProductionFlowImageFilters": ProductionFlowImageFilters,
    "ProductionFlowFaceSquare": ProductionFlowFaceSquare,
    "ProductionFlowPasteMask": ProductionFlowPasteMask,
    "ProductionFlowEulerAncestral": ProductionFlowEulerAncestral,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProductionFlowPromptFolderLoop": "ProductionFlow Prompt Folder Loop",
    "ProductionFlowPromptFolderSelector": "ProductionFlow Prompt Folder Loop",
    "ProductionFlowTextFolderLoad": "ProductionFlow Text Folder Load",
    "ProductionFlowShowTexts": "ProductionFlow Show Texts",
    "ProductionFlowImageFolderLoad": "ProductionFlow Image Folder Load",
    "ProductionFlowImageFolderLoop": "ProductionFlow Image Folder Loop",
    "ProductionFlowShowImages": "ProductionFlow Show Images",
    "ProductionFlowLoraFolderLoader": "ProductionFlow LoRA Folder Loader",
    "ProductionFlowLoraTestSaveImage": "ProductionFlow LoRA Test Save Image",
    "ProductionFlowNoisyLatentImage": "ProductionFlow Noisy Latent Image",
    "ProductionFlowVLMLoader": "ProductionFlow VLM Loader (Local)",
    "ProductionFlowVLMCloudLoader": "ProductionFlow VLM Loader (Cloud API)",
    "ProductionFlowVLMGenerate": "ProductionFlow VLM Generate",
    "ProductionFlowMotionBlurFilmGrain": "ProductionFlow Motion Blur Film Grain",
    "ProductionFlowImageFilters": "ProductionFlow Image Filters",
    "ProductionFlowFaceSquare": "ProductionFlow Face Square",
    "ProductionFlowPasteMask": "ProductionFlow Paste Mask",
    "ProductionFlowEulerAncestral": "ProductionFlow Euler Ancestral",
}
