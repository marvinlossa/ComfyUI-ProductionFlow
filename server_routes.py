from aiohttp import web

from server import PromptServer

from .nodes import (
    all_text_folders,
    folder_output_name,
    list_text_folders,
    prompt_output_name,
    scan_image_files,
    scan_loras,
    scan_prompts,
    scan_text_files,
)


@PromptServer.instance.routes.post("/productionflow/lora-folder-info")
async def lora_folder_info(request):
    data = await request.json()
    lora_folder = data.get("lora_folder", ".")
    recursive = bool(data.get("recursive", False))
    loras = scan_loras(lora_folder, "", recursive)
    return web.json_response(
        {
            "count": len(loras),
            "loras": loras,
            "output_folder": folder_output_name(lora_folder),
        }
    )


@PromptServer.instance.routes.post("/productionflow/prompt-folder-info")
async def prompt_folder_info(request):
    data = await request.json()
    prompt_folder = data.get("prompt_folder", "none")
    recursive = bool(data.get("recursive", False))
    prompts = scan_prompts(prompt_folder, "", recursive)
    return web.json_response(
        {
            "count": len(prompts),
            "prompts": prompts,
            "output_folders": [prompt_output_name(prompt) for prompt in prompts],
        }
    )


@PromptServer.instance.routes.get("/productionflow/text-folders")
async def text_folders(request):
    root = request.rel_url.query.get("root")
    try:
        folders = list_text_folders(root) if root else all_text_folders()
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response(folders)


@PromptServer.instance.routes.post("/productionflow/text-folders")
async def text_folders_post(request):
    data = await request.json()
    root = data.get("root", "input")
    try:
        folders = list_text_folders(root)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"folders": folders})


@PromptServer.instance.routes.post("/productionflow/text-folder-info")
async def text_folder_info(request):
    data = await request.json()
    root = data.get("root", "input")
    folder = data.get("folder", ".")
    recursive = bool(data.get("recursive", False))
    try:
        files = scan_text_files(root, folder, recursive)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"count": len(files), "files": files})


@PromptServer.instance.routes.post("/productionflow/image-folders")
async def image_folders_post(request):
    data = await request.json()
    root = data.get("root", "input")
    try:
        folders = list_text_folders(root)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"folders": folders})


@PromptServer.instance.routes.post("/productionflow/image-folder-info")
async def image_folder_info(request):
    data = await request.json()
    root = data.get("root", "input")
    folder = data.get("folder", ".")
    recursive = bool(data.get("recursive", False))
    try:
        files = scan_image_files(root, folder, recursive)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"count": len(files), "files": files})
