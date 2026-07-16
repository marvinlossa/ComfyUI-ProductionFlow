from aiohttp import web

from server import PromptServer

from .nodes import folder_output_name, prompt_output_name, scan_loras, scan_prompts


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
