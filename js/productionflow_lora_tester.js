import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";


function widgetValue(node, name, fallback = null) {
  const widget = node.widgets?.find((w) => w.name === name);
  return widget ? widget.value : fallback;
}


function setPromptInput(prompt, nodeId, inputName, value) {
  const item = prompt[String(nodeId)] || prompt[nodeId];
  if (!item?.inputs) throw new Error(`Could not find prompt inputs for node ${nodeId}`);
  item.inputs[inputName] = value;
}


function removeStaleSaveInputs(prompt) {
  for (const item of Object.values(prompt)) {
    if (item?.class_type === "ProductionFlowLoraTestSaveImage") {
      delete item.inputs?.prompt_folder_name;
    }
  }
}


function removeStaleSaveNodeInput(node) {
  if (node.comfyClass !== "ProductionFlowLoraTestSaveImage") return;
  const slot = node.findInputSlot?.("prompt_folder_name");
  if (slot !== undefined && slot !== -1) {
    node.removeInput(slot);
    app.graph.setDirtyCanvas(true, true);
  }
}


function findPromptSelector() {
  const selectors = app.graph._nodes.filter((n) => n.comfyClass === "ProductionFlowPromptFolderSelector");
  return selectors.find((n) => widgetValue(n, "prompt_folder", "none") !== "none") || selectors[0] || null;
}


function promptSourceForLoraNode(node) {
  if (widgetValue(node, "prompt_folder", "none") !== "none") {
    return {
      node,
      indexInput: "prompt_index",
      folder: widgetValue(node, "prompt_folder", "none"),
      recursive: !!widgetValue(node, "prompt_recursive", false),
    };
  }

  const selector = findPromptSelector();
  if (!selector || widgetValue(selector, "prompt_folder", "none") === "none") return null;
  return {
    node: selector,
    indexInput: "index",
    folder: widgetValue(selector, "prompt_folder", "none"),
    recursive: !!widgetValue(selector, "recursive", false),
  };
}


async function queuePrompt(prompt, workflow) {
  const body = {
    prompt,
    extra_data: { extra_pnginfo: { workflow } },
    client_id: api.clientId,
  };
  const response = await api.fetchApi("/prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Queue failed with HTTP ${response.status}`);
  }
  return await response.json();
}


async function queueAllLoras(node) {
  const loraFolder = widgetValue(node, "lora_folder", ".");
  const recursive = !!widgetValue(node, "recursive", false);

  const infoResponse = await api.fetchApi("/productionflow/lora-folder-info", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lora_folder: loraFolder, recursive }),
  });
  if (!infoResponse.ok) throw new Error(await infoResponse.text());
  const info = await infoResponse.json();
  if (!info.count) throw new Error(`No LoRAs found in ${loraFolder}`);

  const promptSource = promptSourceForLoraNode(node);
  let promptInfo = { count: 1, output_folders: ["single_prompt"] };
  if (promptSource) {
    const promptInfoResponse = await api.fetchApi("/productionflow/prompt-folder-info", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt_folder: promptSource.folder, recursive: promptSource.recursive }),
    });
    if (!promptInfoResponse.ok) throw new Error(await promptInfoResponse.text());
    promptInfo = await promptInfoResponse.json();
    if (!promptInfo.count) throw new Error(`No prompts found in ${promptSource.folder}`);
  }

  const graphPrompt = await app.graphToPrompt();
  const basePrompt = graphPrompt.output;
  const workflow = graphPrompt.workflow;

  for (let promptIndex = 0; promptIndex < promptInfo.count; promptIndex++) {
    for (let loraIndex = 0; loraIndex < info.count; loraIndex++) {
      const prompt = structuredClone(basePrompt);
      if (promptSource) {
        setPromptInput(prompt, promptSource.node.id, promptSource.indexInput, promptIndex);
      }
      setPromptInput(prompt, node.id, "index", loraIndex);
      removeStaleSaveInputs(prompt);
      await queuePrompt(prompt, workflow);
    }
  }

  const total = promptInfo.count * info.count;
  alert(`ProductionFlow queued ${total} LoRA test jobs (${promptInfo.count} prompts x ${info.count} LoRAs). Images will save to output/ProductionFlow/<prompt>/<lora>.`);
}


app.registerExtension({
  name: "ComfyUI-ProductionFlow.LoRATester",

  async nodeCreated(node) {
    removeStaleSaveNodeInput(node);
    if (node.comfyClass !== "ProductionFlowLoraFolderLoader") return;
    node.addWidget("button", "Queue All LoRAs/Prompts", null, async () => {
      try {
        await queueAllLoras(node);
      } catch (error) {
        console.error(error);
        alert(`ProductionFlow LoRA queue failed: ${error.message || error}`);
      }
    });
  },

  async loadedGraphNode(node) {
    removeStaleSaveNodeInput(node);
  },
});
