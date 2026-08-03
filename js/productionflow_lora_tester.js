import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";


const PROMPT_LOOP_CLASSES = new Set([
  "ProductionFlowPromptFolderLoop",
  "ProductionFlowPromptFolderSelector", // legacy type name
]);


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


function isPromptLoopNode(node) {
  return PROMPT_LOOP_CLASSES.has(node.comfyClass);
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


async function queueAllPrompts(node) {
  const folder = widgetValue(node, "prompt_folder", "none");
  const recursive = !!widgetValue(node, "recursive", false);

  if (!folder || folder === "none") {
    throw new Error(
      "Select a prompt folder (not none) before Queue All Prompts. " +
        "Use fallback_prompt for a single connected prompt."
    );
  }

  const infoResponse = await api.fetchApi("/productionflow/prompt-folder-info", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt_folder: folder, recursive }),
  });
  if (!infoResponse.ok) throw new Error(await infoResponse.text());
  const info = await infoResponse.json();
  if (!info.count) throw new Error(`No prompt files found in ${folder}`);

  const graphPrompt = await app.graphToPrompt();
  const basePrompt = graphPrompt.output;
  const workflow = graphPrompt.workflow;

  for (let promptIndex = 0; promptIndex < info.count; promptIndex++) {
    const prompt = structuredClone(basePrompt);
    setPromptInput(prompt, node.id, "index", promptIndex);
    removeStaleSaveInputs(prompt);
    await queuePrompt(prompt, workflow);
  }

  alert(
    `ProductionFlow queued ${info.count} prompt job(s) from "${folder}". ` +
      `Each run uses index 0…${info.count - 1} (sorted file order).`
  );
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

  // Prompt loop lives on this LoRA node only (prompt_folder / prompt_index).
  let promptInfo = { count: 1 };
  const promptFolder = widgetValue(node, "prompt_folder", "none");
  const promptRecursive = !!widgetValue(node, "prompt_recursive", false);
  if (promptFolder && promptFolder !== "none") {
    const promptInfoResponse = await api.fetchApi("/productionflow/prompt-folder-info", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt_folder: promptFolder, recursive: promptRecursive }),
    });
    if (!promptInfoResponse.ok) throw new Error(await promptInfoResponse.text());
    promptInfo = await promptInfoResponse.json();
    if (!promptInfo.count) throw new Error(`No prompts found in ${promptFolder}`);
  }

  const graphPrompt = await app.graphToPrompt();
  const basePrompt = graphPrompt.output;
  const workflow = graphPrompt.workflow;

  for (let promptIndex = 0; promptIndex < promptInfo.count; promptIndex++) {
    for (let loraIndex = 0; loraIndex < info.count; loraIndex++) {
      const prompt = structuredClone(basePrompt);
      if (promptFolder && promptFolder !== "none") {
        setPromptInput(prompt, node.id, "prompt_index", promptIndex);
      }
      setPromptInput(prompt, node.id, "index", loraIndex);
      removeStaleSaveInputs(prompt);
      await queuePrompt(prompt, workflow);
    }
  }

  const total = promptInfo.count * info.count;
  alert(
    `ProductionFlow queued ${total} LoRA test jobs ` +
      `(${promptInfo.count} prompts x ${info.count} LoRAs). ` +
      `Images save under output/ProductionFlow/<prompt>/<lora>.`
  );
}


app.registerExtension({
  name: "ComfyUI-ProductionFlow.LoRATester",

  async nodeCreated(node) {
    removeStaleSaveNodeInput(node);

    if (node.comfyClass === "ProductionFlowLoraFolderLoader") {
      node.addWidget("button", "Queue All LoRAs/Prompts", null, async () => {
        try {
          await queueAllLoras(node);
        } catch (error) {
          console.error(error);
          alert(`ProductionFlow LoRA queue failed: ${error.message || error}`);
        }
      });
      return;
    }

    if (isPromptLoopNode(node)) {
      node.addWidget("button", "Queue All Prompts", null, async () => {
        try {
          await queueAllPrompts(node);
        } catch (error) {
          console.error(error);
          alert(`ProductionFlow prompt queue failed: ${error.message || error}`);
        }
      });
    }
  },

  async loadedGraphNode(node) {
    removeStaleSaveNodeInput(node);
  },
});
