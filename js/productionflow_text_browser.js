import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";


function widgetByName(node, name) {
  return node.widgets?.find((w) => w.name === name);
}


function widgetValue(node, name, fallback = null) {
  const widget = widgetByName(node, name);
  return widget ? widget.value : fallback;
}


function wrapCallback(widget, extra) {
  if (!widget) return;
  const prev = widget.callback;
  widget.callback = function () {
    const result = prev?.apply(this, arguments);
    extra();
    return result;
  };
}


function disableSerialize(widget) {
  if (!widget) return;
  widget.serialize = false;
  if (widget.options) widget.options.serialize = false;
  widget.serializeValue = async () => "";
}


async function postJson(path, body) {
  const response = await api.fetchApi(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json();
}


function folderLoadRoutes(kind) {
  if (kind === "image") {
    return {
      folders: "/productionflow/image-folders",
      info: "/productionflow/image-folder-info",
      fallback: ".",
    };
  }
  return {
    folders: "/productionflow/text-folders",
    info: "/productionflow/text-folder-info",
    fallback: "Prompts",
  };
}


async function refreshFileCount(node) {
  const infoWidget = node.pfFileCountWidget;
  const routes = node.pfFolderRoutes;
  if (!infoWidget || !routes) return;

  try {
    const data = await postJson(routes.info, {
      root: widgetValue(node, "root", "input"),
      folder: widgetValue(node, "folder", "."),
      recursive: !!widgetValue(node, "recursive", false),
    });
    const n = data.count || 0;
    infoWidget.value = n === 1 ? "1 file" : `${n} files`;
  } catch (error) {
    infoWidget.value = "Could not scan folder";
    console.error(error);
  }
}


async function refreshFolderList(node) {
  const folderWidget = widgetByName(node, "folder");
  const routes = node.pfFolderRoutes;
  if (!folderWidget || !routes) return;

  try {
    const data = await postJson(routes.folders, {
      root: widgetValue(node, "root", "input"),
    });
    const folders = data.folders?.length ? data.folders : ["."];
    if (folderWidget.options) folderWidget.options.values = folders;
    if (!folders.includes(folderWidget.value)) {
      folderWidget.value = folders.includes(routes.fallback)
        ? routes.fallback
        : folders[0];
    }
  } catch (error) {
    console.error(error);
  }

  await refreshFileCount(node);
  node.setDirtyCanvas?.(true, true);
}


function setupFolderLoad(node, kind) {
  if (node.pfFolderSetup) {
    refreshFolderList(node);
    return;
  }
  node.pfFolderSetup = true;
  node.pfFolderRoutes = folderLoadRoutes(kind);

  const infoWidget = node.addWidget("text", "file_count", "0 files", () => {});
  infoWidget.disabled = true;
  disableSerialize(infoWidget);
  node.pfFileCountWidget = infoWidget;

  node.addWidget("button", "refresh", "refresh", () => {
    refreshFolderList(node);
  });

  wrapCallback(widgetByName(node, "root"), () => {
    refreshFolderList(node);
  });
  wrapCallback(widgetByName(node, "folder"), () => {
    refreshFileCount(node);
  });
  wrapCallback(widgetByName(node, "recursive"), () => {
    refreshFileCount(node);
  });

  refreshFolderList(node);
}


function asList(value) {
  if (value == null) return [];
  return Array.isArray(value) ? value : [value];
}


function nodeOutputMessage(node) {
  const outputs = app.nodeOutputs || {};
  return outputs[node.id] || outputs[String(node.id)] || null;
}


function leftoverHeight(node, skipWidget, minHeight) {
  const title = 30;
  let used = title + 10;
  for (const widget of node.widgets || []) {
    if (widget === skipWidget) continue;
    used += 24;
  }
  return Math.max(minHeight, (node.size?.[1] || minHeight) - used);
}


function markNodeDirty(node) {
  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
}


function refreshShowTexts(node) {
  const names = node.pfTextNames || [];
  const bodies = node.pfTextBodies || [];
  const n = names.length;
  let index = Number(node.pfTextIndex) || 0;
  if (n === 0) {
    index = 0;
  } else {
    index = ((index % n) + n) % n;
  }
  node.pfTextIndex = index;

  if (node.pfFileWidget) {
    if (node.pfFileWidget.options) {
      node.pfFileWidget.options.values = n ? names : [""];
    }
    node.pfFileWidget.value = n ? names[index] : "";
  }
  if (node.pfIndexWidget) {
    node.pfIndexWidget.value = n ? `${index + 1} / ${n}` : "0 / 0";
  }
  const preview = currentShowText(node);
  if (node.pfTextWidget) {
    node.pfTextWidget.value = preview;
  }
  markNodeDirty(node);
}


function currentShowText(node) {
  const bodies = node.pfTextBodies || [];
  const n = bodies.length;
  if (!n) return "";
  const index = Number(node.pfTextIndex) || 0;
  return bodies[((index % n) + n) % n] ?? "";
}


function stepShowTexts(node, delta) {
  const n = node.pfTextNames?.length || 0;
  if (!n) return;
  node.pfTextIndex = (Number(node.pfTextIndex) || 0) + delta;
  refreshShowTexts(node);
}


function setupShowTexts(node) {
  if (node.pfShowSetup) {
    applyShowFromOutputs(node, applyShowTexts);
    return;
  }
  node.pfShowSetup = true;
  node.pfTextNames = node.pfTextNames || [];
  node.pfTextBodies = node.pfTextBodies || [];
  node.pfTextIndex = Number(node.pfTextIndex) || 0;

  node.addWidget("button", "Previous", null, () => stepShowTexts(node, -1));
  node.addWidget("button", "Next", null, () => stepShowTexts(node, 1));

  const fileWidget = node.addWidget(
    "combo",
    "file",
    "",
    () => {
      const idx = (node.pfTextNames || []).indexOf(fileWidget.value);
      if (idx >= 0) {
        node.pfTextIndex = idx;
        refreshShowTexts(node);
      }
    },
    { values: [""] }
  );
  disableSerialize(fileWidget);
  node.pfFileWidget = fileWidget;

  const indexWidget = node.addWidget("text", "file_index", "0 / 0", () => {});
  indexWidget.disabled = true;
  disableSerialize(indexWidget);
  node.pfIndexWidget = indexWidget;

  const box = document.createElement("textarea");
  box.readOnly = true;
  box.placeholder = "Queue a run to load text files.";
  box.style.width = "100%";
  box.style.height = "100%";
  box.style.minHeight = "0";
  box.style.boxSizing = "border-box";
  box.style.resize = "none";
  const textHeight = () => leftoverHeight(node, domWidget, 160);
  const domWidget = node.addDOMWidget("text", "text", box, {
    hideOnZoom: false,
    getHeight: textHeight,
    getMinHeight: textHeight,
    getMaxHeight: textHeight,
  });
  disableSerialize(domWidget);
  Object.defineProperty(domWidget, "value", {
    get() {
      return box.value;
    },
    set(v) {
      box.value = v ?? "";
    },
  });
  node.pfTextWidget = domWidget;
  node.pfPreviewWidget = domWidget;

  if (node.size[1] < 320) {
    node.setSize([Math.max(node.size[0], 360), 360]);
  }
  applyShowFromOutputs(node, applyShowTexts);
}


function applyShowTexts(node, message) {
  if (!message) return;
  node.pfTextNames = asList(message.names).map((v) => String(v));
  node.pfTextBodies = asList(message.texts).map((v) => String(v));
  if (!node.pfTextNames.length) {
    node.pfTextIndex = 0;
  } else if (node.pfTextIndex >= node.pfTextNames.length) {
    node.pfTextIndex = 0;
  }
  refreshShowTexts(node);
}


function imageViewUrl(info) {
  if (!info?.filename) return "";
  const params = new URLSearchParams({
    filename: info.filename,
    type: info.type || "input",
    subfolder: info.subfolder || "",
  });
  return api.apiURL(`/view?${params}`);
}


function refreshShowImages(node) {
  const names = node.pfImageNames || [];
  const infos = node.pfImageInfos || [];
  const n = names.length;
  let index = Number(node.pfImageIndex) || 0;
  if (n === 0) {
    index = 0;
  } else {
    index = ((index % n) + n) % n;
  }
  node.pfImageIndex = index;

  if (node.pfFileWidget) {
    if (node.pfFileWidget.options) {
      node.pfFileWidget.options.values = n ? names : [""];
    }
    node.pfFileWidget.value = n ? names[index] : "";
  }
  if (node.pfIndexWidget) {
    node.pfIndexWidget.value = n ? `${index + 1} / ${n}` : "0 / 0";
  }
  if (node.pfImageEl) {
    const url = n ? imageViewUrl(infos[index]) : "";
    if (url && node.pfImageEl.src !== url) {
      node.pfImageEl.src = url;
    } else if (!url) {
      node.pfImageEl.removeAttribute("src");
    }
    node.pfImageEl.style.visibility = url ? "visible" : "hidden";
  }
  markNodeDirty(node);
}


function stepShowImages(node, delta) {
  const n = node.pfImageNames?.length || 0;
  if (!n) return;
  node.pfImageIndex = (Number(node.pfImageIndex) || 0) + delta;
  refreshShowImages(node);
}


function setupShowImages(node) {
  if (node.pfShowSetup) {
    applyShowFromOutputs(node, applyShowImages);
    return;
  }
  node.pfShowSetup = true;
  node.pfImageNames = node.pfImageNames || [];
  node.pfImageInfos = node.pfImageInfos || [];
  node.pfImageIndex = Number(node.pfImageIndex) || 0;

  node.addWidget("button", "Previous", null, () => stepShowImages(node, -1));
  node.addWidget("button", "Next", null, () => stepShowImages(node, 1));

  const fileWidget = node.addWidget(
    "combo",
    "file",
    "",
    () => {
      const idx = (node.pfImageNames || []).indexOf(fileWidget.value);
      if (idx >= 0) {
        node.pfImageIndex = idx;
        refreshShowImages(node);
      }
    },
    { values: [""] }
  );
  disableSerialize(fileWidget);
  node.pfFileWidget = fileWidget;

  const indexWidget = node.addWidget("text", "file_index", "0 / 0", () => {});
  indexWidget.disabled = true;
  disableSerialize(indexWidget);
  node.pfIndexWidget = indexWidget;

  const wrap = document.createElement("div");
  wrap.style.cssText =
    "width:100%;height:100%;display:flex;align-items:center;justify-content:center;overflow:hidden;background:#111;";
  const img = document.createElement("img");
  img.alt = "Queue a run to load images.";
  img.style.cssText =
    "max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;display:block;";
  img.addEventListener("load", () => markNodeDirty(node));
  wrap.appendChild(img);

  const imageHeight = () => leftoverHeight(node, domWidget, 160);
  const domWidget = node.addDOMWidget("image_preview", "image_preview", wrap, {
    hideOnZoom: false,
    getHeight: imageHeight,
    getMinHeight: imageHeight,
    getMaxHeight: imageHeight,
  });
  disableSerialize(domWidget);
  node.pfImageEl = img;
  node.pfPreviewWidget = domWidget;

  if (node.size[1] < 420) {
    node.setSize([Math.max(node.size[0], 380), 520]);
  }
  applyShowFromOutputs(node, applyShowImages);
}


function applyShowImages(node, message) {
  if (!message) return;
  node.pfImageNames = asList(message.names).map((v) => String(v));
  node.pfImageInfos = asList(message.pf_images);
  if (!node.pfImageNames.length) {
    node.pfImageIndex = 0;
  } else if (node.pfImageIndex >= node.pfImageNames.length) {
    node.pfImageIndex = 0;
  }
  refreshShowImages(node);
}


function applyShowFromOutputs(node, apply) {
  const message = nodeOutputMessage(node);
  if (message) apply(node, message);
  requestAnimationFrame(() => {
    const later = nodeOutputMessage(node);
    if (later) apply(node, later);
  });
}


function hookShowNode(nodeType, apply) {
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    onNodeCreated?.apply(this, arguments);
    if (apply === applyShowTexts) setupShowTexts(this);
    else setupShowImages(this);
  };

  const onExecuted = nodeType.prototype.onExecuted;
  nodeType.prototype.onExecuted = function (message) {
    onExecuted?.apply(this, arguments);
    apply(this, message);
    requestAnimationFrame(() => apply(this, message));
  };

  const onConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function () {
    onConfigure?.apply(this, arguments);
    requestAnimationFrame(() => applyShowFromOutputs(this, apply));
  };
}


app.registerExtension({
  name: "ComfyUI-ProductionFlow.TextBrowser",

  async setup() {
    api.addEventListener("executed", ({ detail }) => {
      const id = detail?.display_node ?? detail?.node;
      if (id == null) return;
      const graph = app.graph;
      const node =
        graph?.getNodeById?.(id) ||
        graph?.getNodeById?.(String(id)) ||
        graph?.getNodeById?.(Number(id));
      if (!node) return;
      if (node.comfyClass === "ProductionFlowShowTexts") {
        applyShowTexts(node, detail.output);
      } else if (node.comfyClass === "ProductionFlowShowImages") {
        applyShowImages(node, detail.output);
      }
    });
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "ProductionFlowTextFolderLoad") {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);
        setupFolderLoad(this, "text");
      };
    }

    if (nodeData.name === "ProductionFlowImageFolderLoad" || nodeData.name === "ProductionFlowImageFolderLoop") {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);
        setupFolderLoad(this, "image");
      };
    }

    if (nodeData.name === "ProductionFlowShowTexts") {
      hookShowNode(nodeType, applyShowTexts);
    }

    if (nodeData.name === "ProductionFlowShowImages") {
      hookShowNode(nodeType, applyShowImages);
    }
  },
});
