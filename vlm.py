import base64
import gc
import io
import logging
import os
import re
from dataclasses import dataclass

import folder_paths
import numpy as np
import torch
from PIL import Image

import comfy.model_management
import comfy.sd
import comfy.utils


logger = logging.getLogger("ProductionFlow.VLM")

# Track live GGUF sessions so we can free CUDA memory that Comfy's
# "Clear VRAM" does not know about (llama-cpp is outside ModelPatcher).
_active_gguf_sessions: list = []

SAFETENSORS_EXTS = (".safetensors", ".sft")
GGUF_EXTS = (".gguf",)

# Model search roots relative to Comfy models dir
VLM_SUBDIRS = (
    "text_encoders",
    "clip",
    os.path.join("LLM", "GGUF"),
    "LLM",
)


@dataclass
class VlmModelInfo:
    label: str
    path: str
    backend: str  # "comfy" | "gguf"
    clip_type: str = "krea2"
    mmproj_path: str | None = None


def _models_dir():
    return folder_paths.models_dir


def _is_mmproj(name: str) -> bool:
    n = name.lower()
    return "mmproj" in n or n.startswith("mmproj")


def _clip_type_for_safetensors(filename: str) -> str:
    name = filename.lower()
    if "8b" in name or "8B" in filename:
        return "ideogram4"
    if "gemma" in name:
        return "stable_diffusion"  # unused for gemma native; placeholder
    return "krea2"


def _find_mmproj(model_path: str) -> str | None:
    root = os.path.dirname(model_path)
    base = os.path.splitext(os.path.basename(model_path))[0].lower()
    # strip common quant suffixes for fuzzy match
    base_stem = re.sub(
        r"[-_]?(?:q[2-8]_[a-z0-9_]+|iq[2-4]_[a-z0-9_]+|f16|bf16|fp16|q6_k_p|q5_k_p|q4_k_p|q8_0|q6_k|q5_k_m|q4_k_m)+$",
        "",
        base,
        flags=re.I,
    )
    candidates = []
    if not os.path.isdir(root):
        return None
    for fname in os.listdir(root):
        if not fname.lower().endswith(".gguf"):
            continue
        if not _is_mmproj(fname):
            continue
        full = os.path.join(root, fname)
        fl = fname.lower()
        score = 0
        if base_stem and base_stem[:20] in fl:
            score += 2
        if "qwen3.5" in base and "qwen3.5" in fl:
            score += 3
        if "qwen3.5" in base and "qwen3" in fl and "3.5" in fl:
            score += 3
        if "gemma-4" in base or "gemma4" in base.replace("-", ""):
            if "gemma-4" in fl or "gemma4" in fl.replace("-", ""):
                score += 3
        if "e4b" in base and "e4b" in fl:
            score += 1
        if "9b" in base and "9b" in fl:
            score += 1
        candidates.append((score, full))
    if not candidates:
        # single mmproj in folder
        mm = [os.path.join(root, f) for f in os.listdir(root) if _is_mmproj(f) and f.lower().endswith(".gguf")]
        return mm[0] if len(mm) == 1 else None
    candidates.sort(key=lambda x: (-x[0], x[1].lower()))
    if candidates[0][0] <= 0:
        return candidates[0][1] if len(candidates) == 1 else None
    return candidates[0][1]


def scan_vlm_models() -> list[VlmModelInfo]:
    found: list[VlmModelInfo] = []
    seen = set()
    models_dir = _models_dir()

    for sub in VLM_SUBDIRS:
        root = os.path.join(models_dir, sub)
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                lower = fname.lower()
                full = os.path.join(dirpath, fname)
                if full in seen:
                    continue
                if lower.endswith(SAFETENSORS_EXTS):
                    # skip unrelated TEs if clearly not VL — still list qwen/gemma/heretic/vl
                    if not any(k in lower for k in ("qwen", "vl", "gemma", "heretic", "krea")):
                        continue
                    seen.add(full)
                    rel = os.path.relpath(full, models_dir).replace("\\", "/")
                    found.append(
                        VlmModelInfo(
                            label=f"[TE] {rel}",
                            path=full,
                            backend="comfy",
                            clip_type=_clip_type_for_safetensors(fname),
                        )
                    )
                elif lower.endswith(GGUF_EXTS):
                    if _is_mmproj(fname):
                        continue
                    seen.add(full)
                    rel = os.path.relpath(full, models_dir).replace("\\", "/")
                    mmproj = _find_mmproj(full)
                    found.append(
                        VlmModelInfo(
                            label=f"[GGUF] {rel}",
                            path=full,
                            backend="gguf",
                            mmproj_path=mmproj,
                        )
                    )

    found.sort(key=lambda m: m.label.lower())
    return found


def vlm_model_labels() -> list[str]:
    models = scan_vlm_models()
    if not models:
        return ["(no VLM models found)"]
    return [m.label for m in models]


def get_model_by_label(label: str) -> VlmModelInfo:
    for m in scan_vlm_models():
        if m.label == label:
            return m
    raise FileNotFoundError(f"VLM model not found: {label}")


def tensor_to_pil(image) -> Image.Image:
    # Comfy IMAGE: [B,H,W,C] float 0-1
    if isinstance(image, torch.Tensor):
        t = image
    else:
        t = torch.as_tensor(image)
    if t.ndim == 4:
        t = t[0]
    arr = (t.detach().cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def pil_to_data_url(pil: Image.Image, max_side: int = 1280) -> str:
    img = pil.convert("RGB")
    w, h = img.size
    scale = min(1.0, float(max_side) / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# Qwen3 / 3.5 thinking markers (template tags and free-form traces)
_THINK_TAG_RE = re.compile(
    r"<think>.*?</think\s*>|"
    r"</?think>|"
    r"<\|?think\|?>.*?<\|?/think\|?>|"
    r"<channel>.*?</channel>",
    re.DOTALL | re.IGNORECASE,
)
# Colon may be followed by space OR newline (HauhauCS often uses "Thinking Process:\n")
_THINK_HEADER_RE = re.compile(
    r"^\s*(Thinking Process|Reasoning|Thoughts|Analysis|Chain of Thought|Inner Monologue)\s*:?\s*",
    re.IGNORECASE,
)
_FINAL_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*(?:Final Answer|Final Response|Answer|Response|Output|Result)\s*:\s*",
    re.IGNORECASE,
)


def strip_thinking_text(text: str) -> str:
    """Remove Qwen-style thinking blocks; keep the final answer only."""
    if not text:
        return text

    text = _THINK_TAG_RE.sub("", text).strip()

    # Free-form "Thinking Process: ..." traces (HauhauCS / some GGUF templates)
    if _THINK_HEADER_RE.match(text):
        parts = _FINAL_ANSWER_RE.split(text)
        if len(parts) >= 2 and parts[-1].strip():
            text = parts[-1].strip()
        else:
            # Prefer last non-reasoning paragraph
            chunks = re.split(r"\n\s*\n", text)
            kept = None
            for chunk in reversed(chunks):
                c = chunk.strip()
                if not c:
                    continue
                if _THINK_HEADER_RE.match(c):
                    continue
                if re.match(r"^(\d+\.|[-*]|\*\*|Step\s+\d+|Task:|Constraint:)", c, re.I):
                    continue
                kept = c
                break
            if kept:
                text = kept
            else:
                # Entire completion was thinking — nothing useful left
                text = ""

    return text.strip()


def apply_no_think_to_messages(messages: list, enable_thinking: bool, model_path: str = "") -> list:
    """Steer Qwen templates away from thinking when enable_thinking is False."""
    if enable_thinking:
        return messages

    path_l = (model_path or "").lower()
    is_qwen = "qwen" in path_l
    # /no_think is the Qwen3 / 3.5 template switch; plain English alone is often ignored.
    no_think_suffix = " /no_think" if is_qwen else ""

    out = []
    if is_qwen:
        out.append(
            {
                "role": "system",
                "content": (
                    "You answer directly and concisely. "
                    "Do not output reasoning, thinking process, analysis steps, or chain-of-thought. "
                    "Output only the final answer. /no_think"
                ),
            }
        )

    for msg in messages:
        m = dict(msg)
        content = m.get("content")
        if isinstance(content, str):
            if no_think_suffix and "/no_think" not in content:
                m["content"] = content.rstrip() + no_think_suffix
        elif isinstance(content, list):
            # Multimodal: append /no_think to the last text part
            new_parts = list(content)
            for i in range(len(new_parts) - 1, -1, -1):
                part = new_parts[i]
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text") or ""
                    if no_think_suffix and "/no_think" not in t:
                        new_parts[i] = {**part, "text": t.rstrip() + no_think_suffix}
                    break
            else:
                if no_think_suffix:
                    new_parts.append({"type": "text", "text": no_think_suffix.strip()})
            m["content"] = new_parts
        out.append(m)
    return out


def _vram_free_bytes() -> int | None:
    """Driver-reported free VRAM (not just torch's cache)."""
    if not torch.cuda.is_available():
        return None
    try:
        free_b, _total_b = torch.cuda.mem_get_info()
        return int(free_b)
    except Exception:
        try:
            return int(comfy.model_management.get_free_memory())
        except Exception:
            return None


def _purge_torch_cuda():
    """Return as much VRAM as possible to the driver for llama-cpp cudaMalloc."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        # Drop dead model refs first so empty_cache can reclaim
        comfy.model_management.cleanup_models_gc()
    except Exception:
        pass
    try:
        comfy.model_management.soft_empty_cache(force=True)
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass
    # Second pass: cudaMallocAsync often keeps pools until another sync+empty
    gc.collect()
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:
        pass


def _free_comfy_vram(reason: str = ""):
    """Unload Comfy-managed models so llama-cpp / TE can allocate."""
    try:
        logger.info("Freeing Comfy models%s", f" ({reason})" if reason else "")
        comfy.model_management.unload_all_models()
    except Exception as e:
        logger.warning("Comfy unload_all_models failed: %s", e)
    _purge_torch_cuda()
    free_b = _vram_free_bytes()
    if free_b is not None:
        logger.info("VRAM free after Comfy unload: %.2f GiB", free_b / (1024**3))


def _free_all_gguf_sessions(except_session=None):
    dead = []
    for s in list(_active_gguf_sessions):
        if s is except_session:
            continue
        try:
            s.unload()
        except Exception as e:
            logger.warning("GGUF unload failed: %s", e)
        dead.append(s)
    for s in dead:
        if s in _active_gguf_sessions:
            _active_gguf_sessions.remove(s)


def _is_oom_error(err: BaseException) -> bool:
    msg = str(err).lower()
    return any(
        s in msg
        for s in (
            "out of memory",
            "cudaMalloc failed",
            "failed to allocate",
            "oom",
            "insufficient memory",
            "ggml_gallocr",
        )
    )


def _llama_has_gpu_offload() -> bool:
    try:
        from llama_cpp import llama_supports_gpu_offload

        return bool(llama_supports_gpu_offload())
    except Exception:
        return False


def _pick_n_gpu_layers(requested: int, needs_vision: bool) -> int:
    """
    Leave headroom for the vision encoder graph (~4.5+ GiB on Qwen3.5 mmproj).
    Full Q6_K 9B weights ~6.2 GiB; together they need a mostly-empty 16 GiB card.
    """
    if requested == 0:
        return 0
    free_b = _vram_free_bytes()
    if free_b is None:
        return requested

    free_gib = free_b / (1024**3)
    # Reserve space for vision compute buffer + KV + fragmentation
    vision_reserve = 5.2 if needs_vision else 1.0
    # Rough weight budget if fully on GPU (Q6 9B-class); smaller models just use less
    weight_budget = 7.0
    usable_for_weights = free_gib - vision_reserve - 0.6

    if usable_for_weights >= weight_budget:
        n = requested if requested > 0 else -1
        logger.info(
            "VRAM plan: free=%.2f GiB, full GPU offload (layers=%s, vision=%s)",
            free_gib,
            n,
            needs_vision,
        )
        return n

    if usable_for_weights < 1.5:
        logger.warning(
            "VRAM plan: free=%.2f GiB too low for GPU VLM (need ~%.1f GiB free); "
            "using CPU (slow).",
            free_gib,
            weight_budget + vision_reserve,
        )
        return 0

    # Partial offload: ~0.2 GiB per layer for this class of model
    layers = max(4, min(32, int(usable_for_weights / 0.22)))
    if requested > 0:
        layers = min(layers, requested)
    logger.warning(
        "VRAM plan: free=%.2f GiB — partial offload n_gpu_layers=%s "
        "(leaves ~%.1f GiB for vision graph). Prefer clearing other models first.",
        free_gib,
        layers,
        free_gib - layers * 0.22,
    )
    return layers


class ComfyVlmSession:
    def __init__(self, info: VlmModelInfo):
        self.info = info
        self.clip = None

    def unload(self):
        self.clip = None
        gc.collect()
        _purge_torch_cuda()

    def load(self, pbar=None):
        if self.clip is not None:
            return self
        if pbar is not None:
            pbar.update_absolute(0, total=3)
        # Drop foreign GGUF CUDA allocations + free Comfy models first
        _free_all_gguf_sessions()
        _free_comfy_vram("before TE VLM load")
        logger.info("Loading TE VLM: %s", self.info.path)
        clip_type = getattr(comfy.sd.CLIPType, self.info.clip_type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        self.clip = comfy.sd.load_clip(
            ckpt_paths=[self.info.path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
        )
        try:
            self.clip.load_model()
        except Exception as e:
            logger.debug("clip.load_model preload skipped: %s", e)
        if pbar is not None:
            pbar.update_absolute(1, total=3)
        return self

    def generate(
        self,
        prompt,
        image=None,
        max_tokens=512,
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        seed=0,
        enable_thinking=False,
    ):
        pbar = comfy.utils.ProgressBar(3)
        try:
            self.load(pbar=pbar)

            te_prompt = prompt
            if not enable_thinking and "qwen" in self.info.path.lower():
                if "/no_think" not in te_prompt:
                    te_prompt = te_prompt.rstrip() + " /no_think"

            logger.info("Tokenizing / encoding image for TE VLM...")
            tokens = self.clip.tokenize(
                te_prompt,
                image=image,
                skip_template=False,
                min_length=1,
                thinking=bool(enable_thinking),
            )
            pbar.update_absolute(2, total=3)

            logger.info("Generating up to %s tokens (UI progress from token loop)...", max_tokens)
            generated_ids = self.clip.generate(
                tokens,
                do_sample=True,
                max_length=max_tokens,
                temperature=max(temperature, 0.01),
                top_k=top_k,
                top_p=top_p,
                min_p=0.05,
                repetition_penalty=1.05,
                seed=seed,
                presence_penalty=0.0,
            )
            pbar.update_absolute(3, total=3)
            text = self.clip.decode(generated_ids)
            if not enable_thinking:
                text = strip_thinking_text(text)
            return text
        finally:
            # TE weights are managed by Comfy, but drop our ref so free-memory can reclaim.
            self.unload()


class GgufVlmSession:
    def __init__(self, info: VlmModelInfo, n_gpu_layers=-1, n_ctx=4096):
        self.info = info
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.llm = None
        self._base_chat_handler = None
        self._enable_thinking = False
        self._loaded_n_gpu = None
        self._loaded_n_ctx = None
        self._loaded_needs_vision = None

    def unload(self):
        """Fully release llama-cpp CUDA/host memory."""
        if self.llm is not None:
            logger.info("Unloading GGUF VLM: %s", self.info.path)
            try:
                self.llm.close()
            except Exception as e:
                logger.warning("Llama.close failed: %s", e)
            try:
                del self.llm
            except Exception:
                pass
            self.llm = None
        # MTMD / clip handlers hold GPU context too
        handler = self._base_chat_handler
        self._base_chat_handler = None
        if handler is not None:
            try:
                if hasattr(handler, "_exit_stack") and handler._exit_stack is not None:
                    handler._exit_stack.close()
            except Exception as e:
                logger.warning("chat handler cleanup failed: %s", e)
            try:
                if hasattr(handler, "mtmd_ctx"):
                    handler.mtmd_ctx = None
            except Exception:
                pass
            try:
                del handler
            except Exception:
                pass
        self._loaded_n_gpu = None
        self._loaded_n_ctx = None
        self._loaded_needs_vision = None
        if self in _active_gguf_sessions:
            _active_gguf_sessions.remove(self)
        _purge_torch_cuda()

    def load(self, needs_vision: bool = True, n_gpu_layers=None, n_ctx=None, force: bool = False):
        n_ctx = int(n_ctx if n_ctx is not None else self.n_ctx)
        requested_gpu = self.n_gpu_layers if n_gpu_layers is None else n_gpu_layers

        # Reload if vision requirements / offload plan changed
        if self.llm is not None and not force:
            if (
                self._loaded_needs_vision == needs_vision
                and self._loaded_n_ctx == n_ctx
                and self._loaded_n_gpu == requested_gpu
            ):
                return self
            self.unload()

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "GGUF VLM requires llama-cpp-python in the ComfyUI venv.\n"
                "Install: pip install llama-cpp-python"
            ) from e

        if needs_vision and (
            not self.info.mmproj_path or not os.path.isfile(self.info.mmproj_path)
        ):
            raise FileNotFoundError(
                f"No mmproj GGUF found next to {self.info.path}. "
                "Place the matching mmproj-*.gguf in the same folder."
            )

        # Critical: llama-cpp is invisible to Comfy's VRAM manager. Always clear
        # other GGUF sessions and Comfy models before allocating.
        _free_all_gguf_sessions(except_session=self)
        _free_comfy_vram("before GGUF VLM load")

        has_gpu = _llama_has_gpu_offload()
        if not has_gpu:
            n_gpu = 0
            logger.warning(
                "llama-cpp-python has no GPU offload (CPU build). "
                "GGUF vision will be very slow. Prefer [TE] Qwen3-VL safetensors, "
                "or install a CUDA build of llama-cpp-python."
            )
        else:
            n_gpu = _pick_n_gpu_layers(requested_gpu, needs_vision=needs_vision)

        if not has_gpu:
            n_ctx = min(n_ctx, 4096)
        # Cap context: large n_ctx + vision graphs are a common OOM after other workflows
        n_ctx = min(int(n_ctx), 4096 if needs_vision else 8192)
        # Smaller batches cut llama compute buffers (secondary; vision graph is the big one)
        n_batch = 256 if needs_vision else 512

        logger.info(
            "Loading GGUF VLM %s (vision=%s, n_gpu_layers=%s, n_ctx=%s, n_batch=%s, gpu_offload=%s)",
            self.info.path,
            needs_vision,
            n_gpu,
            n_ctx,
            n_batch,
            has_gpu,
        )

        def _construct(ctx, layers, with_vision):
            chat_handler = None
            if with_vision:
                self._base_chat_handler = self._make_chat_handler(self.info.mmproj_path)
                chat_handler = self._wrap_chat_handler(self._base_chat_handler)
            else:
                self._base_chat_handler = None
            kwargs = dict(
                model_path=self.info.path,
                n_ctx=ctx,
                n_batch=n_batch,
                n_ubatch=min(n_batch, 256),
                n_gpu_layers=layers,
                logits_all=False,
                embedding=False,
                verbose=False,
            )
            if chat_handler is not None:
                kwargs["chat_handler"] = chat_handler
            return Llama(**kwargs)

        try:
            self.llm = _construct(n_ctx, n_gpu, needs_vision)
        except Exception as e:
            logger.warning("GGUF load failed (%s); retrying after hard free + fewer layers...", e)
            self.unload()
            _free_all_gguf_sessions()
            _free_comfy_vram("GGUF load retry")
            retry_layers = 0 if n_gpu == 0 else max(8, (n_gpu if n_gpu > 0 else 24) // 2)
            if not has_gpu:
                retry_layers = 0
            self.llm = _construct(min(n_ctx, 2048 if needs_vision else 4096), retry_layers, needs_vision)
            n_gpu = retry_layers
            n_ctx = min(n_ctx, 2048 if needs_vision else 4096)

        self._loaded_n_gpu = n_gpu
        self._loaded_n_ctx = n_ctx
        self._loaded_needs_vision = needs_vision
        if self not in _active_gguf_sessions:
            _active_gguf_sessions.append(self)
        return self

    def _wrap_chat_handler(self, handler):
        """Inject enable_thinking into the GGUF jinja chat template (Qwen3.5)."""
        session = self

        def wrapped(**kwargs):
            # Template: {% if enable_thinking is defined and enable_thinking is false %}
            # emits empty <think></think> and skips open-ended reasoning.
            kwargs["enable_thinking"] = bool(session._enable_thinking)
            return handler(**kwargs)

        return wrapped

    def _make_chat_handler(self, mmproj_path: str):
        import llama_cpp.llama_chat_format as formats

        path_l = self.info.path.lower()
        preferred = []
        # MTMD is the modern path for Qwen3.5 / Gemma4 multimodal GGUFs
        if "gemma" in path_l:
            preferred.extend(["Gemma4ChatHandler", "MTMDChatHandler"])
        elif "qwen3.5" in path_l or "qwen35" in path_l or "qwen3_5" in path_l:
            preferred.extend(["MTMDChatHandler", "Qwen25VLChatHandler"])
        elif "qwen" in path_l:
            preferred.extend(["Qwen25VLChatHandler", "MTMDChatHandler"])
        else:
            preferred.extend(["MTMDChatHandler", "Gemma4ChatHandler", "Qwen25VLChatHandler"])

        seen = set()
        names = []
        for n in preferred:
            if n not in seen:
                seen.add(n)
                names.append(n)

        last_err = None
        for name in names:
            cls = getattr(formats, name, None)
            if cls is None:
                continue
            try:
                handler = cls(clip_model_path=mmproj_path)
                logger.info("Using chat handler %s", name)
                return handler
            except TypeError:
                try:
                    handler = cls(mmproj_path)
                    logger.info("Using chat handler %s", name)
                    return handler
                except Exception as e:
                    last_err = e
                    logger.warning("chat handler %s failed: %s", name, e)
            except Exception as e:
                last_err = e
                logger.warning("chat handler %s failed: %s", name, e)

        raise RuntimeError(
            f"No multimodal chat handler could load mmproj for {self.info.path}. "
            f"Tried: {', '.join(names)}. Last error: {last_err}"
        )

    def _build_messages(self, prompt, image, enable_thinking, max_side: int):
        if image is not None:
            pil = tensor_to_pil(image)
            content = [
                {"type": "image_url", "image_url": {"url": pil_to_data_url(pil, max_side=max_side)}},
                {"type": "text", "text": prompt},
            ]
            messages = [{"role": "user", "content": content}]
            mode = "vision+text"
        else:
            messages = [{"role": "user", "content": prompt}]
            mode = "text-only"

        messages = apply_no_think_to_messages(
            messages,
            enable_thinking=enable_thinking,
            model_path=self.info.path,
        )
        return messages, mode

    def _stream_completion(self, messages, max_tokens, temperature, top_p, top_k, seed, pbar):
        stream = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            top_k=top_k if top_k > 0 else None,
            seed=int(seed) if seed else None,
            stream=True,
        )

        parts = []
        step = 0
        for chunk in stream:
            try:
                delta = chunk["choices"][0].get("delta") or {}
                piece = delta.get("content") or ""
            except (KeyError, IndexError, TypeError):
                piece = ""
            if piece:
                parts.append(piece)
                step += 1
                pbar.update_absolute(min(step, max_tokens), total=max_tokens)

        pbar.update_absolute(max_tokens, total=max_tokens)
        return "".join(parts)

    def generate(
        self,
        prompt,
        image=None,
        max_tokens=512,
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        seed=0,
        enable_thinking=False,
    ):
        pbar = comfy.utils.ProgressBar(max(1, int(max_tokens)))
        pbar.update_absolute(0)

        needs_vision = image is not None
        # Controls jinja chat template (see tokenizer.chat_template in GGUF)
        self._enable_thinking = bool(enable_thinking)

        # Cap generation hard on CPU so a run cannot look infinite
        has_gpu = _llama_has_gpu_offload()
        if not has_gpu:
            max_tokens = min(int(max_tokens), 384)
            pbar = comfy.utils.ProgressBar(max_tokens)

        # Attempt plan: full → smaller image → fewer GPU layers → even leaner
        attempts = []
        if needs_vision:
            attempts = [
                dict(max_side=768, n_gpu_layers=self.n_gpu_layers, n_ctx=min(int(self.n_ctx), 4096)),
                dict(max_side=512, n_gpu_layers=self.n_gpu_layers, n_ctx=2048),
                dict(max_side=448, n_gpu_layers=16 if (self.n_gpu_layers < 0 or self.n_gpu_layers > 16) else self.n_gpu_layers, n_ctx=2048),
                dict(max_side=384, n_gpu_layers=8 if (self.n_gpu_layers < 0 or self.n_gpu_layers > 8) else max(0, self.n_gpu_layers), n_ctx=2048),
            ]
        else:
            attempts = [
                dict(max_side=768, n_gpu_layers=self.n_gpu_layers, n_ctx=min(int(self.n_ctx), 8192)),
                dict(max_side=768, n_gpu_layers=16 if (self.n_gpu_layers < 0 or self.n_gpu_layers > 16) else self.n_gpu_layers, n_ctx=4096),
            ]

        last_err = None
        try:
            for i, plan in enumerate(attempts):
                try:
                    self.load(
                        needs_vision=needs_vision,
                        n_gpu_layers=plan["n_gpu_layers"],
                        n_ctx=plan["n_ctx"],
                        force=(i > 0),
                    )
                    messages, mode = self._build_messages(
                        prompt, image, enable_thinking, max_side=plan["max_side"]
                    )
                    logger.info(
                        "GGUF generate starting (%s, thinking=%s, max_tokens=%s, "
                        "max_side=%s, n_gpu=%s, n_ctx=%s, attempt=%s/%s)...",
                        mode,
                        enable_thinking,
                        max_tokens,
                        plan["max_side"],
                        self._loaded_n_gpu,
                        self._loaded_n_ctx,
                        i + 1,
                        len(attempts),
                    )
                    text = self._stream_completion(
                        messages, max_tokens, temperature, top_p, top_k, seed, pbar
                    )
                    if not enable_thinking:
                        text = strip_thinking_text(text)
                    logger.info("GGUF generate finished (%s chars)", len(text))
                    return text
                except Exception as e:
                    last_err = e
                    oom = _is_oom_error(e)
                    logger.warning(
                        "GGUF generate attempt %s failed%s: %s",
                        i + 1,
                        " (OOM)" if oom else "",
                        e,
                    )
                    self.unload()
                    _free_all_gguf_sessions()
                    _free_comfy_vram("after GGUF generate failure")
                    # Non-OOM: one free+retry only; leaner plans won't fix template bugs
                    if not oom and i >= 1:
                        raise
                    # OOM / first failure: continue to next leaner plan
            raise RuntimeError(
                f"GGUF VLM failed after {len(attempts)} attempts (likely VRAM). "
                f"Last error: {last_err}"
            ) from last_err
        finally:
            # Always release llama-cpp VRAM after the run so other Comfy workflows
            # (and the next VLM load) are not fighting a hidden CUDA allocation.
            self.unload()


def load_vlm_session(label: str, n_gpu_layers=-1, n_ctx=4096):
    """Return a session handle. GGUF is not pre-loaded (loads on generate, unloads after)."""
    info = get_model_by_label(label)
    if info.backend == "comfy":
        # TE uses Comfy model management; load lazily on generate as well
        return ComfyVlmSession(info)
    if info.backend == "gguf":
        return GgufVlmSession(info, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
    raise ValueError(f"Unknown backend: {info.backend}")
