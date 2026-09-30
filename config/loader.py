import os
from dataclasses import dataclass, field, fields
from typing import List

import yaml

VALID_KINDS = ("chat", "agent", "rag", "rag-verified")
# names only; implementations live in scripts/agent_demo.py
VALID_TOOLS = ("get_time", "get_device_stats", "calculator", "search_documents",
               "get_power_modes", "set_power_mode", "free_memory",
               "get_network_info",
               "get_telemetry_history", "get_recent_alerts", "get_action_history",
               "check_anomalies", "remember_note", "recall_notes",
               "explain_recent_errors", "generate_report",
               "list_containers", "inspect_container", "container_stats")


@dataclass
class LLMProfile:
    backend: str                       # llamacpp (C++ llama-server) | python (llama_cpp.server)
    hf_repo: str
    gguf_file: str
    revision: str = ""                 # pinned commit of hf_repo; empty follows the repo's default branch
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    parallel: int = 1
    jinja: bool = False
    # a model too small to emit native tool_calls reliably echoes the tool
    # schema as text instead; profiles set this false to keep agent apps off
    tool_calls: bool = True
    flash_attn: bool = False
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    extra_serve_args: List[str] = field(default_factory=list)
    model_cache_dir: str = "/models/huggingface"
    name: str = ""


@dataclass
class EmbeddingConfig:
    hf_repo: str
    gguf_file: str
    revision: str = ""
    pooling: str = "mean"
    n_ctx: int = 2048
    n_gpu_layers: int = -1
    model_cache_dir: str = "/models/huggingface"


@dataclass
class AppConfig:
    kind: str
    profile: str = ""          # empty = follow the config's default_profile
    system_prompt: str = ""
    tools: List[str] = field(default_factory=list)
    collection: str = ""
    top_k: int = 0
    chunk_chars: int = 0
    chunk_overlap: int = 0
    docs_dir: str = "."
    endpoints: dict = field(default_factory=dict)
    sampling: dict = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    # retrieval: dense + BM25 fused (hybrid), optional cross-encoder rerank,
    # and for kind rag-verified the number of retry rounds the graph may take
    hybrid: bool = True
    rerank: bool = False
    verify_rounds: int = 1
    # False disables the model's reasoning phase per request
    # (chat_template_kwargs enable_thinking; honored by Qwen3-class templates)
    thinking: bool = True
    # True when the block names a profile; an unpinned app follows whatever
    # profile is being served (profile then only holds the config default)
    pinned: bool = False
    name: str = ""

    def generation_base(self) -> str:
        return (self.endpoints or {}).get(
            "generation", f"http://127.0.0.1:{os.environ.get('WEDA_LLM_PORT', '9000')}/v1")

    def embeddings_base(self) -> str:
        return (self.endpoints or {}).get(
            "embeddings", f"http://127.0.0.1:{os.environ.get('WEDA_EMBED_PORT', '9001')}/v1")

    def needs_embeddings(self) -> bool:
        return self.kind in ("rag", "rag-verified") or "search_documents" in (self.tools or [])


def _config_path(config_path=None):
    return config_path or os.path.join(os.path.dirname(__file__), "llm_config.yaml")


def _load(config_path=None):
    with open(_config_path(config_path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def _filtered_init(cls, raw: dict, label: str):
    """Warn on unknown keys (typo net, forward-compatible), then construct."""
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    for k in sorted(unknown):
        print(f"[config] WARN: {label}: unknown key '{k}' ignored")
    try:
        return cls(**{k: v for k, v in raw.items() if k in known})
    except TypeError:
        required = {f.name for f in fields(cls)
                    if f.default is f.default_factory}  # no default either way
        missing = sorted(k for k in required - set(raw) if k != "name")
        raise ValueError(f"{label}: missing required key(s) {missing}") from None


KV_TYPES = ("f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl")


def _validate_profile(p: "LLMProfile"):
    """Range/enum checks a typo would otherwise carry into llama-server
    (which exits) or into the fit check (which would guess)."""
    for key in ("n_ctx", "parallel", "n_gpu_layers"):
        v = getattr(p, key)
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"profile '{p.name}': {key} must be an integer, got {v!r}")
    if p.n_ctx < 512:
        raise ValueError(f"profile '{p.name}': n_ctx must be >= 512, got {p.n_ctx}")
    if p.parallel < 1:
        raise ValueError(f"profile '{p.name}': parallel must be >= 1, got {p.parallel}")
    for key in ("cache_type_k", "cache_type_v"):
        v = getattr(p, key)
        if v not in KV_TYPES:
            raise ValueError(f"profile '{p.name}': {key} '{v}' invalid (valid: {', '.join(KV_TYPES)})")
    if p.backend not in ("llamacpp", "python"):
        raise ValueError(f"profile '{p.name}': backend '{p.backend}' invalid (valid: llamacpp, python)")


def load_llm_config(profile_name: str = None, config_path: str = None) -> LLMProfile:
    return _profile_from_data(_load(config_path), profile_name)


def _profile_from_data(data: dict, profile_name: str = None) -> LLMProfile:
    profile_name = profile_name or data.get("default_profile")
    if not profile_name:
        raise ValueError("No profile_name provided and no default_profile set in config.")
    profiles = data.get("profiles", {})
    if profile_name not in profiles:
        raise KeyError(
            f"Profile '{profile_name}' not found. Available: {', '.join(sorted(profiles))}")
    p = dict(profiles[profile_name])
    if "model_cache_dir" in data:
        p.setdefault("model_cache_dir", data["model_cache_dir"])
    p["name"] = profile_name
    prof = _filtered_init(LLMProfile, p, f"profile '{profile_name}'")
    _validate_profile(prof)
    return prof


def load_embedding_config(config_path: str = None) -> EmbeddingConfig:
    data = _load(config_path)
    emb = data.get("embedding")
    if not emb:
        raise KeyError("No 'embedding' section in llm_config.yaml — RAG needs one.")
    e = dict(emb)
    if "model_cache_dir" in data:
        e.setdefault("model_cache_dir", data["model_cache_dir"])
    return _filtered_init(EmbeddingConfig, e, "embedding")


def load_monitoring_config(config_path=None):
    """Optional monitoring: block -- recorder interval and alert thresholds.
    Absent keys fall back to the recorder's built-in defaults."""
    return _load(config_path).get("monitoring") or {}


def load_reranker_config(config_path: str = None) -> dict:
    """Cross-encoder used when an app sets rerank: true. Repo and cache dir,
    with the bundle defaults."""
    data = _load(config_path)
    r = dict(data.get("reranker") or {})
    r.setdefault("hf_repo", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    r.setdefault("model_cache_dir", data.get("model_cache_dir", "/models/huggingface"))
    return r


def list_apps(config_path: str = None) -> List[str]:
    return sorted((_load(config_path).get("applications") or {}))


def load_app(app_name: str, config_path: str = None) -> AppConfig:
    """Resolve an application block; every failure is a config error with a
    named remedy, raised BEFORE anything talks to a server."""
    data = _load(config_path)
    apps = data.get("applications") or {}
    if not apps:
        raise KeyError("No 'applications' section in llm_config.yaml — add one "
                       "(see the shipped blocks) to use app.sh.")
    if app_name not in apps:
        raise KeyError(f"Application '{app_name}' not found. "
                       f"Available: {', '.join(sorted(apps))}")
    raw = dict(apps[app_name])
    # rag defaults fill only unset retrieval keys
    for k, v in (data.get("rag") or {}).items():
        raw.setdefault(k, v)
    raw["name"] = app_name
    raw["pinned"] = bool(raw.get("profile"))
    app = _filtered_init(AppConfig, raw, f"application '{app_name}'")
    if app.chunk_chars and app.chunk_overlap >= app.chunk_chars:
        raise ValueError(f"application '{app_name}': chunk_overlap ({app.chunk_overlap}) "
                         f"must be smaller than chunk_chars ({app.chunk_chars})")
    if not app.profile:
        app.profile = data.get("default_profile") or ""
        if not app.profile:
            raise ValueError(f"application '{app_name}': no profile and no "
                             "default_profile in config")

    if app.kind not in VALID_KINDS:
        raise ValueError(f"application '{app_name}': kind '{app.kind}' invalid "
                         f"(valid: {', '.join(VALID_KINDS)})")
    unknown_tools = [t for t in (app.tools or []) if t not in VALID_TOOLS]
    if unknown_tools:
        raise ValueError(f"application '{app_name}': unknown tool(s) {unknown_tools} "
                         f"(registry: {', '.join(VALID_TOOLS)}). New tools are code — "
                         "add them in scripts/agent_demo.py, never in config.")
    profile = _profile_from_data(data, app.profile)
    if app.kind == "agent" and profile.backend != "llamacpp":
        raise ValueError(f"application '{app_name}': kind 'agent' needs a "
                         f"backend: llamacpp profile — '{profile.name}' is "
                         f"'{profile.backend}', which cannot emit tool_calls.")
    _SAMPLING_KEYS = {"temperature", "top_p", "top_k", "min_p", "max_tokens",
                      "repeat_penalty"}
    bad_sampling = set(app.sampling or {}) - _SAMPLING_KEYS
    if bad_sampling:
        raise ValueError(f"application '{app_name}': unknown sampling key(s) "
                         f"{sorted(bad_sampling)} (valid: {sorted(_SAMPLING_KEYS)})")
    if app.kind == "agent" and not app.tools:
        print(f"[config] WARN: application '{app_name}' is kind agent with no "
              "tools — it will behave as chat")
    return app


if __name__ == "__main__":
    p = load_llm_config()
    print(f"Loaded default profile: {p.name} ({p.backend})")
    for a in list_apps():
        app = load_app(a)
        print(f"  app {a}: kind={app.kind} profile={app.profile} tools={app.tools}")
