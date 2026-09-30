# ==============================================================================
# WISE-Edge WEDA LLM Suite — jp6 / Jetson Orin (8 GB UMA)
# Base: WEDA-LLM Platform (Orin)
# ------------------------------------------------------------------------------
# The bundle's default operating model runs the PLATFORM image with this
# directory mounted read-only over /wise-edge/app (see docker-compose.yml) --
# edit on the host, restart the service, nothing baked in. That still works and
# is unchanged.
#
# This image bakes the same tree in, for the cases a mount cannot serve:
# air-gapped fleets that receive one artefact, and CI, which wants a single
# immutable thing to deploy and bench. Compose still mounts over these paths
# when it is driven from a checkout, so a baked file is the fallback, not an
# override.
#
# The base lives in weda-release ONLY -- the rolling weda-ai project carries the
# jp7 Thor bases but not the Orin one (build 31867 failed on exactly that), so
# this default does not track the branch the way the jp7 demos' do.
# ==============================================================================

ARG BASE_IMAGE=harbor.edgesync.cloud/weda-release/weda-llm-base:1.0.0-orin
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="WISE-Edge WEDA LLM Suite (jp6/Orin)" \
      org.opencontainers.image.description="llama.cpp / GGUF LLM + RAG suite for Jetson Orin" \
      com.advantech.layer="demo" \
      com.advantech.type="application"

WORKDIR /wise-edge/app

# Bake the tree the compose file otherwise mounts. Paths match its bind targets
# exactly, so a mounted checkout shadows like-for-like instead of half-shadowing.
COPY config/      /wise-edge/app/config/
COPY scripts/     /wise-edge/app/scripts/
COPY docs_sample/ /wise-edge/app/docs/
COPY entrypoint.sh /wise-edge/app/entrypoint.sh

# rag_store and test_logs are named/bind volumes at runtime; create them so the
# image is runnable bare (air-gapped) without compose supplying them.
RUN mkdir -p /wise-edge/app/rag_store /wise-edge/app/test_logs \
    && chmod +x /wise-edge/app/entrypoint.sh \
    && if [ -f /wise-edge/app/scripts/wise-bench ]; then \
         chmod +x /wise-edge/app/scripts/wise-bench; \
       fi

# Mirrors .env's defaults. HF_HUB_OFFLINE stays 0: the suite downloads its GGUF
# weights on first start via scripts/download_models.py, and pinning it to 1
# here would turn a fresh container into a hard failure rather than a fetch.
ENV WEDA_LLM_PORT=9000 \
    WEDA_EMBED_PORT=9001 \
    WEDA_GATEWAY_PORT=9002 \
    HF_HUB_OFFLINE=0

# generation, embedding, gateway
EXPOSE 9000 9001 9002

ENTRYPOINT ["/wise-edge/app/entrypoint.sh"]
CMD ["serve"]
