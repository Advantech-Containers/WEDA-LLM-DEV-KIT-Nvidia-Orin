# WISE-Edge platform overview

WISE-Edge is Advantech's edge-AI container platform for NVIDIA Jetson
devices. Applications run from prebuilt, hardware-verified container images
published through the Advantech registry, so a device goes from image pull
to serving without compiling anything on-site.

On JetPack 6 (Orin) devices the LLM serving path uses llama.cpp with GGUF
quantized models, because vLLM requires more than 8 GB of memory. On JetPack 7
(Thor) devices the serving path is vLLM.

The generation server exposes an OpenAI-compatible API on port 9000. RAG
deployments additionally run an embedding server on port 9001 using the
nomic-embed-text model with mean pooling. Applications — chat, document
Q&A, device assistant — are published by the application gateway on port
9002, where each is selectable as a model.
