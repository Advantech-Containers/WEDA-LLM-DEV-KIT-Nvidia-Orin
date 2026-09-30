# Jetson Orin memory guidance

The 8 GB Orin module has unified memory shared between CPU and GPU. A 4B
parameter model quantized to Q4_K_M occupies roughly 2.3 GiB, leaving headroom
for the KV cache and the operating system. Always stop a previous serving
container before starting a new one, because a stale server keeps its memory
allocated. If MemAvailable drops below 3000 MiB, restart the compose stack.
