FROM vllm/vllm-openai:latest
# Pin vllm 0.22.0: 0.22.1 has a _merge_multimodal_embeddings bug for MiniCPM-V-4.6 (is_embed mask ignored).
# Pin transformers to 5.10.2 (gemma4_unified + MiniCPMV4_6 support). Do NOT `--upgrade` to bleeding edge:
# transformers 5.11.0 rewrote the MiniCPM-V processor, producing more image-placeholder tokens than
# vllm 0.22.0's resampler emits -> "Attempted to assign N multimodal tokens to M placeholders" engine crash.
# Install vllm first so the explicit transformers pin wins dependency resolution.
RUN pip install "vllm==0.22.0" && pip install "transformers==5.10.2"
