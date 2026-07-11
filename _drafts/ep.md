
## pplx-Kernels

What does pplx-kernels do differently?

## MegaMoE and FlashDMoE
These are not optimized dispatch kernel implementations: these are more like MEGAKERNELS that do everything at once!


# Why NCCL EP?

There are already a lot of MoE communication libraries out there, e.g. DeepSeek's DeepEP or Hybrid-EP. These all operate outside of NCCL and create complex deployments outside of the NVIDIA ecosystem.

- NCCL EP is just an implementation of these EP libraries, just using the standard NCCL GIN Device API, adding vendor support!
- Literally just a library that handles the dispatch and combine operations.

There is a unified API:

- `ncclEpDispatch` and `ncclEpCombine`
  - Each has:
    - Low Latency mode: 
      - Inference decode
      - Small batch sizes of 1-128 tokns
    - High Throughput mode: 
      - training and inference prefill
      - Large batch sizes of 4096+ tokens
- Replaces NVSHMEM and IBGDA kernels with standard NCCL LSA (intra-node) + GIN (inter-node)

It has better buffer management compared to DeepEP, allowing for up to 14x less memory allocation (TODO: understand).