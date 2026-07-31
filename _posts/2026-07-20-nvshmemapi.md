---
layout: post
title: The NVSHMEM API
pdf: /pdfs/nvshmemapi1.pdf
---

I made a tree diagram and overview of some of the most important parts of the NVSHMEM functional API. I couldn't find a satisfactory breakdown online: hopefully you can get some use out of this!

- nvshmem_{} denotes SHMEM-compatible functions
- nvshmemx_{} are NVIDIA/GPU-specific extensions
- nvshmemi_{} refers to the lower level APIs (typically users don't have to deal with these) that the above two function types use.

I highly recommend the demystifying NVSHMEM paper for a more comprehensive overview.

## NVSHMEM API Notes: