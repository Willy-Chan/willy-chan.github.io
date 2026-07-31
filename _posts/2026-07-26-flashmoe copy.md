---
layout: post
title: Notes on FlashMoE's Kernel Design
pdf: /pdfs/FlashMoE.pdf
---

These are my kernel visualizations of the FlashMoE megakernel that was released: this kernel does a really clever thing in that it treats the GPU like a virtual machine: many SMs run a dedicated "processor" function, while a single SM performs all the scheduling (and network) tasks. This, coupled with a clever lock-free symmetric memory design I'm still looking into to, allows you to achieve blazing fast performance with just some CUDA and NVSHMEM!

## "Teaching" Notes:

- Important: every SM is a processor, except for one which has a scheduler warp and a bunch of network subscriber warps.

- Refer to the "life of a tile" to see how data is processed as it comes in: this prevents SMs from being stalled for work!



## FlashMoE Kernel Design: