---
layout: post
title: Some Interesting Multi-GPU Problems Pt. 1
pdf: /pdfs/pkb_problem_visualizations.pdf
---

*Note*: Testing a new visualization framework: the idea is to step through code by using "blocks" to visualize the most important lines. Here are a few ParallelKernelBench problems I present in this new format - I found the process of creating the notes to be very informative in getting me to understand the problems better!


## "Teaching" Notes:

- I start with a review of Data Parallelism and building up ZeRO from scratch. Increasing sharding of different things, just like the PKB problem (in the future, we might want to emphasize memory savings rather than wall clock speedup).

- That transitions into an explanation of Ring Attention: the problem category which immediately follows. The notes on flashattention really helped me out here by clarifying the core "5 things" (QKV, running/local max, and running/local sum). Might work on this a bit more.

- Then I explain the highlighted "net new" PKB problems with intuitive visualizations, trying to show what the tensors look like exactly. Might be better if I could step through it like a slideshow, but this is the best I can do with some static notes. I explain the vocab parallel top-k problem, Hyena Context Parallel, and SAM3 IoU suppression.

## ParallelKernelBench Problem Visualizations: