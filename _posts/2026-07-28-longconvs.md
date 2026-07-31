---
layout: post
title: FFT & Long Convolutions
pdf: /pdfs/fft_longconvs.pdf
math: true
---

In the [previous post on SSMs](/2026/07/11/ssms/), we saw that models like S4 unroll into a convolution. That leaves a practical question: how do you actually run long convolutions fast on modern GPUs?

This post is about the FFT, moving between coefficient and value form. The key idea is that a degree-\(n\) polynomial is uniquely determined by \(n+1\) points. Treat two coefficient sequences as polynomials of degrees \(n\) and \(m\); their product is then uniquely determined by \(n+m+1\) points, and that product is exactly a 1-D convolution of the coefficients.

See below for this really cool connection!

## Convolutions + the FFT: