---
layout: post
title: Hyena Models, Convolutions, & Signal Processing
pdf: /pdfs/HyenaModels.pdf
---

Hyena models are really quite fascinating: they've been used in biology for next-sequence prediction and perform very well for long-context tasks. I took some notes, but the broad takeaway is their usage of both short and long convolutions as an attention replacement and sequence mixer! This way you mix channels via the MLP, short-range interactions via the shortconvs, and long-range attention-like interactions via the longconvs.

I also threw in a section on the Signal Processing "take" on convs as LTI systems, because I just find that perspective interesting. Feel free to skip if you're not interested.

## Hyena Model Notes: