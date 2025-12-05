---
layout: post
title: Notes from the Smol Training Playbook
tags: ["llm", "pretraining", "sft", "rlhf"]
mathjax: true
published: false
---

These are my notes from reading the [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) by HuggingFace. These are only meant for quick review and summary of concepts.

## Training compass

* Before training, you need to have a clear answer to *why you are training*:
  1. To answer a research question, e.g., does this new optimizer scale well?
  2. Production use case not achievable with prompting or fine-tuning existing models
  3. Strategic open-source, e.g., "the first small model with 1M context"
* What to train: model type, size, atchitecture, data mixture, etc.
*
