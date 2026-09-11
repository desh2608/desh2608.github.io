---
layout: post
title: "Dissecting GPT-Live"
subtitle: "What public information tells us about OpenAI's full-duplex model"
tags: ["speech", "full-duplex", "voice-agents", "llm"]
mathjax: true
published: true
css:
  - /static/posts/gpt-live-1/figures.css
js:
  - /static/posts/gpt-live-1/figures.js
---

*This post was jointly written by Desh and Claude, based on [Desh's notes](https://drive.google.com/file/d/1gYegqWkq98baazm1SlXBf4K8hsj4sSuS/view?usp=drive_link).*

OpenAI released [GPT-Live-1 in the API](https://openai.com/index/introducing-gpt-live-1-in-the-api/) yesterday, with launch partners already using it in production: [Hatch](https://x.com/usehatchapp/status/2098102253106118700) for customer service calls, [Yelp](https://x.com/Yelp/status/2098110294727999934) for restaurant reservations, and [Speak](https://x.com/speak/status/2098095986606551481) for language tutoring. It is the API version of the model that powers voice in ChatGPT, Codex, and Work.

OpenAI has not published anything about the architecture, but we can still infer a lot from the API, the pricing, the docs, and third-party evals. In this post, let us go through what we know and what we can guess.

<div class="glf-guide">
<p><strong>How to read this page.</strong> Everything in the main text is sourced, with links to the docs, tweets, and leaderboards it comes from. Our guesses are kept out of the main text: they appear as <span class="sn-tag">guess</span> notes in the right margin, anchored to the sentence they comment on. On a narrow screen, tap the <span style="color:#0e7a57">✦</span> marker to expand a note inline. Things we could not resolve from public information are called out in amber <strong style="color:#92400e">? Open question</strong> boxes where they come up.</p>
<p>The figures are interactive. Figure 1 and Figure 2 play a scripted sequence and can be stepped through; Figure 3 has sliders for the assumptions behind the pricing model; the benchmark charts link to their sources.</p>
</div>

<nav class="glf-toc">
<p class="glf-toc-title">Contents</p>
<ol>
<li><a href="#full-duplex">What does "full duplex" mean here?</a></li>
<li><a href="#frontend-and-backend">Why split the model into a frontend and a backend?</a> <span class="glf-toc-sub"><a href="#training-for-delegation">training</a> · <a href="#prompting-the-frontend">prompting</a></span></li>
<li><a href="#what-the-frontend-models">What is the frontend actually modeling?</a></li>
<li><a href="#latency-and-chunking">Why is the turn-taking latency 0.8 seconds?</a></li>
<li><a href="#can-we-infer-model-size-from-token-pricing">Can we infer model size from token pricing?</a></li>
<li><a href="#evals">How good is it, and how do you evaluate full-duplex voice agents?</a> <span class="glf-toc-sub"><a href="#τ-voice">τ-Voice</a> · <a href="#full-duplex-bench">Full-Duplex-Bench</a> · <a href="#evaluation-cookbook">evaluation cookbook</a></span></li>
</ol>
</nav>

## Full duplex

From the launch post, GPT-Live-1 is "a single model that reasons over incoming and outgoing audio together." [Kwindla Kramer](https://x.com/kwindla) calls it "the first production speech-to-speech model that can talk and listen at the same time."

The API reflects this. [Justin Uberti noted](https://x.com/juberti/status/2098105712987791624) that it is very different from the Realtime API. There are no turn or audio-done events. You stream audio in, audio streams out, and the model decides when to speak.

## Frontend and backend

The bigger change is that GPT-Live only handles the interaction. GPT-Realtime is a single model that does the conversation, the reasoning, and the tool calls. GPT-Live "can delegate reasoning and tool calls to a backend text model like GPT-6 Astra or a third-party model."

<div class="glf" id="glf-arch">
  <noscript>Animated figure: GPT-Realtime handles interaction, reasoning, and tools in one model; GPT-Live-1 handles only the interaction and delegates the rest to a backend.</noscript>
</div>

There are two [delegation modes](https://developers.openai.com/api/docs/guides/live-delegation):

1. **Responses delegation.** An OpenAI backend model gets the conversation context through the Responses API.
2. **Client delegation.** GPT-Live emits a delegation event and your code handles it, so you can route to your own agents.

Client delegation in particular is very powerful: it lets you add full-duplex behavior on top of an existing system of agents. Kwindla: "We can now build things like state machine conversation graphs, threaded/backtracking modes, and structured data input that were difficult with the first generation of speech-to-speech APIs."

This kind of frontend-backend separation is pretty much accepted as the correct way of building voice agents.<span class="sn"><button class="sn-ref" aria-label="guess">✦</button><span class="sn-body"><span class="sn-tag">guess</span>One could imagine the frontend being small enough to run on-device, talking to a large server-side backend.</span></span> What is new is that the behavior is trained into the model. Kwindla again: "It's great to see this pattern trained into a production model, so that we can leverage it natively at the prompt level."

Details from the docs:

- In client delegation, the frontend sends no task description to the backend. From the [delegation guide](https://developers.openai.com/api/docs/guides/live-delegation): "The delegation object contains metadata, not task text." You collect user and agent transcript deltas, and the backend infers the task from them. "A transcript fragment is not a complete user turn, and transcripts may contain mistakes."
- Backend results go into the frontend's context in [three ways](https://developers.openai.com/api/docs/guides/live-delegation): `session.instructions.append` ("Stop speaking about that request."), `session.commentary.append` for things to be spoken ("Your appointment is confirmed for Thursday at 2:00 PM"), and `session.thinking.append` for internal context ("Checking Thursday availability. No appointment has been booked.").
- Each append is [limited to 500 tokens](https://developers.openai.com/api/docs/guides/live-delegation). The frontend acknowledges each one with a corresponding `*.appended` event.
- The frontend verbalizes structured results itself. From the [delegation guide](https://developers.openai.com/api/docs/guides/live-delegation): "A concise tool result doesn't need an additional model call to rewrite it for speech."
- Images go to a "vision-capable backend", and typed exact values such as an order number go "to the backend that handles the task" ([delegation guide](https://developers.openai.com/api/docs/guides/live-delegation)). UI context is passed to the frontend only as a concise text summary of the current page. The frontend itself only handles voice and text.

### Training for delegation

The frontend has to learn three things:<span class="sn"><button class="sn-ref" aria-label="guess">✦</button><span class="sn-body"><span class="sn-tag">guess</span>How would you train this? In SFT, with sandboxed backends and scripted responses. In RL, with environments containing real or mocked backends, which is expensive.</span></span>

1. When to delegate a task to the backend
2. How to carry on the conversation while the task runs asynchronously
3. How to gracefully incorporate the backend response

> **Open question:** What did the training setup for delegation look like? In particular, how were asynchronous backends, with realistic latency and occasional bad responses, simulated at RL scale? This is the most valuable thing to know for anyone trying to build a comparable model.
{: .oq}

### Prompting the frontend

The [prompting guide](https://developers.openai.com/api/docs/guides/live-prompting) is mostly about turn-taking behavior. Through the frontend prompt you control response length, language and pronunciation of names (via IPA), interpreter mode, when to respond (e.g. only to certain kinds of queries), handling of background noise, and asking for clarifications.

## What the frontend models

**Audio tokens directly.** You can "shape an agent's tone, pace, and conversational style through the system prompt," e.g. "Speak warmly and naturally, at an unhurried pace." You cannot change the voice. There are 12 fixed voices (Quartz, Ripple, Vesper, Willow, Stone, Gleam, Meridian, Bossa, Tempo, Beacon, Delta, Cinder), disjoint from the 10 Realtime voices. This is consistent with a single model generating acoustic tokens rather than a TTS stage.<span class="sn"><button class="sn-ref" aria-label="guess">✦</button><span class="sn-body"><span class="sn-tag">guess</span>The promotion and evals focus on delegation and turn-taking, so most post-training effort probably went into the frontend's conversational behavior, with the audio encoder and vocoder kept close to GPT-Realtime's. The voice sets differ, so the vocoder was at least retrained. <a href="https://x.com/kundan2510/status/2098149959073935711">Kundan Kumar</a> from OpenAI: "we did specific training to further improve the controllability and instruction following of the model."</span></span>

> **Open question:** Did the audio encoder and vocoder change from GPT-Realtime? The completely new voice set says the vocoder was at least retrained. Nothing public says anything about the encoder.
{: .oq}

**User and agent text.** GPT-Live-1 "natively provides ASR transcripts and response text" and "supports keyword biasing." The model predicts user text as it hears audio, and generates agent text before agent audio. [We did the same for Moshi in our paper](https://arxiv.org/abs/2510.07497), which let us match S2S intelligence with T2T.

**A reasoning trace.** The backend can inject text through `session.thinking.append`, which is described as context for the model's internal reasoning. **This means the frontend also maintains a reasoning trace, in addition to the user and agent audio streams and the user and agent text streams.**

**Turn tokens.** "Although GPT-Live-1 is not a turn-based model, it natively supports turn detection." So it is trained with special tokens for user turn start and end.

Putting this together, Figure 2 shows what the frontend's sequence might look like, including a delegation. We have drawn the streams as separate rows for clarity, but the model would put them all in a single interleaved sequence, with special tokens to demarcate the boundaries. Note that the frontend keeps consuming user audio in chunks throughout, including while the backend is working.

<div class="glf" id="glf-stream">
  <noscript>Animated figure: the frontend's interleaved token stream. User audio arrives in chunks and is prefilled; the model emits user transcript, thinking, a delegation event, agent text, and agent audio in between; the backend result is prefilled into the context while user chunks keep arriving. A slider controls the chunk size.</noscript>
</div>

## Latency and chunking

Turn-taking latency on [Full-Duplex-Bench v1](https://github.com/DanielLin94144/Full-Duplex-Bench/tree/main/v1_v1.5/dataset) is 0.8 s. Moshi gets 0.25 s and PersonaPlex under 0.2 s. But those are multi-stream models operating on an 80 ms clock (see [my earlier article](https://x.com/rdesh26/article/2054954720024801280) on multi-stream v/s interleaved models). This suggests that GPT-Live-1 is operating on a larger clock.<span class="sn"><button class="sn-ref" aria-label="guess">✦</button><span class="sn-body"><span class="sn-tag">guess</span>Most likely it still uses 80 ms tokens, but prefills user audio in chunks of 2 to 5 tokens (160 to 400 ms) and decodes between chunks. TML uses 200 ms chunks, for comparison.</span></span>

Two things that are *not* evidence for the chunk size:

- The [eval harness](https://developers.openai.com/cookbook/examples/audio/voice_agent_evaluation) streams audio in 20 ms PCM appends. That is a transport choice. Audio is buffered on both sides.
- The RUN harness "builds a 400-millisecond playback reserve." In the code, this is a jitter buffer on the output side: GPT-Live's output audio deltas have no timestamps, so the relay holds up to 400 ms before forwarding. It says something about how bursty the output is, not about input chunking.

> **Open question:** What are the actual frame rate and chunk size? The 0.8 s turn-taking latency and the per-minute pricing both fit chunked prefill on an 80 ms clock, but a coarser tokenizer would fit too. A direct answer would settle most of the guesses in this post.
{: .oq}

## Can we infer model size from token pricing?

GPT-Live-1 costs $0.05 per minute, billed per second. The backend is billed separately, and rate limits are per concurrent session rather than per token. There is no per-token price for the frontend, but we can try to back one out. Some assumptions:

- The audio tokenizer runs at 12.5 Hz (80 ms frames), which is the most common rate for full-duplex models. The user and agent streams then produce 750 tokens per minute each.
- User audio is prefilled in chunks of 5 tokens, so one minute needs 150 prefills. Agent audio is decoded.
- Text (user transcript, thinking, agent text) adds about 1 token per 5 audio tokens, i.e. ~300 decoded tokens per minute.

That gives 750 prefilled and ~1000 decoded tokens per minute. Let us model the token pricing under these assumptions and see which GPT model would charge $0.05 for that mix. The sliders change the assumptions.

<div class="glf" id="glf-cost">
  <noscript>Interactive figure: a cost calculator. Sliders for frame rate, chunk size, and text ratio recompute the frontend's tokens per minute and compare the implied cost to each OpenAI model's list price at the same token mix.</noscript>
</div>

At 12.5 Hz, the implied price is close to GPT-5.5 / GPT-5.6 Sol. At 25 Hz, it is close to GPT-5.4. But serving an S2S model is more expensive than serving a text model of the same size:

- Prefix caching is a no-op. Every audio input is different.
- Prefills are tiny (a few tokens per chunk), so disaggregated prefill/decode buys little. A 5-token prefill is a decode step as far as the GPU is concerned.
- Requests cannot wait in the continuous-batching queue without hurting real-time latency.

So for 12.5 or 25 Hz, the frontend seems to be similar in size to GPT-5.6 Terra or GPT-5.4.<span class="sn"><button class="sn-ref" aria-label="guess">✦</button><span class="sn-body"><span class="sn-tag">guess</span>Probably initialized from the same backbone as Terra or GPT-5.4, with the price premium over those models paying for real-time serving.</span></span>

> **Open question:** How big is the frontend, really? The pricing argument above depends on the frame rate and on how much of the $0.05 is margin for reserved real-time capacity rather than compute. The gap between Sol-priced and Terra-sized is exactly the cost of real-time serving, and we do not know that number.
{: .oq}

Context also fills fast: ~2000 tokens per minute plus ~500 per backend call. The [prompting guide](https://developers.openai.com/api/docs/guides/live-prompting) notes that "the live model has a small context window."

## Evals

### τ-Voice

τ-Voice is built on τ²-bench's customer support tasks across retail, airline, and banking. It has 278 scenarios with a dynamic simulated user, and correctness is judged by comparing the final DB state to the expected state. OpenAI's launch post reports 86.2%. The [τ-bench leaderboard](https://taubench.com/leaderboard?benchmark=voice) and [Artificial Analysis](https://artificialanalysis.ai/speech-to-speech#agentic-performance-voice) report lower numbers:

<div class="glf" id="glf-tau">
  <noscript>Bar charts: τ-Voice results from the τ-bench leaderboard (GPT-Live-1 81.7%, Pine Voice Preview 75.4%, grok-voice-think-fast-1.0 67.3%, ...) and from Artificial Analysis (GPT-Live-1 with Astra 67.9%, with Sol 59.3%, ...).</noscript>
</div>

> **Open questions:**
>
> - Why is AA's number so different from the leaderboard and the blog? There is randomness from the dynamic user, but does that explain 20 points?
> - How much of the score is the backend? Astra-medium vs Sol-low is 8.6 points, even though the tasks are simple. How do Astra and Sol score on text τ²-bench, which is the source data? That would isolate the frontend's contribution.
> - What kind of errors improved over GPT-Realtime? Did delegation fix task execution, or did the interaction quality also change outcomes?
{: .oq}

### Full-Duplex-Bench

Artificial Analysis's [conversational dynamics index](https://artificialanalysis.ai/speech-to-speech#sts-quality-index) is a weighted average of pause handling, turn-taking, interruption handling, and backchannel handling on [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench/tree/main/v1_v1.5/dataset) v1 and v1.5 (Candor, ICC, and synthetic). This is saturated at the top. The FDB v1.5 interactivity subset (background speech, speech to another person, listener backchannels, interruptions), which OpenAI reports in the launch post, is more discriminative:

<div class="glf" id="glf-fdb">
  <noscript>Bar charts: Full-Duplex-Bench conversational dynamics from Artificial Analysis (Qwen Audio 3.0 Realtime Plus 98.4%, GPT-Live-1 with Sol 97.3%, ...) and FDB v1.5 interactivity from OpenAI (gpt-live-1 80.1%, gpt-realtime-2.1 45.4%, gpt-realtime-2 47.8%).</noscript>
</div>

Agent-directed speech is handled correctly: this is the clearest quantitative evidence that the model knows whether it is being spoken to.

### Evaluation cookbook

Anyone building evals for full-duplex voice agents should read OpenAI's [evaluation cookbook](https://developers.openai.com/cookbook/examples/audio/voice_agent_evaluation). They published the full [harness](https://github.com/openai/openai-cookbook/tree/main/examples/audio/duplex_voice_agent_evaluation) they use, same protocol as for GPT-Realtime. Three modes separate the failure sources:

<div class="glf" id="glf-evals">
  <noscript>Animated figure: the three evaluation modes. CRAWL streams synthetic TTS audio then continuous silence; WALK streams a recorded WAV with optional acoustic degradation then continuous silence; RUN connects two GPT-Live sessions in a live full-duplex conversation.</noscript>
</div>

The harness reports one primary metric, task completion, and a set of diagnostic metrics that explain *why* a task failed. The diagnostics split into task-completion metrics (did the agent reason, delegate, and call tools correctly?) and experience metrics (did it sound like a good conversation?). Hover over a metric for its definition.

<table class="mgrid">
<thead>
<tr><th></th><th>Primary metric</th><th>Diagnostic metrics</th></tr>
</thead>
<tbody>
<tr>
<td class="mrow">Task completion<small>Did the agent reason, delegate, and call tools correctly?</small></td>
<td><span class="mchip mprimary" tabindex="0" data-tip="Whether the final application state matches the expected state while satisfying mandatory constraints. Pass/fail. All modes.">Task completion</span></td>
<td>
<span class="mchip" tabindex="0" data-tip="LLM-judged score for task understanding, context fidelity, clarification quality, grounded communication, and conversational coherence. All modes.">Semantic quality</span>
<span class="mchip" tabindex="0" data-tip="Deterministic comparison of executed tools and arguments against expected calls. Missing, extra, or failed calls reduce the score. All modes.">Tool accuracy</span>
<span class="mchip" tabindex="0" data-tip="Actual vs. expected tool calls. Surfaces missing work, redundant calls, and unnecessary loops. All modes.">Tool calls</span>
<span class="mchip" tabindex="0" data-tip="Whether the assistant delegates when required, avoids delegation when forbidden, and may do either when optional. All modes.">Delegation accuracy</span>
<span class="mchip" tabindex="0" data-tip="Actual vs. expected frontend-to-backend handoffs. Multiple backend responses or tool calls within one delegation count as one. All modes.">Delegations</span>
<span class="mchip m-run" tabindex="0" data-tip="Actual vs. expected substantive turns, excluding backchannels. All modes, but most useful in RUN.">Conversation turns</span>
</td>
</tr>
<tr>
<td class="mrow">Experience<small>Did it sound like a good conversation?</small></td>
<td class="mnone">none</td>
<td>
<span class="mchip" tabindex="0" data-tip="Fraction of completed, acoustically eligible requests answered by the configured first-audio deadline. A spoken preamble counts; late, censored, and excluded requests are tracked separately. All modes.">Response rate</span>
<span class="mchip" tabindex="0" data-tip="Mean time from the end of audible caller speech to the first qualifying assistant audio, including a spoken preamble. Not time to task completion. All modes.">Response latency</span>
<span class="mchip m-cond" tabindex="0" data-tip="Observed assistant interruptions per response-eligible caller turn. Only defined when eligible caller turns occur.">Interruption rate</span>
<span class="mchip" tabindex="0" data-tip="Total audible assistant speech, and the largest speech total within one assistant turn. A verbosity check. All modes.">Speaking duration</span>
<span class="mchip m-cond" tabindex="0" data-tip="Total silence and the longest uninterrupted silent stretch while a delegation is active, excluding speech from either participant. Zero when no delegation is active.">Silence during delegation</span>
</td>
</tr>
<tr>
<td class="mrow">Usage<small>What did it cost?</small></td>
<td class="mnone">none</td>
<td>
<span class="mchip" tabindex="0" data-tip="Cumulative Live session duration from usage.seconds. The API does not report frontend token counts. All modes.">Frontend usage</span>
<span class="mchip m-cond" tabindex="0" data-tip="Delegated reasoning usage: input, output, reasoning, and caching tokens. Only when a delegation occurs.">Backend usage</span>
</td>
</tr>
</tbody>
</table>

<p class="mlegend"><span class="mchip mdemo">plain</span> applies in all modes &nbsp; <span class="mchip mdemo m-run">blue</span> most useful in RUN &nbsp; <span class="mchip mdemo m-cond">amber</span> only when a delegation or interruption occurs. Definitions condensed from the <a href="https://developers.openai.com/cookbook/examples/audio/voice_agent_evaluation">cookbook's metrics table</a>.</p>

## Links

- [Launch post](https://openai.com/index/introducing-gpt-live-1-in-the-api/)
- [Live API guide](https://developers.openai.com/api/docs/guides/live), [delegation](https://developers.openai.com/api/docs/guides/live-delegation), [prompting](https://developers.openai.com/api/docs/guides/live-prompting)
- [Model page](https://developers.openai.com/api/docs/models/gpt-live-1)
- [Evaluation cookbook](https://developers.openai.com/cookbook/examples/audio/voice_agent_evaluation) and [harness source](https://github.com/openai/openai-cookbook/tree/main/examples/audio/duplex_voice_agent_evaluation)
- [τ-Voice leaderboard](https://taubench.com/leaderboard?benchmark=voice), [Artificial Analysis](https://artificialanalysis.ai/speech-to-speech), [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)

---

*Some of the design elements on this page (margin notes for guesses, figures wider than the text column, and the note boxes) are inspired by Edward Yang's [interactive adaptation of "How to Parallelize a Transformer for Training"](https://ezyang.github.io/interactive-parallelize-transformer/).*
