# ClipForge — Competitor Study (June 2026)

A focused look at the leading "long video → viral shorts" tools, and the concrete,
**locally-achievable** ideas worth borrowing. ClipForge's constraints are
deliberate: **100% local, no cloud APIs, explainable scoring.** So we only adopt
ideas that survive those constraints.

## The landscape

| Tool | Positioning | Standout |
| --- | --- | --- |
| **OpusClip** | Market leader, solo creators | Mature highlight detection, 0–99 *Virality Score*, ReframeAnything subject tracking, "AI B-roll", animated captions |
| **Submagic** | Caption-first, "viral look" | 30+ animated caption styles, word-by-word highlight, **auto emoji**, **auto B-roll**, **auto zoom punch-ins**, sound effects — the MrBeast/Hormozi look in one click |
| **Vizard** | Teams / agencies | Collaboration, brand kits, workspace management, multi-language |
| **Klap / Munch / Crayo / 2short** | Various | Templates, scheduling, hook libraries |
| **OSS** (SamurAIGPT AI-Youtube-Shorts, openshorts, ShortGPT, short-video-maker) | Self-host | LLM-on-transcript highlight picking; **none** combine OCR + audio-cue + per-speaker like ClipForge |

**Reality check on virality scores:** independent reviews repeatedly flag OpusClip's
0–99 score as *unreliable* — clips rated 40 sometimes outperform 85s, and
selection accuracy reportedly drops from ~60–70% (solo) to ~40% (multi-speaker).
This is genuinely a place ClipForge's **explainable, multi-signal, per-speaker**
approach can be *better*, not just cheaper — provided we lean into transparency.

## Feature-gap matrix

| Capability | OpusClip | Submagic | Vizard | **ClipForge today** |
| --- | :-: | :-: | :-: | :-: |
| Auto highlight detection | yes | yes | yes | yes (audio+OCR+LLM+PANNs) |
| Virality score | yes | partial | partial | yes, **explainable** |
| 9:16 subject reframe | yes | yes | yes | yes (face/YOLO/active-speaker) |
| Animated word-by-word captions | yes | yes+ | yes | yes (karaoke pop) |
| **Auto emoji in captions** | yes | yes | partial | no |
| **Auto B-roll** | yes | yes | partial | no |
| **Auto zoom / punch-in** | partial | yes | partial | partial (slow push-in only) |
| **Keyword emphasis (color/size)** | yes | yes | yes | partial (active word only) |
| Filler / silence removal | yes | yes | yes | yes (jump cuts) |
| Per-speaker handling | partial | partial | partial | yes, **toggle per speaker** |
| Multi-language captions | yes (25+) | yes (100+) | yes | partial (en/de tuned) |
| Direct publish / scheduling | yes | yes | yes | no (out of scope) |
| Local / private / no API | no | no | no | **yes — unique** |
| Learns your taste | partial | no | no | yes (local feedback) |

ClipForge already matches or beats the field on *detection breadth*, *explainability*,
*per-speaker control*, and *privacy*. The gaps are almost entirely in **caption
production value** and **dynamic visual editing** — the things that make a clip
*look* professionally edited.

## Top 8 ideas to borrow (locally achievable)

Rated **effort** (S/M/L) × **impact** (low/med/high).

1. **Auto zoom / punch-in on emphasis** — S × high. Submagic's signature. We already
   have keyframes + ffmpeg `zoompan`; trigger a quick punch-in at high-energy
   words (we already compute energy/emotion) or scene cuts. Biggest "looks edited"
   win per line of code.
2. **Keyword emphasis in captions** — S × high. Color/scale the *important* word, not
   just the spoken one. We have the transcript + signal lexicons (hook/emotion/
   payoff words) — reuse them to mark emphasis tokens in the ASS builder.
3. **Auto emoji on keywords** — S × med. A small keyword→emoji map (money 💰, fire 🔥,
   mind-blown 🤯) injected next to matched words. Pure, offline, tasteful cap of
   1–2 per line. Fits the existing caption pipeline.
4. **More caption style presets** — S × med. Submagic's edge is *choice*. We have a
   clean ASS engine; add 6–8 presets (Hormozi bold-yellow, MrBeast outline,
   minimal, TikTok bubble) — pure data in `styles.py`.
5. **Local auto B-roll from the source** — M × med. Full generative B-roll needs cloud;
   the *local* version is "smart cutaways" — detect the strongest visual moments
   (scene/motion) and cut to them over a continuing VO. Honest, offline, useful.
6. **Hook / first-3-seconds analysis** — M × med. Score the *opening* separately and
   warn "weak hook"; suggest a stronger on-screen hook line. We already have
   `hook_strength` — surface it as a first-3s gauge in the editor.
7. **Speaker-aware caption color** — S × med. We already attribute words to speakers;
   give each kept speaker a caption color (the multi-person podcast look). Tiny
   change in the ASS builder, leans on a strength competitors lack.
8. **"AI Boost" panel grouping** — S × med (UX). Submagic's one-panel "toggle the viral
   effects" (captions / silences / zoom / B-roll) is worth copying for discoverability.

## What makes competitor *captions* feel premium (implementable specifics)

- **Word-by-word with the active word emphasized** (color + slight scale) — we do this.
- **Keyword emphasis independent of timing** — the *meaningful* word is bigger/colored
  for the whole line, not just when spoken. (Idea #2)
- **Tasteful auto emoji** beside power words — 1–2 per line max. (Idea #3)
- **Tight word grouping** — 2–4 words on screen, big, centered in the safe zone. (done)
- **Punch-in zoom** synced to emphasis/cuts so the frame feels alive. (Idea #1)
- **Pop / scale-in animation** per word rather than a hard cut-in — a short ASS `\t`
  scale transition softens the entrance.
- **Consistent loudness + clean voice** so it *sounds* pro. (loudnorm + new Demucs)

## What makes their *UX* feel polished

- **One "AI Boost" panel** to toggle the viral effects.
- **At-a-glance ranking** — "Top pick" badges on the strongest clips. (added this pass)
- **Virality shown with the *why*** — our explainable factors are already a
  differentiator; lead with them harder in the card.
- **Instant preview** of caption-style changes without a full re-render (harder
  locally; a static thumbnail overlay preview is a cheap approximation).
- **Platform presets** that set aspect + caption style + length in one click.

## Recommended next slice (highest ROI, all local)

Ship **#1 punch-in zoom**, **#2 keyword emphasis**, **#3 auto emoji**, and **#4 more
presets** together as a "production value" pass — they share the caption/render
pipeline, are all S-effort, and collectively close most of the *perceived* gap with
Submagic/OpusClip while staying 100% local and explainable.

---

### Sources
- [Opus Clip review (2026) — ScaleReach](https://www.scalereach.ai/blog/opus-clip-review)
- [Opus Clip vs Descript vs Submagic vs Captions — Forasoft](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/opus-clip-descript-submagic-captions-ai-video-editor-tools-2026)
- [Submagic review 2026 — Max Productive](https://max-productive.ai/ai-tools/submagic/)
- [Submagic AI B-Roll](https://www.submagic.co/features/b-roll) · [Submagic Magic Clips](https://www.submagic.co/features/magic-clips)
- [Vizard vs OpusClip vs quso.ai — Transcriptr](https://transcriptr.ai/compare/vizard-vs-opusclip-vs-quso)
- [Top AI clipping tools 2026 — Reap](https://reap.video/reports/state-of-top-ai-video-clipping-tools-2026)
