"""Process-wide serialization for torch model loading.

Several providers load PyTorch models from different threads (the ASR loader,
the CLAP audio-event loader, the background resume worker). The CLAP loader has
to temporarily patch ``torch.load`` to tolerate older checkpoint formats, and a
global mutation like that is only safe if no other thread calls ``torch.load``
during the patch window. Routing every heavy torch load through this single
lock makes those windows mutually exclusive.
"""
from __future__ import annotations

import threading

# Shared by transcribe.py (ASR) and audio_events.py (CLAP). Keep it module-level
# so every importer gets the same lock object.
TORCH_LOAD_LOCK = threading.RLock()
