"""Pluggable providers for the AI-shaped pipeline stages.

Each stage that *could* be swapped for a hosted model or a different vendor
(transcription, moment detection, scoring) lives behind a small function here.
The defaults run fully locally so the core loop works with no API keys; richer
providers can be slotted in without touching the orchestrator.
"""
