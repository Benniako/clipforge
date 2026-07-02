import { describe, expect, it } from "vitest";
import { capabilityReadiness, capabilitySummary } from "./capabilities";
import type { Health } from "./types";

const baseCaps: Health["capabilities"] = {
  ffmpeg: false,
  ffprobe: false,
  transcription: "synthetic",
  diarization: false,
  ocr: false,
  vad: false,
  scene_detect: false,
  emotion: false,
  denoise: false,
  audio_events: false,
  panns_audio: false,
  clap_audio: false,
  reframe_engine: "haar",
  active_speaker: false,
  face_tracking: false,
  url_import: true,
  gpu: false,
  gpu_detected: true,
  gpu_encode: false,
  device: "cpu",
  llm: false,
  llm_model: null,
  vlm: false,
  vlm_model: null,
  whisper_model: "small",
  diarization_model: null,
  auto_model: true,
  vram_gb: 15.9,
  cpu: 12,
  recommended_power_mode: "max_gpu",
  deno: false,
  ollama: false,
  ollama_models: "",
  torchaudio: false,
  paddleocr: false,
  easyocr: false,
  tesseract: false,
};

describe("capability badge helpers", () => {
  it("scores core readiness instead of every optional detector flag", () => {
    expect(capabilityReadiness(baseCaps)).toEqual({ count: 1, total: 4, pct: 25 });
  });

  it("shows detected GPU hardware even when CUDA runtime is not ready", () => {
    expect(capabilitySummary(baseCaps)).toContain("GPU detected");
    expect(capabilitySummary(baseCaps)).not.toContain("CPU");
  });
});
