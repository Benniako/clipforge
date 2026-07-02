import type { Health } from "./types";

type CapabilityMap = Health["capabilities"];
type SummaryLabels = {
  synthetic?: string;
  gpuDetected?: string;
  ocrOff?: string;
};

export function capabilityReadiness(c: CapabilityMap) {
  const checks = [
    Boolean(c.ffmpeg),
    Boolean(c.ffprobe),
    c.transcription !== "synthetic",
    Boolean(c.url_import),
  ];
  const count = checks.filter(Boolean).length;
  const total = checks.length;
  return { count, total, pct: total > 0 ? Math.round((count / total) * 100) : 0 };
}

export function capabilitySummary(c: CapabilityMap, labels: SummaryLabels = {}) {
  const engine =
    c.transcription === "whisperx" ? "WhisperX" : c.transcription === "whisper" ? "Whisper" : labels.synthetic ?? "Synthetic";
  const compute = c.gpu_encode ? "NVENC" : c.gpu ? "CUDA" : c.gpu_detected ? labels.gpuDetected ?? "GPU detected" : "CPU";
  const ocr = c.ocr ? `OCR ${c.ocr}` : labels.ocrOff ?? "OCR off";
  return [
    `${engine}${c.diarization ? "+DZ" : ""}`,
    compute,
    c.audio_events ? "Audio" : "",
    ocr,
  ].filter(Boolean).join(" · ");
}
