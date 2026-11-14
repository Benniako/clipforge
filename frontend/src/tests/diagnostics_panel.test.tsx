import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import DiagnosticsPanel from "../components/DiagnosticsPanel";
import { api } from "../lib/api";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("DiagnosticsPanel", () => {
  it("says why an installed engine is not working", async () => {
    vi.spyOn(api, "capabilities").mockResolvedValue({
      flat: {} as never,
      detail: {
        categories: [
          {
            name: "audio",
            items: [
              { key: "emotion", available: true, label: "emotion2vec/FunASR", impact: "Voice excitement.",
                problem: "The model could not be loaded: HTTP 403." },
              { key: "clap", available: true, label: "CLAP", impact: "Audio cues.", problem: null },
              { key: "demucs", available: false, label: "Demucs", impact: "Voice isolation." },
            ],
          },
        ],
      },
    });
    render(<DiagnosticsPanel onClose={() => {}} />);

    expect(await screen.findByText("The model could not be loaded: HTTP 403.")).toBeTruthy();
    expect(screen.getByText("läuft nicht")).toBeTruthy();         // emotion: installed but failing
    expect(screen.getByText("verfügbar")).toBeTruthy();           // CLAP works
    expect(screen.getByText("fehlt")).toBeTruthy();               // Demucs not installed
    expect(screen.getByTitle("The model could not be loaded: HTTP 403.").className).toContain("warn");
  });
});
