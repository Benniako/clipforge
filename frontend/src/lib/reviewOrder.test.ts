import { describe, it, expect } from "vitest";
import { orderReviewClips } from "./reviewOrder";

describe("orderReviewClips", () => {
  it("drops clips without an export and sorts by score desc", () => {
    const out = orderReviewClips([
      { id: "a", score: 50, export_url: "/a.mp4" },
      { id: "b", score: 90, export_url: null },
      { id: "c", score: 70, export_url: "/c.mp4" },
    ]);
    expect(out.map((c) => c.id)).toEqual(["c", "a"]);
  });

  it("keeps a deterministic order for equal scores across refreshes", () => {
    const first = orderReviewClips([
      { id: "x", score: 80, export_url: "/x.mp4" },
      { id: "y", score: 80, export_url: "/y.mp4" },
      { id: "z", score: 80, export_url: "/z.mp4" },
    ]);
    // Same clips arriving in a different order (a refresh) must order the same.
    const second = orderReviewClips([
      { id: "z", score: 80, export_url: "/z.mp4" },
      { id: "x", score: 80, export_url: "/x.mp4" },
      { id: "y", score: 80, export_url: "/y.mp4" },
    ]);
    expect(first.map((c) => c.id)).toEqual(["x", "y", "z"]);
    expect(second.map((c) => c.id)).toEqual(first.map((c) => c.id));
  });
});
