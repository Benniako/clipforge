import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useUndo, type UndoState } from "./useUndo";

const base: UndoState = {
  title: "Clip",
  start: 0,
  end: 10,
  styleId: "bold-pop",
  cx: null,
  words: [],
  layout: "center",
  cam: null,
  aspect: "",
  capSpeakers: null,
  loopPreviewSeconds: 0,
};

describe("useUndo", () => {
  beforeEach(() => { vi.useFakeTimers(); });
  afterEach(() => { vi.useRealTimers(); });

  it("starts with canUndo/canRedo false", () => {
    const { result } = renderHook(() => useUndo(base));
    expect(result.current.canUndo).toBe(false);
    expect(result.current.canRedo).toBe(false);
    expect(result.current.state).toEqual(base);
  });

  it("pushes to history on set and reverts on undo", () => {
    const { result } = renderHook(() => useUndo(base));
    act(() => result.current.set({ ...base, title: "Edited" }));
    expect(result.current.state.title).toBe("Edited");
    expect(result.current.canUndo).toBe(true);

    let prev: UndoState | undefined;
    act(() => { prev = result.current.undo(); });
    expect(prev?.title).toBe("Clip");
    expect(result.current.state.title).toBe("Clip");
    expect(result.current.canUndo).toBe(false);
    expect(result.current.canRedo).toBe(true);
  });

  it("reapplies on redo", () => {
    const { result } = renderHook(() => useUndo(base));
    act(() => result.current.set({ ...base, title: "Edited" }));
    act(() => result.current.undo());
    expect(result.current.state.title).toBe("Clip");
    act(() => result.current.redo());
    expect(result.current.state.title).toBe("Edited");
    expect(result.current.canRedo).toBe(false);
  });

  it("clears the redo stack when a new edit follows an undo", () => {
    const { result } = renderHook(() => useUndo(base));
    act(() => result.current.set({ ...base, title: "A" }));
    act(() => vi.advanceTimersByTime(350)); // past the 300ms batching window
    act(() => result.current.set({ ...base, title: "B" }));
    act(() => vi.advanceTimersByTime(350));
    act(() => result.current.undo()); // back to A, redo has B
    expect(result.current.canRedo).toBe(true);
    act(() => vi.advanceTimersByTime(350)); // past batching before new edit
    act(() => result.current.set({ ...base, title: "C" })); // new edit
    expect(result.current.canRedo).toBe(false); // B is gone
    act(() => result.current.redo()); // nothing to redo
    expect(result.current.state.title).toBe("C");
  });

  it("undo/redo are no-ops at the history boundary", () => {
    const { result } = renderHook(() => useUndo(base));
    act(() => result.current.undo()); // nothing to undo
    expect(result.current.state).toEqual(base);
    act(() => result.current.redo()); // nothing to redo
    expect(result.current.state).toEqual(base);
  });

  it("reset clears both stacks and sets new state", () => {
    const { result } = renderHook(() => useUndo(base));
    act(() => result.current.set({ ...base, title: "A" }));
    act(() => result.current.set({ ...base, title: "B" }));
    act(() => result.current.reset({ ...base, title: "Reset" }));
    expect(result.current.state.title).toBe("Reset");
    expect(result.current.canUndo).toBe(false);
    expect(result.current.canRedo).toBe(false);
  });

  it("setSilent updates state without recording history", () => {
    const { result } = renderHook(() => useUndo(base));
    act(() => result.current.setSilent!({ ...base, title: "Silent" }));
    expect(result.current.state.title).toBe("Silent");
    expect(result.current.canUndo).toBe(false); // no history entry
  });
});
