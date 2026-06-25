/** Simple undo/redo history hook for the clip editor.

Tracks snapshots of editor state and allows navigating back/forward.
History is bounded to prevent memory issues with long editing sessions.

Usage:
  const { state, set, undo, redo, canUndo, canRedo, reset } = useUndo(initialState);

  set({ ...state, title: "new title" });  // pushes to history
  undo();  // reverts to previous state
  redo();  // goes forward again
*/
import { useCallback, useRef, useState } from "react";

const MAX_HISTORY = 50;

export interface UndoState {
  title: string;
  start: number;
  end: number;
  styleId: string;
  cx: number | null;
  words: { t: number; d: number; text: string }[];
  layout: string;
  cam: { x: number; y: number; w: number; h: number } | null;
  aspect: string;
  capSpeakers: number[] | null;
}

export function useUndo(initial: UndoState) {
  const [state, setState] = useState<UndoState>(initial);
  const past = useRef<UndoState[]>([]);
  const future = useRef<UndoState[]>([]);
  const ticking = useRef(false);
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);

  const sync = useCallback(() => {
    setCanUndo(past.current.length > 0);
    setCanRedo(future.current.length > 0);
  }, []);

  const push = useCallback((next: UndoState) => {
    past.current = [...past.current.slice(-(MAX_HISTORY - 1)), state];
    future.current = [];
    setState(next);
    sync();
  }, [state, sync]);

  const set = useCallback((next: UndoState) => {
    // Batch rapid changes (e.g. slider drags) — only record every 300ms.
    if (ticking.current) {
      setState(next);
      return;
    }
    ticking.current = true;
    push(next);
    setTimeout(() => { ticking.current = false; }, 300);
  }, [push]);

  const undo = useCallback(() => {
    const prev = past.current.pop();
    if (!prev) return state;
    future.current = [...future.current, state];
    setState(prev);
    sync();
    return prev;
  }, [state, sync]);

  const redo = useCallback(() => {
    const next = future.current.pop();
    if (!next) return state;
    past.current = [...past.current, state];
    setState(next);
    sync();
    return next;
  }, [state, sync]);

  const reset = useCallback((newState: UndoState) => {
    past.current = [];
    future.current = [];
    setState(newState);
    sync();
  }, [sync]);

  return {
    state,
    set,
    undo,
    redo,
    canUndo,
    canRedo,
    reset,
    // Direct setter without history (for initial hydration).
    setSilent: setState,
  };
}
