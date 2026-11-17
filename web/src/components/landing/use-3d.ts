"use client";

import { useEffect, useRef, useState } from "react";

let webgl: boolean | undefined;

function hasWebGL(): boolean {
  if (webgl === undefined) {
    try {
      const c = document.createElement("canvas");
      webgl = !!(c.getContext("webgl2") ?? c.getContext("webgl"));
    } catch {
      webgl = false;
    }
  }
  return webgl;
}

export type Mode = "pending" | "3d" | "static";

/**
 * "3d" once the browser is idle, if WebGL works and the visitor has not asked for reduced motion;
 * otherwise "static". Until then "pending", so the server render and first paint show the static art
 * and the text never waits for three.js.
 */
export function use3DMode(): Mode {
  const [mode, setMode] = useState<Mode>("pending");
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const decide = () => setMode(mq.matches || !hasWebGL() ? "static" : "3d");
    const idle = window.requestIdleCallback?.(decide, { timeout: 1500 });
    const timer = idle === undefined ? window.setTimeout(decide, 200) : undefined;
    mq.addEventListener("change", decide);
    return () => {
      if (idle !== undefined) window.cancelIdleCallback(idle);
      if (timer !== undefined) window.clearTimeout(timer);
      mq.removeEventListener("change", decide);
    };
  }, []);
  return mode;
}

/** Whether the element is on (or near) the screen; canvases stop rendering frames when it is not. */
export function useOnScreen<T extends Element>(margin = "120px"): [React.RefObject<T | null>, boolean] {
  const ref = useRef<T>(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    if (!ref.current) return;
    const io = new IntersectionObserver(([e]) => setVisible(!!e?.isIntersecting), { rootMargin: margin });
    io.observe(ref.current);
    return () => io.disconnect();
  }, [margin]);
  return [ref, visible];
}

/** Device pixel ratio for canvases: at least 1, at most 2. */
export const DPR: [number, number] = [1, 2];

// Landing palette (dark surface). Scenes need raw colours; they mirror the .landing tokens.
export const COLORS = {
  background: "#07090d",
  accent: "#3987e5",
  accentBright: "#7fb2f5",
  critical: "#e66767",
  risk: "#f2a7a7",
  warning: "#fab219",
  node: "#5b6475",
  edge: "#2c3341",
} as const;
