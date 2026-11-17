"use client";

import { useEffect, useRef, useState } from "react";

let webgl: boolean | undefined;

// Probing creates a WebGL context, which is expensive (over a second at 4x CPU throttling on
// Windows/ANGLE), so it runs once, and only when a scene is about to load.
function hasWebGL(): boolean {
  if (webgl === undefined) {
    try {
      const c = document.createElement("canvas");
      const gl = c.getContext("webgl2") ?? c.getContext("webgl");
      webgl = !!gl;
      gl?.getExtension("WEBGL_lose_context")?.loseContext();
    } catch {
      webgl = false;
    }
  }
  return webgl;
}

/** True once `wanted` and WebGL works; false when it does not; null until checked. */
export function useWebGL(wanted: boolean): boolean | null {
  const [ok, setOk] = useState<boolean | null>(null);
  useEffect(() => {
    if (wanted && ok === null) setOk(hasWebGL());
  }, [wanted, ok]);
  return wanted ? ok : null;
}

export type Mode = "pending" | "3d" | "static";

/**
 * "3d" (motion allowed) once the browser is idle, unless the visitor prefers reduced motion
 * ("static"). Until then "pending", so the server render and first paint show the SVG art and the
 * text never waits for three.js. Whether WebGL works is checked per scene with `useWebGL`.
 */
export function use3DMode(): Mode {
  const [mode, setMode] = useState<Mode>("pending");
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const decide = () => setMode(mq.matches ? "static" : "3d");
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

/**
 * `onScreen`: the element is on (or just off) the screen, so its canvas should render frames.
 * `near`: it has come within about a screen of the viewport at least once, so its scene may load.
 * Scenes far down the page therefore cost nothing until the visitor scrolls towards them.
 */
export function useViewport<T extends Element>(): { ref: React.RefObject<T | null>; onScreen: boolean; near: boolean } {
  const ref = useRef<T>(null);
  const [onScreen, setOnScreen] = useState(false);
  const [near, setNear] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const visible = new IntersectionObserver(([e]) => setOnScreen(!!e?.isIntersecting), { rootMargin: "120px" });
    const approach = new IntersectionObserver(
      ([e]) => {
        if (e?.isIntersecting) {
          setNear(true);
          approach.disconnect();
        }
      },
      { rootMargin: "100% 0px" },
    );
    visible.observe(el);
    approach.observe(el);
    return () => {
      visible.disconnect();
      approach.disconnect();
    };
  }, []);
  return { ref, onScreen, near };
}

/** Device pixel ratio for canvases: at least 1, at most 2. */
export const DPR: [number, number] = [1, 2];

/**
 * Phones, tablets and machines with few cores: starting three.js and rendering the globe there
 * blocks the main thread for seconds (measured at 4x CPU throttling), so the hero stays SVG.
 */
export function isLowPower(): boolean {
  if (typeof window === "undefined") return false;
  const coarse = window.matchMedia("(pointer: coarse)").matches;
  const cores = navigator.hardwareConcurrency || 4;
  return coarse || window.innerWidth < 768 || cores <= 4;
}

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
