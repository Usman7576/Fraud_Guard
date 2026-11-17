"use client";

import dynamic from "next/dynamic";

import { StaticGlobe } from "./static-art";
import { use3DMode, useOnScreen } from "./use-3d";

const HeroScene = dynamic(() => import("./hero-scene"), { ssr: false, loading: () => <StaticGlobe /> });

/** The hero's globe: static SVG first, the WebGL scene once the browser is idle (if allowed). */
export function HeroArt() {
  const mode = use3DMode();
  const [ref, onScreen] = useOnScreen<HTMLDivElement>();
  return (
    <div ref={ref} className="size-full">
      {mode === "3d" ? <HeroScene active={onScreen} /> : <StaticGlobe />}
    </div>
  );
}
