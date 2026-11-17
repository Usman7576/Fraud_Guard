"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

import { StaticGlobe } from "./static-art";
import { isLowPower, use3DMode, useViewport, useWebGL } from "./use-3d";

const HeroScene = dynamic(() => import("./hero-scene"), { ssr: false, loading: () => <StaticGlobe animated /> });

/**
 * The hero's globe. The SVG version paints first. Laptops and desktops then get the WebGL scene
 * once the browser is idle; phones and low-power machines keep the SVG, animated with CSS; reduced
 * motion keeps it still.
 */
export function HeroArt() {
  const mode = use3DMode();
  const { ref, onScreen, near } = useViewport<HTMLDivElement>();
  const [lowPower, setLowPower] = useState(true);
  useEffect(() => setLowPower(isLowPower()), []);
  const webgl = useWebGL(mode === "3d" && near && !lowPower);
  return (
    <div ref={ref} className="size-full" data-hero-art={webgl ? "webgl" : "svg"}>
      {webgl ? <HeroScene active={onScreen} /> : <StaticGlobe animated={mode !== "static"} />}
    </div>
  );
}
