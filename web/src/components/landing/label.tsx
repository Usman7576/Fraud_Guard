"use client";

// Text inside a WebGL scene, drawn once to a canvas texture (no extra React roots or DOM nodes).
import { useEffect, useMemo } from "react";
import * as THREE from "three";

export function Label({
  text,
  position,
  height = 0.12,
  color,
  weight = 600,
}: {
  text: string;
  position: [number, number, number];
  height?: number;
  color: string;
  weight?: number;
}) {
  const { texture, aspect } = useMemo(() => {
    const px = 64;
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d")!;
    const font = `${weight} ${px}px system-ui, -apple-system, "Segoe UI", sans-serif`;
    ctx.font = font;
    canvas.width = Math.ceil(ctx.measureText(text).width) + 8;
    canvas.height = Math.ceil(px * 1.25);
    ctx.font = font; // resizing the canvas resets its state
    ctx.fillStyle = color;
    ctx.textBaseline = "middle";
    ctx.fillText(text, 4, canvas.height / 2);
    const t = new THREE.CanvasTexture(canvas);
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 4;
    return { texture: t, aspect: canvas.width / canvas.height };
  }, [text, color, weight]);
  useEffect(() => () => texture.dispose(), [texture]);
  return (
    <mesh position={position}>
      <planeGeometry args={[height * aspect, height]} />
      <meshBasicMaterial map={texture} transparent depthWrite={false} toneMapped={false} />
    </mesh>
  );
}
