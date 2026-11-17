"use client";

// Score gauge: a half ring on a log scale, filled up to the score, and a needle that swings there
// on a spring (a little overshoot, then it settles).
import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

import { GAUGE_TICKS, gaugePosition, tickLabel } from "@/lib/landing";

import { Label } from "./label";
import { COLORS, DPR } from "./use-3d";

const R = 1.25;

const arcVertex = /* glsl */ `
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const arcFragment = /* glsl */ `
  uniform float uValue;
  uniform vec3 uFill;
  uniform vec3 uTrack;
  varying vec2 vUv;
  void main() {
    // TorusGeometry's arc runs from the right (uv.x = 0) to the left (1); the gauge reads left to right.
    float pos = 1.0 - vUv.x;
    vec3 c = pos <= uValue ? uFill * (0.6 + 0.6 * pos / max(uValue, 0.001)) : uTrack;
    gl_FragColor = vec4(c, 1.0);
  }`;

function Gauge({ target }: { target: number }) {
  const needle = useRef<THREE.Group>(null);
  const state = useRef({ value: 0, velocity: 0 });
  const uniforms = useMemo(
    () => ({
      uValue: { value: 0 },
      uFill: { value: new THREE.Color(COLORS.accent) },
      uTrack: { value: new THREE.Color(COLORS.edge) },
    }),
    [],
  );

  useFrame((_, dt) => {
    const s = state.current;
    const step = Math.min(dt, 1 / 30);
    s.velocity += (target - s.value) * 90 * step; // spring
    s.velocity *= Math.exp(-11 * step); // damping (under-damped: a small overshoot)
    s.value += s.velocity * step;
    const v = THREE.MathUtils.clamp(s.value, 0, 1);
    uniforms.uValue.value = v;
    if (needle.current) needle.current.rotation.z = Math.PI / 2 - v * Math.PI;
  });

  return (
    <group position={[0, -0.45, 0]} rotation={[-0.32, 0, 0]}>
      <mesh>
        <torusGeometry args={[R, 0.075, 24, 160, Math.PI]} />
        <shaderMaterial vertexShader={arcVertex} fragmentShader={arcFragment} uniforms={uniforms} />
      </mesh>
      {GAUGE_TICKS.map((t) => {
        const a = Math.PI - gaugePosition(t) * Math.PI;
        const x = Math.cos(a);
        const y = Math.sin(a);
        return (
          <group key={t}>
            <mesh position={[x * (R + 0.16), y * (R + 0.16), 0]} rotation={[0, 0, a]}>
              <boxGeometry args={[0.1, 0.018, 0.018]} />
              <meshBasicMaterial color={COLORS.node} />
            </mesh>
            <Label text={tickLabel(t)} position={[x * (R + 0.5), y * (R + 0.5), 0]} height={0.19} color="#8d8c86" weight={500} />
          </group>
        );
      })}
      <group ref={needle} rotation={[0, 0, Math.PI / 2]}>
        <mesh position={[0, R * 0.46, 0.06]}>
          <coneGeometry args={[0.045, R * 0.92, 4]} />
          <meshStandardMaterial color="#f0efec" emissive="#f0efec" emissiveIntensity={0.25} />
        </mesh>
      </group>
      <mesh position={[0, 0, 0.06]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.1, 0.1, 0.08, 32]} />
        <meshStandardMaterial color={COLORS.accent} metalness={0.3} roughness={0.4} />
      </mesh>
    </group>
  );
}

export default function GaugeScene({ value, active }: { value: number; active: boolean }) {
  return (
    <Canvas dpr={DPR} frameloop={active ? "always" : "never"} camera={{ position: [0, 0, 4.2], fov: 40 }} gl={{ antialias: true, alpha: true }} aria-hidden>
      <ambientLight intensity={0.8} />
      <directionalLight position={[2, 3, 4]} intensity={1.4} />
      <Gauge target={value} />
    </Canvas>
  );
}
