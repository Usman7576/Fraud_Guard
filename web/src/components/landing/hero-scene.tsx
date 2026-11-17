"use client";

// Hero: a slowly turning globe of payments. Most arcs pulse blue end to end; a few are fraud,
// flash red, and stop where a shield ring catches them.
import { Canvas, useFrame } from "@react-three/fiber";
import { Bloom, EffectComposer } from "@react-three/postprocessing";
import { useMemo, useRef } from "react";
import * as THREE from "three";

import { COLORS, DPR } from "./use-3d";

const R = 1.55;
const N_DOTS = 2600;
const N_ARCS = 30;
const N_FRAUD = 5;
const STOP = 0.62; // fraud pulses are caught at this point along their arc
const TAIL = 0.28;

function rng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
}

function fibonacciSphere(n: number, r: number): Float32Array {
  const out = new Float32Array(n * 3);
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / (n - 1)) * 2;
    const rad = Math.sqrt(1 - y * y);
    const th = golden * i;
    out.set([Math.cos(th) * rad * r, y * r, Math.sin(th) * rad * r], i * 3);
  }
  return out;
}

const dotVertex = /* glsl */ `
  uniform float uSize;
  varying float vFacing;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vFacing = dot(normalize(normalMatrix * position), vec3(0.0, 0.0, 1.0));
    gl_PointSize = uSize * (6.0 / -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;
const dotFragment = /* glsl */ `
  uniform vec3 uColor;
  varying float vFacing;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;
    float a = smoothstep(0.5, 0.15, d) * mix(0.1, 0.8, smoothstep(-0.2, 0.6, vFacing));
    gl_FragColor = vec4(uColor, a);
  }`;

const glowVertex = /* glsl */ `
  varying vec3 vNormal;
  void main() {
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;
const glowFragment = /* glsl */ `
  uniform vec3 uColor;
  varying vec3 vNormal;
  void main() {
    float rim = pow(clamp(0.62 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 0.0, 1.0), 4.0);
    gl_FragColor = vec4(uColor, rim * 0.55);
  }`;

const arcVertex = /* glsl */ `
  varying float vU;
  void main() {
    vU = uv.x;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;
const arcFragment = /* glsl */ `
  uniform vec3 uColor;
  uniform float uHead;
  uniform float uStop;
  uniform float uFade;
  uniform float uBase;
  varying float vU;
  void main() {
    if (vU > uStop) discard;
    float trail = smoothstep(uHead - ${TAIL.toFixed(2)}, uHead, vU) * step(vU, uHead);
    float a = max(trail * uFade, uBase);
    gl_FragColor = vec4(uColor * (1.0 + trail * 1.6), a);
  }`;

interface ArcSpec {
  curve: THREE.QuadraticBezierCurve3;
  fraud: boolean;
  period: number;
  offset: number;
}

function makeArcs(): ArcSpec[] {
  const rand = rng(7);
  const point = () => {
    const u = rand() * 2 - 1;
    const th = rand() * Math.PI * 2;
    const s = Math.sqrt(1 - u * u);
    return new THREE.Vector3(Math.cos(th) * s, u * 0.85, Math.sin(th) * s).normalize().multiplyScalar(R);
  };
  return Array.from({ length: N_ARCS }, (_, i) => {
    const a = point();
    let b = point();
    while (a.distanceTo(b) < R * 0.6 || a.distanceTo(b) > R * 1.7) b = point();
    const lift = 1 + 0.22 * a.distanceTo(b);
    const mid = a.clone().add(b).normalize().multiplyScalar(R * lift);
    return {
      curve: new THREE.QuadraticBezierCurve3(a, mid, b),
      fraud: i % Math.round(N_ARCS / N_FRAUD) === 2,
      period: 2.6 + rand() * 2.4,
      offset: rand() * 10,
    };
  });
}

function Arc({ spec }: { spec: ArcSpec }) {
  const mat = useRef<THREE.ShaderMaterial>(null);
  const burst = useRef<THREE.Mesh>(null);
  const geometry = useMemo(() => new THREE.TubeGeometry(spec.curve, 64, spec.fraud ? 0.007 : 0.005, 6), [spec]);
  const uniforms = useMemo(
    () => ({
      uColor: { value: new THREE.Color(spec.fraud ? COLORS.critical : COLORS.accent) },
      uHead: { value: 0 },
      uStop: { value: spec.fraud ? STOP : 1.01 },
      uFade: { value: 1 },
      uBase: { value: spec.fraud ? 0 : 0.06 },
    }),
    [spec],
  );
  const stopAt = useMemo(() => {
    const p = spec.curve.getPoint(STOP);
    const t = spec.curve.getTangent(STOP);
    return { p, look: p.clone().add(t) };
  }, [spec]);

  useFrame(({ clock }) => {
    const t = ((clock.elapsedTime + spec.offset) % spec.period) / spec.period;
    const travel = t * (1 + TAIL + 0.1);
    if (!mat.current) return;
    const u = mat.current.uniforms;
    if (spec.fraud) {
      u.uHead!.value = Math.min(travel, STOP);
      const past = (travel - STOP) / 0.35;
      u.uFade!.value = past > 0 ? Math.max(1 - past, 0) : 1;
      if (burst.current) {
        const k = past > 0 && past < 1 ? past : -1;
        burst.current.visible = k >= 0;
        if (k >= 0) {
          burst.current.scale.setScalar(0.04 + k * 0.16);
          (burst.current.material as THREE.MeshBasicMaterial).opacity = 1 - k;
        }
      }
    } else {
      u.uHead!.value = travel;
    }
  });

  return (
    <>
      <mesh geometry={geometry}>
        <shaderMaterial
          ref={mat}
          vertexShader={arcVertex}
          fragmentShader={arcFragment}
          uniforms={uniforms}
          transparent
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      {spec.fraud && (
        <mesh ref={burst} position={stopAt.p} onUpdate={(m) => m.lookAt(stopAt.look)} visible={false}>
          <ringGeometry args={[0.8, 1, 48]} />
          <meshBasicMaterial color={COLORS.accentBright} transparent side={THREE.DoubleSide} toneMapped={false} />
        </mesh>
      )}
    </>
  );
}

function Globe() {
  const group = useRef<THREE.Group>(null);
  const shield = useRef<THREE.Mesh>(null);
  const dots = useMemo(() => fibonacciSphere(N_DOTS, R), []);
  const arcs = useMemo(makeArcs, []);
  const dotUniforms = useMemo(() => ({ uColor: { value: new THREE.Color(COLORS.accentBright) }, uSize: { value: 3.2 } }), []);
  const glowUniforms = useMemo(() => ({ uColor: { value: new THREE.Color(COLORS.accent) } }), []);

  useFrame(({ pointer }, delta) => {
    if (!group.current) return;
    group.current.rotation.y += delta * 0.07;
    group.current.rotation.x = THREE.MathUtils.damp(group.current.rotation.x, 0.25 - pointer.y * 0.12, 2, delta);
    group.current.rotation.z = THREE.MathUtils.damp(group.current.rotation.z, pointer.x * 0.06, 2, delta);
    if (shield.current) shield.current.rotation.z -= delta * 0.12;
  });

  return (
    <group ref={group}>
      <points>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[dots, 3]} />
        </bufferGeometry>
        <shaderMaterial
          vertexShader={dotVertex}
          fragmentShader={dotFragment}
          uniforms={dotUniforms}
          transparent
          depthWrite={false}
        />
      </points>
      <mesh scale={1.06}>
        <sphereGeometry args={[R, 48, 48]} />
        <shaderMaterial
          vertexShader={glowVertex}
          fragmentShader={glowFragment}
          uniforms={glowUniforms}
          side={THREE.BackSide}
          transparent
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      <mesh>
        <sphereGeometry args={[R * 0.985, 48, 48]} />
        <meshBasicMaterial color={COLORS.background} />
      </mesh>
      {arcs.map((a, i) => (
        <Arc key={i} spec={a} />
      ))}
      <mesh ref={shield} rotation={[Math.PI / 2.3, 0.2, 0]}>
        <torusGeometry args={[R * 1.3, 0.003, 8, 200]} />
        <meshBasicMaterial color={COLORS.accentBright} transparent opacity={0.3} toneMapped={false} />
      </mesh>
    </group>
  );
}

export default function HeroScene({ active }: { active: boolean }) {
  return (
    <Canvas
      dpr={DPR}
      frameloop={active ? "always" : "never"}
      camera={{ position: [0, 0, 6.6], fov: 42 }}
      gl={{ antialias: false, powerPreference: "high-performance" }}
      aria-hidden
    >
      <color attach="background" args={[COLORS.background]} />
      <Globe />
      <EffectComposer multisampling={0}>
        <Bloom mipmapBlur intensity={0.7} luminanceThreshold={0.35} luminanceSmoothing={0.25} />
      </EffectComposer>
    </Canvas>
  );
}
