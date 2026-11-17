"use client";

import { MotionConfig, animate, motion, useInView, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";

import { fmt } from "@/lib/landing";

/** Framer Motion honours the visitor's reduced-motion setting everywhere below this. */
export function Motion({ children }: { children: React.ReactNode }) {
  return <MotionConfig reducedMotion="user">{children}</MotionConfig>;
}

/** Children fade and rise into place, one after another, when the list scrolls into view. */
export function Stagger({ children, className }: { children: React.ReactNode[]; className?: string }) {
  return (
    <motion.ol
      className={className}
      initial="hidden"
      whileInView="shown"
      viewport={{ once: true, margin: "-80px" }}
      variants={{ shown: { transition: { staggerChildren: 0.12 } } }}
    >
      {children.map((c, i) => (
        <motion.li
          key={i}
          variants={{ hidden: { opacity: 0, y: 24 }, shown: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" } } }}
        >
          {c}
        </motion.li>
      ))}
    </motion.ol>
  );
}

/** A number that counts up from zero the first time it is seen. */
export function Counter({ value, format }: { value: number; format: "percent" | "money" | "decimal" }) {
  const ref = useRef<HTMLSpanElement>(null);
  const seen = useInView(ref, { once: true, margin: "-60px" });
  const reduce = useReducedMotion();
  const [shown, setShown] = useState(0);
  useEffect(() => {
    if (!seen) return;
    if (reduce) {
      setShown(value);
      return;
    }
    const controls = animate(0, value, { duration: 1.4, ease: [0.16, 1, 0.3, 1], onUpdate: setShown });
    return () => controls.stop();
  }, [seen, reduce, value]);
  return (
    <span ref={ref} className="tabular">
      {fmt(shown, format)}
    </span>
  );
}
