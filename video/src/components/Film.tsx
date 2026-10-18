import React from 'react';
import {AbsoluteFill, random, useCurrentFrame, useVideoConfig} from 'remotion';
import {prog} from '../theme';

/**
 * Old-film treatment for the history section: per-frame gate weave and flicker on the content, plus dust,
 * hairs, scratches and a burnt vignette on top. Everything is seeded by frame, so renders are repeatable.
 */
export const FilmLook: React.FC<{
  children: React.ReactNode;
  mono: number; // 1 = black and white, 0 = full colour
  strength?: number; // 0..1 overall amount of damage, flicker and weave
}> = ({children, mono, strength = 1}) => {
  const frame = useCurrentFrame();
  const r = (k: string) => random(`${k}-${frame}`);
  const weaveX = (r('wx') - 0.5) * 3 * strength;
  const weaveY = (r('wy') - 0.5) * 4 * strength + (r('jump') > 0.985 ? 6 * strength : 0);
  const flicker = 1 + (r('fl') - 0.5) * 0.09 * strength;
  const filter = `grayscale(${mono}) sepia(${0.08 * mono}) contrast(${1 + 0.16 * strength}) brightness(${flicker})`;
  return (
    <AbsoluteFill style={{overflow: 'hidden'}}>
      <AbsoluteFill style={{transform: `translate(${weaveX}px, ${weaveY}px)`, filter}}>{children}</AbsoluteFill>
      <FilmDamage strength={strength} />
    </AbsoluteFill>
  );
};

const FilmDamage: React.FC<{strength: number}> = ({strength}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  if (strength <= 0) return null;
  const specks = Array.from({length: 14}, (_, i) => {
    const seed = `sp-${frame}-${i}`;
    if (random(`${seed}-on`) > 0.55) return null;
    const x = random(`${seed}-x`) * width;
    const y = random(`${seed}-y`) * height;
    const s = 1.5 + random(`${seed}-s`) ** 3 * 7;
    const light = random(`${seed}-c`) > 0.6;
    return <circle key={i} cx={x} cy={y} r={s} fill={light ? 'rgba(255,248,230,0.75)' : 'rgba(20,14,8,0.8)'} />;
  });
  const hair = random(`hair-${Math.floor(frame / 3)}`) > 0.7;
  const hx = random(`hx-${Math.floor(frame / 3)}`) * width;
  const hy = random(`hy-${Math.floor(frame / 3)}`) * height;
  // A scratch lives for a few frames and drifts slightly, like a real one on a running print.
  const scratches = [0, 1].map((k) => {
    const block = Math.floor(frame / 9) + k * 7;
    if (random(`scr-${block}`) > 0.55) return null;
    const x = random(`scrx-${block}`) * width + Math.sin(frame / 3 + k) * 3;
    return (
      <rect key={k} x={x} y={0} width={1.2} height={height}
        fill={random(`scrc-${block}`) > 0.5 ? 'rgba(255,250,235,0.35)' : 'rgba(30,20,10,0.35)'} />
    );
  });
  return (
    <AbsoluteFill style={{pointerEvents: 'none', opacity: strength}}>
      <svg width={width} height={height} style={{position: 'absolute', inset: 0}}>
        {specks}
        {scratches}
        {hair && (
          <path d={`M${hx},${hy} q 30,${-40} 70,${-10} t 60,${30}`} stroke="rgba(15,10,5,0.7)" strokeWidth={1.4} fill="none" />
        )}
      </svg>
      <AbsoluteFill style={{background: 'radial-gradient(ellipse 72% 68% at 50% 50%, transparent 50%, rgba(25,15,5,0.55) 85%, rgba(0,0,0,0.85))'}} />
    </AbsoluteFill>
  );
};

/** Projector beam with dust motes drifting through it, behind the book. */
export const DustBeam: React.FC<{opacity: number}> = ({opacity}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const motes = Array.from({length: 46}, (_, i) => {
    const sx = random(`m-x-${i}`);
    const sy = random(`m-y-${i}`);
    const sp = 0.15 + random(`m-s-${i}`) * 0.5;
    const x = ((sx * width + frame * sp * 1.4 + Math.sin(frame / 40 + i) * 20) % (width + 40)) - 20;
    const y = ((sy * height - frame * sp * 0.8 + height * 4) % height);
    const size = 1 + random(`m-r-${i}`) * 3.2;
    const tw = 0.4 + 0.6 * Math.abs(Math.sin(frame / (12 + i) + i));
    return (
      <div key={i} style={{position: 'absolute', left: x, top: y, width: size, height: size, borderRadius: size,
        background: 'rgba(255,236,200,0.9)', opacity: tw * 0.6, filter: size > 3 ? 'blur(1px)' : undefined}} />
    );
  });
  return (
    <AbsoluteFill style={{opacity, pointerEvents: 'none'}}>
      <AbsoluteFill style={{background: 'linear-gradient(115deg, transparent 20%, rgba(255,226,170,0.10) 42%, rgba(255,226,170,0.04) 58%, transparent 75%)'}} />
      {motes}
    </AbsoluteFill>
  );
};

/** Film burn: a hot spot blooms, eats the frame to white, then the reel runs out to black. */
export const FilmBurn: React.FC<{at: number; dur: number}> = ({at, dur}) => {
  const frame = useCurrentFrame();
  const p = prog(frame, at, dur);
  if (p <= 0) return null;
  const grow = Math.min(1, p / 0.7);
  const r = 6 + grow * 120;
  const wobble = Math.sin(frame * 1.7) * 2;
  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      <AbsoluteFill style={{
        background: `radial-gradient(ellipse ${r + wobble}% ${r * 0.8}% at 64% 42%, rgba(255,255,255,${Math.min(1, grow * 1.4)}) 0%, rgba(235,232,225,${Math.min(1, grow * 1.3)}) 30%, rgba(150,146,140,${Math.min(0.95, grow * 1.2)}) 55%, rgba(40,38,36,${Math.min(0.85, grow)}) 75%, transparent 100%)`}} />
      <AbsoluteFill style={{background: 'rgba(255,255,255,0.2)', mixBlendMode: 'overlay', opacity: grow}} />
      <AbsoluteFill style={{background: '#000', opacity: Math.max(0, (p - 0.78) / 0.22)}} />
    </AbsoluteFill>
  );
};
