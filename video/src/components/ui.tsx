import React from 'react';
import {AbsoluteFill, interpolate, random, Sequence, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, MONO, SANS, easeIn, easeInOut, easeOut, prog, sec} from '../theme';

/** Persistent dark canvas: soft accent glow, dot grid, vignette and animated grain. */
export const Backdrop: React.FC = () => {
  const frame = useCurrentFrame();
  const drift = Math.sin(frame / 90) * 6;
  return (
    <AbsoluteFill style={{background: C.bg}}>
      <AbsoluteFill
        style={{
          background: `radial-gradient(ellipse 60% 50% at ${50 + drift}% 38%, ${C.accent}14, transparent 70%)`,
        }}
      />
      <AbsoluteFill
        style={{
          backgroundImage: 'radial-gradient(rgba(255,255,255,0.07) 1px, transparent 1.4px)',
          backgroundSize: '36px 36px',
          maskImage: 'radial-gradient(ellipse 75% 70% at 50% 45%, black 20%, transparent 85%)',
          WebkitMaskImage: 'radial-gradient(ellipse 75% 70% at 50% 45%, black 20%, transparent 85%)',
          opacity: 0.6,
        }}
      />
    </AbsoluteFill>
  );
};

/** Grain and vignette on top of everything, so footage from different cameras sits in one texture. */
export const FilmFinish: React.FC = () => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const seed = Math.floor(random(`grain-${frame}`) * 1000);
  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      <svg width={width} height={height} style={{position: 'absolute', inset: 0, opacity: 0.075, mixBlendMode: 'overlay'}}>
        <filter id={`g${frame}`}>
          <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed={seed} />
          <feColorMatrix type="saturate" values="0" />
        </filter>
        <rect width="100%" height="100%" filter={`url(#g${frame})`} />
      </svg>
      <AbsoluteFill style={{background: 'radial-gradient(ellipse 85% 80% at 50% 50%, transparent 55%, rgba(0,0,0,0.55))'}} />
    </AbsoluteFill>
  );
};

type Transition = 'cut' | 'whip' | 'zoom' | 'fade';

/**
 * A shot on the timeline. Whip = fast lateral move with directional blur on both sides of the cut,
 * zoom = punch-in with blur, fade = soft dissolve. The cut itself always lands on the frame given.
 */
export const Shot: React.FC<{
  start: number;
  end: number;
  enter?: Transition;
  exit?: Transition;
  children: React.ReactNode;
}> = ({start, end, enter = 'cut', exit = 'cut', children}) => {
  const from = sec(start);
  const dur = sec(end) - from;
  return (
    <Sequence from={from} durationInFrames={dur} layout="none">
      <ShotMotion dur={dur} enter={enter} exit={exit}>
        {children}
      </ShotMotion>
    </Sequence>
  );
};

const ShotMotion: React.FC<{dur: number; enter: Transition; exit: Transition; children: React.ReactNode}> = ({
  dur,
  enter,
  exit,
  children,
}) => {
  const frame = useCurrentFrame();
  const {width} = useVideoConfig();
  let x = 0;
  let scale = 1;
  let blur = 0;
  let opacity = 1;
  const inN = 7;
  const outN = 6;
  if (enter === 'whip') {
    const p = prog(frame, 0, inN, easeOut);
    x += (1 - p) * width * 0.45;
    blur += (1 - p) * 40;
  } else if (enter === 'zoom') {
    const p = prog(frame, 0, inN + 2, easeOut);
    scale *= 1.18 - 0.18 * p;
    blur += (1 - p) * 24;
    opacity *= interpolate(frame, [0, 3], [0, 1], {extrapolateRight: 'clamp'});
  } else if (enter === 'fade') {
    opacity *= prog(frame, 0, 10, easeInOut);
  }
  if (exit === 'whip') {
    const p = prog(frame, dur - outN, outN, easeIn);
    x -= p * width * 0.45;
    blur += p * 40;
  } else if (exit === 'zoom') {
    const p = prog(frame, dur - outN, outN, easeIn);
    scale *= 1 + 0.2 * p;
    blur += p * 20;
  } else if (exit === 'fade') {
    opacity *= 1 - prog(frame, dur - 10, 10, easeInOut);
  }
  // Horizontal-only blur reads as motion blur: stretch a normal blur along the move.
  const filter = blur > 0.3 ? `blur(${(blur * 0.35).toFixed(2)}px)` : undefined;
  return (
    <AbsoluteFill
      style={{
        transform: `translateX(${x}px) scale(${scale}) ${blur > 0.3 ? `scaleX(${1 + blur / 400})` : ''}`,
        filter,
        opacity,
      }}
    >
      {children}
    </AbsoluteFill>
  );
};

export type TagKind = 'real' | 'sim' | 'model';

/** Source tag, always in the same corner of a card. SIM is the only filled tag so it can never be missed. */
export const SourceTag: React.FC<{kind: TagKind; label: string; style?: React.CSSProperties}> = ({kind, label, style}) => {
  const sim = kind === 'sim';
  return (
    <div
      style={{
        position: 'absolute',
        top: 18,
        left: 18,
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        height: 34,
        padding: '0 14px 0 12px',
        borderRadius: 9,
        fontFamily: MONO,
        fontSize: 15,
        fontWeight: 500,
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        color: sim ? C.bg : C.text,
        background: sim ? C.text : 'rgba(6,8,7,0.62)',
        border: `1px solid ${sim ? C.text : 'rgba(255,255,255,0.16)'}`,
        backdropFilter: 'blur(12px)',
        zIndex: 5,
        ...style,
      }}
    >
      {sim ? (
        <span style={{fontWeight: 700}}>SIM</span>
      ) : (
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: 8,
            background: kind === 'real' ? C.accent : 'transparent',
            border: kind === 'real' ? 'none' : `1.5px solid ${C.accent}`,
            boxShadow: kind === 'real' ? `0 0 10px ${C.accent}` : 'none',
          }}
        />
      )}
      <span style={{opacity: sim ? 0.75 : 0.9}}>{kind === 'real' ? `Real · ${label}` : sim ? `· ${label}` : label}</span>
    </div>
  );
};

/** A floating media card: rounded, bordered, deep shadow, optional 3D tilt. */
export const Card: React.FC<{
  x: number;
  y: number;
  w: number;
  h: number;
  rotX?: number;
  rotY?: number;
  z?: number;
  radius?: number;
  opacity?: number;
  blur?: number;
  children: React.ReactNode;
  overlay?: React.ReactNode;
  style?: React.CSSProperties;
}> = ({x, y, w, h, rotX = 0, rotY = 0, z = 0, radius = 22, opacity = 1, blur = 0, children, overlay, style}) => (
  <div
    style={{
      position: 'absolute',
      left: x,
      top: y,
      width: w,
      height: h,
      transform: `perspective(2200px) translateZ(${z}px) rotateX(${rotX}deg) rotateY(${rotY}deg)`,
      transformStyle: 'preserve-3d',
      opacity,
      filter: blur > 0.2 ? `blur(${blur}px)` : undefined,
      ...style,
    }}
  >
    <div
      style={{
        position: 'absolute',
        inset: 0,
        borderRadius: radius,
        overflow: 'hidden',
        background: C.surface,
        boxShadow: `0 60px 120px -30px rgba(0,0,0,0.85), 0 24px 48px -24px rgba(0,0,0,0.7), 0 0 0 1px ${C.line}`,
      }}
    >
      {children}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          borderRadius: radius,
          boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.10), inset 0 0 0 1px rgba(255,255,255,0.04)',
          background: 'linear-gradient(160deg, rgba(255,255,255,0.06), transparent 35%)',
          pointerEvents: 'none',
        }}
      />
    </div>
    {overlay}
  </div>
);

/** Small UI chip used for labels and status. */
export const Chip: React.FC<{children: React.ReactNode; accent?: boolean; style?: React.CSSProperties}> = ({
  children,
  accent,
  style,
}) => (
  <div
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 10,
      height: 44,
      padding: '0 18px',
      borderRadius: 12,
      fontFamily: MONO,
      fontSize: 19,
      fontWeight: 500,
      letterSpacing: '0.02em',
      color: accent ? C.accent : C.text,
      background: accent ? `${C.accent}14` : 'rgba(255,255,255,0.04)',
      border: `1px solid ${accent ? `${C.accent}55` : C.line}`,
      boxShadow: accent ? `0 0 30px ${C.accent}22` : 'none',
      whiteSpace: 'nowrap',
      ...style,
    }}
  >
    {children}
  </div>
);

/** Small uppercase mono section label with an accent tick. */
export const Eyebrow: React.FC<{children: React.ReactNode; style?: React.CSSProperties}> = ({children, style}) => (
  <div
    style={{
      display: 'flex',
      alignItems: 'center',
      gap: 14,
      fontFamily: MONO,
      fontSize: 18,
      fontWeight: 500,
      letterSpacing: '0.18em',
      textTransform: 'uppercase',
      color: C.muted,
      ...style,
    }}
  >
    <span style={{width: 22, height: 2, background: C.accent, boxShadow: `0 0 10px ${C.accent}`}} />
    {children}
  </div>
);

/** Words rise out of a mask, staggered; optional exit lifts them away. */
export const MaskWords: React.FC<{
  text: string;
  at: number;
  out?: number;
  stagger?: number;
  style?: React.CSSProperties;
  wordStyle?: (i: number) => React.CSSProperties;
}> = ({text, at, out, stagger = 3, style, wordStyle}) => {
  const frame = useCurrentFrame();
  const words = text.split(' ');
  return (
    <div style={{display: 'flex', flexWrap: 'wrap', justifyContent: 'center', columnGap: '0.26em', ...style}}>
      {words.map((w, i) => {
        const pin = prog(frame, at + i * stagger, 16, easeOut);
        const pout = out === undefined ? 0 : prog(frame, out + i * 1.5, 9, easeIn);
        return (
          <span key={i} style={{display: 'inline-block', overflow: 'hidden', paddingBottom: '0.08em', marginBottom: '-0.08em'}}>
            <span
              style={{
                display: 'inline-block',
                transform: `translateY(${(1 - pin) * 105 - pout * 105}%)`,
                ...wordStyle?.(i),
              }}
            >
              {w}
            </span>
          </span>
        );
      })}
    </div>
  );
};

/** Typed text with a block caret that blinks on the beat. */
export const Typewriter: React.FC<{text: string; at: number; cps?: number; caret?: boolean; style?: React.CSSProperties}> = ({
  text,
  at,
  cps = 38,
  caret = true,
  style,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const n = Math.max(0, Math.min(text.length, Math.floor(((frame - at) / fps) * cps)));
  const typing = n < text.length && frame >= at;
  const on = typing || Math.floor(frame / (fps / 2)) % 2 === 0;
  return (
    <span style={style}>
      {text.slice(0, n)}
      {caret && frame >= at - 6 && (
        <span
          style={{
            display: 'inline-block',
            width: '0.55em',
            height: '1.05em',
            marginLeft: 2,
            verticalAlign: '-0.15em',
            background: C.accent,
            opacity: on ? 0.9 : 0,
          }}
        />
      )}
    </span>
  );
};

/** Corner-bracket HUD box. Coordinates in px within the parent. */
export const Brackets: React.FC<{
  x: number;
  y: number;
  w: number;
  h: number;
  color?: string;
  label?: string;
  opacity?: number;
  dashed?: boolean;
  glow?: number;
}> = ({x, y, w, h, color = C.accent, label, opacity = 1, dashed, glow = 0}) => {
  const L = Math.max(10, Math.min(26, Math.min(w, h) * 0.28));
  const s = 2.5;
  const corner = (pos: React.CSSProperties, bt: string, bl: string): React.CSSProperties => ({
    position: 'absolute',
    width: L,
    height: L,
    borderTop: bt,
    borderLeft: bl,
    ...pos,
  });
  const line = `${s}px solid ${color}`;
  return (
    <div style={{position: 'absolute', left: x, top: y, width: w, height: h, opacity, zIndex: 4}}>
      {dashed && (
        <div style={{position: 'absolute', inset: 0, border: `1.5px dashed ${color}`, opacity: 0.55, borderRadius: 4}} />
      )}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          filter: glow ? `drop-shadow(0 0 ${8 * glow}px ${color})` : undefined,
        }}
      >
        <div style={corner({left: 0, top: 0}, line, line)} />
        <div style={{...corner({right: 0, top: 0}, line, 'none'), borderRight: line}} />
        <div style={{...corner({left: 0, bottom: 0}, 'none', line), borderBottom: line}} />
        <div style={{...corner({right: 0, bottom: 0}, 'none', 'none'), borderRight: line, borderBottom: line}} />
      </div>
      {label && (
        <div
          style={{
            position: 'absolute',
            left: 0,
            top: -34,
            height: 26,
            padding: '0 9px',
            display: 'flex',
            alignItems: 'center',
            borderRadius: 6,
            background: color,
            color: C.bg,
            fontFamily: MONO,
            fontSize: 15,
            fontWeight: 500,
            letterSpacing: '0.04em',
            whiteSpace: 'nowrap',
          }}
        >
          {label}
        </div>
      )}
    </div>
  );
};

/** The Liminal mark: two arm bars around a threshold gap, one lit. */
export const Mark: React.FC<{size: number; reveal?: number}> = ({size, reveal = 1}) => {
  const bw = size * 0.2;
  const gap = size * 0.16;
  return (
    <div style={{position: 'relative', width: bw * 2 + gap, height: size}}>
      <div
        style={{
          position: 'absolute',
          left: 0,
          bottom: 0,
          width: bw,
          height: size * reveal,
          borderRadius: bw / 2,
          background: C.text,
        }}
      />
      <div
        style={{
          position: 'absolute',
          left: bw + gap,
          top: 0,
          width: bw,
          height: size * reveal,
          borderRadius: bw / 2,
          background: C.accent,
          boxShadow: `0 0 ${size * 0.4}px ${C.accent}88`,
        }}
      />
    </div>
  );
};

/** Wordmark: mark grows, letters resolve through a mask with tightening tracking. */
export const Wordmark: React.FC<{text: string; at: number; size: number}> = ({text, at, size}) => {
  const frame = useCurrentFrame();
  const p = prog(frame, at, 22, easeOut);
  const markP = prog(frame, at - 4, 18, easeOut);
  return (
    <div style={{display: 'flex', alignItems: 'center', gap: size * 0.42}}>
      <Mark size={size * 0.86} reveal={markP} />
      <div style={{display: 'flex', overflow: 'hidden', paddingRight: '0.1em'}}>
        {text.split('').map((ch, i) => {
          const pi = prog(frame, at + 2 + i * 1.6, 16, easeOut);
          return (
            <span
              key={i}
              style={{
                display: 'inline-block',
                fontFamily: SANS,
                fontWeight: 700,
                fontSize: size,
                lineHeight: 1,
                color: C.text,
                letterSpacing: `${0.34 - 0.2 * p}em`,
                transform: `translateY(${(1 - pi) * 110}%)`,
                textShadow: `0 0 ${40 * p}px rgba(62,227,163,${0.18 * p})`,
              }}
            >
              {ch}
            </span>
          );
        })}
      </div>
    </div>
  );
};

/** Full-bleed still with a slow push and the shared grade. */
export const KenBurns: React.FC<{
  src: string;
  dur: number;
  from: number;
  to: number;
  pan?: [number, number];
  position?: string;
  filter?: string;
}> = ({src, dur, from, to, pan = [0, 0], position = 'center', filter}) => {
  const frame = useCurrentFrame();
  const p = prog(frame, 0, dur, easeInOut);
  const s = from + (to - from) * p;
  return (
    <AbsoluteFill style={{overflow: 'hidden'}}>
      <img
        src={src}
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          objectPosition: position,
          transform: `scale(${s}) translate(${pan[0] * p}px, ${pan[1] * p}px)`,
          filter,
        }}
      />
    </AbsoluteFill>
  );
};
