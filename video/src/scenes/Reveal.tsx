import React from 'react';
import {AbsoluteFill, Img, staticFile, useCurrentFrame} from 'remotion';
import {Wordmark} from '../components/ui';
import {C, MONO, T, cfg, easeInOut, easeOut, prog, sec, useLayout} from '../theme';

const R = cfg.reveal;
const IMG_W = 830;
const IMG_H = 900;

/** Where the robot photo sits: centred while the callouts play, then moved aside for the wordmark. */
export const robotBox = (portrait: boolean, settle: number) => {
  const h = portrait ? 720 : 900;
  const w = (h * IMG_W) / IMG_H;
  const cx = portrait ? 540 : 960 + 300 * settle;
  const top = portrait ? 420 - 170 * settle : 120;
  const s = 1 - 0.08 * settle;
  return {x: cx - (w * s) / 2, y: top + (h * (1 - s)) / 2, w: w * s, h: h * s};
};

const SETTLE_AT = 2.2;

/** The real Haller rig, cut out of the team's photo, lit like a product shot. */
export const RobotStage: React.FC<{settle: number; frame: number; showLight?: boolean}> = ({settle, frame, showLight = true}) => {
  const {portrait} = useLayout();
  const b = robotBox(portrait, settle);
  const sweep = prog(frame, 8, 34, easeInOut);
  const src = staticFile('robot_cutout.png');
  const mask = {maskImage: `url(${src})`, WebkitMaskImage: `url(${src})`, maskSize: '100% 100%', WebkitMaskSize: '100% 100%'};
  return (
    <>
      <div style={{position: 'absolute', left: b.x - b.w * 0.4, top: b.y - b.h * 0.15, width: b.w * 1.8, height: b.h * 1.3,
        background: `radial-gradient(ellipse 45% 45% at 50% 50%, ${C.accent}2a, transparent 70%)`}} />
      <div style={{position: 'absolute', left: b.x - b.w * 0.25, top: 0, width: b.w * 1.5, height: b.y + b.h,
        background: 'linear-gradient(180deg, rgba(255,255,255,0.07), transparent 85%)',
        clipPath: 'polygon(42% 0, 58% 0, 100% 100%, 0 100%)', opacity: 0.7}} />
      {/* Floor reflection. */}
      <Img src={src} style={{position: 'absolute', left: b.x, top: b.y + b.h - 4, width: b.w, height: b.h,
        transform: 'scaleY(-1)', opacity: 0.16 * (portrait ? 1 - settle : 1), filter: 'blur(2px)',
        maskImage: 'linear-gradient(0deg, transparent 72%, black 100%)', WebkitMaskImage: 'linear-gradient(0deg, transparent 72%, black 100%)'}} />
      <Img src={src} style={{position: 'absolute', left: b.x, top: b.y, width: b.w, height: b.h, filter: cfg.grade}} />
      {showLight && sweep > 0 && sweep < 1 && (
        <div style={{position: 'absolute', left: b.x, top: b.y, width: b.w, height: b.h, ...mask, mixBlendMode: 'screen'}}>
          <div style={{position: 'absolute', inset: 0,
            background: `linear-gradient(105deg, transparent ${sweep * 140 - 40}%, rgba(255,255,255,0.35) ${sweep * 140 - 25}%, transparent ${sweep * 140 - 10}%)`}} />
        </div>
      )}
      <div style={{position: 'absolute', left: b.x - 60, width: b.w + 120, top: b.y + b.h - 6, height: 2,
        background: `linear-gradient(90deg, transparent, ${C.accent}66, transparent)`}} />
    </>
  );
};

/** 14-18 s: match cut from the book plate to the real robot, with callouts, then the wordmark. */
export const Reveal: React.FC = () => {
  const frame = useCurrentFrame();
  const {portrait, width} = useLayout();
  const settle = prog(frame, sec(SETTLE_AT), 18, easeInOut);
  const enter = prog(frame, 0, 16, easeOut);
  const calloutsOut = prog(frame, sec(SETTLE_AT) - 6, 8);
  const b = robotBox(portrait, 0);
  return (
    <AbsoluteFill>
      <AbsoluteFill style={{transform: `scale(${1.12 - 0.12 * enter})`, filter: enter < 0.95 ? `blur(${(1 - enter) * 12}px)` : undefined}}>
        <RobotStage settle={settle} frame={frame} />
      </AbsoluteFill>
      {R.callouts.map((c, i) => {
        const p = prog(frame, 16 + i * 5, 14, easeOut);
        const px = b.x + c.at[0] * b.w;
        const py = b.y + c.at[1] * b.h;
        const right = c.side === 'right';
        const edge = right ? b.x + b.w + (portrait ? 4 : 40) : b.x + (portrait ? -4 : -40);
        const len = Math.abs(edge - px);
        const op = p * (1 - calloutsOut);
        return (
          <div key={i} style={{opacity: op}}>
            <div style={{position: 'absolute', left: px - 7, top: py - 7, width: 14, height: 14, borderRadius: 14,
              background: C.accent, boxShadow: `0 0 0 ${6 + 10 * (1 - p)}px ${C.accent}33, 0 0 18px ${C.accent}`}} />
            <div style={{position: 'absolute', top: py - 1, left: right ? px : px - len * p, width: len * p, height: 2,
              background: `linear-gradient(${right ? 90 : 270}deg, ${C.accent}, ${C.accent}88)`}} />
            <div style={{position: 'absolute', top: py - 19, ...(right ? {left: edge + 12} : {right: width - edge + 12}),
              height: 38, display: 'flex', alignItems: 'center', padding: '0 14px', borderRadius: 10, whiteSpace: 'nowrap',
              background: 'rgba(6,8,7,0.78)', border: `1px solid ${C.accent}55`, fontFamily: MONO, fontSize: portrait ? 16 : 19,
              color: C.text, letterSpacing: '0.04em', transform: `translateX(${(1 - p) * (right ? -14 : 14)}px)`}}>
              {c.label}
            </div>
          </div>
        );
      })}
      <Title opacity={settle} at={sec(SETTLE_AT) + 4} subOpacity={prog(frame, sec(SETTLE_AT) + 16, 14)} />
    </AbsoluteFill>
  );
};

const Title: React.FC<{opacity: number; at: number; subOpacity: number}> = ({opacity, at, subOpacity}) => {
  const {portrait} = useLayout();
  return (
    <div style={{position: 'absolute', ...(portrait ? {left: 0, right: 0, top: 1300, alignItems: 'center'} : {left: 150, top: 430}),
      display: 'flex', flexDirection: 'column', gap: 26, opacity}}>
      <Wordmark text={T.wordmark} at={at} size={portrait ? 110 : 120} />
      <div style={{fontFamily: MONO, fontSize: portrait ? 22 : 23, letterSpacing: '0.12em', textTransform: 'uppercase',
        color: C.muted, opacity: subOpacity, whiteSpace: 'nowrap'}}>
        {T.revealSub}
      </div>
    </div>
  );
};

/** The reveal's last frame, held under the scan into the twin. */
export const RevealFinal: React.FC = () => (
  <AbsoluteFill>
    <RobotStage settle={1} frame={0} showLight={false} />
    <Title opacity={1} at={-1000} subOpacity={1} />
  </AbsoluteFill>
);
