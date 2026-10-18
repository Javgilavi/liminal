import React from 'react';
import {AbsoluteFill, Img, staticFile, useCurrentFrame} from 'remotion';
import track16 from '../../public/twin_16x9/track.json';
import track9 from '../../public/twin_9x16/track.json';
import {Brackets, MaskWords, SourceTag} from '../components/ui';
import {C, MONO, SANS, T, cfg, easeInOut, easeOut, prog, sec, useLayout} from '../theme';
import {RevealFinal} from './Reveal';

const S = cfg.sim;

type Frame = {left: number[]; right: number[]; lid: number; placed: boolean; rightActive: boolean};
type Status = 'pending' | 'running' | 'done';

const StatusPill: React.FC<{status: Status; frame: number; doneLabel?: string; runLabel?: string}> = ({
  status,
  frame,
  doneLabel = 'success',
  runLabel = 'running',
}) => {
  const pulse = 0.5 + 0.5 * Math.sin(frame / 4);
  const base: React.CSSProperties = {display: 'inline-flex', alignItems: 'center', gap: 8, height: 28, padding: '0 10px',
    borderRadius: 8, fontFamily: MONO, fontSize: 14, letterSpacing: '0.04em', whiteSpace: 'nowrap'};
  if (status === 'done') {
    return <span style={{...base, background: C.accent, color: C.bg, fontWeight: 500}}>✓ {doneLabel}</span>;
  }
  if (status === 'running') {
    return (
      <span style={{...base, color: C.accent, border: `1px solid ${C.accent}66`, background: `${C.accent}14`}}>
        <span style={{width: 8, height: 8, borderRadius: 8, background: C.accent, boxShadow: `0 0 ${4 + 8 * pulse}px ${C.accent}`}} />
        {runLabel}
      </span>
    );
  }
  return <span style={{...base, color: C.muted, border: `1px solid ${C.line}`}}>waiting</span>;
};

/** 18-26 s: the real rig scans into its MuJoCo twin, which plays lid_hold_insert with the live task tree. */
export const Sim: React.FC = () => {
  const frame = useCurrentFrame();
  const {portrait, width, height} = useLayout();
  const frames = (portrait ? track9 : track16).frames as Frame[];
  const i = Math.min(frames.length - 1, frame);
  const f = frames[i];
  const folder = portrait ? 'twin_9x16' : 'twin_16x9';
  const scan = prog(frame, 0, sec(S.scanSec), easeInOut);
  const hud = prog(frame, sec(S.hudFrom), 12, easeOut);
  const lidOpen = f.lid >= 65; // the twin expert's own open_box success threshold
  const pull = 1.0 + 0.03 * prog(frame, 0, sec(8), easeInOut);

  const rows: {arm: string; node: string; status: Status; indent: number; done: string; run: string}[] = [
    {arm: 'left', node: 'open_lid', indent: 0, done: 'lid_open', run: 'running',
      status: lidOpen ? 'done' : 'running'},
    {arm: 'left', node: 'while · maintain lid_open', indent: 0, done: 'until met', run: 'holding',
      status: f.placed ? 'done' : lidOpen ? 'running' : 'pending'},
    {arm: 'right', node: 'pick_place', indent: 1, done: 'object_inserted', run: 'running',
      status: f.placed ? 'done' : f.rightActive ? 'running' : 'pending'},
  ];

  const box = (p: number[], label: string, show: number) => {
    const size = portrait ? 130 : 150;
    const x = Math.max(10, Math.min(width - size - 10, p[0] * width - size / 2));
    const y = Math.max(50, Math.min(height - size - 10, p[1] * height - size / 2));
    return <Brackets x={x} y={y} w={size} h={size} label={label} opacity={show} />;
  };

  const panel = portrait ? {x: 60, y: 1440, w: 960} : {x: 1290, y: 120, w: 560};

  return (
    <AbsoluteFill>
      {scan < 1 && <RevealFinal />}
      <AbsoluteFill style={{clipPath: `inset(0 0 ${(1 - scan) * 100}% 0)`}}>
        <Img src={staticFile(`${folder}/f${String(i).padStart(4, '0')}.jpg`)}
          style={{width: '100%', height: '100%', objectFit: 'cover', transform: `scale(${pull})`}} />
        <AbsoluteFill style={{background: 'radial-gradient(ellipse 80% 75% at 50% 50%, transparent 55%, rgba(0,0,0,0.5))'}} />
      </AbsoluteFill>
      {scan > 0 && scan < 1 && (
        <>
          <div style={{position: 'absolute', left: 0, right: 0, top: scan * height - 140, height: 140,
            background: `linear-gradient(180deg, transparent, ${C.accent}22)`,
            backgroundImage: `repeating-linear-gradient(0deg, ${C.accent}22 0 1px, transparent 1px 6px)`}} />
          <div style={{position: 'absolute', left: 0, right: 0, top: scan * height - 1.5, height: 3, background: C.accent,
            boxShadow: `0 0 24px 6px ${C.accent}88`}} />
        </>
      )}

      <div style={{opacity: prog(frame, 8, 10)}}>
        <SourceTag kind="sim" label={T.simTag} style={{top: 56, left: 56}} />
      </div>

      {frame >= sec(S.captionFrom) && frame < sec(S.captionTo) + 12 && (
        <AbsoluteFill style={{justifyContent: portrait ? 'flex-start' : 'flex-end', alignItems: 'center',
          paddingBottom: portrait ? 0 : 120, paddingTop: portrait ? 230 : 0,
          background: `linear-gradient(${portrait ? 180 : 0}deg, rgba(6,8,7,0.8), transparent 45%)`,
          opacity: 1 - prog(frame, sec(S.captionTo), 10)}}>
          <MaskWords text={T.twinCaption} at={sec(S.captionFrom) + 2}
            style={{fontFamily: SANS, fontWeight: 600, fontSize: portrait ? 72 : 96, letterSpacing: '-0.035em', color: C.text,
              maxWidth: portrait ? 860 : 1700,
              textShadow: '0 4px 40px rgba(0,0,0,0.6)'}}
            wordStyle={(k) => (k >= 3 ? {color: C.accent} : {})} />
        </AbsoluteFill>
      )}

      {hud > 0 && (
        <>
          {box(f.left, lidOpen ? 'left · hold lid' : 'left · open_lid', hud)}
          {f.rightActive && box(f.right, 'right · pick_place', hud * prog(frame, frames.findIndex((x) => x.rightActive), 8))}
          <div style={{position: 'absolute', left: panel.x, top: panel.y, width: panel.w, borderRadius: 18, padding: '18px 22px 20px',
            background: 'rgba(8,11,10,0.72)', border: `1px solid ${C.line}`, backdropFilter: 'blur(14px)',
            boxShadow: '0 40px 80px -30px rgba(0,0,0,0.9)', opacity: hud, transform: `translateY(${(1 - hud) * 24}px)`}}>
            <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontFamily: MONO, fontSize: 15,
              color: C.muted, letterSpacing: '0.06em', marginBottom: 14}}>
              <span>{S.taskFile}</span>
              <span style={{color: C.accent}}>sequence</span>
            </div>
            {rows.map((r, k) => (
              <div key={k} style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12,
                padding: `10px 0 10px ${r.indent * 26}px`, borderTop: k ? `1px solid ${C.line}` : 'none'}}>
                <span style={{display: 'flex', alignItems: 'center', gap: 12, fontFamily: MONO, fontSize: 18, color: C.text}}>
                  {r.indent > 0 && <span style={{color: C.muted}}>└</span>}
                  <span style={{fontSize: 13, padding: '2px 7px', borderRadius: 6, border: `1px solid ${C.line}`,
                    color: r.arm === 'left' ? C.text : C.accent}}>{r.arm}</span>
                  {r.node}
                </span>
                <StatusPill status={r.status} frame={frame} doneLabel={r.done} runLabel={r.run} />
              </div>
            ))}
          </div>
        </>
      )}
      <div style={{position: 'absolute', right: 56, bottom: 48, fontFamily: MONO, fontSize: 15, color: C.muted,
        letterSpacing: '0.06em', opacity: hud * (portrait ? 0 : 1)}}>
        keyframed from the twin’s IK planner · not a policy rollout
      </div>
    </AbsoluteFill>
  );
};
