import React from 'react';
import {AbsoluteFill, OffthreadVideo, Sequence, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import track from '../../public/hero_wrist_track.json';
import {Brackets, Card, Chip, Eyebrow, SourceTag, Typewriter} from '../components/ui';
import {C, MONO, T, cfg, easeOut, prog, sec, useLayout} from '../theme';

const H = cfg.hero;
const segDur = H.segments.map((s) => (s.to - s.from) / s.speed);
const segStart = segDur.map((_, i) => segDur.slice(0, i).reduce((a, b) => a + b, 0));

/** Source-clip seconds shown at a given second of the hero shot. */
export const sourceTimeAt = (t: number) => {
  for (let i = 0; i < H.segments.length; i++) {
    if (t < segStart[i] + segDur[i] || i === H.segments.length - 1) {
      const s = H.segments[i];
      return Math.min(s.to, s.from + Math.max(0, t - segStart[i]) * s.speed);
    }
  }
  return 0;
};

const segmentAt = (t: number) => Math.max(0, segStart.filter((s) => s <= t).length - 1);

/** The same source segments, time-remapped, for any clip cut from the red-pen run. */
const RemappedClip: React.FC<{src: string}> = ({src}) => (
  <>
    {H.segments.map((s, i) => (
      <Sequence key={i} from={sec(segStart[i])} durationInFrames={sec(segDur[i])}>
        <OffthreadVideo
          src={staticFile(src)}
          trimBefore={sec(s.from)}
          playbackRate={s.speed}
          muted
          style={{width: '100%', height: '100%', objectFit: 'cover', filter: cfg.grade}}
        />
      </Sequence>
    ))}
  </>
);

/** Smoothed red-cap box from the per-frame track, normalised to the wrist crop. */
const capBox = (srcSec: number): number[] | null => {
  const boxes = track.boxes as (number[] | null)[];
  const i = Math.round(srcSec * track.fps);
  const win = boxes.slice(Math.max(0, i - 3), i + 4).filter((b): b is number[] => b !== null);
  if (win.length < 3) return null;
  return [0, 1, 2, 3].map((k) => win.reduce((a, b) => a + b[k], 0) / win.length);
};

const useRise = (delay: number) => {
  const frame = useCurrentFrame();
  const p = prog(frame, delay, 20, easeOut);
  return {opacity: p, transform: `translateY(${(1 - p) * 50}px)`, filter: p < 0.98 ? `blur(${(1 - p) * 10}px)` : undefined};
};

export const Hero: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const {portrait} = useLayout();
  const t = frame / fps;
  const src = sourceTimeAt(t);
  const seg = segmentAt(t);
  const dur = sec(cfg.shots.hero.end - cfg.shots.hero.start);
  const released = src >= H.releasedAt;
  const releaseFrame = sec(segStart[1] + (H.releasedAt - H.segments[1].from) / H.segments[1].speed);

  // Layout: landscape puts the wrist camera in a right column, portrait stacks it under the head camera.
  const L = portrait
    ? {
        eyebrow: [70, 300], prompt: [70, 360, 940], head: [70, 480, 940, 646], wrist: [400, 1050, 610, 419],
        pred: [70, 1520], stat: [70, 1590], policy: [70, 1660],
      }
    : {
        eyebrow: [120, 92], prompt: [1230, 150, 570], head: [120, 150, 1040, 715], wrist: [1230, 262, 570, 392],
        pred: [1230, 690], stat: [1230, 760], policy: [120, 893],
      };

  // Speed-ramp accent at the segment change: a short punch-in and flash.
  const bumpAt = sec(segStart[1]);
  const bump = Math.max(0, 1 - Math.abs(frame - bumpAt) / 6);
  const flash = frame >= bumpAt ? Math.max(0, 1 - (frame - bumpAt) / 7) * 0.16 : 0;
  const driftX = 18 - (36 * frame) / dur;

  const [hx, hy, hw, hh] = L.head;
  const [wx, wy, ww, wh] = L.wrist;
  const cap = src >= H.trackFrom && src <= H.trackUntil ? capBox(src) : null;
  const capFade = Math.min(prog(src * 30, H.trackFrom * 30, 6), 1 - prog(src * 30, H.trackUntil * 30 - 8, 8));
  const slotOn = prog(src * 30, H.slotShowFrom * 30, 10);
  const releasePulse = released ? Math.max(0, 1 - (frame - releaseFrame) / 20) : 0;

  const rise = [useRise(0), useRise(4), useRise(8), useRise(12), useRise(16), useRise(20)];
  const statRise = useRise(sec(H.statFrom - cfg.shots.hero.start));

  return (
    <AbsoluteFill style={{transform: `translateX(${driftX}px) scale(${1 + bump * 0.025})`}}>
      <div style={{position: 'absolute', left: L.eyebrow[0], top: L.eyebrow[1], ...rise[0]}}>
        <Eyebrow>{T.heroLabel}</Eyebrow>
      </div>

      <div style={{position: 'absolute', left: L.prompt[0], top: L.prompt[1], width: L.prompt[2], ...rise[1]}}>
        <div style={{fontFamily: MONO, fontSize: 15, letterSpacing: '0.14em', color: C.muted, marginBottom: 10}}>
          PROMPT
        </div>
        <div
          style={{
            height: 62,
            display: 'flex',
            alignItems: 'center',
            padding: '0 20px',
            borderRadius: 14,
            background: C.surfaceHi,
            border: `1px solid ${C.line}`,
            fontFamily: MONO,
            fontSize: 22,
            color: C.text,
            boxShadow: '0 20px 50px -20px rgba(0,0,0,0.8)',
          }}
        >
          <span style={{color: C.accent, marginRight: 14}}>›</span>
          <Typewriter text={T.prompt} at={10} cps={30} />
        </div>
      </div>

      <Card x={hx} y={hy} w={hw} h={hh} rotY={portrait ? 0 : 5 - (3 * frame) / dur} rotX={portrait ? 3 - (2 * frame) / dur : 0}
        style={rise[2]}>
        <RemappedClip src="hero_head.mp4" />
        <SourceTag kind="real" label="Head cam · D455" />
        <div style={{position: 'absolute', top: 18, right: 18, zIndex: 5}}>
          <Chip style={{height: 34, fontSize: 15, padding: '0 12px', background: 'rgba(6,8,7,0.62)'}}>
            <span style={{color: C.accent}}>▶▶</span>
            {H.segments[seg].speed}× speed
          </Chip>
        </div>
        {slotOn > 0 && (
          <Brackets
            x={H.slotBox[0] * hw}
            y={H.slotBox[1] * hh}
            w={(H.slotBox[2] - H.slotBox[0]) * hw}
            h={(H.slotBox[3] - H.slotBox[1]) * hh}
            color={released ? C.accent : C.text}
            dashed={!released}
            opacity={slotOn}
            glow={released ? 1 + releasePulse * 2 : 0}
            label={released ? `${T.slotLabel} ✓` : T.slotLabel}
          />
        )}
      </Card>

      <Card x={wx} y={wy} w={ww} h={wh} z={40} rotY={portrait ? 0 : -8 + (4 * frame) / dur}
        rotX={portrait ? -3 + (2 * frame) / dur : 0} style={rise[3]}>
        <RemappedClip src="hero_wrist.mp4" />
        <SourceTag kind="real" label="Wrist cam" />
        {cap && capFade > 0 && (
          <Brackets
            x={cap[0] * ww - 8}
            y={Math.max(4, cap[1] * wh - 8)}
            w={(cap[2] - cap[0]) * ww + 16}
            h={Math.min((cap[3] - cap[1]) * wh + 16, wh * 0.6)}
            label={cap[1] * wh > 50 ? T.trackLabel : undefined}
            opacity={capFade}
          />
        )}
      </Card>

      <div style={{position: 'absolute', left: L.pred[0], top: L.pred[1], ...rise[4]}}>
        <Chip
          accent={released}
          style={{transform: `scale(${1 + releasePulse * 0.06})`, transformOrigin: 'left center', gap: 14}}
        >
          <span style={{color: released ? C.accent : C.muted}}>predicate</span>
          <span>{T.predicate}</span>
          <span
            style={{
              marginLeft: 6,
              padding: '3px 10px',
              borderRadius: 8,
              fontSize: 16,
              background: released ? C.accent : 'rgba(255,255,255,0.06)',
              color: released ? C.bg : C.muted,
            }}
          >
            {released ? `✓ ${T.predicateDone}` : `${T.predicatePending}${'.'.repeat(1 + (Math.floor(frame / 8) % 3))}`}
          </span>
        </Chip>
      </div>

      <div style={{position: 'absolute', left: L.stat[0], top: L.stat[1], ...statRise}}>
        <Chip>
          <span style={{color: C.accent}}>●</span>
          {T.statChip}
        </Chip>
      </div>

      <div style={{position: 'absolute', left: L.policy[0], top: L.policy[1], ...rise[5]}}>
        <Chip>
          <span style={{color: C.muted}}>policy</span>
          {T.policyChip}
        </Chip>
      </div>

      <AbsoluteFill style={{background: '#fff', opacity: flash, pointerEvents: 'none'}} />
    </AbsoluteFill>
  );
};
