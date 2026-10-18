import React from 'react';
import {AbsoluteFill, Img, OffthreadVideo, Sequence, staticFile, useCurrentFrame} from 'remotion';
import {Brackets, Card, Chip, Eyebrow, MaskWords, SourceTag} from '../components/ui';
import {C, SANS, T, cfg, easeInOut, easeOut, prog, sec, useLayout} from '../theme';

const P = cfg.problem;

const rise = (frame: number, delay: number, dist = 50) => {
  const p = prog(frame, delay, 18, easeOut);
  return {opacity: p, transform: `translateY(${(1 - p) * dist}px)`, filter: p < 0.98 ? `blur(${(1 - p) * 10}px)` : undefined};
};

const headline = (portrait: boolean, size = 84): React.CSSProperties => ({
  fontFamily: SANS,
  fontWeight: 600,
  fontSize: portrait ? size * 0.82 : size,
  lineHeight: 1.04,
  letterSpacing: '-0.035em',
  color: C.text,
  justifyContent: 'flex-start',
});

/** Beat 1: a learned policy drives one arm. */
const OneArm: React.FC = () => {
  const frame = useCurrentFrame();
  const {portrait} = useLayout();
  const w = portrait ? 900 : 980;
  const h = Math.round(w / 1.4545);
  const x = portrait ? 90 : 120;
  const y = portrait ? 620 : 170;
  return (
    <AbsoluteFill>
      <Card x={x} y={y} w={w} h={h} rotY={portrait ? 0 : 6 - 3 * prog(frame, 0, 60, easeInOut)} style={rise(frame, 0)}>
        <Sequence from={0}>
          <OffthreadVideo src={staticFile('hero_head.mp4')} trimBefore={sec(P.oneArmFrom)} playbackRate={3} muted
            style={{width: '100%', height: '100%', objectFit: 'cover', filter: cfg.grade}} />
        </Sequence>
        <SourceTag kind="real" label="Head cam · FLUX run" />
      </Card>
      <div style={{position: 'absolute', left: portrait ? 90 : 1180, top: portrait ? 1300 : 330, width: portrait ? 900 : 640}}>
        <MaskWords text={T.problemOneArm[0]} at={4} style={headline(portrait)} />
        <MaskWords text={T.problemOneArm[1]} at={8} style={headline(portrait)} wordStyle={(i) => (i === 1 ? {color: C.accent} : {})} />
        <div style={{marginTop: 34, ...rise(frame, 14, 20)}}>
          <Chip>{T.problemOneArmChip}</Chip>
        </div>
      </div>
    </AbsoluteFill>
  );
};

/** Beat 2: in the team's photo, a person holds the lid open. */
const HoldsLid: React.FC = () => {
  const frame = useCurrentFrame();
  const {portrait, width, height} = useLayout();
  const push = prog(frame, 0, sec(2), easeInOut);
  // The photo is 2040x1530; show it large and drift toward the hand on the lid.
  // Portrait fills the height and frames the hand and the rig together.
  const w = portrait ? (height * 2040) / 1530 : width * 1.04;
  const h = (w * 1530) / 2040;
  const ox = portrait ? width / 2 - 0.38 * w : (width - w) / 2;
  const oy = (height - h) / 2;
  const s = 1.04 + 0.08 * push;
  const box = P.handBox;
  const bx = (v: number) => ox + v * w;
  const by = (v: number) => oy + v * h;
  return (
    <AbsoluteFill>
      <AbsoluteFill style={{transform: `scale(${s})`, transformOrigin: `${(bx(0.24) / width) * 100}% ${(by(0.72) / height) * 100}%`}}>
        <Img src={staticFile('rig_front.jpg')} style={{position: 'absolute', left: ox, top: oy, width: w, height: h,
          filter: `${cfg.grade} brightness(0.72)`}} />
        <Brackets x={bx(box[0])} y={by(box[1])} w={(box[2] - box[0]) * w} h={(box[3] - box[1]) * h} label={T.problemHandLabel}
          opacity={prog(frame, 8, 10)} glow={1} />
      </AbsoluteFill>
      <AbsoluteFill style={{background: 'linear-gradient(0deg, rgba(6,8,7,0.92), rgba(6,8,7,0.2) 45%, transparent 70%)'}} />
      <SourceTag kind="real" label="Team photo · Haller rig" style={{top: 56, left: 56}} />
      <div style={{position: 'absolute', left: portrait ? 70 : 120, right: 70, bottom: portrait ? 300 : 110}}>
        <MaskWords text={T.problemHoldsLid} at={6} style={headline(portrait, 96)} wordStyle={(i) => (i >= 3 ? {color: C.accent} : {})} />
      </div>
    </AbsoluteFill>
  );
};

/** Beat 3: stock FLUX keeps executing and replanning; nothing checks the goal. */
const NoCheck: React.FC = () => {
  const frame = useCurrentFrame();
  const {portrait} = useLayout();
  const cols = portrait ? 2 : 4;
  const gap = 28;
  const cw = portrait ? 440 : 400;
  const ch = Math.round((cw * 220) / 320);
  const total = cols * cw + (cols - 1) * gap;
  const x0 = ((portrait ? 1080 : 1920) - total) / 2;
  const y0 = portrait ? 760 : 430;
  return (
    <AbsoluteFill>
      <div style={{position: 'absolute', left: portrait ? 70 : 120, right: 70, top: portrait ? 420 : 150}}>
        <MaskWords text={T.problemNoCheck} at={2} style={headline(portrait, 88)} wordStyle={(i) => (i === 0 ? {color: C.accent} : {})} />
      </div>
      {P.strip.map((label, i) => {
        const x = x0 + (i % cols) * (cw + gap);
        const y = y0 + Math.floor(i / cols) * (ch + 90);
        return (
          <React.Fragment key={i}>
            <Card x={x} y={y} w={cw} h={ch} radius={16} style={rise(frame, 4 + i * 3)}>
              <Img src={staticFile(`strip_${i}.png`)} style={{width: '100%', height: '100%', objectFit: 'cover', filter: cfg.grade}} />
              {i === 0 && <SourceTag kind="real" label="Stock FLUX" />}
            </Card>
            <div style={{position: 'absolute', left: x, top: y + ch + 16, ...rise(frame, 8 + i * 3, 12)}}>
              <Chip style={{height: 36, fontSize: 16}}>
                <span style={{color: C.muted}}>{String(i + 1).padStart(2, '0')}</span>
                {label}
              </Chip>
            </div>
          </React.Fragment>
        );
      })}
      <div style={{position: 'absolute', left: portrait ? 70 : 120, bottom: portrait ? 330 : 110, ...rise(frame, 20, 16)}}>
        <Chip accent>{T.problemNoCheckChip}</Chip>
      </div>
    </AbsoluteFill>
  );
};

/** Beat 4: the three questions Liminal answers. */
const Questions: React.FC = () => {
  const {portrait} = useLayout();
  return (
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', flexDirection: portrait ? 'column' : 'row', gap: portrait ? 10 : 40}}>
      {T.problemQuestions.map((q, i) => (
        <MaskWords key={i} text={q} at={i * 9}
          style={{fontFamily: SANS, fontWeight: 600, fontSize: portrait ? 110 : 116, letterSpacing: '-0.04em', color: C.text}}
          wordStyle={() => (i === 2 ? {color: C.accent, textShadow: `0 0 40px ${C.accent}55`} : {})} />
      ))}
    </AbsoluteFill>
  );
};

/** 15.5-23.5 s: the gap Liminal fills, in four beats on the bar grid. */
export const Problem: React.FC = () => {
  const frame = useCurrentFrame();
  const {portrait} = useLayout();
  const beat = sec(P.beatSec);
  const beats = [OneArm, HoldsLid, NoCheck, Questions];
  return (
    <AbsoluteFill>
      {beats.map((Beat, i) => (
        <Sequence key={i} from={i * beat} durationInFrames={beat}>
          <BeatMotion dur={beat}>
            <Beat />
          </BeatMotion>
        </Sequence>
      ))}
      {frame < 3 * beat && (
        <div style={{position: 'absolute', left: portrait ? 70 : 120, top: portrait ? 200 : 80, opacity: prog(frame, 0, 10)}}>
          <Eyebrow>{T.problemLabel}</Eyebrow>
        </div>
      )}
    </AbsoluteFill>
  );
};

/** Quick lateral push between beats, so each lands like a cut on the downbeat. */
const BeatMotion: React.FC<{dur: number; children: React.ReactNode}> = ({dur, children}) => {
  const frame = useCurrentFrame();
  const inP = prog(frame, 0, 7, easeOut);
  const outP = prog(frame, dur - 5, 5, easeInOut);
  return (
    <AbsoluteFill style={{transform: `translateX(${(1 - inP) * 140 - outP * 140}px)`, opacity: inP * (1 - outP),
      filter: inP < 0.95 || outP > 0.05 ? `blur(${(1 - inP) * 10 + outP * 10}px)` : undefined}}>
      {children}
    </AbsoluteFill>
  );
};
