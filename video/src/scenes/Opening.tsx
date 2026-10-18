import React from 'react';
import {AbsoluteFill, staticFile, useCurrentFrame} from 'remotion';
import {Eyebrow, KenBurns, MaskWords, SourceTag, Wordmark} from '../components/ui';
import {C, MONO, SANS, T, cfg, easeInOut, prog, sec, useLayout} from '../theme';

const S = cfg.shots;

/** 0-1.5 s: macro push on the real gripper. */
export const HookMacro: React.FC = () => {
  const frame = useCurrentFrame();
  const dur = sec(S.hookMacro.end - S.hookMacro.start);
  const lift = prog(frame, 0, 10, easeInOut);
  return (
    <AbsoluteFill style={{opacity: lift}}>
      <KenBurns src={staticFile('gripper.jpg')} dur={dur} from={1.32} to={1.62} pan={[0, -40]} position="50% 62%"
        filter={cfg.grade} />
      <AbsoluteFill style={{background: 'linear-gradient(180deg, rgba(6,8,7,0.55), transparent 30%, transparent 60%, rgba(6,8,7,0.8))'}} />
      <SourceTag kind="real" label="SO-101 gripper" style={{top: 56, left: 56}} />
    </AbsoluteFill>
  );
};

/** 1.5-3 s: hard cut to the full rig; the wordmark resolves over it. */
export const HookReveal: React.FC = () => {
  const {portrait} = useLayout();
  const frame = useCurrentFrame();
  const dur = sec(S.hookReveal.end - S.hookReveal.start);
  const sub = prog(frame, 16, 14);
  return (
    <AbsoluteFill>
      <KenBurns src={staticFile('station.jpg')} dur={dur} from={1.16} to={1.05} pan={[0, 10]}
        position={portrait ? '58% 50%' : '50% 40%'} filter={`${cfg.grade} brightness(0.78)`} />
      <AbsoluteFill
        style={{
          background: `radial-gradient(ellipse 70% 55% at 50% ${portrait ? 50 : 55}%, rgba(6,8,7,0.82), rgba(6,8,7,0.35) 70%)`,
        }}
      />
      <SourceTag kind="real" label="Haller · XLeRobot" style={{top: 56, left: 56}} />
      <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 34}}>
        <Wordmark text={T.wordmark} at={6} size={portrait ? 118 : 150} />
        <div
          style={{
            fontFamily: MONO,
            fontSize: portrait ? 22 : 24,
            letterSpacing: '0.16em',
            textTransform: 'uppercase',
            color: C.muted,
            opacity: sub,
            transform: `translateY(${(1 - sub) * 14}px)`,
            whiteSpace: 'pre',
          }}
        >
          {T.hookSub}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/** 3-6 s: the problem in one line, then the answer as the tagline. */
export const Problem: React.FC = () => {
  const frame = useCurrentFrame();
  const {portrait} = useLayout();
  const swap = sec(S.problem.swapAt - S.problem.start);
  const size = portrait ? 92 : 112;
  const head: React.CSSProperties = {
    fontFamily: SANS,
    fontWeight: 600,
    fontSize: size,
    lineHeight: 1.06,
    letterSpacing: '-0.035em',
    color: C.text,
    maxWidth: portrait ? 900 : 1600,
    textAlign: 'center',
  };
  return (
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center'}}>
      <Eyebrow style={{position: 'absolute', top: portrait ? 640 : 300, opacity: 1 - prog(frame, swap - 8, 8)}}>
        The problem
      </Eyebrow>
      <MaskWords text={T.problem} at={2} out={swap - 8} style={head} />
      <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', flexDirection: 'column'}}>
        <MaskWords text={T.tagline[0]} at={swap} style={head} wordStyle={() => ({color: C.accent, textShadow: `0 0 40px ${C.accent}55`})} />
        <MaskWords text={T.tagline[1]} at={swap + 5} style={head} />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
