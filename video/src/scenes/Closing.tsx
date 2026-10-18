import React from 'react';
import {AbsoluteFill, staticFile, useCurrentFrame} from 'remotion';
import {Eyebrow, KenBurns, MaskWords, Typewriter, Wordmark} from '../components/ui';
import {C, MONO, SANS, T, cfg, easeInOut, easeOut, prog, sec, useLayout} from '../theme';

/** 23-26 s: the real quick-start commands and their real output, typed into a terminal card. */
export const TryIt: React.FC = () => {
  const frame = useCurrentFrame();
  const {portrait} = useLayout();
  const w = portrait ? 980 : 1240;
  const fs = portrait ? 21 : 26;
  const cps = 46;
  const lineAt = [6, 6 + Math.ceil((T.commands[0].length / cps) * 30) + 5];
  const outAt = lineAt[1] + Math.ceil((T.commands[1].length / cps) * 30) + 6;
  const p = prog(frame, 0, 20, easeOut);
  const captionAt = sec(1.5);
  return (
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center'}}>
      <div style={{position: 'absolute', left: portrait ? 70 : 120, top: portrait ? 560 : 120, opacity: p}}>
        <Eyebrow>{T.tryLabel}</Eyebrow>
      </div>
      <div
        style={{
          width: w,
          borderRadius: 22,
          background: C.surface,
          border: `1px solid ${C.line}`,
          boxShadow: `0 70px 140px -40px rgba(0,0,0,0.9), 0 0 80px ${C.accent}10, inset 0 1px 0 rgba(255,255,255,0.08)`,
          overflow: 'hidden',
          opacity: p,
          transform: `perspective(2200px) rotateX(${(1 - p) * 14}deg) translateY(${(1 - p) * 60}px) scale(${0.98 + 0.02 * p})`,
          marginTop: portrait ? -120 : -40,
        }}
      >
        <div style={{height: 52, display: 'flex', alignItems: 'center', gap: 9, padding: '0 22px',
          borderBottom: `1px solid ${C.line}`, background: 'rgba(255,255,255,0.02)'}}>
          {[0, 1, 2].map((i) => (
            <span key={i} style={{width: 12, height: 12, borderRadius: 12, background: 'rgba(255,255,255,0.14)'}} />
          ))}
          <span style={{marginLeft: 14, fontFamily: MONO, fontSize: 15, color: C.muted, letterSpacing: '0.06em'}}>
            liminal — zsh
          </span>
        </div>
        <div style={{padding: portrait ? '30px 30px 34px' : '36px 40px 40px', fontFamily: MONO, fontSize: fs,
          lineHeight: 1.75, color: C.text}}>
          {T.commands.map((cmd, i) => (
            <div key={i} style={{opacity: frame >= lineAt[i] - 4 ? 1 : 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all'}}>
              <span style={{color: C.accent}}>$ </span>
              <Typewriter text={cmd} at={lineAt[i]} cps={cps}
                caret={i === T.commands.length - 1 ? frame < outAt + 4 : frame < lineAt[i + 1]} />
            </div>
          ))}
          <div style={{marginTop: 8}}>
            {T.commandOutput.map((line, i) => {
              const q = prog(frame, outAt + i * 3, 8, easeOut);
              const [k, v] = line.split(': ');
              return (
                <div key={i} style={{opacity: q, transform: `translateX(${(1 - q) * -12}px)`, color: C.muted}}>
                  {'  '}
                  <span>{k}: </span>
                  <span style={{color: C.accent, textShadow: `0 0 18px ${C.accent}66`}}>{v}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
      <div style={{position: 'absolute', bottom: portrait ? 560 : 150, width: '100%'}}>
        <MaskWords text={T.tryCaption} at={captionAt}
          style={{fontFamily: SANS, fontWeight: 600, fontSize: portrait ? 54 : 52, letterSpacing: '-0.025em', color: C.text}}
          wordStyle={(i) => (i >= 4 ? {color: C.muted} : {})} />
      </div>
    </AbsoluteFill>
  );
};

/** 26-30 s: end card, fully composed by 27 s and held. */
export const EndCard: React.FC = () => {
  const frame = useCurrentFrame();
  const {portrait} = useLayout();
  const dur = sec(cfg.shots.end.end - cfg.shots.end.start);
  const line = (delay: number) => {
    const p = prog(frame, delay, 18, easeOut);
    return {opacity: p, transform: `translateY(${(1 - p) * 18}px)`};
  };
  const rule = prog(frame, 14, 20, easeInOut);
  const fadeOut = 1 - prog(frame, dur - 8, 8, easeInOut);
  const meta = [T.event, T.track, T.result, T.team].filter(Boolean);
  return (
    <AbsoluteFill style={{opacity: fadeOut}}>
      <AbsoluteFill style={{opacity: 0.22}}>
        <KenBurns src={staticFile('station.jpg')} dur={dur} from={1.12} to={1.04} position="50% 40%"
          filter={`${cfg.grade} blur(14px) brightness(0.5)`} />
      </AbsoluteFill>
      <AbsoluteFill style={{background: `radial-gradient(ellipse 60% 55% at 50% 50%, rgba(6,8,7,0.2), ${C.bg} 85%)`}} />
      <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', flexDirection: 'column'}}>
        <Wordmark text={T.wordmark} at={0} size={portrait ? 110 : 128} />
        <div style={{marginTop: 34, fontFamily: SANS, fontWeight: 500, fontSize: portrait ? 40 : 42,
          letterSpacing: '-0.02em', color: C.text, ...line(8)}}>
          {T.endTagline}
        </div>
        <div style={{width: portrait ? 560 : 720, height: 1, margin: '46px 0 40px',
          background: `linear-gradient(90deg, transparent, ${C.line} 15%, ${C.accent}88 50%, ${C.line} 85%, transparent)`,
          transform: `scaleX(${rule})`}} />
        {meta.map((m, i) => (
          <div key={m} style={{fontFamily: MONO, fontSize: portrait ? 23 : 22, letterSpacing: '0.1em',
            textTransform: 'uppercase', color: C.muted, lineHeight: 1.9, textAlign: 'center', ...line(16 + i * 3)}}>
            {m}
          </div>
        ))}
        <div style={{marginTop: 36, ...line(24)}}>
          <div style={{display: 'flex', alignItems: 'center', gap: 14, height: 62, padding: '0 26px', borderRadius: 16,
            border: `1px solid ${C.accent}55`, background: `${C.accent}12`, boxShadow: `0 0 50px ${C.accent}22`,
            fontFamily: MONO, fontSize: portrait ? 28 : 28, color: C.accent}}>
            <svg width="26" height="26" viewBox="0 0 16 16" fill={C.accent}>
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
            </svg>
            {T.repo}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
