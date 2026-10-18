import React from 'react';
import {AbsoluteFill, Img, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {INK, INK_GREEN, Plate} from '../components/Art';
import {MONO, SANS, cfg, easeIn, easeInOut, easeOut, prog, sec, useLayout} from '../theme';

const B = cfg.book;
const SERIF = '"Instrument Serif", serif';
const PAPER = 'linear-gradient(180deg, #F3EDE0 0%, #EAE2D0 100%)';
const FOIL = '#E3D3A6';

type Page = (typeof B.pages)[number];

/** Left page: the year and what changed, set like a history book. */
const LeftPage: React.FC<{page: Page; index: number; pw: number; ph: number}> = ({page, index, pw, ph}) => {
  const k = pw / 620;
  return (
    <div style={{position: 'absolute', inset: 0, background: PAPER, padding: `${58 * k}px ${56 * k}px`, overflow: 'hidden'}}>
      <div style={{position: 'absolute', inset: 0, background: 'linear-gradient(270deg, rgba(60,45,20,0.22), transparent 14%)'}} />
      <div style={{fontFamily: MONO, fontSize: 13 * k, letterSpacing: '0.22em', color: '#8d8572', textTransform: 'uppercase'}}>
        {B.runningHead}
      </div>
      <div style={{fontFamily: SERIF, fontSize: 236 * k, lineHeight: 0.86, color: INK, marginTop: 70 * k, letterSpacing: '-0.02em'}}>
        {page.year}
      </div>
      <div style={{width: 64 * k, height: 3 * k, background: INK_GREEN, margin: `${34 * k}px 0 ${26 * k}px`}} />
      <div style={{fontFamily: SANS, fontWeight: 600, fontSize: 44 * k, color: INK, letterSpacing: '-0.02em', lineHeight: 1.05}}>
        {page.name}
      </div>
      <div style={{fontFamily: SERIF, fontStyle: 'italic', fontSize: 38 * k, color: '#3b3e3c', marginTop: 14 * k, lineHeight: 1.12}}>
        {page.line}
      </div>
      <div style={{position: 'absolute', left: 56 * k, bottom: 58 * k, fontFamily: MONO, fontSize: 14 * k, color: '#857d6a',
        letterSpacing: '0.06em'}}>
        {page.who}
      </div>
      <div style={{position: 'absolute', right: 40 * k, bottom: 26 * k, fontFamily: SERIF, fontSize: 20 * k, color: '#8d8572'}}>
        {index * 2 + 2}
      </div>
      <div style={{position: 'absolute', left: 0, right: 0, top: ph, height: 0}} />
    </div>
  );
};

/** Right page: the plate, drawn on when the page lands, or the robot photo on the last page. */
const RightPage: React.FC<{page: Page; index: number; pw: number; drawAt: number; last: boolean}> = ({
  page,
  index,
  pw,
  drawAt,
  last,
}) => {
  const frame = useCurrentFrame();
  const k = pw / 620;
  const p = prog(frame, drawAt, sec(B.drawSec), easeInOut);
  return (
    <div style={{position: 'absolute', inset: 0, background: PAPER, overflow: 'hidden'}}>
      <div style={{position: 'absolute', inset: 0, background: 'linear-gradient(90deg, rgba(60,45,20,0.25), transparent 12%)'}} />
      {last ? (
        <div style={{position: 'absolute', inset: `${70 * k}px ${50 * k}px ${110 * k}px`, borderRadius: 6 * k, overflow: 'hidden',
          background: 'radial-gradient(ellipse 70% 60% at 50% 55%, #17382d, #070908 75%)',
          boxShadow: `inset 0 0 0 1px rgba(0,0,0,0.25)`}}>
          <Img src={staticFile('robot_cutout.png')} style={{position: 'absolute', left: '6%', top: '8%', width: '88%',
            opacity: prog(frame, drawAt - 6, 14), transform: `scale(${1.04 - 0.04 * p})`}} />
        </div>
      ) : (
        <div style={{position: 'absolute', left: (pw - 520 * k) / 2, top: 96 * k}}>
          <Plate art={page.art} progress={p} size={520 * k} />
        </div>
      )}
      <div style={{position: 'absolute', left: 0, right: 0, bottom: 58 * k, textAlign: 'center', fontFamily: SERIF,
        fontStyle: 'italic', fontSize: 22 * k, color: '#6d675a'}}>
        {last ? 'pl. 8 — the Haller rig, ERH Physical AI Hackathon' : `pl. ${index + 1} — ${page.name}`}
      </div>
      <div style={{position: 'absolute', left: 40 * k, bottom: 26 * k, fontFamily: SERIF, fontSize: 20 * k, color: '#8d8572'}}>
        {index * 2 + 3}
      </div>
    </div>
  );
};

const Cover: React.FC<{pw: number}> = ({pw}) => {
  const frame = useCurrentFrame();
  const k = pw / 620;
  const t = prog(frame, 10, 24, easeOut);
  return (
    <div style={{position: 'absolute', inset: 0, borderRadius: `${4 * k}px ${14 * k}px ${14 * k}px ${4 * k}px`, overflow: 'hidden',
      background: 'linear-gradient(135deg, #12352a 0%, #0b241c 55%, #081a14 100%)'}}>
      <div style={{position: 'absolute', inset: 0, opacity: 0.35,
        backgroundImage: 'repeating-linear-gradient(45deg, rgba(255,255,255,0.025) 0 2px, transparent 2px 5px)'}} />
      <div style={{position: 'absolute', inset: 34 * k, border: `${1.5 * k}px solid ${FOIL}55`, borderRadius: 6 * k}} />
      <div style={{position: 'absolute', inset: 44 * k, border: `${1 * k}px solid ${FOIL}33`, borderRadius: 4 * k}} />
      <div style={{position: 'absolute', left: 0, top: 0, bottom: 0, width: 26 * k,
        background: 'linear-gradient(90deg, rgba(0,0,0,0.45), rgba(255,255,255,0.05) 60%, transparent)'}} />
      <div style={{position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', textAlign: 'center', opacity: t, transform: `translateY(${(1 - t) * 14}px)`}}>
        <div style={{fontFamily: MONO, fontSize: 16 * k, letterSpacing: '0.4em', color: `${FOIL}bb`}}>VOLUME I</div>
        <div style={{fontFamily: SERIF, fontSize: 112 * k, lineHeight: 0.95, color: FOIL, marginTop: 30 * k,
          textShadow: `0 ${1 * k}px 0 rgba(0,0,0,0.5), 0 0 ${30 * k}px rgba(227,211,166,0.18)`}}>
          {B.coverTitle[0]}
          <br />
          {B.coverTitle[1]}
        </div>
        <div style={{fontFamily: SERIF, fontStyle: 'italic', fontSize: 40 * k, color: `${FOIL}cc`, marginTop: 22 * k}}>
          {B.coverSub}
        </div>
        <div style={{width: 120 * k, height: 1, background: `${FOIL}88`, margin: `${34 * k}px 0`}} />
        <div style={{fontFamily: MONO, fontSize: 20 * k, letterSpacing: '0.3em', color: `${FOIL}bb`}}>{B.coverYears}</div>
      </div>
    </div>
  );
};

/** A sheet rotating about the spine: front face on the right, back face lands on the left. */
const Leaf: React.FC<{pw: number; ph: number; angle: number; front: React.ReactNode; back: React.ReactNode}> = ({
  pw,
  ph,
  angle,
  front,
  back,
}) => {
  const lift = Math.sin((Math.abs(angle) * Math.PI) / 180);
  return (
    <div style={{position: 'absolute', left: 0, top: 0, width: pw, height: ph, transformOrigin: '0% 50%',
      transform: `rotateY(${angle}deg) translateZ(${1 + lift * 2}px)`, transformStyle: 'preserve-3d', zIndex: 5}}>
      <div style={{position: 'absolute', inset: 0, backfaceVisibility: 'hidden'}}>
        {front}
        <div style={{position: 'absolute', inset: 0, background: `linear-gradient(90deg, rgba(0,0,0,${0.35 * lift}), rgba(0,0,0,${0.08 * lift}))`}} />
      </div>
      <div style={{position: 'absolute', inset: 0, backfaceVisibility: 'hidden', transform: 'rotateY(180deg)'}}>
        {back}
        <div style={{position: 'absolute', inset: 0, background: `linear-gradient(270deg, rgba(0,0,0,${0.3 * lift}), transparent)`}} />
      </div>
    </div>
  );
};

/** 0-14 s: a book opens on a short history of robot arms; the last page is our robot, and we dive into it. */
export const Book: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const {portrait} = useLayout();
  const t = frame / fps;
  const pw = portrait ? 486 : 620;
  const ph = Math.round(pw * (820 / 620));
  const pages = B.pages;
  const flipAt = (i: number) => B.coverOpenAt + i * B.flipEvery; // i = 0 is the cover
  const flipDur = B.flipSec;

  // Which spread is under the reader, and whether a sheet is mid-turn.
  let current = -1;
  for (let i = 0; i < pages.length; i++) if (t >= flipAt(i)) current = i;
  const turning = current >= 0 && t < flipAt(current) + flipDur;
  const turnP = turning ? easeInOut(Math.min(1, (t - flipAt(current)) / flipDur)) : 1;
  const angle = -180 * turnP;
  const opened = current >= 0 ? (current === 0 ? turnP : 1) : 0;

  const drawAt = (i: number) => sec(flipAt(i) + flipDur * 0.55);
  const left = (i: number) => (i < 0 ? null : <LeftPage page={pages[i]} index={i} pw={pw} ph={ph} />);
  const right = (i: number) => (
    <RightPage page={pages[i]} index={i} pw={pw} drawAt={drawAt(i)} last={i === pages.length - 1} />
  );

  // Camera: the closed book rises in, tilts toward us as it opens, then dives into the final plate.
  const rise = prog(frame, 0, 36, easeOut);
  const dive = prog(frame, sec(B.diveAt), sec(B.end - B.diveAt), easeIn);
  // Pan onto the plate first, then push in, so the move lands on the robot photo.
  const pan = prog(frame, sec(B.diveAt) - 6, sec(B.end - B.diveAt) * 0.7, easeInOut);
  const tilt = 22 - 10 * prog(frame, sec(B.coverOpenAt), sec(3), easeInOut);
  const shiftX = (-pw / 2) * (1 - opened);
  const scale = (portrait ? 0.98 : 1.0) + 0.06 * prog(frame, 0, sec(B.diveAt), easeInOut);
  const diveScale = 1 + dive * (portrait ? 3.2 : 2.6);
  const diveX = -pan * (pw / 2) * scale * diveScale;
  const board = 14;

  return (
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', perspective: 2600}}>
      <AbsoluteFill style={{background: 'radial-gradient(ellipse 55% 50% at 50% 55%, rgba(227,211,166,0.10), transparent 70%)',
        opacity: rise}} />
      <div style={{position: 'relative', width: 0, height: ph, transformStyle: 'preserve-3d', opacity: rise,
        filter: dive > 0.6 ? `blur(${(dive - 0.6) * 20}px)` : undefined,
        transform: `translateY(${(1 - rise) * 140 + (portrait ? 0 : 20)}px) translateX(${shiftX + diveX}px) rotateX(${tilt * (1 - dive)}deg) scale(${scale * diveScale})`}}>
        {/* Boards and page edges under the paper. */}
        <div style={{position: 'absolute', top: -board, height: ph + board * 2, left: (opened > 0 ? -pw : 0) - board,
          width: (opened > 0 ? pw * 2 : pw) + board * 2, borderRadius: 16, background: '#0a2019',
          boxShadow: '0 80px 120px -30px rgba(0,0,0,0.9), 0 30px 60px -20px rgba(0,0,0,0.8)', transform: 'translateZ(-6px)'}} />
        {[3, 2, 1].map((d) => (
          <React.Fragment key={d}>
            <div style={{position: 'absolute', left: d * 2, top: d * 2, width: pw, height: ph, background: '#d8cfba',
              borderRadius: 3, transform: `translateZ(${-d}px)`}} />
            {opened > 0.5 && (
              <div style={{position: 'absolute', left: -pw - d * 2, top: d * 2, width: pw, height: ph, background: '#d8cfba',
                borderRadius: 3, transform: `translateZ(${-d}px)`}} />
            )}
          </React.Fragment>
        ))}
        {/* Static left page (the previous spread until the turning sheet covers it). */}
        {current >= 1 && (
          <div style={{position: 'absolute', left: -pw, top: 0, width: pw, height: ph}}>
            {left(turning ? current - 1 : current)}
          </div>
        )}
        {current === 0 && turnP > 0.5 && (
          <div style={{position: 'absolute', left: -pw, top: 0, width: pw, height: ph}}>{left(0)}</div>
        )}
        {/* Static right page: the spread being turned to. */}
        <div style={{position: 'absolute', left: 0, top: 0, width: pw, height: ph}}>{right(Math.max(0, current))}</div>
        {/* The turning sheet, or the closed cover. */}
        {current < 0 && (
          <div style={{position: 'absolute', left: 0, top: 0, width: pw, height: ph, transform: 'translateZ(3px)'}}>
            <Cover pw={pw} />
          </div>
        )}
        {turning && (
          <Leaf pw={pw} ph={ph} angle={angle}
            front={current === 0 ? <Cover pw={pw} /> : right(current - 1)}
            back={left(current)} />
        )}
        {/* Soft gutter shadow. */}
        {opened > 0 && (
          <div style={{position: 'absolute', left: -40, top: 0, width: 80, height: ph, transform: 'translateZ(2px)',
            background: 'linear-gradient(90deg, transparent, rgba(40,30,15,0.18) 45%, rgba(40,30,15,0.18) 55%, transparent)',
            opacity: opened}} />
        )}
      </div>
      <div style={{position: 'absolute', bottom: portrait ? 300 : 46, width: '100%', textAlign: 'center', fontFamily: MONO,
        fontSize: 16, letterSpacing: '0.2em', color: 'rgba(227,211,166,0.45)', opacity: rise * (1 - dive) * (portrait ? 1 : 1),
        textTransform: 'uppercase'}}>
        {B.footnote}
      </div>
    </AbsoluteFill>
  );
};
