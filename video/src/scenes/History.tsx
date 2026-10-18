import React from 'react';
import {AbsoluteFill, Img, random, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {DustBeam, FilmBurn, FilmLook} from '../components/Film';
import {MONO, SANS, cfg, easeInOut, easeOut, prog, sec, useLayout} from '../theme';

const B = cfg.book;
const SERIF = '"Instrument Serif", serif';
const INK = '#1b1a17';
const GOLD = '#c9a45c';

type Page = {
  year: string;
  name: string;
  line: string;
  who: string;
  photo?: string;
  photoAspect?: number;
  caption?: string;
  credit?: string;
  coda?: string;
};

const smooth = (a: number, b: number, t: number) => {
  const x = Math.min(1, Math.max(0, (t - a) / (b - a)));
  return x * x * (3 - 2 * x);
};

/** Laid paper: fibre noise, foxing, a curved page shaded toward the gutter and darkened at the fore-edge. */
const Paper: React.FC<{seed: string; side: 'left' | 'right'}> = ({seed, side}) => {
  const spots = Array.from({length: 7}, (_, i) => {
    const x = random(`${seed}-fx-${i}`) * 100;
    const y = random(`${seed}-fy-${i}`) * 100;
    const r = 0.5 + random(`${seed}-fr-${i}`) ** 2 * 2.6;
    return `radial-gradient(circle at ${x}% ${y}%, rgba(90,60,25,0.20) 0, rgba(90,60,25,0.10) ${r}%, transparent ${r * 1.8}%)`;
  });
  const toGutter = side === 'left' ? 90 : 270;
  return (
    <>
      <div style={{position: 'absolute', inset: 0, background: 'linear-gradient(180deg, #ebe3d0, #ddd3bb)'}} />
      <svg width="100%" height="100%" style={{position: 'absolute', inset: 0, mixBlendMode: 'multiply', opacity: 0.5}}>
        <filter id={`paper-${seed}`}>
          <feTurbulence type="fractalNoise" baseFrequency="0.75 0.35" numOctaves="4" seed={seed.length * 7} />
          <feColorMatrix type="matrix" values="0 0 0 0 0.55  0 0 0 0 0.5  0 0 0 0 0.42  0 0 0 0.55 0" />
        </filter>
        <rect width="100%" height="100%" filter={`url(#paper-${seed})`} />
      </svg>
      <div style={{position: 'absolute', inset: 0, background: spots.join(',')}} />
      {/* Curvature: shadow into the gutter, a soft highlight where the page rises, a darker fore-edge. */}
      <div style={{position: 'absolute', inset: 0, background: `linear-gradient(${toGutter}deg, transparent 0%, transparent 62%, rgba(255,255,255,0.18) 76%, rgba(40,28,12,0.25) 92%, rgba(25,15,5,0.55) 100%)`}} />
      <div style={{position: 'absolute', inset: 0, background: `linear-gradient(${toGutter}deg, rgba(60,40,15,0.28), transparent 7%)`}} />
      <div style={{position: 'absolute', inset: 0, boxShadow: 'inset 0 0 40px rgba(80,55,20,0.35)'}} />
    </>
  );
};

const ink: React.CSSProperties = {color: INK, textShadow: '0 0 0.6px rgba(20,15,10,0.8)'};

const LeftPage: React.FC<{page: Page; index: number; pw: number}> = ({page, index, pw}) => {
  const k = pw / 620;
  return (
    <div style={{position: 'absolute', inset: 0, overflow: 'hidden'}}>
      <Paper seed={`L${index}`} side="left" />
      <div style={{position: 'absolute', inset: `${60 * k}px ${58 * k}px`}}>
        <div style={{fontFamily: MONO, fontSize: 13 * k, letterSpacing: '0.24em', color: '#5d5445', textTransform: 'uppercase'}}>
          {B.runningHead}
        </div>
        <div style={{...ink, fontFamily: SERIF, fontSize: 250 * k, lineHeight: 0.86, marginTop: 64 * k, letterSpacing: '-0.02em'}}>
          {page.year}
        </div>
        <div style={{width: 70 * k, height: 3 * k, background: INK, margin: `${32 * k}px 0 ${24 * k}px`}} />
        <div style={{...ink, fontFamily: SANS, fontWeight: 600, fontSize: 46 * k, letterSpacing: '-0.02em', lineHeight: 1.05}}>
          {page.name}
        </div>
        <div style={{...ink, fontFamily: SERIF, fontStyle: 'italic', fontSize: 40 * k, marginTop: 14 * k, lineHeight: 1.12,
          color: '#2e2b26'}}>
          {page.line}
        </div>
      </div>
      <div style={{position: 'absolute', left: 58 * k, bottom: 56 * k, fontFamily: MONO, fontSize: 14 * k, color: '#5d5445',
        letterSpacing: '0.06em'}}>
        {page.who}
      </div>
      <div style={{position: 'absolute', right: 40 * k, bottom: 26 * k, fontFamily: SERIF, fontSize: 20 * k, color: '#5d5445'}}>
        {index * 2 + 2}
      </div>
    </div>
  );
};

/** Right page: a mounted photographic print, or, while a photo is missing, a catalogue entry set in type. */
const RightPage: React.FC<{page: Page; index: number; pw: number}> = ({page, index, pw}) => {
  const k = pw / 620;
  const photo = page.photo;
  const w = 480 * k;
  const h = w * (page.photoAspect ?? 0.75);
  const corner = (pos: React.CSSProperties, rot: number) => (
    <div style={{position: 'absolute', width: 36 * k, height: 36 * k, background: '#1e1a15',
      clipPath: 'polygon(0 0, 100% 0, 0 100%)', transform: `rotate(${rot}deg)`, ...pos}} />
  );
  return (
    <div style={{position: 'absolute', inset: 0, overflow: 'hidden'}}>
      <Paper seed={`R${index}`} side="right" />
      {photo ? (
        <div style={{position: 'absolute', left: (pw - w) / 2 - 13 * k, top: 110 * k, width: w + 26 * k,
          transform: `rotate(${index % 2 ? 1.4 : -1.2}deg)`}}>
          <div style={{background: '#f2eee6', padding: 13 * k, boxShadow: '0 8px 18px rgba(40,25,10,0.45), 0 1px 2px rgba(0,0,0,0.35)'}}>
            <Img src={staticFile(photo)} style={{display: 'block', width: w, height: h, objectFit: 'cover',
              filter: 'contrast(1.15) brightness(0.98)'}} />
          </div>
          {corner({left: -7 * k, top: -7 * k}, 0)}
          {corner({right: -7 * k, top: -7 * k}, 90)}
          {corner({right: -7 * k, bottom: -7 * k}, 180)}
          {corner({left: -7 * k, bottom: -7 * k}, 270)}
          <div style={{marginTop: 18 * k, textAlign: 'center', fontFamily: SERIF, fontStyle: 'italic', fontSize: 23 * k, color: '#3d382f'}}>
            pl. {index + 1} — {page.caption ?? page.name}
          </div>
          {page.credit && (
            <div style={{marginTop: 6 * k, textAlign: 'center', fontFamily: MONO, fontSize: 11 * k, color: '#5d5445'}}>{page.credit}</div>
          )}
        </div>
      ) : (
        <div style={{position: 'absolute', inset: `${150 * k}px ${70 * k}px`, display: 'flex', flexDirection: 'column',
          justifyContent: 'center'}}>
          <div style={{fontFamily: MONO, fontSize: 14 * k, letterSpacing: '0.3em', color: '#5d5445'}}>
            NO. {String(index + 1).padStart(3, '0')}
          </div>
          <div style={{...ink, fontFamily: SERIF, fontStyle: 'italic', fontSize: 78 * k, lineHeight: 1, marginTop: 20 * k}}>
            {page.name}
          </div>
          <div style={{width: '100%', height: 1, background: '#5d5445', margin: `${26 * k}px 0`}} />
          <div style={{fontFamily: MONO, fontSize: 15 * k, color: '#3d382f', lineHeight: 1.7}}>{page.who}</div>
        </div>
      )}
      {page.coda && (
        <div style={{position: 'absolute', left: 0, right: 0, bottom: 64 * k, textAlign: 'center', fontFamily: SERIF,
          fontStyle: 'italic', fontSize: 44 * k, color: '#2e2b26', transform: 'rotate(-2deg)'}}>
          {page.coda}
        </div>
      )}
      <div style={{position: 'absolute', left: 40 * k, bottom: 26 * k, fontFamily: SERIF, fontSize: 20 * k, color: '#5d5445'}}>
        {index * 2 + 3}
      </div>
    </div>
  );
};

/** Worn leather boards, gold tooling rubbed at the edges. */
const Cover: React.FC<{pw: number}> = ({pw}) => {
  const k = pw / 620;
  return (
    <div style={{position: 'absolute', inset: 0, borderRadius: `${4 * k}px ${12 * k}px ${12 * k}px ${4 * k}px`, overflow: 'hidden',
      background: 'radial-gradient(ellipse 90% 80% at 45% 40%, #5a2f1a 0%, #3d1f11 55%, #1d0e07 100%)'}}>
      <svg width="100%" height="100%" style={{position: 'absolute', inset: 0, opacity: 0.6, mixBlendMode: 'overlay'}}>
        <filter id="leather">
          <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="4" seed="4" />
          <feColorMatrix type="saturate" values="0" />
        </filter>
        <rect width="100%" height="100%" filter="url(#leather)" />
      </svg>
      <div style={{position: 'absolute', inset: 0,
        background: 'radial-gradient(circle at 97% 3%, rgba(255,225,180,0.22), transparent 13%), radial-gradient(circle at 98% 97%, rgba(255,225,180,0.18), transparent 12%), radial-gradient(ellipse at 30% 75%, rgba(0,0,0,0.3), transparent 35%), linear-gradient(115deg, transparent 40%, rgba(255,240,210,0.07) 50%, transparent 60%)'}} />
      <div style={{position: 'absolute', inset: 30 * k, border: `${2 * k}px solid ${GOLD}99`, borderRadius: 4 * k}} />
      <div style={{position: 'absolute', inset: 40 * k, border: `${1 * k}px solid ${GOLD}55`, borderRadius: 3 * k}} />
      <div style={{position: 'absolute', left: 0, top: 0, bottom: 0, width: 32 * k,
        background: 'linear-gradient(90deg, rgba(0,0,0,0.6), rgba(255,255,255,0.06) 60%, transparent)'}} />
      <div style={{position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', textAlign: 'center'}}>
        <div style={{fontFamily: MONO, fontSize: 16 * k, letterSpacing: '0.4em', color: `${GOLD}bb`}}>VOLUME I</div>
        <div style={{fontFamily: SERIF, fontSize: 120 * k, lineHeight: 0.95, color: GOLD, marginTop: 30 * k,
          textShadow: `0 ${2 * k}px 0 rgba(0,0,0,0.65), 0 -${1 * k}px 0 rgba(255,230,180,0.25)`}}>
          {B.coverTitle[0]}
          <br />
          {B.coverTitle[1]}
        </div>
        <div style={{fontFamily: SERIF, fontStyle: 'italic', fontSize: 40 * k, color: `${GOLD}dd`, marginTop: 22 * k}}>{B.coverSub}</div>
        <div style={{width: 120 * k, height: 1, background: `${GOLD}88`, margin: `${34 * k}px 0`}} />
        <div style={{fontFamily: MONO, fontSize: 20 * k, letterSpacing: '0.3em', color: `${GOLD}bb`}}>{B.coverYears}</div>
      </div>
    </div>
  );
};

/** A sheet turning about the spine, bending slightly as it goes. */
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
      transform: `rotateY(${angle}deg) translateZ(${1 + lift * 3}px) skewY(${-lift * 2}deg)`, transformStyle: 'preserve-3d', zIndex: 5}}>
      <div style={{position: 'absolute', inset: 0, backfaceVisibility: 'hidden'}}>
        {front}
        <div style={{position: 'absolute', inset: 0, background: `linear-gradient(90deg, rgba(0,0,0,${0.45 * lift}), rgba(255,255,255,${0.12 * lift}) 70%, rgba(0,0,0,${0.15 * lift}))`}} />
      </div>
      <div style={{position: 'absolute', inset: 0, backfaceVisibility: 'hidden', transform: 'rotateY(180deg)'}}>
        {back}
        <div style={{position: 'absolute', inset: 0, background: `linear-gradient(270deg, rgba(0,0,0,${0.4 * lift}), transparent)`}} />
      </div>
    </div>
  );
};

/** Out-of-focus dust drifting between the lens and the book. */
const ForegroundDust: React.FC = () => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      {Array.from({length: 9}, (_, i) => {
        const size = 26 + random(`fd-s-${i}`) * 70;
        const x = ((random(`fd-x-${i}`) * width + frame * (0.6 + random(`fd-v-${i}`))) % (width + 200)) - 100;
        const y = random(`fd-y-${i}`) * height + Math.sin(frame / 25 + i) * 20;
        return (
          <div key={i} style={{position: 'absolute', left: x, top: y, width: size, height: size, borderRadius: size,
            background: 'radial-gradient(circle, rgba(255,240,215,0.22), rgba(255,240,215,0.06) 60%, transparent 70%)',
            filter: 'blur(4px)'}} />
        );
      })}
    </AbsoluteFill>
  );
};

/** 0-7.5 s: an old book of robot-arm history, in black and white, shot like an archive reel. */
export const History: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const {portrait} = useLayout();
  const t = frame / fps;
  const pw = portrait ? 486 : 620;
  const ph = Math.round(pw * (820 / 620));
  const pages = B.pages as Page[];
  const flipAt = (i: number) => B.coverOpenAt + i * B.flipEvery; // i = 0 opens the cover
  const flipDur = B.flipSec;

  let current = -1;
  for (let i = 0; i < pages.length; i++) if (t >= flipAt(i)) current = i;
  const turning = current >= 0 && t < flipAt(current) + flipDur;
  const turnP = turning ? easeInOut(Math.min(1, (t - flipAt(current)) / flipDur)) : 1;
  const angle = -180 * turnP;
  const opened = current >= 0 ? (current === 0 ? turnP : 1) : 0;
  const left = (i: number) => (i < 0 ? null : <LeftPage page={pages[i]} index={i} pw={pw} />);
  const right = (i: number) => <RightPage page={pages[i]} index={i} pw={pw} />;

  // One continuous camera move: in from the dark, over the open book, and down into the last photograph.
  const lightOn = prog(frame, 0, 12, easeOut);
  const travel = smooth(B.coverOpenAt, B.burnAt, t);
  const last = smooth(flipAt(pages.length - 1) + flipDur, B.end, t);
  const s = (portrait ? 1.0 : 0.92) + (portrait ? 0.55 : 0.32) * travel + 0.5 * last;
  const fx = (portrait ? -pw * 0.5 : -pw * 0.18) * travel * (1 + last);
  const fy = 20 - 40 * travel;
  const tilt = 28 - 12 * smooth(0, B.coverOpenAt + 0.6, t);
  const rz = -2 + 3 * travel;
  const shake = turning ? Math.sin(turnP * Math.PI) : 0;
  const shiftX = (-pw / 2) * (1 - opened);
  const board = 14;
  const edgeLines = 'repeating-linear-gradient(90deg, #d8ccb0 0 1px, #b9ac8f 1px 2px)';

  const book = (
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', perspective: 2400,
      background: 'radial-gradient(ellipse 65% 55% at 50% 55%, #2c241b 0%, #120e0a 60%, #040303 100%)'}}>
      <DustBeam opacity={lightOn} />
      <div style={{position: 'relative', width: 0, height: ph, transformStyle: 'preserve-3d', opacity: lightOn,
        filter: shake > 0.2 ? `blur(${shake * 2.5}px)` : undefined,
        transform: `translate(${shiftX + fx + shake * 6}px, ${fy}px) rotateZ(${rz}deg) rotateX(${tilt}deg) scale(${s})`}}>
        <div style={{position: 'absolute', top: -board, height: ph + board * 2, left: (opened > 0 ? -pw : 0) - board,
          width: (opened > 0 ? pw * 2 : pw) + board * 2, borderRadius: 14, background: '#25140b',
          boxShadow: '0 90px 130px -30px rgba(0,0,0,0.95), 0 30px 60px -20px rgba(0,0,0,0.85)', transform: 'translateZ(-8px)'}} />
        {/* Page block: many thin edges at the fore-edge and foot. */}
        <div style={{position: 'absolute', left: pw - 2, top: 4, width: 9, height: ph - 2, background: edgeLines,
          transform: 'translateZ(-4px)'}} />
        <div style={{position: 'absolute', left: 4, top: ph - 2, width: pw - 4, height: 8,
          background: 'repeating-linear-gradient(0deg, #d8ccb0 0 1px, #b9ac8f 1px 2px)', transform: 'translateZ(-4px)'}} />
        {opened > 0.5 && (
          <div style={{position: 'absolute', left: -pw - 7, top: 4, width: 9, height: ph - 2, background: edgeLines,
            transform: 'translateZ(-4px)'}} />
        )}
        {current >= 1 && (
          <div style={{position: 'absolute', left: -pw, top: 0, width: pw, height: ph}}>{left(turning ? current - 1 : current)}</div>
        )}
        {current === 0 && turnP > 0.5 && <div style={{position: 'absolute', left: -pw, top: 0, width: pw, height: ph}}>{left(0)}</div>}
        <div style={{position: 'absolute', left: 0, top: 0, width: pw, height: ph}}>{right(Math.max(0, current))}</div>
        {current < 0 && (
          <div style={{position: 'absolute', left: 0, top: 0, width: pw, height: ph, transform: 'translateZ(3px)'}}>
            <Cover pw={pw} />
          </div>
        )}
        {turning && (
          <Leaf pw={pw} ph={ph} angle={angle} front={current === 0 ? <Cover pw={pw} /> : right(current - 1)} back={left(current)} />
        )}
        {opened > 0 && (
          <div style={{position: 'absolute', left: -50, top: 0, width: 100, height: ph, transform: 'translateZ(2px)',
            background: 'linear-gradient(90deg, transparent, rgba(30,20,8,0.35) 46%, rgba(30,20,8,0.35) 54%, transparent)',
            opacity: opened}} />
        )}
      </div>
      {/* Shallow depth of field: the frame edges fall out of focus. */}
      <AbsoluteFill style={{backdropFilter: 'blur(5px)', WebkitBackdropFilter: 'blur(5px)',
        maskImage: 'radial-gradient(ellipse 55% 50% at 50% 50%, transparent 55%, black 100%)',
        WebkitMaskImage: 'radial-gradient(ellipse 55% 50% at 50% 50%, transparent 55%, black 100%)'}} />
      <ForegroundDust />
    </AbsoluteFill>
  );

  return (
    <AbsoluteFill>
      <FilmLook mono={1} strength={1}>
        {book}
        <AbsoluteFill style={{background: '#000', opacity: 1 - lightOn}} />
      </FilmLook>
      <FilmBurn at={sec(B.burnAt)} dur={sec(B.end - B.burnAt)} />
    </AbsoluteFill>
  );
};
