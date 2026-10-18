import React from 'react';
import {AbsoluteFill, Img, random, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {INK, Plate} from '../components/Art';
import {Countdown, DustBeam, FilmBurn, FilmLook} from '../components/Film';
import {MONO, SANS, cfg, easeIn, easeInOut, easeOut, prog, sec, useLayout} from '../theme';

const B = cfg.book;
const SERIF = '"Instrument Serif", serif';
const GOLD = '#c9a45c';
const PAPER = 'linear-gradient(180deg, #ece0c3 0%, #e2d3b0 100%)';

type Page = (typeof B.pages)[number];

const smooth = (a: number, b: number, t: number) => {
  const x = Math.min(1, Math.max(0, (t - a) / (b - a)));
  return x * x * (3 - 2 * x);
};

/** Yellowed paper: foxing spots, darkened edges, a few stains, seeded per page. */
const Aging: React.FC<{seed: string; side: 'left' | 'right'}> = ({seed, side}) => {
  const spots = Array.from({length: 9}, (_, i) => {
    const x = random(`${seed}-fx-${i}`) * 100;
    const y = random(`${seed}-fy-${i}`) * 100;
    const r = 0.6 + random(`${seed}-fr-${i}`) ** 2 * 3.2;
    return `radial-gradient(circle at ${x}% ${y}%, rgba(120,72,25,0.22) 0, rgba(120,72,25,0.12) ${r}%, transparent ${r * 1.8}%)`;
  });
  const ring = random(`${seed}-ring`) > 0.6;
  return (
    <>
      <div style={{position: 'absolute', inset: 0, background: spots.join(',')}} />
      {ring && (
        <div style={{position: 'absolute', left: `${20 + random(`${seed}-rx`) * 50}%`, top: `${55 + random(`${seed}-ry`) * 25}%`,
          width: 130, height: 130, borderRadius: 130, border: '6px solid rgba(110,65,20,0.10)', filter: 'blur(1.5px)'}} />
      )}
      <div style={{position: 'absolute', inset: 0, boxShadow: 'inset 0 0 60px rgba(110,70,25,0.35), inset 0 0 14px rgba(80,45,10,0.35)'}} />
      <div style={{position: 'absolute', inset: 0,
        background: `linear-gradient(${side === 'left' ? 270 : 90}deg, rgba(60,40,15,0.32), transparent 13%)`}} />
    </>
  );
};

const LeftPage: React.FC<{page: Page; index: number; pw: number}> = ({page, index, pw}) => {
  const k = pw / 620;
  return (
    <div style={{position: 'absolute', inset: 0, background: PAPER, padding: `${58 * k}px ${56 * k}px`, overflow: 'hidden'}}>
      <Aging seed={`L${index}`} side="left" />
      <div style={{position: 'relative', fontFamily: MONO, fontSize: 13 * k, letterSpacing: '0.22em', color: '#7d6c4f',
        textTransform: 'uppercase'}}>
        {B.runningHead}
      </div>
      <div style={{position: 'relative', fontFamily: SERIF, fontSize: (page.year.length > 4 ? 170 : 236) * k, lineHeight: 0.86,
        color: INK, marginTop: 70 * k, letterSpacing: '-0.02em'}}>
        {page.year}
      </div>
      <div style={{position: 'relative', width: 64 * k, height: 3 * k, background: '#7a1f12', margin: `${34 * k}px 0 ${26 * k}px`}} />
      <div style={{position: 'relative', fontFamily: SANS, fontWeight: 600, fontSize: 44 * k, color: INK, letterSpacing: '-0.02em',
        lineHeight: 1.05}}>
        {page.name}
      </div>
      <div style={{position: 'relative', fontFamily: SERIF, fontStyle: 'italic', fontSize: 38 * k, color: '#3b3a33',
        marginTop: 14 * k, lineHeight: 1.12}}>
        {page.line}
      </div>
      <div style={{position: 'absolute', left: 56 * k, bottom: 58 * k, fontFamily: MONO, fontSize: 14 * k, color: '#7d6c4f',
        letterSpacing: '0.06em'}}>
        {page.who}
      </div>
      <div style={{position: 'absolute', right: 40 * k, bottom: 26 * k, fontFamily: SERIF, fontSize: 20 * k, color: '#7d6c4f'}}>
        {index * 2 + 2}
      </div>
    </div>
  );
};

/** A glued-in photographic print with corner mounts. */
const Print: React.FC<{src: string; pw: number; tilt: number; credit: string}> = ({src, pw, tilt, credit}) => {
  const k = pw / 620;
  const w = 470 * k;
  const h = w * 0.75;
  const corner = (pos: React.CSSProperties, rot: number) => (
    <div style={{position: 'absolute', width: 34 * k, height: 34 * k, background: '#2b2219',
      clipPath: 'polygon(0 0, 100% 0, 0 100%)', transform: `rotate(${rot}deg)`, ...pos}} />
  );
  return (
    <div style={{position: 'absolute', left: (pw - w) / 2 - 12 * k, top: 120 * k, width: w + 24 * k, transform: `rotate(${tilt}deg)`}}>
      <div style={{background: '#f4efe4', padding: 12 * k, boxShadow: '0 6px 14px rgba(60,35,10,0.35), 0 1px 2px rgba(0,0,0,0.3)'}}>
        <Img src={staticFile(src)} style={{display: 'block', width: w, height: h, objectFit: 'cover'}} />
      </div>
      {corner({left: -6 * k, top: -6 * k}, 0)}
      {corner({right: -6 * k, top: -6 * k}, 90)}
      {corner({right: -6 * k, bottom: -6 * k}, 180)}
      {corner({left: -6 * k, bottom: -6 * k}, 270)}
      <div style={{marginTop: 14 * k, textAlign: 'center', fontFamily: MONO, fontSize: 11 * k, color: '#7d6c4f'}}>{credit}</div>
    </div>
  );
};

const RightPage: React.FC<{page: Page; index: number; pw: number; drawAt: number}> = ({page, index, pw, drawAt}) => {
  const frame = useCurrentFrame();
  const k = pw / 620;
  const p = prog(frame, drawAt, sec(B.drawSec), easeInOut);
  const photo = 'photo' in page ? (page.photo as string) : '';
  const open = page.art === 'open';
  return (
    <div style={{position: 'absolute', inset: 0, background: PAPER, overflow: 'hidden'}}>
      <Aging seed={`R${index}`} side="right" />
      {photo ? (
        <Print src={photo} pw={pw} tilt={index % 2 ? 1.6 : -1.4} credit={'credit' in page ? (page.credit as string) : ''} />
      ) : open ? (
        <div style={{position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
          <div style={{fontFamily: SERIF, fontStyle: 'italic', fontSize: 64 * k, color: 'rgba(40,30,20,0.55)', opacity: p,
            transform: `rotate(-4deg) translateY(${(1 - p) * 10}px)`}}>
            {B.openPageText}
          </div>
        </div>
      ) : (
        <div style={{position: 'absolute', left: (pw - 520 * k) / 2, top: 96 * k, filter: 'sepia(0.5)'}}>
          <Plate art={page.art} progress={p} size={520 * k} />
        </div>
      )}
      {!open && (
        <div style={{position: 'absolute', left: 0, right: 0, bottom: 58 * k, textAlign: 'center', fontFamily: SERIF,
          fontStyle: 'italic', fontSize: 22 * k, color: '#6a5a40'}}>
          {photo ? `pl. ${index + 1} — ${page.name}` : `pl. ${index + 1} — ${page.name}, after contemporary photographs`}
        </div>
      )}
      <div style={{position: 'absolute', left: 40 * k, bottom: 26 * k, fontFamily: SERIF, fontSize: 20 * k, color: '#7d6c4f'}}>
        {index * 2 + 3}
      </div>
    </div>
  );
};

/** Worn leather boards with gold tooling. */
const Cover: React.FC<{pw: number}> = ({pw}) => {
  const k = pw / 620;
  return (
    <div style={{position: 'absolute', inset: 0, borderRadius: `${4 * k}px ${12 * k}px ${12 * k}px ${4 * k}px`, overflow: 'hidden',
      background: 'radial-gradient(ellipse 90% 80% at 45% 40%, #5a2f1a 0%, #3d1f11 55%, #24120a 100%)'}}>
      <svg width="100%" height="100%" style={{position: 'absolute', inset: 0, opacity: 0.55, mixBlendMode: 'overlay'}}>
        <filter id="leather">
          <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="3" seed="4" />
          <feColorMatrix type="saturate" values="0" />
        </filter>
        <rect width="100%" height="100%" filter="url(#leather)" />
      </svg>
      <div style={{position: 'absolute', inset: 0,
        background: 'radial-gradient(circle at 96% 4%, rgba(255,220,170,0.18), transparent 14%), radial-gradient(circle at 98% 97%, rgba(255,220,170,0.16), transparent 12%), radial-gradient(circle at 30% 70%, rgba(0,0,0,0.25), transparent 30%)'}} />
      <div style={{position: 'absolute', inset: 30 * k, border: `${2 * k}px solid ${GOLD}aa`, borderRadius: 4 * k}} />
      <div style={{position: 'absolute', inset: 40 * k, border: `${1 * k}px solid ${GOLD}66`, borderRadius: 3 * k}} />
      <div style={{position: 'absolute', left: 0, top: 0, bottom: 0, width: 30 * k,
        background: 'linear-gradient(90deg, rgba(0,0,0,0.55), rgba(255,255,255,0.06) 60%, transparent)'}} />
      <div style={{position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', textAlign: 'center'}}>
        <div style={{fontFamily: MONO, fontSize: 16 * k, letterSpacing: '0.4em', color: `${GOLD}cc`}}>VOLUME I</div>
        <div style={{fontFamily: SERIF, fontSize: 116 * k, lineHeight: 0.95, color: GOLD, marginTop: 30 * k,
          textShadow: `0 ${2 * k}px 0 rgba(0,0,0,0.6), 0 -${1 * k}px 0 rgba(255,230,180,0.25)`}}>
          {B.coverTitle[0]}
          <br />
          {B.coverTitle[1]}
        </div>
        <div style={{fontFamily: SERIF, fontStyle: 'italic', fontSize: 40 * k, color: `${GOLD}dd`, marginTop: 22 * k}}>
          {B.coverSub}
        </div>
        <div style={{width: 120 * k, height: 1, background: `${GOLD}99`, margin: `${34 * k}px 0`}} />
        <div style={{fontFamily: MONO, fontSize: 20 * k, letterSpacing: '0.3em', color: `${GOLD}cc`}}>{B.coverYears}</div>
      </div>
    </div>
  );
};

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
        <div style={{position: 'absolute', inset: 0, background: `linear-gradient(90deg, rgba(0,0,0,${0.4 * lift}), rgba(0,0,0,${0.1 * lift}))`}} />
      </div>
      <div style={{position: 'absolute', inset: 0, backfaceVisibility: 'hidden', transform: 'rotateY(180deg)'}}>
        {back}
        <div style={{position: 'absolute', inset: 0, background: `linear-gradient(270deg, rgba(0,0,0,${0.35 * lift}), transparent)`}} />
      </div>
    </div>
  );
};

/** 0-15.5 s: film leader, then an old book of robot-arm history, shot like an archive reel. */
export const History: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const {portrait} = useLayout();
  const t = frame / fps;
  const pw = portrait ? 486 : 620;
  const ph = Math.round(pw * (820 / 620));
  const pages = B.pages;
  const flipAt = (i: number) => B.coverOpenAt + i * B.flipEvery; // i = 0 opens the cover
  const flipDur = B.flipSec;
  const countdownEnd = B.countdownBeats * 0.5;

  let current = -1;
  for (let i = 0; i < pages.length; i++) if (t >= flipAt(i)) current = i;
  const turning = current >= 0 && t < flipAt(current) + flipDur;
  const turnP = turning ? easeInOut(Math.min(1, (t - flipAt(current)) / flipDur)) : 1;
  const angle = -180 * turnP;
  const opened = current >= 0 ? (current === 0 ? turnP : 1) : 0;
  const drawAt = (i: number) => sec(flipAt(i) + flipDur * 0.6);
  const left = (i: number) => (i < 0 ? null : <LeftPage page={pages[i]} index={i} pw={pw} />);
  const right = (i: number) => <RightPage page={pages[i]} index={i} pw={pw} drawAt={drawAt(i)} />;

  // Colour arrives with the years: each page carries how black-and-white its era should look.
  const pageMono = (i: number) => (i < 0 ? 1 : pages[i].mono);
  const mono = current < 0 ? 1 : pageMono(current - 1) + (pageMono(current) - pageMono(current - 1)) * smooth(flipAt(current),
    flipAt(current) + 0.8, t);

  // Camera: book rises in, then on every page a push toward the plate (or the year), and back out for the turn.
  const rise = prog(frame, sec(countdownEnd), 30, easeOut);
  let s = (portrait ? 1.05 : 1.0) + 0.06 * prog(frame, sec(countdownEnd), sec(B.coverOpenAt - countdownEnd), easeInOut);
  let fx = 0;
  let fy = 0;
  let rz = 0;
  if (current >= 0) {
    const u = t - flipAt(current);
    const push = smooth(flipDur * 0.8, flipDur + 0.55, u) * (1 - smooth(B.flipEvery - 0.3, B.flipEvery, u));
    const last = current === pages.length - 1;
    const toPlate = current % 2 === 0 || last;
    const target = toPlate ? {x: pw * (portrait ? 0.5 : 0.32), y: -ph * 0.04} : {x: -pw * (portrait ? 0.5 : 0.32), y: -ph * 0.12};
    const zoom = (portrait ? 1.7 : 1.36) * (last ? 1.12 : 1);
    const lastHold = last ? smooth(flipDur * 0.8, flipDur + 0.55, u) : 0;
    const amt = last ? lastHold : push;
    s = 1 + (zoom - 1) * amt;
    // Bring the focus point most of the way to the centre as the push lands.
    fx = -target.x * s * amt * 0.85;
    fy = -target.y * s * amt * 0.85;
    rz = (current % 2 ? 1 : -1) * 1.2 * amt;
  }
  const shiftX = (-pw / 2) * (1 - opened);
  const tilt = 20 - 10 * prog(frame, sec(B.coverOpenAt), sec(1.2), easeInOut);
  const turnBlur = turning && current > 0 ? Math.sin(turnP * Math.PI) * 3 : 0;
  const board = 14;
  const finalPush = prog(frame, sec(B.burnAt) - 10, sec(B.end - B.burnAt) + 10, easeIn);

  const book = (
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', perspective: 2600,
      background: 'radial-gradient(ellipse 70% 60% at 50% 55%, #2a2016 0%, #120d08 60%, #050403 100%)'}}>
      <DustBeam opacity={rise} />
      <div style={{position: 'relative', width: 0, height: ph, transformStyle: 'preserve-3d', opacity: rise,
        filter: turnBlur > 0.3 ? `blur(${turnBlur}px)` : undefined,
        transform: `translateY(${(1 - rise) * 160 + (portrait ? 0 : 20)}px) translate(${shiftX + fx}px, ${fy}px) rotateZ(${rz}deg) rotateX(${tilt}deg) scale(${s * (1 + finalPush * 0.6)})`}}>
        <div style={{position: 'absolute', top: -board, height: ph + board * 2, left: (opened > 0 ? -pw : 0) - board,
          width: (opened > 0 ? pw * 2 : pw) + board * 2, borderRadius: 14, background: '#2a160c',
          boxShadow: '0 80px 120px -30px rgba(0,0,0,0.95), 0 30px 60px -20px rgba(0,0,0,0.85)', transform: 'translateZ(-6px)'}} />
        {[3, 2, 1].map((d) => (
          <React.Fragment key={d}>
            <div style={{position: 'absolute', left: d * 2, top: d * 2, width: pw, height: ph, background: '#cdbb94',
              borderRadius: 3, transform: `translateZ(${-d}px)`}} />
            {opened > 0.5 && (
              <div style={{position: 'absolute', left: -pw - d * 2, top: d * 2, width: pw, height: ph, background: '#cdbb94',
                borderRadius: 3, transform: `translateZ(${-d}px)`}} />
            )}
          </React.Fragment>
        ))}
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
          <div style={{position: 'absolute', left: -40, top: 0, width: 80, height: ph, transform: 'translateZ(2px)',
            background: 'linear-gradient(90deg, transparent, rgba(40,25,10,0.25) 45%, rgba(40,25,10,0.25) 55%, transparent)',
            opacity: opened}} />
        )}
      </div>
    </AbsoluteFill>
  );

  return (
    <AbsoluteFill>
      <FilmLook mono={mono} strength={1 - 0.55 * (1 - mono)}>
        {t < countdownEnd ? <Countdown beats={B.countdownBeats} beatFrames={sec(0.5)} /> : book}
      </FilmLook>
      <FilmBurn at={sec(B.burnAt)} dur={sec(B.end - B.burnAt)} />
    </AbsoluteFill>
  );
};
