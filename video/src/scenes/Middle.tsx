import React from 'react';
import {AbsoluteFill, Img, OffthreadVideo, staticFile, useCurrentFrame} from 'remotion';
import {Card, Eyebrow, MaskWords, SourceTag} from '../components/ui';
import {C, MONO, SANS, T, cfg, easeInOut, easeOut, prog, sec, useLayout} from '../theme';

const rise = (frame: number, delay: number, dist = 50) => {
  const p = prog(frame, delay, 20, easeOut);
  return {opacity: p, transform: `translateY(${(1 - p) * dist}px)`, filter: p < 0.98 ? `blur(${(1 - p) * 10}px)` : undefined};
};

/** 14.5-16 s: what FLUX imagined next to what the camera saw, same held-out take. */
export const Imagination: React.FC = () => {
  const frame = useCurrentFrame();
  const {portrait} = useLayout();
  const size = portrait ? 640 : 600;
  const gap = portrait ? 40 : 56;
  const clip = (file: string) => (
    <OffthreadVideo src={staticFile(file)} trimBefore={sec(cfg.imagination.from)} muted
      style={{width: '100%', height: '100%', objectFit: 'cover', filter: cfg.grade}} />
  );
  const pos = portrait
    ? [[(1080 - size) / 2, 250], [(1080 - size) / 2, 250 + size + gap]]
    : [[(1920 - size * 2 - gap) / 2, 150], [(1920 - size * 2 - gap) / 2 + size + gap, 150]];
  const push = 1 + 0.04 * prog(frame, 0, 45, easeInOut);
  return (
    <AbsoluteFill style={{transform: `scale(${push})`}}>
      <Card x={pos[0][0]} y={pos[0][1]} w={size} h={size} rotY={portrait ? 0 : 6} style={rise(frame, 0)}>
        {clip('imag_imagined.mp4')}
        <SourceTag kind="model" label={T.imaginedLabel} />
      </Card>
      <Card x={pos[1][0]} y={pos[1][1]} w={size} h={size} rotY={portrait ? 0 : -6} style={rise(frame, 3)}>
        {clip('imag_actual.mp4')}
        <SourceTag kind="real" label={T.actualLabel} />
      </Card>
      <AbsoluteFill style={{justifyContent: 'flex-end', alignItems: 'center', paddingBottom: portrait ? 230 : 110}}>
        <MaskWords text={T.imagineCaption} at={4}
          style={{fontFamily: SANS, fontWeight: 600, fontSize: portrait ? 76 : 64, letterSpacing: '-0.03em', color: C.text}}
          wordStyle={(i) => (i === 0 ? {color: C.accent} : {})} />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

type Pt = [number, number];
const NODE_W = 236;
const NODE_H = 88;

/** Edge path between two node centres: straight when aligned, otherwise one clean elbow. */
const edgePath = (a: Pt, b: Pt, portrait: boolean): string => {
  const [ax, ay] = a;
  const [bx, by] = b;
  const hw = NODE_W / 2;
  const hh = NODE_H / 2;
  if (Math.abs(ay - by) < 4) {
    const s = Math.sign(bx - ax);
    return `M ${ax + s * hw} ${ay} L ${bx - s * hw} ${by}`;
  }
  if (Math.abs(ax - bx) < 4) {
    const s = Math.sign(by - ay);
    return `M ${ax} ${ay + s * hh} L ${bx} ${by - s * hh}`;
  }
  if (portrait) {
    const s = Math.sign(by - ay);
    const my = (ay + s * hh + by - s * hh) / 2;
    return `M ${ax} ${ay + s * hh} L ${ax} ${my} L ${bx} ${my} L ${bx} ${by - s * hh}`;
  }
  const s = Math.sign(bx - ax);
  const mx = (ax + s * hw + bx - s * hw) / 2;
  return `M ${ax + s * hw} ${ay} L ${mx} ${ay} L ${mx} ${by} L ${bx - s * hw} ${by}`;
};

/** The runtime as a diagram that draws itself, then loops a signal through perception back into the tree. */
export const Diagram: React.FC<{dim?: number; startFrame?: number}> = ({dim = 0, startFrame = 0}) => {
  const frame = useCurrentFrame() + startFrame;
  const {portrait} = useLayout();
  const D = cfg.diagram;
  const pos = (portrait ? D.portrait : D.landscape) as unknown as Record<string, Pt>;
  const box = portrait ? {w: 1000, h: 1260, x: 100, y: 390} : {w: 1700, h: 650, x: 51, y: 250};
  const order = D.nodes.map((n) => n.id);
  const nodeAt = (id: string) => 4 + order.indexOf(id) * 6;
  const flow = (frame / 30) % 1;
  return (
    <AbsoluteFill style={{opacity: 1 - dim * 0.75, filter: dim ? `blur(${dim * 8}px)` : undefined}}>
      <div style={{position: 'absolute', left: portrait ? 70 : 120, top: portrait ? 260 : 120, ...rise(frame, 0, 30)}}>
        <Eyebrow>{T.howLabel}</Eyebrow>
      </div>
      <div style={{position: 'absolute', left: box.x, top: box.y, width: box.w, height: box.h}}>
        <svg width={box.w} height={box.h} style={{position: 'absolute', overflow: 'visible'}}>
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
              <path d="M 0 1 L 8 5 L 0 9" fill="none" stroke={C.muted} strokeWidth="1.6" />
            </marker>
          </defs>
          {D.edges.map(([a, b], i) => {
            const d = edgePath(pos[a], pos[b], portrait);
            const p = prog(frame, nodeAt(a) + 5, 10, easeInOut);
            const loop = b === 'bt' || a === 'arms' || a === 'cams' || a === 'vlm';
            return (
              <g key={i}>
                <path d={d} pathLength={1} fill="none" stroke="rgba(255,255,255,0.22)" strokeWidth={2}
                  strokeDasharray="1 1" strokeDashoffset={1 - p} markerEnd={p > 0.95 ? 'url(#arrow)' : undefined} />
                {p >= 1 && (
                  <path d={d} pathLength={1} fill="none" stroke={C.accent} strokeWidth={3} strokeLinecap="round"
                    strokeDasharray="0.06 0.94" strokeDashoffset={-((flow + i * 0.13) % 1)}
                    style={{filter: `drop-shadow(0 0 6px ${C.accent})`}} opacity={loop ? 1 : 0.8} />
                )}
              </g>
            );
          })}
        </svg>
        {D.nodes.map((n) => {
          const [cx, cy] = pos[n.id];
          const p = prog(frame, nodeAt(n.id), 14, easeOut);
          const accent = 'accent' in n && n.accent;
          return (
            <div
              key={n.id}
              style={{
                position: 'absolute',
                left: cx - NODE_W / 2,
                top: cy - NODE_H / 2,
                width: NODE_W,
                height: NODE_H,
                borderRadius: 18,
                background: accent ? `linear-gradient(180deg, ${C.accent}1f, ${C.accent}0a)` : C.surfaceHi,
                border: `1px solid ${accent ? `${C.accent}66` : C.line}`,
                boxShadow: accent
                  ? `0 0 40px ${C.accent}22, 0 30px 60px -30px rgba(0,0,0,0.9)`
                  : '0 30px 60px -30px rgba(0,0,0,0.9), inset 0 1px 0 rgba(255,255,255,0.06)',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                padding: '0 22px',
                opacity: p,
                transform: `translateY(${(1 - p) * 24}px) scale(${0.92 + 0.08 * p})`,
              }}
            >
              <div style={{fontFamily: SANS, fontWeight: 600, fontSize: 25, letterSpacing: '-0.015em', color: C.text}}>
                {n.label}
              </div>
              <div style={{fontFamily: MONO, fontSize: 15, color: accent ? C.accent : C.muted, marginTop: 4}}>{n.sub}</div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

/** 19.5-21 s: the live UI's reactive task tree, floating over the dimmed diagram. */
export const LiveUi: React.FC = () => {
  const frame = useCurrentFrame();
  const {portrait} = useLayout();
  const dur = sec(cfg.shots.ui.end - cfg.shots.ui.start);
  const w = portrait ? 960 : 1320;
  const h = Math.round((w * 655) / 1010);
  const scroll = prog(frame, 0, dur, easeInOut);
  return (
    <AbsoluteFill>
      <Diagram dim={1} startFrame={sec(cfg.shots.diagram.end - cfg.shots.diagram.start)} />
      <Card x={(portrait ? 1080 : 1920) / 2 - w / 2} y={portrait ? (1920 - h) / 2 : 110} w={w} h={h} rotX={portrait ? 4 : 8}
        rotY={portrait ? 0 : -4 + 3 * scroll} style={rise(frame, 0, 80)}>
        <Img src={staticFile('ui_task_tree.png')}
          style={{width: '100%', position: 'absolute', top: 0, transform: `scale(${1.02 + 0.03 * scroll})`, transformOrigin: '50% 0%'}} />
        <SourceTag kind="sim" label={T.uiLabel} />
      </Card>
    </AbsoluteFill>
  );
};

/** 21-23 s: the real rig next to its MuJoCo twin. */
export const Twin: React.FC = () => {
  const frame = useCurrentFrame();
  const {portrait} = useLayout();
  const dur = sec(cfg.shots.twin.end - cfg.shots.twin.start);
  const w = portrait ? 900 : 820;
  const h = Math.round((w * 338) / 600);
  const gap = portrait ? 50 : 60;
  const p = prog(frame, 0, dur, easeInOut);
  const pos = portrait
    ? [[90, 1920 / 2 - h - gap / 2], [90, 1920 / 2 + gap / 2]]
    : [[1920 / 2 - w - gap / 2, 1080 / 2 - h / 2], [1920 / 2 + gap / 2, 1080 / 2 - h / 2]];
  const img = (file: string, dir: number) => (
    <Img src={staticFile(file)} style={{width: '100%', height: '100%', objectFit: 'cover', filter: cfg.grade,
      transform: `scale(${1.08 - 0.04 * p}) translateX(${dir * 10 * p}px)`}} />
  );
  return (
    <AbsoluteFill>
      <div style={{position: 'absolute', left: portrait ? 90 : 120, top: portrait ? 340 : 120, ...rise(frame, 0, 30)}}>
        <Eyebrow>{T.twinLabel}</Eyebrow>
      </div>
      <Card x={pos[0][0]} y={pos[0][1]} w={w} h={h} rotY={portrait ? 0 : 7} style={rise(frame, 0)}>
        {img('twin_real.png', -1)}
        <SourceTag kind="real" label={T.twinRealLabel} />
      </Card>
      <Card x={pos[1][0]} y={pos[1][1]} w={w} h={h} rotY={portrait ? 0 : -7} style={rise(frame, 4)}>
        {img('twin_sim.png', 1)}
        <SourceTag kind="sim" label={T.twinSimLabel} />
      </Card>
      <AbsoluteFill style={{justifyContent: 'flex-end', alignItems: 'center', paddingBottom: portrait ? 290 : 150}}>
        <MaskWords text={T.twinCaption} at={8}
          style={{fontFamily: SANS, fontWeight: 600, fontSize: portrait ? 56 : 52, letterSpacing: '-0.025em', color: C.text}} />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
