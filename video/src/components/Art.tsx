import React from 'react';

/**
 * Line-art plates for the history book: each robot is drawn from capsules, circles and paths in a
 * 400x400 box, and every stroke draws itself on in order. Illustrations, not technical drawings.
 */

export const INK = '#1c1f1d';
export const INK_GREEN = '#137a55';

type El =
  | {kind: 'path'; d: string; fill?: string; width?: number; color?: string}
  | {kind: 'circle'; cx: number; cy: number; r: number; fill?: string; width?: number; color?: string};

class Pen {
  els: El[] = [];

  path(d: string, opts: {fill?: string; width?: number; color?: string} = {}) {
    this.els.push({kind: 'path', d, ...opts});
    return this;
  }

  circle(cx: number, cy: number, r: number, opts: {fill?: string; width?: number; color?: string} = {}) {
    this.els.push({kind: 'circle', cx, cy, r, ...opts});
    return this;
  }

  rect(x: number, y: number, w: number, h: number, r = 4, opts: {fill?: string; width?: number; color?: string} = {}) {
    const d = `M${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h - r} Q${x + w},${y + h} ${
      x + w - r},${y + h} H${x + r} Q${x},${y + h} ${x},${y + h - r} V${y + r} Q${x},${y} ${x + r},${y} Z`;
    return this.path(d, opts);
  }

  /** A rounded link between two joints. */
  capsule(x1: number, y1: number, x2: number, y2: number, w: number, opts: {fill?: string; width?: number} = {}) {
    const len = Math.hypot(x2 - x1, y2 - y1) || 1;
    const nx = (-(y2 - y1) / len) * (w / 2);
    const ny = ((x2 - x1) / len) * (w / 2);
    const r = w / 2;
    const d = `M${x1 + nx},${y1 + ny} L${x2 + nx},${y2 + ny} A${r},${r} 0 0 0 ${x2 - nx},${y2 - ny} L${x1 - nx},${
      y1 - ny} A${r},${r} 0 0 0 ${x1 + nx},${y1 + ny} Z`;
    return this.path(d, opts);
  }

  line(...pts: number[]) {
    let d = `M${pts[0]},${pts[1]}`;
    for (let i = 2; i < pts.length; i += 2) d += ` L${pts[i]},${pts[i + 1]}`;
    return this.path(d);
  }

  /** Engraving-style ground: a base line and short hatch strokes under it. */
  ground(x1: number, x2: number, y: number) {
    this.line(x1, y, x2, y);
    let d = '';
    for (let x = x1 + 6; x < x2 - 4; x += 14) d += `M${x},${y + 4} L${x - 9},${y + 14} `;
    return this.path(d, {width: 1.6});
  }
}

const tint = 'rgba(19,122,85,0.16)';

const plates: Record<string, (p: Pen) => void> = {
  unimate: (p) => {
    p.ground(40, 370, 352);
    p.rect(70, 238, 172, 114, 6);
    p.rect(86, 254, 62, 82, 3, {fill: tint});
    p.line(162, 266, 226, 266).line(162, 282, 226, 282).line(162, 298, 226, 298);
    p.rect(118, 192, 96, 46, 8);
    p.circle(166, 186, 18);
    p.capsule(166, 182, 318, 182, 30);
    p.capsule(250, 182, 342, 182, 18, {fill: tint});
    p.circle(348, 182, 9);
    p.path('M346,192 L352,224 L342,228', {color: INK_GREEN, width: 3.4});
    p.path('M356,188 L372,214 L364,220', {color: INK_GREEN, width: 3.4});
  },
  stanford: (p) => {
    p.ground(60, 340, 352);
    p.rect(140, 334, 120, 18, 4);
    p.rect(178, 232, 44, 102, 6);
    p.circle(200, 214, 30);
    p.circle(200, 214, 11, {fill: tint});
    p.capsule(200, 214, 326, 146, 26);
    p.capsule(268, 177, 356, 129, 14, {fill: tint});
    p.circle(358, 128, 9);
    p.path('M362,136 L370,164 L362,168', {color: INK_GREEN, width: 3.4});
    p.path('M352,138 L350,168 L342,168', {color: INK_GREEN, width: 3.4});
    p.path('M188,300 C158,268 168,236 184,224', {width: 1.8});
  },
  puma: (p) => {
    p.ground(60, 360, 352);
    p.rect(146, 312, 108, 40, 8);
    p.rect(172, 218, 56, 94, 10);
    p.circle(200, 202, 32);
    p.circle(200, 202, 12, {fill: tint});
    p.capsule(200, 202, 318, 170, 40);
    p.circle(318, 170, 22);
    p.capsule(318, 170, 344, 268, 26, {fill: tint});
    p.circle(345, 272, 10);
    p.path('M338,282 L332,308', {color: INK_GREEN, width: 3.4});
    p.path('M352,282 L358,308', {color: INK_GREEN, width: 3.4});
  },
  baxter: (p) => {
    p.ground(80, 320, 356);
    p.capsule(200, 326, 200, 350, 18);
    p.line(150, 352, 250, 352);
    p.rect(148, 176, 104, 150, 16);
    p.circle(200, 218, 10, {fill: tint});
    p.capsule(200, 158, 200, 176, 14);
    p.rect(158, 92, 84, 62, 10, {fill: tint});
    p.circle(184, 122, 6, {fill: INK});
    p.circle(216, 122, 6, {fill: INK});
    p.capsule(150, 192, 98, 252, 22);
    p.circle(98, 252, 12);
    p.capsule(98, 252, 118, 312, 18);
    p.path('M112,322 L106,344 M126,320 L132,342', {color: INK_GREEN, width: 3.4});
    p.capsule(250, 192, 302, 252, 22);
    p.circle(302, 252, 12);
    p.capsule(302, 252, 282, 312, 18);
    p.path('M288,322 L294,344 M274,320 L268,342', {color: INK_GREEN, width: 3.4});
  },
  aloha: (p) => {
    p.line(60, 124, 340, 124).line(60, 124, 60, 300).line(340, 124, 340, 300);
    p.rect(188, 124, 24, 16, 3, {fill: tint});
    p.rect(40, 300, 320, 14, 3);
    p.line(70, 314, 70, 356).line(330, 314, 330, 356);
    p.rect(58, 278, 34, 22, 4);
    p.capsule(75, 280, 108, 226, 14);
    p.circle(108, 226, 7);
    p.capsule(108, 226, 164, 258, 11);
    p.path('M168,262 L180,282 M172,252 L188,268', {color: INK_GREEN, width: 3});
    p.rect(308, 278, 34, 22, 4);
    p.capsule(325, 280, 292, 226, 14);
    p.circle(292, 226, 7);
    p.capsule(292, 226, 236, 258, 11);
    p.path('M232,262 L220,282 M228,252 L212,268', {color: INK_GREEN, width: 3});
    p.rect(188, 280, 24, 20, 3, {fill: tint});
  },
  rt2: (p) => {
    p.rect(36, 82, 178, 112, 14);
    p.path('M70,194 L62,222 L96,194', {});
    p.line(58, 114, 192, 114).line(58, 138, 176, 138).line(58, 162, 150, 162);
    p.rect(52, 238, 92, 70, 8, {fill: tint});
    p.path('M60,296 L88,266 L104,282 L118,262 L138,296', {width: 1.8});
    p.circle(124, 254, 6);
    p.path('M222,140 C258,140 254,196 252,206', {color: INK_GREEN, width: 3});
    p.path('M244,198 L252,210 L262,198', {color: INK_GREEN, width: 3});
    p.path('M152,272 C190,272 206,250 236,240', {color: INK_GREEN, width: 3});
    p.ground(250, 380, 352);
    p.rect(270, 322, 92, 30, 6);
    p.capsule(316, 322, 316, 252, 28);
    p.circle(316, 250, 16, {fill: tint});
    p.capsule(316, 250, 258, 224, 22);
    p.path('M254,216 L236,212 M254,232 L238,240', {color: INK_GREEN, width: 3.4});
  },
  so101: (p) => {
    p.ground(80, 380, 352);
    p.rect(148, 300, 94, 52, 8);
    p.circle(168, 326, 4).circle(184, 326, 4).circle(200, 326, 4);
    p.rect(164, 262, 62, 40, 6, {fill: tint});
    p.capsule(196, 270, 236, 168, 30);
    for (const t of [0.25, 0.45, 0.65]) p.circle(196 + 40 * t, 270 - 102 * t, 4);
    p.circle(236, 166, 14, {fill: tint});
    p.capsule(236, 166, 318, 204, 24);
    p.circle(262, 178, 3.5).circle(282, 188, 3.5);
    p.rect(310, 194, 34, 36, 5);
    p.rect(316, 178, 16, 12, 3, {fill: INK});
    p.path('M316,230 L330,296 L342,230 Z', {color: INK_GREEN, width: 3});
    p.path('M348,226 L370,282 L360,286 L340,232', {color: INK_GREEN, width: 3});
  },
};

/** Draws one plate; `progress` 0..1 reveals the strokes in order, fills fade in at the end. */
export const Plate: React.FC<{art: string; progress: number; size: number}> = ({art, progress, size}) => {
  const pen = new Pen();
  plates[art]?.(pen);
  const n = pen.els.length;
  const fillP = Math.min(1, Math.max(0, (progress - 0.65) / 0.35));
  return (
    <svg width={size} height={size} viewBox="0 0 400 400" style={{overflow: 'visible'}}>
      {pen.els.map((el, i) => {
        const local = Math.min(1, Math.max(0, progress * (1 + n * 0.08) - i * 0.08));
        const common = {
          pathLength: 1,
          fill: el.fill ? el.fill : 'none',
          fillOpacity: el.fill ? fillP : 0,
          stroke: el.color ?? INK,
          strokeWidth: el.width ?? 2.6,
          strokeLinecap: 'round' as const,
          strokeLinejoin: 'round' as const,
          strokeDasharray: '1 1',
          strokeDashoffset: 1 - local,
        };
        return el.kind === 'path' ? (
          <path key={i} d={el.d} {...common} />
        ) : (
          <circle key={i} cx={el.cx} cy={el.cy} r={el.r} {...common} />
        );
      })}
    </svg>
  );
};
