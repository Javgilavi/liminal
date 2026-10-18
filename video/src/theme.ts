import {Easing, interpolate, useVideoConfig} from 'remotion';
import config from '../config.json';

export const cfg = config;
export const C = config.colors;
export const T = config.text;
export const SANS = 'Inter, sans-serif';
export const MONO = '"JetBrains Mono", monospace';

export const sec = (s: number) => Math.round(s * config.fps);

// One curve family everywhere keeps the motion consistent: fast out, long soft settle.
export const easeOut = Easing.bezier(0.16, 1, 0.3, 1);
export const easeInOut = Easing.bezier(0.65, 0, 0.35, 1);
export const easeIn = Easing.bezier(0.7, 0, 0.84, 0);

/** 0..1 progress between two frames, eased and clamped. */
export const prog = (frame: number, from: number, dur: number, ease = easeOut) =>
  interpolate(frame, [from, from + Math.max(1, dur)], [0, 1], {
    easing: ease,
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

export const useLayout = () => {
  const {width, height} = useVideoConfig();
  return {portrait: height > width, width, height};
};
