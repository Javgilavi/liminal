import React from 'react';
import {AbsoluteFill, Audio, staticFile} from 'remotion';
import './fonts';
import {Backdrop, FilmFinish, Shot} from './components/ui';
import {Hero} from './scenes/Hero';
import {Diagram, Imagination, LiveUi, Twin} from './scenes/Middle';
import {HookMacro, HookReveal, Problem} from './scenes/Opening';
import {EndCard, TryIt} from './scenes/Closing';
import {cfg} from './theme';

const S = cfg.shots;

export const Main: React.FC = () => (
  <AbsoluteFill style={{background: cfg.colors.bg, overflow: 'hidden'}}>
    {cfg.music && <Audio src={staticFile(cfg.music)} />}
    <Backdrop />
    <Shot {...S.hookMacro}>
      <HookMacro />
    </Shot>
    <Shot {...S.hookReveal} exit="whip">
      <HookReveal />
    </Shot>
    <Shot {...S.problem} enter="whip" exit="whip">
      <Problem />
    </Shot>
    <Shot {...S.hero} enter="whip" exit="zoom">
      <Hero />
    </Shot>
    <Shot {...S.imagination} enter="zoom" exit="whip">
      <Imagination />
    </Shot>
    <Shot {...S.diagram} enter="whip">
      <Diagram />
    </Shot>
    <Shot {...S.ui} exit="whip">
      <LiveUi />
    </Shot>
    <Shot {...S.twin} enter="whip" exit="zoom">
      <Twin />
    </Shot>
    <Shot {...S.tryIt} enter="zoom" exit="fade">
      <TryIt />
    </Shot>
    <Shot {...S.end} enter="fade">
      <EndCard />
    </Shot>
    <FilmFinish />
  </AbsoluteFill>
);
