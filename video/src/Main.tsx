import React from 'react';
import {AbsoluteFill, Audio, staticFile} from 'remotion';
import './fonts';
import {Backdrop, FilmFinish, Shot} from './components/ui';
import {EndCard, TryIt} from './scenes/Closing';
import {Hero} from './scenes/Hero';
import {History} from './scenes/History';
import {Diagram, Imagination} from './scenes/Middle';
import {Problem} from './scenes/Problem';
import {Reveal} from './scenes/Reveal';
import {Sim} from './scenes/Sim';
import {cfg} from './theme';

const S = cfg.shots;

export const Main: React.FC = () => (
  <AbsoluteFill style={{background: cfg.colors.bg, overflow: 'hidden'}}>
    {cfg.music && <Audio src={staticFile(cfg.music)} />}
    <Backdrop />
    <Shot {...S.history}>
      <History />
    </Shot>
    <Shot {...S.problem}>
      <Problem />
    </Shot>
    <Shot {...S.reveal} enter="zoom">
      <Reveal />
    </Shot>
    <Shot {...S.sim} exit="whip">
      <Sim />
    </Shot>
    <Shot {...S.hero} enter="whip" exit="zoom">
      <Hero />
    </Shot>
    <Shot {...S.imagination} enter="zoom" exit="whip">
      <Imagination />
    </Shot>
    <Shot {...S.diagram} enter="whip" exit="zoom">
      <Diagram />
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
