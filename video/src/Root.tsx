import {Composition} from 'remotion';
import {Main} from './Main';
import {cfg} from './theme';

export const Root: React.FC = () => (
  <>
    <Composition id="Liminal16x9" component={Main} width={1920} height={1080} fps={cfg.fps}
      durationInFrames={cfg.fps * cfg.durationSec} />
    <Composition id="Liminal9x16" component={Main} width={1080} height={1920} fps={cfg.fps}
      durationInFrames={cfg.fps * cfg.durationSec} />
  </>
);
