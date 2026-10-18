// Render review stills at given seconds: node scripts/stills.mjs <composition> <outDir> <sec...>
import {bundle} from '@remotion/bundler';
import {renderStill, selectComposition} from '@remotion/renderer';
import path from 'node:path';

const [comp, outDir, ...secs] = process.argv.slice(2);
const serveUrl = await bundle({entryPoint: path.resolve('src/index.ts')});
const browserExecutable = process.env.REMOTION_CHROME || null;
const composition = await selectComposition({serveUrl, id: comp, browserExecutable});
for (const s of secs) {
  const frame = Math.round(Number(s) * composition.fps);
  const output = path.join(outDir, `${comp}_${String(s).replace('.', '_')}.jpg`);
  await renderStill({serveUrl, composition, frame, output, imageFormat: 'jpeg', jpegQuality: 85, browserExecutable,
    scale: Number(process.env.SCALE || 0.5)});
  console.log(output);
}
