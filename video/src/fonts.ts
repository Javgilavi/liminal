import {continueRender, delayRender, staticFile} from 'remotion';

const faces: [string, string, string][] = [
  ['Inter', 'fonts/inter-400.woff2', '400'],
  ['Inter', 'fonts/inter-500.woff2', '500'],
  ['Inter', 'fonts/inter-600.woff2', '600'],
  ['Inter', 'fonts/inter-700.woff2', '700'],
  ['JetBrains Mono', 'fonts/mono-400.woff2', '400'],
  ['JetBrains Mono', 'fonts/mono-500.woff2', '500'],
];

const handle = delayRender('fonts');
Promise.all(
  faces.map(([family, file, weight]) => {
    const face = new FontFace(family, `url(${staticFile(file)}) format('woff2')`, {weight});
    document.fonts.add(face);
    return face.load();
  }),
).then(() => continueRender(handle));
