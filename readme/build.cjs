// node readme/build.cjs <ecg-convnextv2-classifier checkout> : pipeline figure (light/dark) into .github/readme/
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const repo = process.argv[2];
if (!repo) throw new Error('usage: node readme/build.cjs <path to a checkout of master>');
const root = path.join(__dirname, '..');
const font = (style, file) => `@font-face { font-family: 'Instrument Serif'; font-style: ${style}; font-weight: 400;
  src: url(data:font/woff2;base64,${fs.readFileSync(path.join(root, 'fonts', file)).toString('base64')}) format('woff2'); }`;

// Synthetic 10 s window at 100 Hz, as preprocess.py hands it on: Gaussian P, Q, R, S and T waves per beat,
// slight baseline wander, min-max scaled to [0, 1]. Fixed beat times and R heights; the beat at 5.04 s is early.
const rate = 100, n = 1000;
const beats = [[0.36, 1], [1.2, 1.01], [2.04, 0.97], [2.87, 0.93], [3.69, 0.95], [4.53, 1], [5.04, 1.01],
  [5.37, 1.07], [6.2, 0.96], [7.04, 0.93], [7.87, 0.96], [8.71, 1.01], [9.54, 1]];
const waves = [[-0.2, 0.12, 0.025], [-0.03, -0.12, 0.01], [0, 1, 0.012], [0.035, -0.3, 0.012], [0.25, 0.3, 0.04]]; // [offset s, amplitude, width s]
const raw = Array.from({ length: n }, (_, i) => {
  const t = i / rate;
  let v = 0.02 * Math.sin(2 * Math.PI * 0.15 * t);
  for (const [at, r] of beats) for (const [dt, a, w] of waves) v += (dt === 0 ? a * r : a) * Math.exp(-(((t - at - dt) / w) ** 2) / 2);
  return v;
});
const lo = Math.min(...raw), hi = Math.max(...raw);
const signal = raw.map(v => (v - lo) / (hi - lo));

// Port of perform_s_transform in src/signal_processing.py, 0 to 40 Hz: row 0 is the mean, row f is
// ifft(fft(signal) * exp(-2 pi^2 sigma^2 (freqs - f)^2)) with sigma = 1 / f. Plain DFT; n is small.
const cos = Array.from({ length: n }, (_, k) => Math.cos((2 * Math.PI * k) / n));
const sin = Array.from({ length: n }, (_, k) => Math.sin((2 * Math.PI * k) / n));
const re = new Float64Array(n), im = new Float64Array(n);
for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) { re[k] += signal[j] * cos[(k * j) % n]; im[k] -= signal[j] * sin[(k * j) % n]; }
const freqs = Array.from({ length: n }, (_, k) => ((k < n / 2 ? k : k - n) * rate) / n);
const mean = signal.reduce((a, b) => a + b) / n;
const st = [new Array(n).fill(mean)];
for (let f = 1; f <= 40; f++) {
  const sigma = 1 / f;
  const g = freqs.map(x => Math.exp(-2 * Math.PI ** 2 * sigma ** 2 * (x - f) ** 2));
  const used = g.map((v, k) => (v > 1e-12 ? k : -1)).filter(k => k >= 0);
  st.push(Array.from({ length: n }, (_, t) => {
    let a = 0, b = 0;
    for (const k of used) {
      const c = cos[(k * t) % n], s = sin[(k * t) % n];
      a += g[k] * (re[k] * c - im[k] * s);
      b += g[k] * (re[k] * s + im[k] * c);
    }
    return Math.hypot(a, b) / n;
  }));
}
// magnitude, each frequency row scaled to its peak
const panel = st.map(row => { const m = Math.max(...row); return row.map(v => v / m); });

const trace = signal.map((v, i) => `${((i / (n - 1)) * 262).toFixed(2)},${(11 + (1 - v) * 107).toFixed(2)}`).join(' ');

const mono = s => `<b class="mono">${s}</b>`;
const cards = [
  { title: 'Clean the signal', visual: `<svg viewBox="0 0 262 130"><polyline points="${trace}"/></svg>`,
    notes: [`MLII lead, ${mono('360 → 100 Hz')}`, '30 Hz low-pass, baseline removed', `${mono('10 s')} windows, min-max scaled`] },
  { title: 'S-Transform', visual: '<canvas id="st"></canvas>',
    notes: [`${mono('0–40 Hz')} × ${mono('1,000')} samples`, 'real and imaginary parts', `input ${mono('2 × 41 × 1000')}`] },
  // bar heights are the original figure's, not to scale
  { title: 'ConvNeXt V2', visual: `<div class="bars">${[[2, 39], [2, 62], [6, 130], [2, 62]].map(([d, h]) => `<div style="height:${h}px">${d}</div>`).join('')}</div>`,
    notes: [`blocks per stage, ${mono('64→512')} ch`, '3×1 then 1×15 depthwise convs', 'GRN, trained from scratch'] },
  { title: 'Classify', visual: `<ul>${[['N', 'Normal'], ['S', 'Supraventricular'], ['V', 'Ventricular'], ['F', 'Fusion'], ['Q', 'Unknown']]
      .map(([c, name]) => `<li><span class="mono">${c}</span>${name}</li>`).join('')}</ul>`,
    notes: ['temporal attention pool', 'AAMI classes'] },
];

const page = dark => `<!doctype html><meta charset="utf-8">
<style>
  ${font('normal', 'InstrumentSerif-Regular.woff2')}
  ${font('italic', 'InstrumentSerif-Italic.woff2')}
  :root { --bg: ${dark ? '#171614' : '#f5f3ee'}; --ink: ${dark ? '#edeae3' : '#1c1b18'}; --muted: ${dark ? '#a09b90' : '#615d55'};
    --accent: ${dark ? '#ff8f7f' : '#c8402f'}; --card: ${dark ? '#211f1c' : '#fffefb'}; --line: ${dark ? '#edeae324' : '#1c1b1824'};
    --shadow: ${dark ? '#00000059' : '#7a281e1f'};
    --stage: ${dark
      ? 'radial-gradient(45% 80% at 0% 0%, #5e271f, transparent), radial-gradient(40% 70% at 100% 45%, #3c1a16, transparent), #2a1613'
      : 'radial-gradient(45% 80% at 0% 0%, #ffd2c5, transparent), radial-gradient(35% 75% at 100% 25%, #f6b2a2, transparent), #f9ddd4'}; }
  * { box-sizing: border-box; }
  html, body { margin: 0; background: transparent; }
  .figure { position: relative; width: 1280px; height: 641px; overflow: hidden; border-radius: 28px; background: var(--bg); color: var(--ink);
    font-family: system-ui, 'Segoe UI', sans-serif; }
  .mono { font-family: ui-monospace, 'Cascadia Code', monospace; }
  h1 { position: absolute; left: 56px; top: 52px; margin: 0; font: 400 60px/1 'Instrument Serif', serif; letter-spacing: -0.01em; }
  h1 i { color: var(--accent); }
  .lede { position: absolute; left: 56px; top: 127px; margin: 0; font-size: 18px; line-height: 24px; color: var(--muted); }
  .stage { position: absolute; left: 56px; top: 184px; width: 1168px; height: 360px; padding: 28px; border-radius: 26px; background: var(--stage);
    display: grid; grid-template-columns: 298px 298px 259px 1fr; gap: 16px; }
  .card { padding: 18px 18px 0; border-radius: 14px; background: var(--card); box-shadow: 0 14px 34px -8px var(--shadow); }
  .num { height: 16px; font: 500 13px/16px ui-monospace, 'Cascadia Code', monospace; color: var(--muted); }
  h2 { margin: 6px 0 0; font-size: 19px; line-height: 25px; font-weight: 700; letter-spacing: -0.01em; }
  .visual { height: 130px; margin-top: 12px; }
  svg, canvas { display: block; width: 262px; height: 130px; }
  polyline { fill: none; stroke: var(--accent); stroke-width: 2.4; stroke-linejoin: round; }
  canvas { border-radius: 2px 2px 5px 5px; }
  .bars { display: flex; align-items: flex-end; gap: 6px; height: 130px; }
  .bars div { flex: 1; display: flex; align-items: flex-end; justify-content: center; padding-bottom: 6px; border-radius: 4px;
    background: color-mix(in srgb, var(--accent) 85%, var(--card)); color: var(--card); font: 700 11px ui-monospace, 'Cascadia Code', monospace; }
  ul { margin: 0; padding: 2.5px 0 0; list-style: none; }
  li { display: flex; align-items: center; gap: 10px; height: 25px; font-size: 15px; }
  li span { display: grid; place-items: center; width: 22px; height: 22px; border: 1px solid var(--line); border-radius: 5px; color: var(--accent); font-size: 12px; font-weight: 700; }
  .notes { margin-top: 13px; font-size: 14px; line-height: 21.5px; color: var(--muted); }
  .notes b { color: var(--ink); font-size: 13px; }
  .training { position: absolute; left: 56px; top: 566px; margin: 0; font-size: 15px; line-height: 21px; color: var(--muted); }
  .training b { color: var(--ink); }
</style>
<div class="figure">
  <h1>From heartbeat to <i>class</i></h1>
  <p class="lede">Ten seconds of single-lead ECG become a time-frequency image, and a ConvNeXt V2 network labels the window.</p>
  <div class="stage">${cards.map((c, i) => `<div class="card"><div class="num">0${i + 1}</div><h2>${c.title}</h2><div class="visual">${c.visual}</div>
    <div class="notes">${c.notes.join('<br>')}</div></div>`).join('')}</div>
  <p class="training"><b>Training</b> · 4-fold patient-wise cross-validation · SMOTE · class-weighted focal loss · AdamW · best fold exported to ONNX and 8-bit quantized ONNX</p>
</div>`;

// low frequencies at the bottom, accent at opacity from magnitude, smoothed up to the panel size
const drawPanel = panel => {
  const rows = panel.length, cols = panel[0].length;
  const src = document.createElement('canvas');
  src.width = cols; src.height = rows;
  const ctx = src.getContext('2d');
  const img = ctx.createImageData(cols, rows);
  const [r, g, b] = getComputedStyle(document.body).getPropertyValue('--accent').trim().match(/\w\w/g).map(h => parseInt(h, 16));
  panel.forEach((row, f) => row.forEach((v, t) => img.data.set([r, g, b, Math.round(255 * v ** 0.6)], ((rows - 1 - f) * cols + t) * 4)));
  ctx.putImageData(img, 0, 0);
  const out = document.getElementById('st');
  out.width = 262 * devicePixelRatio; out.height = 130 * devicePixelRatio;
  const o = out.getContext('2d');
  o.imageSmoothingQuality = 'high';
  o.drawImage(src, 0, 0, out.width, out.height);
};

(async () => {
  const browser = await chromium.launch();
  for (const dark of [false, true]) {
    const tab = await browser.newPage({ viewport: { width: 1280, height: 641 }, deviceScaleFactor: 1.6 });
    await tab.setContent(page(dark), { waitUntil: 'load' });
    await tab.evaluate(() => document.fonts.ready);
    if (!(await tab.evaluate(() => document.fonts.check("60px 'Instrument Serif'") && document.fonts.check("italic 60px 'Instrument Serif'"))))
      throw new Error('Instrument Serif not loaded');
    await tab.evaluate(drawPanel, panel);
    const file = path.join(repo, '.github/readme', `pipeline-${dark ? 'dark' : 'light'}.png`);
    await tab.screenshot({ path: file, omitBackground: true });
    await tab.close();
    console.log(path.relative(process.cwd(), file));
  }
  await browser.close();
})();
