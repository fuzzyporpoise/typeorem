import fs from 'node:fs';
import path from 'node:path';
import { SITE_DATA, requireSite, siteModule } from './site.mjs';

requireSite('the evaluation runs the shipped engine over the shipped corpus');
const { generate, concentration, legibility, DEFAULT_AGG, DEFAULT_LAMBDA, DEFAULT_BETA } = await siteModule('engine.js');
const { contrastAt } = await siteModule('vectors.js');

const DATA = SITE_DATA;

function mulberry(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function loadCorpus() {
  const meta = JSON.parse(fs.readFileSync(path.join(DATA, 'vectors.meta.json'), 'utf8'));
  const catalog = JSON.parse(fs.readFileSync(path.join(DATA, 'catalog.json'), 'utf8'));
  const bin = fs.readFileSync(path.join(DATA, 'vectors.i8.bin'));
  const i8 = new Int8Array(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
  const dim = meta.dim, count = meta.count;
  const units = new Float64Array(count * dim);
  for (let i = 0; i < count; i++) {
    const scale = meta.scale.values[i];
    let sum = 0;
    for (let j = 0; j < dim; j++) { const v = i8[i * dim + j] * scale; units[i * dim + j] = v; sum += v * v; }
    const len = Math.sqrt(sum) || 1;
    for (let j = 0; j < dim; j++) units[i * dim + j] /= len;
  }
  return { dim, count, instances: catalog.instances, units };
}

const corpus = loadCorpus();
const rng = mulberry(7);
const elig = [];
for (let i = 0; i < corpus.count; i++) if (corpus.instances[i].eligible) elig.push(i);
const HEADINGS = Array.from({ length: 60 }, () => elig[Math.floor(rng() * elig.length)]);
const fam = i => corpus.instances[i].family;
const cat = i => corpus.instances[i].category;

const CONFIGS = [
  ['sum l=0.25', { agg: 'sum', lambda: 0.25, beta: 0 }],
  ['min l=0.1', { agg: 'min', lambda: 0.1, beta: 0 }],
  ['min l=0.25', { agg: 'min', lambda: 0.25, beta: 0 }],
  ['min l=0.4', { agg: 'min', lambda: 0.4, beta: 0 }],
  ['min l=0.6', { agg: 'min', lambda: 0.6, beta: 0 }],
  ['min no-leg', { agg: 'min', lambda: 0, beta: 0 }],
  ['min conc', { agg: 'min', lambda: 0.25, beta: 0.5 }],
];

function metrics(opts) {
  let leg = 0, cont = 0, balance = 0, conc = 0, spread = 0, legRate = 0, n = HEADINGS.length;
  for (const h of HEADINGS) {
    const g = generate(corpus, { fixed: { heading: h }, centroid: null, ...opts });
    const [b, a] = [g.body, g.accent];
    const c = [
      contrastAt(corpus.units, corpus.dim, h, b),
      contrastAt(corpus.units, corpus.dim, h, a),
      contrastAt(corpus.units, corpus.dim, b, a),
    ];
    leg += legibility(corpus, null, b);
    if (legibility(corpus, null, b) > 0.5) legRate++;
    cont += c.reduce((x, y) => x + y, 0);
    balance += Math.min(...c) / (Math.max(...c) || 1);
    conc += (concentration(corpus.units, corpus.dim, h, b)
      + concentration(corpus.units, corpus.dim, h, a)
      + concentration(corpus.units, corpus.dim, b, a)) / 3;
    spread += new Set([cat(h), cat(b), cat(a)]).size;
  }
  return {
    leg: leg / n, cont: cont / n, balance: balance / n, conc: conc / n,
    spread: spread / n, legRate: legRate / n,
  };
}

console.log(`evaluate over ${HEADINGS.length} fixed headings (${corpus.count} instances)\n`);
console.log(`${'config'.padEnd(16)}${'bodyLegibility'.padStart(15)}${'overallContrast'.padStart(16)}${'balance'.padStart(9)}${'axisConc'.padStart(10)}${'spread'.padStart(8)}${'legible%'.padStart(10)}`);
for (const [name, opts] of CONFIGS) {
  const m = metrics(opts);
  console.log(`${name.padEnd(16)}${m.leg.toFixed(3).padStart(15)}${m.cont.toFixed(3).padStart(16)}${m.balance.toFixed(2).padStart(9)}${m.conc.toFixed(3).padStart(10)}${m.spread.toFixed(2).padStart(8)}${(100 * m.legRate).toFixed(0).padStart(9)}%`);
}

const def = { agg: DEFAULT_AGG, lambda: DEFAULT_LAMBDA, beta: DEFAULT_BETA };
let hb = 0, ha = 0, ba = 0;
for (const h of HEADINGS) {
  const g = generate(corpus, { fixed: { heading: h }, centroid: null, ...def });
  hb += contrastAt(corpus.units, corpus.dim, h, g.body);
  ha += contrastAt(corpus.units, corpus.dim, h, g.accent);
  ba += contrastAt(corpus.units, corpus.dim, g.body, g.accent);
}
const n = HEADINGS.length;
console.log(`\ndefault (${DEFAULT_AGG}, lambda ${DEFAULT_LAMBDA}): per-pair contrast`);
console.log(`  heading-body ${(hb / n).toFixed(3)}   heading-accent ${(ha / n).toFixed(3)}   body-accent ${(ba / n).toFixed(3)}`);
