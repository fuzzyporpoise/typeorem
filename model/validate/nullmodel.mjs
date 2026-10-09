// Null-model evaluation of the pairing metric, after Choi & Hyun (Sci Rep
// 14:30820, 2024): if the metric tracks design practice, REAL pairs observed in
// fontsinuse.com use cases must outscore randomized pairs. Two null models:
//   frequency  families drawn proportional to their occurrence among real pairs
//     (a high-frequency face is just as likely to be sampled as in the real
//     data, so separation cannot be attributed to popularity alone). This is
//     the PRIMARY baseline: usage data is frequency-skewed, so the metric's job
//     is to beat what family frequency already predicts.
//   uniform    both families drawn uniformly from the eligible catalog
//     (secondary; easier to beat)
// For each null model and metric we report the null mean/sd, the z-score of the
// real mean, and AUC = P(real pair outscores a null pair).
//
// Slices de-skew the usage data: `refined` keeps only repeated, high-lift,
// non-kin pairs (deliberate choices rather than popularity residue), and
// `endorsed` keeps pairs the curated benchmark also endorses (the highest-
// confidence intersection of usage and judgment).
//
// The fontsinuse pairs are unordered usage co-occurrences, so the asymmetric
// shipped metric is applied in both directions and the max is taken. The legacy
// symmetric `product` contrast is reported alongside as a reference.
//
// Run: node model/validate/nullmodel.mjs

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SITE_DATA, requireSite, siteModule } from './site.mjs';

requireSite('the null-model harness scores the shipped metric over the shipped corpus');
const { contrastAt } = await siteModule('vectors.js');
const { roleScore, maxContrast } = await siteModule('engine.js');

const DIR = path.dirname(fileURLToPath(import.meta.url));
const DATA = SITE_DATA;
const SPACE_BIN = process.env.SPACE_BIN || path.join(DATA, 'vectors.i8.bin');
const SPACE_META = process.env.SPACE_META || path.join(DATA, 'vectors.meta.json');
const SPACE_NAME = process.env.SPACE_NAME || 'shipped (dinov2)';
const NULL_SAMPLES = 20000;
const DIGITAL = new Set(['Web', 'Software/Apps', 'Mobile/Tablet', 'Infographics/Maps']);

function loadCorpus() {
  const meta = JSON.parse(fs.readFileSync(SPACE_META, 'utf8'));
  const catalog = JSON.parse(fs.readFileSync(path.join(DATA, 'catalog.json'), 'utf8'));
  const bin = fs.readFileSync(SPACE_BIN);
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
const elig = [];
for (let i = 0; i < corpus.count; i++) if (corpus.instances[i].eligible) elig.push(i);
const cmax = maxContrast(corpus, elig);

function canonId(family) {
  const cands = [];
  for (let i = 0; i < corpus.count; i++) if (corpus.instances[i].eligible && corpus.instances[i].family === family) cands.push(i);
  if (!cands.length) return null;
  return cands.find(i => corpus.instances[i].weight === 400 && corpus.instances[i].style === 'normal')
    ?? cands.find(i => corpus.instances[i].weight === 400)
    ?? cands[0];
}

const fixture = JSON.parse(fs.readFileSync(path.join(DIR, 'fixtures', 'fontsinuse-pairings.json'), 'utf8'));
const pairMap = new Map();
for (const { heading, acceptable } of fixture.pairings) {
  for (const { family, mediums, uses, lift, kin } of acceptable) {
    const key = [heading, family].sort().join('\0');
    if (!pairMap.has(key)) pairMap.set(key, { a: [heading, family].sort(), mediums, uses, lift, kin: !!kin });
  }
}
const realPairs = [...pairMap.values()]
  .map(p => ({ a: canonId(p.a[0]), b: canonId(p.a[1]), mediums: p.mediums, uses: p.uses, lift: p.lift, kin: p.kin }))
  .filter(p => p.a != null && p.b != null && p.a !== p.b);

// Pairs the curated endorsement benchmark also endorses (usage + judgment).
const endorsedKeys = new Set();
for (const file of ['fonts-pairings.json', 'known-pairings.json']) {
  const d = JSON.parse(fs.readFileSync(path.join(DIR, 'fixtures', file), 'utf8'));
  if (d.pairings) {
    for (const { heading, acceptable } of d.pairings) {
      for (const { family } of acceptable) endorsedKeys.add([heading, family].sort().join('\0'));
    }
  }
  if (d.pairs) for (const [h, p] of d.pairs) endorsedKeys.add([h, p].sort().join('\0'));
}

function mulberry32(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeScorers() {
  return {
    'product (legacy)': (a, b) => contrastAt(corpus.units, corpus.dim, a, b),
    'shipped (max role)': (a, b) => Math.max(
      roleScore(corpus, cmax, a, b, 1, 'body'),
      roleScore(corpus, cmax, b, a, 1, 'body')),
  };
}

function summarise(realScores, nullScores, rand) {
  const mean = xs => xs.reduce((a, x) => a + x, 0) / xs.length;
  const sd = xs => Math.sqrt(xs.reduce((a, x) => a + (x - mean(xs)) ** 2, 0) / xs.length);
  const rm = mean(realScores), nm = mean(nullScores), ns = sd(nullScores);
  let wins = 0, ties = 0;
  const draws = Math.min(NULL_SAMPLES, realScores.length * 200);
  for (let i = 0; i < draws; i++) {
    const r = realScores[Math.floor(rand() * realScores.length)];
    const n = nullScores[Math.floor(rand() * nullScores.length)];
    if (r > n) wins++; else if (r === n) ties++;
  }
  return { realMean: rm, nullMean: nm, nullSd: ns, z: (rm - nm) / (ns || 1), auc: (wins + ties / 2) / draws };
}

const scorers = makeScorers();
const slices = [
  ['all', () => true],
  ['web', p => (p.mediums.Web || 0) > 0],
  ['digital', p => Object.keys(p.mediums).some(m => DIGITAL.has(m))],
  ['refined', p => p.uses >= 2 && p.lift >= 2 && !p.kin],
  ['endorsed', p => endorsedKeys.has([corpus.instances[p.a].family, corpus.instances[p.b].family].sort().join('\0'))],
];

console.log(`null-model benchmark over ${realPairs.length} distinct real fontsinuse pairs, space = ${SPACE_NAME} (null samples: ${NULL_SAMPLES}, seeded)\n`);
for (const [sliceName, keep] of slices) {
  const pairs = realPairs.filter(keep);
  if (pairs.length < 5) { console.log(`${sliceName}: too few pairs (${pairs.length}), skipped\n`); continue; }
  const rand = mulberry32(0xC0FFEE);
  const famFreq = new Map();
  for (const p of pairs) {
    famFreq.set(p.a, (famFreq.get(p.a) || 0) + 1);
    famFreq.set(p.b, (famFreq.get(p.b) || 0) + 1);
  }
  const freqTable = [...famFreq.entries()];
  const freqTotal = freqTable.reduce((n, [, c]) => n + c, 0);
  const pickUniform = () => elig[Math.floor(rand() * elig.length)];
  const pickFreq = () => {
    let x = rand() * freqTotal;
    for (const [id, c] of freqTable) { x -= c; if (x <= 0) return id; }
    return freqTable[freqTable.length - 1][0];
  };
  const nulls = { frequency: [], uniform: [] };
  for (const [nullName, pick] of [['frequency', pickFreq], ['uniform', pickUniform]]) {
    let n = 0;
    while (n < NULL_SAMPLES) {
      const a = pick(), b = pick();
      if (a === b) continue;
      nulls[nullName].push([a, b]);
      n++;
    }
  }
  console.log(`${sliceName} (${pairs.length} real pairs)`);
  for (const [metricName, score] of Object.entries(scorers)) {
    const realScores = pairs.map(p => score(p.a, p.b));
    for (const nullName of ['frequency', 'uniform']) {
      const nullScores = nulls[nullName].map(([a, b]) => score(a, b));
      const s = summarise(realScores, nullScores, rand);
      console.log(`  ${metricName.padEnd(18)} vs ${nullName.padEnd(9)}  real ${s.realMean.toFixed(4)}  null ${s.nullMean.toFixed(4)} sd ${s.nullSd.toFixed(4)}  z ${s.z.toFixed(2)}  AUC ${s.auc.toFixed(3)}`);
    }
  }
  console.log('');
}
