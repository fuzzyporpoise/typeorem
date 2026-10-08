import fs from 'node:fs';
import path from 'node:path';
import { SITE_DATA, requireSite, siteModule } from './site.mjs';

requireSite('the engine evaluation runs the shipped engine over the shipped corpus');
const {
  generate, buildBodyCentroid, tripleObjective, aggregate, pairScore, legibility,
  DEFAULT_LAMBDA, DEFAULT_AGG,
} = await siteModule('engine.js');
const { dotAt } = await siteModule('vectors.js');

const DATA = SITE_DATA;

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
const byLabel = new Map(corpus.instances.map((c, i) => [c.label, i]));
const label = i => corpus.instances[i].label;
const category = i => corpus.instances[i].category;
const centroid = buildBodyCentroid(corpus);

console.log(`engine over ${corpus.count} instances, dim ${corpus.dim}`);
console.log(`defaults: agg=${DEFAULT_AGG}, lambda=${DEFAULT_LAMBDA}\n`);

console.log('aggregation comparison (heading fixed, t=0):');
for (const seed of ['Montserrat 400', 'Playfair Display 400', 'Roboto 400', 'Oswald 400']) {
  const h = byLabel.get(seed);
  const line = [];
  for (const agg of ['sum', 'min', 'product']) {
    const tri = generate(corpus, { fixed: { heading: h }, agg, centroid });
    line.push(`${agg}=${label(tri.body)} + ${label(tri.accent)}`);
  }
  console.log(`  ${seed} [${category(h)}]`);
  for (const l of line) console.log(`    ${l}`);
}

console.log('\nhomepage triple in context (heading Montserrat 700):');
const h = byLabel.get('Montserrat 700');
const homeBody = byLabel.get('Lora 400 Italic');
const homeAccent = byLabel.get('Hind Madurai 300');
const { cmax } = generate(corpus, { fixed: { heading: h, body: homeBody, accent: homeAccent }, centroid });
const homeObj = tripleObjective(corpus, cmax, centroid, [h, homeBody, homeAccent], 0, DEFAULT_LAMBDA, DEFAULT_AGG);
const elig = [];
for (let i = 0; i < corpus.count; i++) if (corpus.instances[i].eligible) elig.push(i);
const fam = i => corpus.instances[i].family;
const rng = mulberry(1);
let better = 0, total = 0, betterLeg = 0, totalLeg = 0;
for (let n = 0; n < 8000; n++) {
  const b = elig[Math.floor(rng() * elig.length)];
  const a = elig[Math.floor(rng() * elig.length)];
  if (fam(b) === fam(h) || fam(a) === fam(h) || fam(a) === fam(b)) continue;
  total++;
  const obj = tripleObjective(corpus, cmax, centroid, [h, b, a], 0, DEFAULT_LAMBDA, DEFAULT_AGG);
  if (obj > homeObj) better++;
  if (legibility(corpus, centroid, b) > 0.5) { totalLeg++; if (obj > homeObj) betterLeg++; }
}
const tri = generate(corpus, { fixed: { heading: h }, centroid });
const bestObj = tripleObjective(corpus, cmax, centroid, [h, tri.body, tri.accent], 0, DEFAULT_LAMBDA, DEFAULT_AGG);
const cand = elig.filter(i => fam(i) !== fam(h)).map(i => ({ i, s: pairScore(corpus, cmax, h, i, 0) })).sort((x, y) => y.s - x.s);
const rankOf = id => cand.findIndex(x => x.i === id) + 1;
console.log(`  homepage: ${label(homeBody)} + ${label(homeAccent)}  objective ${homeObj.toFixed(4)}`);
console.log(`  optimizer picks: ${label(tri.body)} + ${label(tri.accent)}  objective ${bestObj.toFixed(4)}`);
console.log(`  homepage percentile among ${total} random triples: ${(100 * (1 - better / total)).toFixed(1)}%`);
console.log(`  homepage percentile among ${totalLeg} legible-body triples: ${(100 * (1 - betterLeg / totalLeg)).toFixed(1)}%`);
console.log(`  gap to the optimizer: ${(bestObj - homeObj).toFixed(4)}`);
console.log(`  example faces as candidates for the heading: body ${label(homeBody)} rank ${rankOf(homeBody)}/${cand.length}, accent ${label(homeAccent)} rank ${rankOf(homeAccent)}/${cand.length}`);

console.log('\ncontrast control monotonicity (heading Montserrat 400):');
const seedH = byLabel.get('Montserrat 400');
for (const t of [-1, -0.5, 0, 0.5, 1]) {
  const g = generate(corpus, { fixed: { heading: seedH }, t, centroid });
  const mean = (dotAt(corpus.units, corpus.dim, seedH, g.body) + dotAt(corpus.units, corpus.dim, seedH, g.accent)) / 2;
  console.log(`  t=${String(t).padStart(4)}  mean cosine to heading ${mean.toFixed(3)}  -> ${label(g.body)} + ${label(g.accent)}`);
}

function mulberry(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
