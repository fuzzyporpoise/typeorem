import fs from 'node:fs';
import path from 'node:path';
import { SITE_DATA, requireSite, siteModule } from './site.mjs';

// TSK-003 engine-interaction check. The learned re-ranker (site/js/reranker.js)
// is a ranking-time scorer: the app will rank its recommendation rows through
// `rankLearned`. Before wiring it in, this measures the risk the metric thread
// already hit once (the multi-axis term took the engine's balance 0.82 -> 0.65):
// what happens if the learned ranking is put *inside* generate()'s candidate
// pre-selection, instead of only in the recommendation rows?
//
// It runs the shipped generate() and a variant whose body/accent candidate
// lists come from rankLearned (everything else identical: same objective, same
// family-diversity loop), over the same 60 fixed headings evaluate.mjs uses,
// and reports the engine metrics both ways. It also measures the actual app
// change: how far the learned top-4 recommendation rows move from rankFor().
//
//     node model/validate/engine_interaction.mjs

requireSite('the engine-interaction check re-ranks generate output over the shipped corpus');
const {
  generate, tripleObjective, legibility, concentration, bodySuitable, bodyOrdered,
  maxContrast, DEFAULT_AGG, DEFAULT_LAMBDA, DEFAULT_BETA, rankFor,
} = await siteModule('engine.js');
const { rankLearned } = await siteModule('reranker.js');
const { contrastAt } = await siteModule('vectors.js');

const BODY_CANDIDATES = 40;
const ACCENT_CANDIDATES = 80;

function loadCorpus() {
  const meta = JSON.parse(fs.readFileSync(path.join(SITE_DATA, 'vectors.meta.json'), 'utf8'));
  const catalog = JSON.parse(fs.readFileSync(path.join(SITE_DATA, 'catalog.json'), 'utf8'));
  const bin = fs.readFileSync(path.join(SITE_DATA, 'vectors.i8.bin'));
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

function mulberry(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const corpus = loadCorpus();
const model = JSON.parse(fs.readFileSync(path.join(SITE_DATA, 'reranker.json'), 'utf8'));
const graph = JSON.parse(fs.readFileSync(path.join(SITE_DATA, 'pairgraph.json'), 'utf8'));

const rng = mulberry(7);
const elig = [];
for (let i = 0; i < corpus.count; i++) if (corpus.instances[i].eligible) elig.push(i);
const HEADINGS = Array.from({ length: 60 }, () => elig[Math.floor(rng() * elig.length)]);
const fam = i => corpus.instances[i].family;
const cat = i => corpus.instances[i].category;
const instances = corpus.instances;

// The shipped generate(), unmodified, at the default configuration.
function shipped(heading) {
  return generate(corpus, { fixed: { heading }, centroid: null, agg: DEFAULT_AGG, lambda: DEFAULT_LAMBDA, beta: DEFAULT_BETA });
}

// The same generate loop with the learned ranking as the candidate pre-filter:
// bodies and accents are the top of rankLearned's order (body pool still text
// weights), while the objective and family-diversity loop are the engine's.
function learnedFilter(heading) {
  const c = corpus;
  const notHeading = elig.filter(i => fam(i) !== fam(heading));
  const bodyPool = bodyOrdered(c, heading, notHeading.filter(i => bodySuitable(instances[i])));
  const rankedBody = rankLearned(c, model, heading, bodyPool, graph);
  const rankedAccent = rankLearned(c, model, heading, notHeading, graph);
  const bodies = rankedBody.slice(0, BODY_CANDIDATES).map(r => r.i);
  const accents = rankedAccent.slice(0, ACCENT_CANDIDATES).map(r => r.i);
  const cmax = maxContrast(c, elig);
  let best = null;
  for (const b of bodies) {
    for (const a of accents) {
      if (a === b || fam(a) === fam(b)) continue;
      const score = tripleObjective(c, cmax, null, [heading, b, a], 0, DEFAULT_LAMBDA, DEFAULT_AGG, DEFAULT_BETA);
      if (!best || score > best.score) best = { score, body: b, accent: a };
    }
  }
  return { heading, body: best.body, accent: best.accent, cmax };
}

function metrics(genFn) {
  let leg = 0, cont = 0, balance = 0, conc = 0, spread = 0, legRate = 0;
  const n = HEADINGS.length;
  for (const h of HEADINGS) {
    const g = genFn(h);
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

console.log(`engine interaction over ${HEADINGS.length} headings, ${corpus.count} instances\n`);

const base = metrics(shipped);
const learned = metrics(learnedFilter);

const row = (name, m) => console.log(
  `${name.padEnd(22)}${m.leg.toFixed(3).padStart(15)}${m.cont.toFixed(3).padStart(16)}`
  + `${m.balance.toFixed(3).padStart(9)}${m.conc.toFixed(3).padStart(10)}`
  + `${m.spread.toFixed(2).padStart(8)}${(100 * m.legRate).toFixed(0).padStart(9)}%`);

console.log(`${'variant'.padEnd(22)}${'bodyLegibility'.padStart(15)}${'overallContrast'.padStart(16)}${'balance'.padStart(9)}${'axisConc'.padStart(10)}${'spread'.padStart(8)}${'legible%'.padStart(10)}`);
row('shipped generate()', base);
row('+ learned pre-filter', learned);
console.log('');
console.log(`balance delta (learned - shipped): ${(learned.balance - base.balance).toFixed(3)}`);
console.log(`spread delta:  ${(learned.spread - base.spread).toFixed(2)}`);

// The app change itself: the top-4 recommendation rows under rankFor (the
// shipped path) vs rankLearned, over each generated heading's pool. This is
// what the browser will actually swap, and its cost if it changed the triple.
let overlap = 0, changed = 0, total = 0, bestRankFor = 0, bestLearned = 0;
for (const h of HEADINGS) {
  const used = new Set([fam(h)]);
  const pool = elig.filter(i => !used.has(fam(i)));
  const bodyPool = pool.filter(i => bodySuitable(instances[i]));
  const hand = rankFor(corpus, maxContrast(corpus, elig), h, bodyPool, 1, 'body').slice(0, 4).map(r => r.i);
  const learnedTop = rankLearned(corpus, model, h, bodyPool, graph).slice(0, 4).map(r => r.i);
  const handSet = new Set(hand);
  overlap += learnedTop.filter(i => handSet.has(i)).length;
  if (hand.join() !== learnedTop.join()) changed++;
  bestRankFor += hand.length;
  bestLearned += learnedTop.length;
  total++;
}
console.log('');
console.log(`recommendation rows (top-4 body): mean overlap ${(overlap / total).toFixed(2)}/4, `
  + `${changed}/${total} headings change order`);
