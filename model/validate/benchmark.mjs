import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SITE_DATA, requireSite, siteModule } from './site.mjs';

// The metric and the corpus live in the site checkout: this harness scores the
// metric the browser actually ships, over the corpus it actually serves.
requireSite('the benchmark scores the shipped metric over the shipped corpus');
const { contrastAt } = await siteModule('vectors.js');
const { roleScore, maxContrast } = await siteModule('engine.js');

const DIR = path.dirname(fileURLToPath(import.meta.url));
const DATA = SITE_DATA;
const SPACE_BIN = process.env.SPACE_BIN || path.join(DATA, 'vectors.i8.bin');
const SPACE_META = process.env.SPACE_META || path.join(DATA, 'vectors.meta.json');
const SPACE_NAME = process.env.SPACE_NAME || 'shipped (dinov2)';

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
// Multi-partner benchmark: each heading carries every body partner the public
// sources endorse (see mine-pairings.mjs). Scoring is strict: a partner family
// only counts at its canonical 400-normal instance (the loose any-instance rule
// flattered every metric).
// FIXTURE overrides the benchmark fixture (e.g. FIXTURE=fontsinuse-pairings.json
// scores the usage-mined pairs instead of the curated endorsement set).
const FIXTURE = process.env.FIXTURE || 'fonts-pairings.json';
const fixtures = JSON.parse(fs.readFileSync(path.join(DIR, 'fixtures', FIXTURE), 'utf8'));
const fam = i => corpus.instances[i].family;

const elig = [];
for (let i = 0; i < corpus.count; i++) if (corpus.instances[i].eligible) elig.push(i);
const cmax = maxContrast(corpus, elig);

function repId(family) {
  const cands = [];
  for (let i = 0; i < corpus.count; i++) if (corpus.instances[i].eligible && corpus.instances[i].family === family) cands.push(i);
  if (!cands.length) return null;
  return cands.find(i => corpus.instances[i].weight === 400 && corpus.instances[i].style === 'normal') ?? cands[0];
}
function familyIds(family) {
  const out = [];
  for (let i = 0; i < corpus.count; i++) if (corpus.instances[i].eligible && corpus.instances[i].family === family) out.push(i);
  return out;
}

// Canonical partner instance: 400 normal, else any 400, else the first.
function canonId(family) {
  const cands = familyIds(family);
  if (!cands.length) return null;
  return cands.find(i => corpus.instances[i].weight === 400 && corpus.instances[i].style === 'normal')
    ?? cands.find(i => corpus.instances[i].weight === 400)
    ?? cands[0];
}

// Score with the legacy embedding contrast, the shipped metric (asymmetric role
// + multi-axis terms), and the shipped metric over guarded pools (the tool
// restricts body recommendations to text weights; the Microsoft rule additionally
// asks that the body not be heavier than the heading, arXiv:2501.10969).
// The body guardrail mirrors BODY_WEIGHT_MIN/MAX in site/js/engine.js.
//
// The contrast control `t`: the app's recommendation rows rank at t=1 (app.js
// `rankSlot(..., 1, slot, learned)`, the hand fallback to the learned re-ranker),
// so t=1 is the shipped ranking this benchmark measures. `generate`'s 3-slot
// objective uses t=0 (its default, app.js state.t), which those rows do not. The
// `generate t=0` control below reports the same metric at the engine default, to
// keep the difference visible rather than folded into the baseline.
const BODY_MIN = 300, BODY_MAX = 700;
const bodyGuard = inst => inst.weight >= BODY_MIN && inst.weight <= BODY_MAX;
const strokeOrder = (inst, heading) => inst.weight <= heading.weight;

const METRICS = [
  ['product (legacy)', i => contrastAt(corpus.units, corpus.dim, corpus._hi, i), null, false],
  ['shipped (role+axis)', i => roleScore(corpus, cmax, corpus._hi, i, 1, 'body'), null, false],
  ['shipped + body guard', i => roleScore(corpus, cmax, corpus._hi, i, 1, 'body'), bodyGuard, true],
  ['shipped + body guard (generate t=0)', i => roleScore(corpus, cmax, corpus._hi, i, 0, 'body'), bodyGuard, false],
  ['shipped + body + stroke', i => roleScore(corpus, cmax, corpus._hi, i, 1, 'body'),
    (inst, heading) => bodyGuard(inst) && strokeOrder(inst, heading), false],
];

// Per heading: rank every eligible instance of other families, then measure
// the acceptable partners' canonical instances against that ranking.
//   hit@k    at least one acceptable partner ranks <= k
//   recall@k fraction of acceptable partners ranking <= k
//   nDCG@k   binary-relevance nDCG over the top k of the ranking
function evaluate(scoreOf, filter) {
  const rows = [];
  const missing = new Set();
  for (const { heading, acceptable } of fixtures.pairings) {
    const hi = repId(heading);
    const rel = [];
    for (const { family } of acceptable) {
      const c = canonId(family);
      if (c != null) rel.push(c); else missing.add(family);
    }
    if (hi == null || !rel.length) { missing.add(heading); continue; }
    corpus._hi = hi;
    const cand = [];
    for (let i = 0; i < corpus.count; i++) {
      if (!corpus.instances[i].eligible || fam(i) === heading) continue;
      if (filter && !filter(corpus.instances[i], corpus.instances[hi])) continue;
      cand.push(i);
    }
    const ranked = cand.map(i => ({ i, s: scoreOf(i) })).sort((a, b) => b.s - a.s);
    const relSet = new Set(rel);
    const pos = new Map(ranked.map((r, k) => [r.i, k + 1]));
    const ranks = rel.map(id => pos.get(id) ?? ranked.length + 1).sort((a, b) => a - b);
    const dcgAt = k => {
      let d = 0;
      for (let p = 1; p <= Math.min(k, ranked.length); p++) if (relSet.has(ranked[p - 1].i)) d += 1 / Math.log2(p + 1);
      return d;
    };
    const idcgAt = k => {
      let d = 0;
      for (let j = 1; j <= Math.min(rel.length, k); j++) d += 1 / Math.log2(j + 1);
      return d;
    };
    rows.push({ h: heading, n: rel.length, ranks, best: ranks[0], total: ranked.length, dcgAt, idcgAt });
  }
  return { rows, missing, used: rows.length };
}

function summarise(rows) {
  const at = k => rows.filter(r => r.best <= k).length;
  const recallAt = k => rows.reduce((a, r) => a + r.ranks.filter(x => x <= k).length / r.n, 0) / rows.length;
  const ndcgAt = k => rows.reduce((a, r) => a + r.dcgAt(k) / r.idcgAt(k), 0) / rows.length;
  const bests = rows.map(r => r.best).sort((a, b) => a - b);
  const median = bests[Math.floor(bests.length / 2)];
  return { at, recallAt, ndcgAt, median };
}

console.log(`multi-partner pairings benchmark, space = ${SPACE_NAME}`);
console.log(`${fixtures.pairings.length} headings, ${fixtures.pairings.reduce((n, e) => n + e.acceptable.length, 0)} acceptable partners (strict: canonical 400-normal instance)\n`);

for (const [name, scoreOf, filter, detail] of METRICS) {
  const { rows, missing, used } = evaluate(scoreOf, filter);
  const s = summarise(rows);
  const f3 = x => x.toFixed(3);
  console.log(`${name}  (${used} headings)`);
  console.log(`  hit@10 ${s.at(10)}/${used}   hit@50 ${s.at(50)}/${used}   hit@100 ${s.at(100)}/${used}   median best-rank ${s.median}`);
  console.log(`  recall@10 ${f3(s.recallAt(10))}   recall@50 ${f3(s.recallAt(50))}   recall@100 ${f3(s.recallAt(100))}`);
  console.log(`  nDCG@10  ${f3(s.ndcgAt(10))}   nDCG@50  ${f3(s.ndcgAt(50))}   nDCG@100  ${f3(s.ndcgAt(100))}`);
  if (detail) {
    console.log(`\n  ${'heading'.padEnd(24)}${'#acc'.padStart(5)}${'best'.padStart(7)}${'rec@100'.padStart(9)}${'of'.padStart(8)}`);
    for (const r of [...rows].sort((a, b) => a.best - b.best)) {
      const rec = r.ranks.filter(x => x <= 100).length / r.n;
      console.log(`  ${r.h.padEnd(24)}${String(r.n).padStart(5)}${String(r.best).padStart(7)}${f3(rec).padStart(9)}${String(r.total).padStart(8)}`);
    }
    if (missing.size) console.log(`  not in catalog: ${[...missing].join(', ')}`);
  }
  console.log('');
}
