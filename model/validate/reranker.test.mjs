import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SITE_DATA, requireSite, siteModule } from './site.mjs';

// The learned re-ranker is fit here (Python) and shipped by the site (JS). This
// is the cross-repo half of that contract: the JS scorer must reproduce the
// Python scores, and the shipped model must surface endorsed partners.
//
//     TYPEOREM_SITE_DIR=/path/to/site node model/validate/reranker.test.mjs

requireSite('the re-ranker parity test scores the shipped JS model');
const { score, maxContrastAgainst, resolveGraph, dsknnColumn, rankLearned } = await siteModule('reranker.js');

const DIR = path.dirname(fileURLToPath(import.meta.url));
const CASES = path.join(DIR, 'fixtures', 'reranker-cases.json');
const GRAPH_CASES = path.join(DIR, 'fixtures', 'pairgraph-cases.json');

let pass = 0, fail = 0;
const ok = (l, c) => { if (c) pass++; else { fail++; console.log('FAIL ' + l); } };

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

const corpus = loadCorpus();
const model = JSON.parse(fs.readFileSync(path.join(SITE_DATA, 'reranker.json'), 'utf8'));
const { cases } = JSON.parse(fs.readFileSync(CASES, 'utf8'));
const graph = JSON.parse(fs.readFileSync(path.join(SITE_DATA, 'pairgraph.json'), 'utf8'));
const resolved = resolveGraph(corpus, graph);

const byLabel = new Map(corpus.instances.map((c, i) => [c.label, i]));
const cmaxCache = new Map();
const cmaxFor = hi => {
  if (!cmaxCache.has(hi)) cmaxCache.set(hi, maxContrastAgainst(corpus, hi));
  return cmaxCache.get(hi);
};
// The shipped model names dsknn/has_graph, so a per-pair score needs the column
// for that heading; a single-candidate pool is enough (the column is per-row).
const colFor = (hi, ci) => {
  const { scores, hasGraph } = dsknnColumn(corpus, resolved, hi, [ci]);
  return { dsknn: () => scores[0], hasGraph };
};

let maxErr = 0, checked = 0;
for (const c of cases) {
  const hi = byLabel.get(c.heading);
  const ci = byLabel.get(c.candidate);
  if (hi == null || ci == null) { ok(`case labels present (${c.heading} / ${c.candidate})`, false); continue; }
  const got = score(corpus, model, hi, ci, cmaxFor(hi), colFor(hi, ci));
  maxErr = Math.max(maxErr, Math.abs(got - c.score));
  checked++;
}
ok(`JS scores match ${cases.length} Python parity cases`, checked === cases.length && maxErr < 1e-4);

// DS-kNN parity (TSK-020 ship vehicle): the JS runtime over the shipped
// family-level pair graph must reproduce the Python twin in make_pairgraph.py.
const graphCases = JSON.parse(fs.readFileSync(GRAPH_CASES, 'utf8'));
const candIdx = graphCases.candidates.map(l => byLabel.get(l));
const columnCache = new Map();
let dsknnErr = 0, dsknnChecked = 0;
for (const c of graphCases.cases) {
  const si = byLabel.get(c.seed);
  const ci = byLabel.get(c.candidate);
  if (si == null || ci == null) { ok(`dsknn case labels present (${c.seed} / ${c.candidate})`, false); continue; }
  if (!columnCache.has(si)) columnCache.set(si, dsknnColumn(corpus, resolved, si, candIdx));
  const got = columnCache.get(si).scores[candIdx.indexOf(ci)];
  dsknnErr = Math.max(dsknnErr, Math.abs(got - c.score));
  dsknnChecked++;
}
ok(`JS dsknnColumn matches ${graphCases.cases.length} Python cases`, dsknnChecked === graphCases.cases.length && dsknnErr < 1e-4);
const flagsOk = graphCases.flags.every(f => {
  const si = byLabel.get(f.seed);
  return si != null && columnCache.get(si) && columnCache.get(si).hasGraph === f.hasGraph;
});
ok('JS dsknnColumn reproduces the has_graph flags', flagsOk);

// End-to-end sanity: the learned model surfaces an endorsed partner near the
// top. Endorsement comes from the multi-partner benchmark
// (fixtures/fonts-pairings.json), not one arbitrary pairing. This exercises the
// shipped path (rankLearned folding the graph column into the score).
const elig = [];
for (let i = 0; i < corpus.count; i++) if (corpus.instances[i].eligible) elig.push(i);
const fx = JSON.parse(fs.readFileSync(path.join(DIR, 'fixtures', 'fonts-pairings.json'), 'utf8'));
const endorsed = new Set(fx.pairings.find(p => p.heading === 'Playfair Display').acceptable.map(a => a.family));
const pf = byLabel.get('Playfair Display 400');
const pool = elig.filter(i => corpus.instances[i].family !== 'Playfair Display');
const ranked = rankLearned(corpus, model, pf, pool, graph);
const best = ranked.findIndex(r => endorsed.has(corpus.instances[r.i].family));
ok('learned model surfaces an endorsed Playfair Display partner in the top 50', best >= 0 && best < 50);

console.log(`${pass} passed, ${fail} failed  (max parity error ${maxErr.toExponential(2)})`);
process.exit(fail ? 1 : 0);
