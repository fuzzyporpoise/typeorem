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
const { score, maxContrastAgainst } = await siteModule('reranker.js');

const DIR = path.dirname(fileURLToPath(import.meta.url));
const CASES = path.join(DIR, 'fixtures', 'reranker-cases.json');

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

const byLabel = new Map(corpus.instances.map((c, i) => [c.label, i]));
const cmaxCache = new Map();
const cmaxFor = hi => {
  if (!cmaxCache.has(hi)) cmaxCache.set(hi, maxContrastAgainst(corpus, hi));
  return cmaxCache.get(hi);
};

let maxErr = 0, checked = 0;
for (const c of cases) {
  const hi = byLabel.get(c.heading);
  const ci = byLabel.get(c.candidate);
  if (hi == null || ci == null) { ok(`case labels present (${c.heading} / ${c.candidate})`, false); continue; }
  const got = score(corpus, model, hi, ci, cmaxFor(hi));
  maxErr = Math.max(maxErr, Math.abs(got - c.score));
  checked++;
}
ok(`JS scores match ${cases.length} Python parity cases`, checked === cases.length && maxErr < 1e-4);

// End-to-end sanity: the learned model surfaces an endorsed partner near the
// top. Endorsement comes from the multi-partner benchmark
// (fixtures/fonts-pairings.json), not one arbitrary pairing.
const elig = [];
for (let i = 0; i < corpus.count; i++) if (corpus.instances[i].eligible) elig.push(i);
const fx = JSON.parse(fs.readFileSync(path.join(DIR, 'fixtures', 'fonts-pairings.json'), 'utf8'));
const endorsed = new Set(fx.pairings.find(p => p.heading === 'Playfair Display').acceptable.map(a => a.family));
const pf = byLabel.get('Playfair Display 400');
const pool = elig.filter(i => corpus.instances[i].family !== 'Playfair Display');
const ranked = pool.map(i => ({ i, s: score(corpus, model, pf, i, cmaxFor(pf)) })).sort((a, b) => b.s - a.s);
const best = ranked.findIndex(r => endorsed.has(corpus.instances[r.i].family));
ok('learned model surfaces an endorsed Playfair Display partner in the top 50', best >= 0 && best < 50);

console.log(`${pass} passed, ${fail} failed  (max parity error ${maxErr.toExponential(2)})`);
process.exit(fail ? 1 : 0);
