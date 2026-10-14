import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SITE_DATA, requireSite, siteModule } from './site.mjs';
import { loadSpace, mergeCorpora, eligibleIds, resolveLabels, evaluate, evaluateFamilies, summarise, defaultRoot } from './private.mjs';

// The axis diagnostic (TSK-001 A3 follow-up): which signals do we already measure
// but never score? The metric ships the embedding contrast, a body-legibility
// term, and a gated x-height axis. The renderer also measures stroke contrast,
// aperture, and 'o' aspect; the catalog carries weight and category. This ranks
// candidates under each axis on its own (and a few combinations) and reports
// hit@k / median, so a missing signal shows up as an axis that beats the shipped
// metric on the owner's slice. Run it on the public benchmark too, to see which
// axes generalize rather than overfit ten pairings.
//
//   model/.venv/bin/python model/validate/features.py --out ~/.cache/typeorem/features.json
//   node model/validate/axes.mjs --private ~/.cache/typeorem --features ~/.cache/typeorem/features.json
//   node model/validate/axes.mjs --labels model/validate/fixtures/fonts-pairings.json

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..');
const DEFAULT_LABELS = path.join(ROOT, 'model', 'validate', 'fixtures', 'user-pairings.json');

function parseArgs(argv) {
  const out = { private: process.env.TYPEOREM_PRIVATE_DIR || null, labels: null,
    features: null, k: 10, matrix: true, shippedOnly: false, familyLevel: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--private') out.private = argv[++i];
    else if (a === '--labels') out.labels = argv[++i];
    else if (a === '--features') out.features = argv[++i];
    else if (a === '--k') out.k = parseInt(argv[++i], 10);
    else if (a === '--no-matrix') out.matrix = false;
    else if (a === '--shipped-only') out.shippedOnly = true;
    else if (a === '--family-level') out.familyLevel = true;
    else throw new Error(`unknown argument ${a}`);
  }
  return out;
}

const num = v => typeof v === 'number' && Number.isFinite(v);
// A per-heading match in [0,1] over an axis, tolerant to `span`; -1 when unknown
// (so a missing feature never beats a real match).
const match = (a, b, span) => (num(a) && num(b)) ? 1 - Math.min(1, Math.abs(a - b) / span) : -1;

function axisCandidates(corpus, cmax, E, features) {
  const inst = i => corpus.instances[i];
  const feat = (i, key) => (features ? features[i]?.[key] ?? null : null);
  const xh = i => inst(i).xh;
  const axis = (name, score) => [name, { score }];
  const serifSplit = (a, b) => (inst(a).category === 'serif') !== (inst(b).category === 'serif');
  const shipped = (i, hi, role) => E.roleScore(corpus, cmax, hi, i, 0, role ?? 'body');
  const xhMatch = (i, hi) => match(xh(hi), xh(i), 0.15);
  const weightMatch = (i, hi) => 1 - Math.min(1, Math.abs(inst(hi).weight - inst(i).weight) / 400);
  const widthMatch = (i, hi) => match(inst(i).width ?? feat(i, 'o_aspect'), inst(hi).width ?? feat(hi, 'o_aspect'), 0.3);
  const strokeMatch = (i, hi) => match(feat(hi, 'contrast'), feat(i, 'contrast'), 0.3);
  const apertureMatch = (i, hi) => match(feat(hi, 'aperture'), feat(i, 'aperture'), 0.3);
  const catSplit = (i, hi) => (inst(hi).category !== inst(i).category ? 1 : 0);
  const legibility = i => E.legibility(corpus, null, i);
  const bodyGuard = x => E.bodySuitable(x);
  const roleGuard = (inst, h, role) => role !== 'body' || bodyGuard(inst);
  const bench = (i, hi) => E.roleScore(corpus, cmax, hi, i, 1, 'body');

  const list = [
    axis('embedding similarity', (i, hi) => E.dotAt(corpus.units, corpus.dim, hi, i)),
    axis('embedding contrast', (i, hi) => E.contrastAt(corpus.units, corpus.dim, hi, i)),
    axis('shipped (role+axis)', (i, hi, role) => shipped(i, hi, role)),
    axis('x-height match', (i, hi) => xhMatch(i, hi)),
    axis('width match', (i, hi) => widthMatch(i, hi)),
    axis('stroke-contrast match', (i, hi) => strokeMatch(i, hi)),
    axis('aperture match', (i, hi) => apertureMatch(i, hi)),
    axis('weight match', (i, hi) => weightMatch(i, hi)),
    axis('category split', (i, hi, role) => catSplit(i, hi)),
    axis('legibility (body)', i => legibility(i)),
    axis('serif axis (split * xh gate)', (i, hi) => (serifSplit(hi, i) ? E.xhGate(corpus, hi, i) : 0)),
    axis('split * xh match', (i, hi) => catSplit(i, hi) * Math.max(0, xhMatch(i, hi))),
    axis('split * (xh+width+stroke)', (i, hi) => catSplit(i, hi) * Math.max(0,
      0.5 * xhMatch(i, hi) + 0.3 * widthMatch(i, hi) + 0.2 * strokeMatch(i, hi))),
    axis('contrast + xh match', (i, hi) => E.contrastAt(corpus.units, corpus.dim, hi, i) / cmax + xhMatch(i, hi)),
    axis('similarity + xh match', (i, hi) => E.dotAt(corpus.units, corpus.dim, hi, i) + xhMatch(i, hi)),
    axis('similarity + xh + width', (i, hi) => E.dotAt(corpus.units, corpus.dim, hi, i) + xhMatch(i, hi) + widthMatch(i, hi)),
    // Additions to the shipped metric: does a signal it ignores lift it? A half
    // weight keeps the addition comparable to the ~1.0-1.9 shipped score. x-height
    // is already in the shipped score (STRUCT_WEIGHT), so only width/stroke/
    // aperture are additions here; width is the remaining lead (see the app's
    // metric-research "Structural term sweep").
    axis('shipped + 0.5 width', (i, hi, role) => shipped(i, hi, role) + 0.5 * widthMatch(i, hi)),
    axis('shipped + 0.5 stroke', (i, hi, role) => shipped(i, hi, role) + 0.5 * strokeMatch(i, hi)),
    axis('shipped + 0.5 aperture', (i, hi, role) => shipped(i, hi, role) + 0.5 * apertureMatch(i, hi)),
    // The public benchmark's own shipped config (contrast t=1, body-weight guard),
    // and the same with the width axis added, against the arbiter.
    ['benchmark (t=1 + guard)', { score: bench, keep: roleGuard }],
    ['default (t=0 + guard)', { score: (i, hi) => E.roleScore(corpus, cmax, hi, i, 0, 'body'), keep: roleGuard }],
    ['bench + 0.5 width', { score: (i, hi) => bench(i, hi) + 0.5 * widthMatch(i, hi), keep: roleGuard }],
    ['bench + 0.75 width', { score: (i, hi) => bench(i, hi) + 0.75 * widthMatch(i, hi), keep: roleGuard }],
  ];
  return list;
}

async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  requireSite('the axis diagnostic scores the shipped metric over the shipped corpus');
  const engine = await siteModule('engine.js');
  const vectors = await siteModule('vectors.js');
  const E = { ...engine, dotAt: vectors.dotAt, contrastAt: vectors.contrastAt };

  const shipped = loadSpace(SITE_DATA);
  const privRoot = args.private || defaultRoot();
  const hasPrivate = !args.shippedOnly && fs.existsSync(path.join(privRoot, 'corpus', 'catalog.json'));
  const corpus = hasPrivate ? mergeCorpora(shipped, loadSpace(path.join(privRoot, 'corpus'))) : shipped;
  const cmax = E.maxContrast(corpus, eligibleIds(corpus));
  E.centroid = E.buildBodyCentroid(corpus);

  let features = null;
  const featPath = args.features || (hasPrivate ? path.join(privRoot, 'features.json') : null);
  if (featPath && fs.existsSync(featPath)) {
    const dump = JSON.parse(fs.readFileSync(featPath, 'utf8'));
    if (dump.count === corpus.count) features = dump.features;
    else console.error(`! features.json has ${dump.count} rows, corpus has ${corpus.count}; ignoring it`);
  } else {
    console.error('! no features.json: the renderer axes (width, stroke, aperture) are skipped');
  }

  const labelsPath = args.labels || [path.join(privRoot, 'pairings.json'), DEFAULT_LABELS]
    .find(p => fs.existsSync(p));
  const labels = JSON.parse(fs.readFileSync(labelsPath, 'utf8'));
  const { rows, missing } = resolveLabels(corpus, labels, m => console.error(`  ! ${m}`));

  console.log(`corpus: ${corpus.count} rows (dim ${corpus.dim})${hasPrivate ? ` = ${shipped.count} shipped + ${corpus.count - shipped.count} private` : ''}`);
  console.log(`labels: ${labelsPath}  ->  ${rows.length} headings, ${rows.reduce((n, r) => n + r.partners.length, 0)} partners`);
  if (missing.length) console.log(`  unresolved: ${[...new Set(missing)].join(', ')}`);
  console.log(`axes from renders: ${features ? 'yes' : 'NO (run features.py)'}`);
  if (!rows.length) { console.log('no labels resolved'); return; }
  console.log('');

  const candidates = axisCandidates(corpus, cmax, E, features);
  const results = (args.familyLevel ? evaluateFamilies : evaluate)(candidates, corpus, rows, args.k);
  const f3 = x => (Number.isFinite(x) ? x.toFixed(3) : '-');

  console.log(`ranking: ${args.familyLevel ? 'family-level (a family scored by its best cut)' : 'instance-level'}`);
  console.log(`${'axis (higher is better)'.padEnd(34)}hit@${args.k}`.padEnd(45) + 'recall'.padStart(8) + 'nDCG'.padStart(8) + 'median'.padStart(9));
  const table = candidates.map(([name]) => [name, summarise(results.get(name), args.k)]);
  for (const [name, s] of table) {
    console.log(`${name.padEnd(34)}${`${s.hit}/${s.n}`.padStart(6)}${f3(s.recall).padStart(13)}${f3(s.ndcg).padStart(8)}${String(s.median ?? '-').padStart(9)}`);
  }

  if (args.matrix) {
    // Per-heading rank under the shipped metric vs the axes most likely to carry
    // the missing signal: structure (xh/width/stroke), category split, and combos.
    const cols = ['shipped (role+axis)', 'x-height match', 'width match', 'stroke-contrast match',
      'category split', 'split * xh match', 'split * (xh+width+stroke)', 'contrast + xh match'];
    const idx = Object.fromEntries([...results].map(([name, per]) => [name, new Map(per.map(r => [r.name, r]))]));
    console.log(`\nbest partner rank per axis (of ~${corpus.count} candidates):`);
    console.log(`  ${'heading'.padEnd(16)}` + cols.map(c => abbr(c).padStart(9)).join(''));
    for (const row of rows) {
      const cells = cols.map(c => {
        const r = idx[c]?.get(row.name);
        return (r ? String(r.best) : '-').padStart(9);
      }).join('');
      console.log(`  ${row.name.padEnd(16)}${cells}`);
    }
    console.log(`\n  ${cols.map(c => `${abbr(c)}=${c}`).join('   ')}`);
  }
}

function abbr(name) {
  return name.split(/[\s(]/)[0].replace(/[^a-z]/gi, '').slice(0, 7).toLowerCase();
}

main().catch(e => { console.error(`\n${e.message}`); process.exit(1); });
