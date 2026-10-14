import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { SITE_DATA, requireSite, siteModule } from './site.mjs';

// The private tuning harness (TSK-001, plan Part A3/B2).
//
// It scores the SHIPPED metric and a set of challengers over the owner's own
// pairings, against their own (possibly licensed) faces. Those faces only exist
// in a private corpus written OUTSIDE both checkouts
// (model/build_corpus.py --private, plan A1); this harness merges that corpus
// with the shipped catalog exactly as the app's dev overlay will (public rows,
// then private rows), so the ranking it measures is the ranking the tool runs.
//
// It reads and prints only. The numbers it prints are the durable output: the
// chosen constants are copied into site/js/engine.js by hand, and the run goes
// into the app's private .todo/metric-research.md. Nothing here ships.
//
//   node model/validate/private.mjs                       # defaults to ~/.cache/typeorem
//   node model/validate/private.mjs --private /some/where --labels pairings.json --sweep

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..');
const DEFAULT_LABELS = path.join(ROOT, 'model', 'validate', 'fixtures', 'user-pairings.json');
const BODY_MIN = 300, BODY_MAX = 700;

// The writer's default root (model/private_corpus.py#default_root): the same
// machine-local cache home, so the two sides agree with no configuration.
export function defaultRoot() {
  const base = process.env.XDG_CACHE_HOME || path.join(process.env.HOME || '.', '.cache');
  return path.resolve(base, 'typeorem');
}

export function norm(s) {
  return String(s).toLowerCase().replace(/[^a-z0-9]+/g, '');
}

export function readJson(p) {
  return JSON.parse(fs.readFileSync(p, 'utf8'));
}

export function sha256File(p) {
  return createHash('sha256').update(fs.readFileSync(p)).digest('hex');
}

// A corpus in the shipped shape (the app's loadCorpus): dim, count, instances,
// and L2-normalized unit vectors in a flat [count, dim] Float64Array.
export function loadSpace(dataDir) {
  const meta = readJson(path.join(dataDir, 'vectors.meta.json'));
  const catalog = readJson(path.join(dataDir, 'catalog.json'));
  const bin = fs.readFileSync(path.join(dataDir, 'vectors.i8.bin'));
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

// Public rows first, then private rows, with the private ids renumbered. This is
// the contract the app overlay reproduces (app ledger TSK-019); keep them equal.
export function mergeCorpora(publicCorpus, privateCorpus) {
  if (publicCorpus.dim !== privateCorpus.dim) {
    throw new Error(`dim mismatch: shipped ${publicCorpus.dim}, private ${privateCorpus.dim}`);
  }
  const base = publicCorpus.count;
  const instances = publicCorpus.instances.concat(
    privateCorpus.instances.map((c, i) => ({ ...c, id: base + i })));
  const units = new Float64Array(instances.length * publicCorpus.dim);
  units.set(publicCorpus.units);
  units.set(privateCorpus.units, base * publicCorpus.dim);
  return { dim: publicCorpus.dim, count: instances.length, instances, units };
}

export function eligibleIds(corpus) {
  const out = [];
  for (let i = 0; i < corpus.count; i++) if (corpus.instances[i].eligible) out.push(i);
  return out;
}

// The families a label name refers to. Exact matches win; else a prefix match
// (so "Neue Haas Grotesk" resolves to its Display and Text cuts). A name that
// matches neither is unresolved, not fuzzy-matched: substring matching would tie
// "Kabel" to "Abel" and "Baskerville" to "Libre Baskerville". `aliases[name]`
// overrides the name, for the cases the fonts' internal names do not match the
// owner's spelling.
export function familiesMatching(instances, elig, name, aliases = {}) {
  const target = norm(aliases[name] ?? name);
  if (!target) return [];
  const exact = new Set(), prefix = new Set();
  for (const i of elig) {
    const family = instances[i].family, nf = norm(family);
    if (nf === target) exact.add(family);
    else if (nf.startsWith(target) || target.startsWith(nf)) prefix.add(family);
  }
  return [...(exact.size ? exact : prefix)].sort();
}

// The instance a `cut` names within `family` ({ weight, style }), else the
// canonical 400-normal cut. A cut the family does not carry falls back to the
// canonical one with a warning, so a typo does not silently render the wrong
// face.
export function pickCut(instances, elig, family, cut, warn = () => {}) {
  const cands = elig.filter(i => instances[i].family === family);
  if (!cands.length) return null;
  if (cut && (cut.weight != null || cut.style != null)) {
    const weight = cut.weight ?? null;
    const style = cut.style == null ? 'normal' : (cut.style === 'italic' ? 'italic' : 'normal');
    let hit = cands.find(i => (weight == null || instances[i].weight === weight)
      && instances[i].style === style);
    if (hit == null && cut.style == null) hit = cands.find(i => weight == null || instances[i].weight === weight);
    if (hit != null) return hit;
    warn(`no ${[cut.weight, cut.style].filter(v => v != null).join(' ')} cut of "${family}"; using the canonical cut`);
  }
  return canonId(instances, elig, family);
}

// Families whose name looks like a missing label name, for a helpful "did you
// mean" line (the matching above returns nothing by construction when missing).
export function nearestFamilies(instances, elig, name, k = 4) {
  const key = norm(name).slice(0, 4);
  if (!key) return [];
  const seen = new Set();
  for (const i of elig) seen.add(instances[i].family);
  return [...seen].filter(f => norm(f).includes(key)).sort().slice(0, k);
}

function canonId(instances, elig, family) {
  const cands = elig.filter(i => instances[i].family === family);
  if (!cands.length) return null;
  return cands.find(i => instances[i].weight === 400 && instances[i].style === 'normal')
    ?? cands.find(i => instances[i].weight === 400) ?? cands[0];
}

// Resolve the labels (the multi-partner schema, or the legacy heading + partners
// list) to instance ids over the merged corpus. `dropped` entries are honoured so
// a face the owner no longer has does not read as a metric miss. A per-partner
// `cut` ({ weight, style }) picks a specific instance; without it the role picks
// the canonical 400-normal cut. `sameFamily` keeps partners in the heading's own
// family (the browser's family-diversity rule excludes them, so the harness
// defaults to false and the by-eye sheet sets it true).
export function resolveLabels(corpus, labels, warn = () => {}, { sameFamily = false } = {}) {
  const elig = eligibleIds(corpus);
  const aliases = labels.aliases || {};
  const dropped = new Set();
  for (const d of labels.dropped || []) {
    dropped.add(d.partner ? `${norm(d.heading)}|${norm(d.partner)}` : `${norm(d.heading)}|`);
  }
  const rows = [], missing = [];
  for (const entry of labels.pairings || []) {
    const isList = Array.isArray(entry);
    const heading = isList ? entry[0] : entry.heading;
    const partnerSpecs = (isList ? entry.slice(1) : (entry.acceptable || []))
      .map(a => (typeof a === 'string' ? { family: a, role: 'body' }
        : { family: a.family, role: a.role || 'body', cut: a.cut }))
      .filter(p => !dropped.has(`${norm(heading)}|${norm(p.family)}`)
        && !dropped.has(`${norm(heading)}|`));
    if (!partnerSpecs.length) continue;

    const hFams = familiesMatching(corpus.instances, elig, heading, aliases);
    if (!hFams.length) { missing.push(heading); continue; }
    if (hFams.length > 1) warn(`heading "${heading}" matches ${hFams.join(', ')}; using "${hFams[0]}" (add an alias to pick a cut)`);
    const headingFamily = hFams[0];
    const hi = pickCut(corpus.instances, elig, headingFamily, entry.cut, warn);

    const partners = new Map();
    for (const spec of partnerSpecs) {
      const fams = familiesMatching(corpus.instances, elig, spec.family, aliases);
      if (!fams.length) { missing.push(spec.family); continue; }
      for (const f of fams) {
        const c = pickCut(corpus.instances, elig, f, spec.cut, warn);
        if (c != null && (sameFamily || corpus.instances[c].family !== headingFamily)) {
          partners.set(c, { id: c, family: f, role: spec.role, label: spec.family });
        }
      }
    }
    if (hi == null || !partners.size) { if (hi == null) missing.push(heading); continue; }
    const partnerList = [...partners.values()];
    const roles = [...new Set(partnerList.map(p => p.role))].sort();
    rows.push({ name: heading, headingId: hi, headingFamily, partners: partnerList, roles,
      role: roles.join('/') });
  }
  return { rows, missing };
}

// The candidate metrics: the shipped metric, the guards the tool applies, and,
// with --sweep, a grid over the shipped constants so a change can be read off.
// `role` is the slot the partners fill (from the label row); the body role is the
// one the legibility term applies to.
export function buildCandidates(corpus, cmax, E, { sweeps = false } = {}) {
  const serifSplit = (a, b) => (corpus.instances[a].category === 'serif')
    !== (corpus.instances[b].category === 'serif');
  const bodyGuard = inst => E.bodySuitable(inst);
  const stroke = (inst, heading) => E.isCustomRow(inst) || inst.weight <= heading.weight;
  const variant = ({ t = 0, lambda = E.ROLE_LAMBDA, serif = E.SERIF_WEIGHT,
                     sigma = E.XH_GATE_SIGMA, leg = 'stored' }) => (i, hi, role) => {
    let s = E.pairScore(corpus, cmax, hi, i, t)
      + serif * (serifSplit(hi, i) ? E.xhGate(corpus, hi, i, sigma) : 0)
      + (E.STRUCT_WEIGHT || 0) * E.xhMatch(corpus, hi, i)
      + (E.WIDTH_WEIGHT || 0) * E.widthMatch(corpus, hi, i);
    if (role === 'body') {
      s += lambda * (leg === 'centroid'
        ? E.legibility(corpus, E.centroid, i) : E.legibility(corpus, null, i));
    }
    return s;
  };

  const list = [
    ['contrast (legacy)', { score: (i, hi) => E.contrastAt(corpus.units, corpus.dim, hi, i) }],
    ['shipped (role+axis)', { score: variant({}) }],
    ['shipped + body guard', { score: variant({}), keep: (inst, h, role) => role !== 'body' || bodyGuard(inst) }],
    ['shipped + body + stroke', { score: variant({}),
      keep: (inst, h, role) => role !== 'body' || (bodyGuard(inst) && stroke(inst, h)) }],
    ['shipped + centroid legibility', { score: variant({ leg: 'centroid' }) }],
  ];
  if (sweeps) {
    for (const lambda of [0, 0.25, 0.5, 0.75, 1.0]) list.push([`role lambda ${lambda}`, { score: variant({ lambda }) }]);
    for (const serif of [0, 0.1, 0.2, 0.4]) list.push([`serif weight ${serif}`, { score: variant({ serif }) }]);
    for (const t of [-0.5, 0, 0.5]) list.push([`contrast t ${t}`, { score: variant({ t }) }]);
    for (const sigma of [0.03, 0.05, 0.1]) list.push([`xh sigma ${sigma}`, { score: variant({ sigma }) }]);
  }
  return list;
}

// Rank the eligible instances of other families for each heading and measure the
// acceptable partners against that ranking. Each partner is scored in its OWN
// slot: a `body` partner is measured in the body ranking (text-weight pool, the
// role legibility term), an `accent` partner in the accent ranking, so a role is
// never penalized by the other slot's guards. Reports hit@k, recall@k, nDCG@k, and
// the best rank per pairing. `sameFamily` admits instances of the heading's own
// family to the pool (the heading instance itself is always excluded, so a partner
// naming the heading's exact cut stays unscorable instead of a vacuous rank 1);
// the default follows the browser's family-diversity rule.
export function evaluate(candidates, corpus, rows, K, { sameFamily = false } = {}) {
  const inst = i => corpus.instances[i];
  const results = new Map();
  for (const [name, spec] of candidates) {
    const per = [];
    for (const row of rows) {
      const hi = row.headingId;
      const hfam = inst(hi).family;
      const byRole = new Map();
      for (const role of row.roles) {
        const cand = [];
        for (let i = 0; i < corpus.count; i++) {
          if (!inst(i).eligible || i === hi) continue;
          if (!sameFamily && inst(i).family === hfam) continue;
          if (spec.keep && !spec.keep(inst(i), inst(hi), role)) continue;
          cand.push(i);
        }
        const ranked = cand.map(i => ({ i, s: spec.score(i, hi, role) })).sort((a, b) => b.s - a.s);
        const pos = new Map(ranked.map((r, k) => [r.i, k + 1]));
        const rel = new Set(row.partners.filter(p => p.role === role).map(p => p.id));
        byRole.set(role, { ranked, pos, rel, total: ranked.length });
      }
      const ranks = row.partners.map(p => {
        const b = byRole.get(p.role);
        return { ...p, rank: b.pos.get(p.id) ?? b.total + 1 };
      }).sort((a, b) => a.rank - b.rank);
      const ndcgAt = k => {
        let sum = 0, n = 0;
        for (const { ranked, rel } of byRole.values()) {
          let d = 0;
          for (let p = 1; p <= Math.min(k, ranked.length); p++) if (rel.has(ranked[p - 1].i)) d += 1 / Math.log2(p + 1);
          let id = 0;
          for (let j = 1; j <= Math.min(rel.size, k); j++) id += 1 / Math.log2(j + 1);
          sum += id ? d / id : 0;
          n++;
        }
        return n ? sum / n : 0;
      };
      per.push({ name: row.name, role: row.role, n: ranks.length, best: ranks[0].rank, ranks,
        total: Math.max(0, ...[...byRole.values()].map(b => b.total)), ndcgAt, pass: ranks[0].rank <= K });
    }
    results.set(name, per);
  }
  return results;
}

// Family-level evaluation. The gold labels name families ("Grenette", "Styrene"),
// while the tool ranks instances; ranking every instance lets one many-cut family
// crowd the head, and relevance at a single canonical cut is arbitrary. Here a
// family's score is its BEST eligible instance for a role (so the family is the
// unit and the cut is chosen within it), families are ranked per role, and a
// partner family is relevant however its cut is named.
export function evaluateFamilies(candidates, corpus, rows, K) {
  const inst = i => corpus.instances[i];
  const elig = eligibleIds(corpus);
  const results = new Map();
  for (const [name, spec] of candidates) {
    const per = [];
    for (const row of rows) {
      const hi = row.headingId;
      const hfam = inst(hi).family;
      const byRole = new Map();
      for (const role of row.roles) {
        const best = new Map();
        for (const i of elig) {
          if (inst(i).family === hfam) continue;
          if (spec.keep && !spec.keep(inst(i), inst(hi), role)) continue;
          const s = spec.score(i, hi, role);
          const cur = best.get(inst(i).family);
          if (cur === undefined || s > cur) best.set(inst(i).family, s);
        }
        const ranked = [...best.entries()].sort((a, b) => b[1] - a[1]).map(e => e[0]);
        const pos = new Map(ranked.map((f, k) => [f, k + 1]));
        const rel = new Set(row.partners.filter(p => p.role === role).map(p => p.family));
        byRole.set(role, { ranked, pos, rel });
      }
      const ranks = row.partners.map(p => {
        const b = byRole.get(p.role);
        return { ...p, rank: b.pos.get(p.family) ?? b.ranked.length + 1 };
      }).sort((a, b) => a.rank - b.rank);
      const ndcgAt = k => {
        let sum = 0, n = 0;
        for (const { ranked, rel } of byRole.values()) {
          let d = 0;
          for (let p = 1; p <= Math.min(k, ranked.length); p++) if (rel.has(ranked[p - 1])) d += 1 / Math.log2(p + 1);
          let id = 0;
          for (let j = 1; j <= Math.min(rel.size, k); j++) id += 1 / Math.log2(j + 1);
          sum += id ? d / id : 0;
          n++;
        }
        return n ? sum / n : 0;
      };
      per.push({ name: row.name, role: row.role, n: ranks.length, best: ranks[0].rank, ranks,
        total: Math.max(0, ...[...byRole.values()].map(b => b.ranked.length)), ndcgAt, pass: ranks[0].rank <= K });
    }
    results.set(name, per);
  }
  return results;
}

export function summarise(per, K) {
  if (!per.length) return { hit: 0, n: 0, recall: 0, ndcg: 0, median: null };
  const ndcgAt = k => per.reduce((a, r) => a + r.ndcgAt(k), 0) / per.length;
  const recallAt = k => per.reduce((a, r) => a + r.ranks.filter(x => x.rank <= k).length / r.n, 0) / per.length;
  const bests = per.map(r => r.best).sort((a, b) => a - b);
  return {
    hit: per.filter(r => r.pass).length, n: per.length,
    recall: recallAt(K), ndcg: ndcgAt(K),
    median: bests[Math.floor(bests.length / 2)],
  };
}

function parseArgs(argv) {
  const out = { private: process.env.TYPEOREM_PRIVATE_DIR || null,
    labels: process.env.TYPEOREM_PRIVATE_LABELS || null, k: 10, sweep: false, familyLevel: false,
    sameFamily: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--private') out.private = argv[++i];
    else if (a === '--labels') out.labels = argv[++i];
    else if (a === '--k') out.k = parseInt(argv[++i], 10);
    else if (a === '--sweep') out.sweep = true;
    else if (a === '--family-level') out.familyLevel = true;
    else if (a === '--same-family') out.sameFamily = true;
    else throw new Error(`unknown argument ${a}`);
  }
  return out;
}

export function checkPca(privateVersionPath, siteData) {
  if (!fs.existsSync(privateVersionPath)) return null;
  const version = readJson(privateVersionPath);
  const want = version.pcaSha256;
  const shipped = path.join(siteData, 'pca.bin');
  if (!want || !fs.existsSync(shipped)) return want || null;
  const got = sha256File(shipped);
  if (got !== want) {
    throw new Error(
      `the private corpus was written in a different projection space.\n` +
      `  private meta pcaSha256 ${want}\n  shipped pca.bin      ${got}\n` +
      `A PCA refit (a corpus rebuild) invalidates the private corpus; re-run ` +
      `model/build_corpus.py --private.`);
  }
  return want;
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (!args.private) {
    args.private = defaultRoot();
    console.log(`no private root given; using the default ${args.private}`);
  }
  const root = path.resolve(args.private.replace(/^~(?=$|\/)/, process.env.HOME || '~'));
  const corpusDir = path.join(root, 'corpus');
  if (!fs.existsSync(path.join(corpusDir, 'catalog.json'))) {
    throw new Error(`no private corpus at ${corpusDir}. Write one first:\n`
      + `    TYPEOREM_PRIVATE_DIR=${args.private} model/.venv/bin/python model/build_corpus.py --private`);
  }
  const explicit = args.labels;
  const resolvedLabels = explicit || [path.join(root, 'pairings.json'), DEFAULT_LABELS]
    .find(p => fs.existsSync(p)) || null;
  if (!resolvedLabels || !fs.existsSync(resolvedLabels)) {
    throw new Error(`no labels file at ${resolvedLabels || explicit}`
      + ` (pass --labels <path>, or keep the owner's pairings in <root>/pairings.json)`);
  }

  requireSite('the private harness scores the shipped metric over the shipped corpus plus a private corpus');
  const E = {
    ...(await siteModule('engine.js')),
    contrastAt: (await siteModule('vectors.js')).contrastAt,
  };

  const pcaSha = checkPca(path.join(corpusDir, 'corpus.version.json'), SITE_DATA);
  const shipped = loadSpace(SITE_DATA);
  const priv = loadSpace(corpusDir);
  const corpus = mergeCorpora(shipped, priv);
  E.centroid = E.buildBodyCentroid(corpus);
  const cmax = E.maxContrast(corpus, eligibleIds(corpus));

  const labels = readJson(resolvedLabels);
  const { rows, missing } = resolveLabels(corpus, labels, m => console.error(`  ! ${m}`),
    { sameFamily: args.sameFamily });
  console.log(`private corpus ${corpusDir}`);
  console.log(`  shipped ${shipped.count} rows + private ${priv.count} rows = ${corpus.count} (dim ${corpus.dim})`);
  if (pcaSha) console.log(`  projection pca.bin sha256 ${pcaSha.slice(0, 16)} (matches the shipped space)`);
  console.log(`labels ${resolvedLabels}: ${rows.length} headings, `
    + `${rows.reduce((n, r) => n + r.partners.length, 0)} acceptable partners`);
  if (missing.length) {
    console.log(`  unresolved (not in the merged corpus, or dropped): ${[...new Set(missing)].join(', ')}`);
    for (const name of [...new Set(missing)]) {
      const near = nearestFamilies(corpus.instances, eligibleIds(corpus), name);
      if (near.length) console.log(`    near "${name}": ${near.join(', ')}`);
    }
  }

  if (!rows.length) {
    console.log('\nno labels resolved against the merged corpus; nothing to score.');
    return new Map();
  }
  console.log('');

  const candidates = buildCandidates(corpus, cmax, E, { sweeps: args.sweep });
  const results = args.familyLevel
    ? evaluateFamilies(candidates, corpus, rows, args.k)
    : evaluate(candidates, corpus, rows, args.k, { sameFamily: args.sameFamily });
  const f3 = x => (Number.isFinite(x) ? x.toFixed(3) : '-');
  const baselineName = 'shipped (role+axis)';
  const baseline = results.get(baselineName);
  const basePass = new Map(baseline.map(r => [r.name, r.pass]));

  console.log(`${'candidate'.padEnd(32)}hit@${args.k}`.padEnd(43) + 'recall'.padStart(8)
    + 'nDCG'.padStart(8) + 'median'.padStart(9));
  for (const [name] of candidates) {
    const s = summarise(results.get(name), args.k);
    console.log(`${name.padEnd(32)}${`${s.hit}/${s.n}`.padStart(6)}`
      + `${f3(s.recall).padStart(13)}${f3(s.ndcg).padStart(8)}${String(s.median ?? '-').padStart(9)}`);
  }

  console.log(`\nper-pairing (baseline = ${baselineName}, rank of each partner in its own slot, of ${baseline[0] ? baseline[0].total : 0}):`);
  for (const r of [...baseline].sort((a, b) => a.best - b.best)) {
    const parts = r.ranks.map(p => `${p.label || p.family} [${p.role}] ${p.rank}`).join('   ');
    console.log(`  ${r.name.padEnd(16)} ${parts}`);
  }

  console.log('\nwhat each change breaks / fixes vs the baseline:');
  let anyFlip = false;
  for (const [name] of candidates) {
    if (name === baselineName) continue;
    const per = new Map(results.get(name).map(r => [r.name, r]));
    const broke = [], fixed = [];
    for (const r of baseline) {
      const now = per.get(r.name);
      if (!now) continue;
      if (basePass.get(r.name) && !now.pass) broke.push(r.name);
      if (!basePass.get(r.name) && now.pass) fixed.push(r.name);
    }
    if (!broke.length && !fixed.length) continue;
    anyFlip = true;
    const bits = [];
    if (broke.length) bits.push(`broke ${broke.join(', ')}`);
    if (fixed.length) bits.push(`fixed ${fixed.join(', ')}`);
    console.log(`  ${name}: ${bits.join('; ')}`);
  }
  if (!anyFlip) console.log('  (no baseline-passing pairing flips under any candidate)');
  return results;
}

const isMain = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  main().catch(e => { console.error(`\n${e.message}`); process.exit(1); });
}
