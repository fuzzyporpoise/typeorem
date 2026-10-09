// Build fixtures/fontsinuse-pairings.json: real-world font-use pairs mined from
// the supplementary data of Choi & Hyun, "Typeface network and the principle of
// font pairing" (Sci Rep 14:30820, 2024), itself collected from fontsinuse.com.
//
// Why: the multi-partner benchmark (mine-pairings.mjs) is curated endorsement
// data; this fixture is observed usage by practicing designers, including pairings
// involving famous commercial faces. Only pairs where BOTH sides resolve to
// eligible families in the shipped catalog survive, so nothing derived from a
// licensed font file is involved: names and co-occurrence counts only.
//
// Differences from the curated fixture:
//   - pairs are unordered usage co-occurrences (fontsinuse records no roles), so
//     each surviving pair is emitted in both directions;
//   - each partner carries `uses` (how many use cases show the pair),
//     `mediums` (per-medium counts), `lift` (observed co-occurrence over what
//     family frequency alone predicts, the de-skew), and `kin` when the pair is
//     same-superfamily (a different bias, excluded in refined slices);
//   - a small alias map substitutes open revivals / metric-compatible faces for
//     commercial names (Futura -> Jost, Arial -> Arimo, ...); every alias is
//     documented below. Contested lookalikes (Helvetica, Univers, Gotham, ...)
//     are deliberately NOT aliased.
//
// The raw CSV is not committed (the article is CC BY-NC-ND); the generator
// re-downloads it from the publisher's stable supplementary URL, or reads a
// local copy: FONTSINUSE_CSV=<path> wins, then model/.cache/fontsinuse/MOESM1.csv.
//
// Re-run: node model/validate/mine-fontsinuse.mjs

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SITE_DATA, requireSite } from './site.mjs';

requireSite('mined partners are validated against the shipped catalog');
const DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(DIR, '..', '..');
const CACHE_CSV = path.join(REPO, 'model', '.cache', 'fontsinuse', 'MOESM1.csv');
const OUT = path.join(DIR, 'fixtures', 'fontsinuse-pairings.json');
const CSV_URL = 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-024-81601-w/MediaObjects/41598_2024_81601_MOESM1_ESM.csv';

async function loadCsvText() {
  if (process.env.FONTSINUSE_CSV) return fs.readFileSync(process.env.FONTSINUSE_CSV, 'utf8');
  if (fs.existsSync(CACHE_CSV)) return fs.readFileSync(CACHE_CSV, 'utf8');
  const r = await fetch(CSV_URL);
  if (!r.ok) throw new Error(`${CSV_URL}: HTTP ${r.status}`);
  const text = await r.text();
  fs.mkdirSync(path.dirname(CACHE_CSV), { recursive: true });
  fs.writeFileSync(CACHE_CSV, text);
  return text;
}

function* csvRows(text) {
  let field = '', row = [], inQ = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQ) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else inQ = false;
      } else field += c;
    } else if (c === '"') inQ = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field); field = '';
      yield row; row = [];
    } else field += c;
  }
  if (field || row.length) { row.push(field); yield row; }
}

const catalog = JSON.parse(fs.readFileSync(path.join(SITE_DATA, 'catalog.json'), 'utf8'));
const elig = new Set(catalog.instances.filter(i => i.eligible).map(i => i.family));

// Commercial name -> catalog family. Every entry is an open revival or a
// metric-compatible face (per its Google Fonts description), never a lookalike.
const FONTINUSE_ALIAS = new Map(Object.entries({
  'futura': 'Jost',
  'franklin gothic': 'Libre Franklin',
  'times new roman': 'Tinos',
  'times': 'Tinos',
  'arial': 'Arimo',
  'georgia': 'Gelasio',
  'bodoni': 'Bodoni Moda',
  'baskerville': 'Libre Baskerville',
  'courier': 'Courier Prime',
  'caslon': 'Libre Caslon Text',
  'adobe caslon': 'Libre Caslon Text',
}));

const QUALIFIER = /^(?:Ultra[-\s]+|Extra[-\s]+|Semi[-\s]+)?(?:Thin|Light|Regular|Medium|Bold|Black|Heavy|Book|Demi|Italic|Roman)$/i;

const lower = new Map();
for (const f of elig) if (!lower.has(f.toLowerCase())) lower.set(f.toLowerCase(), f);
const hit = s => elig.has(s) ? s : lower.get(s.toLowerCase()) ?? null;

function resolve(raw) {
  const s = raw.replace(/\([^)]*\)/g, ' ').replace(/\s+/g, ' ').replace(/[.,;:!?]+$/, '').trim();
  if (!s) return null;
  const exact = hit(s);
  if (exact) return exact;
  const aliased = FONTINUSE_ALIAS.get(s.toLowerCase());
  if (aliased && elig.has(aliased)) return aliased;
  let t = s;
  while (true) {
    const parts = t.split(' ');
    if (parts.length > 1 && QUALIFIER.test(parts[parts.length - 1])) {
      t = parts.slice(0, -1).join(' ');
      const h = hit(t);
      if (h) return h;
      const a = FONTINUSE_ALIAS.get(t.toLowerCase());
      if (a && elig.has(a)) return a;
      continue;
    }
    break;
  }
  return null;
}

const CLASS_TOKENS = new Set(['sans', 'serif', 'mono', 'slab', 'display', 'text', 'gothic', 'grotesk', 'grotesque', 'condensed', 'expanded', 'narrow', 'rounded', 'sc', 'pro']);

// Same-superfamily pairs (Space Grotesk + Space Mono, IBM Plex Sans + IBM Plex
// Mono): shared leading tokens, differ only in class words. High lift but a
// different bias, so the flag lets slices exclude them.
function kin(a, b) {
  const ta = a.toLowerCase().split(' '), tb = b.toLowerCase().split(' ');
  let i = 0;
  while (i < ta.length && i < tb.length && ta[i] === tb[i]) i++;
  if (i === 0) return false;
  return ta.slice(i).every(t => CLASS_TOKENS.has(t)) && tb.slice(i).every(t => CLASS_TOKENS.has(t));
}

const text = await loadCsvText();
let rows = 0, multiRaw = 0;
const pairs = new Map();
const support = new Map();
const unresolved = new Map();
for (const row of csvRows(text.replace(/^﻿/, ''))) {
  if (!row.length || row[0].startsWith('#')) continue;
  const medium = row[0].trim() || 'Unlabeled';
  const raw = row.slice(1).map(c => c.trim()).filter(Boolean);
  if (raw.length < 2) continue;
  rows++;
  multiRaw++;
  const seen = new Set();
  const fams = [];
  for (const name of raw) {
    const f = resolve(name);
    if (f == null) unresolved.set(name, (unresolved.get(name) || 0) + 1);
    else if (!seen.has(f)) { seen.add(f); fams.push(f); }
  }
  for (const f of fams) support.set(f, (support.get(f) || 0) + 1);
  for (let i = 0; i < fams.length; i++) {
    for (let j = i + 1; j < fams.length; j++) {
      const key = [fams[i], fams[j]].sort().join('\0');
      let p = pairs.get(key);
      if (!p) { p = { uses: 0, mediums: {} }; pairs.set(key, p); }
      p.uses++;
      p.mediums[medium] = (p.mediums[medium] || 0) + 1;
    }
  }
}

// Lift over the frequency expectation: observed co-occurrence divided by what
// family popularity alone predicts (supports counted over all multi-font use
// cases, so unresolved commercial partners still count toward popularity).
for (const [key, p] of pairs) {
  const [a, b] = key.split('\0');
  p.lift = (p.uses * multiRaw) / ((support.get(a) || 1) * (support.get(b) || 1));
  p.kin = kin(a, b);
}

const byHeading = new Map();
for (const [key, p] of pairs) {
  const [a, b] = key.split('\0');
  for (const [h, partner] of [[a, b], [b, a]]) {
    let list = byHeading.get(h);
    if (!list) { list = new Map(); byHeading.set(h, list); }
    const cur = list.get(partner) || { uses: 0, mediums: {}, lift: p.lift, kin: p.kin };
    cur.uses += p.uses;
    cur.lift = p.lift;
    cur.kin = p.kin;
    for (const [m, n] of Object.entries(p.mediums)) cur.mediums[m] = (cur.mediums[m] || 0) + n;
    list.set(partner, cur);
  }
}

const pairings = [...byHeading.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([heading, partners]) => ({
  heading,
  acceptable: [...partners.entries()]
    .sort((a, b) => b[1].uses - a[1].uses || a[0].localeCompare(b[0]))
    .map(([family, p]) => ({
      family,
      uses: p.uses,
      lift: Math.round(p.lift * 100) / 100,
      ...(p.kin ? { kin: true } : {}),
      mediums: Object.fromEntries(Object.entries(p.mediums).sort((x, y) => y[1] - x[1])),
      sources: ['fontsinuse'],
    })),
}));

const fixture = {
  version: 1,
  generated: new Date().toISOString().slice(0, 10),
  note: 'Real-world font-use pairs from the supplementary data of Choi & Hyun, "Typeface network and the principle of font pairing" (Sci Rep 14:30820, 2024), collected from fontsinuse.com. Usage co-occurrence, not curated endorsement: pairs are unordered, so each is emitted in both directions. `uses`/`mediums` record how often and where; `lift` is observed co-occurrence over the family-frequency expectation (the usage de-skew); `kin` marks same-superfamily pairs. A documented alias map in mine-fontsinuse.mjs substitutes open revivals/metric-compatible faces for commercial names (Futura -> Jost, Arial -> Arimo, ...). The raw CSV is not committed (article is CC BY-NC-ND); re-run the generator to rebuild.',
  minPartners: 1,
  sources: [
    { name: 'Choi & Hyun, "Typeface network and the principle of font pairing", Sci Rep 14:30820 (2024), supplementary file MOESM1 (22,897 font-use cases from fontsinuse.com)', url: 'https://doi.org/10.1038/s41598-024-81601-w' },
    { name: 'Fonts In Use (fontsinuse.com), the underlying use-case archive', url: 'https://fontsinuse.com/' },
  ],
  pairings,
};

fs.writeFileSync(OUT, JSON.stringify(fixture, null, 2) + '\n');

const partnerTotal = pairings.reduce((n, e) => n + e.acceptable.length, 0);
console.log(`${rows} multi-font use cases -> ${pairs.size} distinct catalog-covered pairs`);
console.log(`${pairings.length} headings, ${partnerTotal} directed partners -> ${path.relative(REPO, OUT)}`);
const top = [...unresolved.entries()].sort((a, b) => b[1] - a[1]).slice(0, 25);
console.log(`top unresolved names: ${top.map(([n, c]) => `${n} (${c})`).join(', ')}`);
