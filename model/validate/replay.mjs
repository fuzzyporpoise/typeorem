import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { requireSite, siteModule } from './site.mjs';

requireSite('the fontjoy replay ranks with the shipped vector math');
const { normalize, FORMS, SHIPPED_FORM, rankAgainst } = await siteModule('vectors.js');

const here = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(here, 'fixtures');

function rows(path) {
  return readFileSync(path, 'utf8').replace(/\n+$/, '').split('\n');
}

function loadCatalog() {
  const lines = rows(join(FIXTURES, 'metadata.tsv'));
  lines.shift();
  return lines.map(line => {
    const [name, variant, category] = line.split('\t');
    return { name, variant, category, family: name.slice(0, name.length - variant.length - 1) };
  });
}

function loadVectors() {
  return rows(join(FIXTURES, 'vectors-200.tsv')).map(line => Float64Array.from(line.split('\t'), Number));
}

function flag(name, fallback) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}

const catalog = loadCatalog();
const vectors = loadVectors();
const ref = JSON.parse(readFileSync(join(FIXTURES, 'reference.json'), 'utf8'));
const indexOfName = new Map(catalog.map((f, i) => [f.name, i]));
const forms = Object.keys(FORMS);
const topN = Number(flag('--top', 10));

function idFor(name) {
  if (!indexOfName.has(name)) throw new Error(`font not in fixture: ${name}`);
  return indexOfName.get(name);
}

function familyExclusion(id) {
  const family = catalog[id].family;
  return new Set(catalog.map((f, i) => (f.family === family ? i : -1)).filter(i => i >= 0));
}

function ranked(id, form) {
  return rankAgainst(vectors[id], vectors, FORMS[form], familyExclusion(id));
}

let pairMatrix = null;
function pairScore(i, j) {
  if (!pairMatrix) {
    const unit = vectors.map(normalize);
    const n = unit.length;
    pairMatrix = new Float32Array(n * n);
    for (let a = 0; a < n; a++) {
      const ua = unit[a];
      for (let b = a + 1; b < n; b++) {
        const ub = unit[b];
        let P = 0, N = 0;
        for (let k = 0; k < ua.length; k++) { const p = ua[k] * ub[k]; if (p > 0) P += p; else N -= p; }
        pairMatrix[a * n + b] = P * N;
        pairMatrix[b * n + a] = P * N;
      }
    }
  }
  return pairMatrix[i * vectors.length + j];
}

function rankOfScore(score, target, pool) {
  const list = pool.map(i => ({ i, s: score(i) }));
  list.sort((a, b) => b.s - a.s);
  return list.findIndex(x => x.i === target) + 1;
}

console.log(`fontjoy metric replay: ${vectors.length} vectors x ${vectors[0].length} dims`);
console.log(`source: ${ref.source}\n`);
console.log('resolved formula');
console.log(`  ${ref.formula}\n`);
console.log(`shipped form: ${SHIPPED_FORM}\n`);

const heading = idFor(ref.homepageDefault.faces[0].name);
const body = idFor(ref.homepageDefault.faces[1].name);
const accent = idFor(ref.homepageDefault.faces[2].name);
console.log('control: homepage default (a curated example; low ranks mean the model is incomplete, see TSK-009)');
console.log(`  ${ref.homepageDefault.note}`);
for (const form of forms) {
  const pos = new Map(ranked(heading, form).map((r, i) => [r.index, i + 1]));
  const cells = [body, accent].map(i => `${catalog[i].name}=${pos.has(i) ? pos.get(i) : 'n/a'}`);
  console.log(`  ${form.padEnd(9)} pairwise vs heading: ${cells.join('  ')}`);
}
const bodyPool = [...Array(catalog.length).keys()].filter(i => i !== heading && catalog[i].family !== catalog[heading].family);
const bodyRank = rankOfScore(i => pairScore(heading, i), body, bodyPool);
const accentPool = bodyPool.filter(i => i !== body);
const accentRank = rankOfScore(i => pairScore(heading, i) + pairScore(body, i), accent, accentPool);
const defScore = pairScore(heading, body) + pairScore(heading, accent) + pairScore(body, accent);
let better = 0, total = 0;
for (const bi of bodyPool) for (const ai of bodyPool) {
  if (ai === bi) continue;
  total++;
  if (pairScore(heading, bi) + pairScore(heading, ai) + pairScore(bi, ai) > defScore) better++;
}
console.log(`  body ${catalog[body].name}: rank ${bodyRank}/${bodyPool.length}`);
console.log(`  accent ${catalog[accent].name} given (heading, body): rank ${accentRank}/${accentPool.length}`);
console.log(`  whole triple: rank ${better + 1}/${total} under summed pairwise contrast (a reconstruction, not fontjoy's exact generator)`);
console.log();

console.log(`structural check (top ${topN}, family excluded): contrast = share of partners in a different category than the seed`);
console.log(`  ${'seed'.padEnd(26)}${'category'.padEnd(12)}${forms.map(f => f.padStart(10)).join('')}   crowd`);
const meanContrast = new Map(forms.map(f => [f, 0]));
for (const seed of ref.structuralSeeds) {
  const sid = idFor(seed);
  const cells = forms.map(form => {
    const r = ranked(sid, form).slice(0, topN);
    const contrast = r.filter(x => catalog[x.index].category !== catalog[sid].category).length / r.length;
    meanContrast.set(form, meanContrast.get(form) + contrast / ref.structuralSeeds.length);
    return contrast.toFixed(2).padStart(10);
  });
  const shipped = ranked(sid, SHIPPED_FORM).slice(0, topN);
  const counts = new Map();
  for (const x of shipped) counts.set(catalog[x.index].family, (counts.get(catalog[x.index].family) || 0) + 1);
  const crowd = Math.max(...counts.values()) / shipped.length;
  console.log(`  ${seed.padEnd(26)}${catalog[sid].category.padEnd(12)}${cells.join('')}   ${crowd.toFixed(2)}`);
}
console.log();
console.log(`  mean contrast: ${forms.map(f => `${f} ${meanContrast.get(f).toFixed(2)}`).join('   ')}`);
console.log('  note: min and balanced are rank-identical by construction (balanced = 2*min)\n');

const form = flag('--form', SHIPPED_FORM);
const inspect = flag('--seed') ? [flag('--seed')] : ref.structuralSeeds;
console.log(`top ${topN} under ${form}:`);
for (const name of inspect) {
  const sid = idFor(name);
  const r = ranked(sid, form).slice(0, topN);
  console.log(`\n  ${name} [${catalog[sid].category}]`);
  r.forEach((x, i) => console.log(`    ${String(i + 1).padStart(2)}. ${catalog[x.index].name} [${catalog[x.index].category}]  ${x.score.toFixed(4)}`));
}
