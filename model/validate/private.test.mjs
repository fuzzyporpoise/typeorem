import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { requireSite, siteModule } from './site.mjs';
import * as P from './private.mjs';

// The private harness's pieces, unit-tested without a private corpus: the merge
// the app overlay reproduces, label resolution (schema, aliases, dropped,
// missing), the ranking math, the PCA-drift guard, and a parity check that the
// parametrized "shipped" candidate reproduces site/js/engine.js's roleScore.

let pass = 0, fail = 0;
const ok = (label, cond) => { if (cond) pass++; else { fail++; console.log('FAIL ' + label); } };

requireSite('the private harness test runs the shipped metric');
const E = { ...(await siteModule('engine.js')), contrastAt: (await siteModule('vectors.js')).contrastAt };

// --- a tiny hand-built corpus -------------------------------------------------
const inst = (over) => ({ id: 0, weight: 400, style: 'normal', eligible: true,
  category: 'sans-serif', xh: 0.7, legibility: 0.5, ...over });
const corpus = {
  dim: 3, count: 4,
  instances: [
    inst({ family: 'Alpha', category: 'serif' }),
    inst({ family: 'Beta', legibility: 0.9 }),
    inst({ family: 'Gamma', legibility: 0.1 }),
    inst({ family: 'Alpha Bold', weight: 700 }),
  ],
  units: Float64Array.from([1, 0, 0, 0.6, 0.8, 0, 0.6, -0.8, 0, 1, 0, 0]),
};
corpus.instances.forEach((c, i) => { c.id = i; });

// --- merge renumbers and concatenates (the app overlay contract) -------------
const pub = { dim: 3, count: 2, instances: [inst({ family: 'P0' }), inst({ family: 'P1' })],
  units: Float64Array.from([1, 0, 0, 0, 1, 0]) };
const priv = { dim: 3, count: 2, instances: [inst({ family: 'X0', source: 'private', css2: null, fontUrl: 'f/X0.ttf' }),
  inst({ family: 'X1', source: 'private', css2: null, fontUrl: 'f/X1.ttf' })],
  units: Float64Array.from([0, 0, 1, 0.1, 0.2, 0.3]) };
const merged = P.mergeCorpora(pub, priv);
ok('merge keeps the public rows first', merged.instances.slice(0, 2).every(c => c.family.startsWith('P')));
ok('merge renumbers the private rows', merged.instances[2].id === 2 && merged.instances[3].id === 3);
ok('merge carries the private provenance', merged.instances[2].source === 'private' && merged.instances[2].css2 === null);
ok('merge concatenates units', merged.units.length === merged.count * merged.dim
  && merged.units[2 * merged.dim] === 0 && merged.units[3 * merged.dim] === 0.1);
try { P.mergeCorpora(pub, { dim: 9, count: 0, instances: [], units: new Float64Array() }); ok('a dim mismatch is refused', false); }
catch { ok('a dim mismatch is refused', true); }

// --- family resolution --------------------------------------------------------
const elig = P.eligibleIds(corpus);
ok('exact family match', JSON.stringify(P.familiesMatching(corpus.instances, elig, 'Beta')) === '["Beta"]');
ok('a prefix with no exact match resolves several cuts',
  JSON.stringify(P.familiesMatching(corpus.instances, elig, 'Alph')) === '["Alpha","Alpha Bold"]');
ok('an exact match beats a prefix sibling',
  JSON.stringify(P.familiesMatching(corpus.instances, elig, 'Alpha')) === '["Alpha"]');
ok('an alias resolves a name', JSON.stringify(P.familiesMatching(corpus.instances, elig, 'Bee', { Bee: 'Beta' })) === '["Beta"]');
ok('an unknown name resolves to nothing', P.familiesMatching(corpus.instances, elig, 'Zeta').length === 0);
ok('nearestFamilies suggests a near miss', P.nearestFamilies(corpus.instances, elig, 'Bet').includes('Beta'));

// --- labels: multi-partner, legacy list, dropped, aliases, missing ------------
const labels = {
  aliases: { Bee: 'Beta' },
  dropped: [{ heading: 'Beta', partner: 'Gamma' }],
  pairings: [
    { heading: 'Alpha', role: 'body', acceptable: [{ family: 'Bee' }, 'Gamma'] },
    ['Beta', 'Gamma', 'Alpha'],
    ['Zeta', 'Beta'],
  ],
};
const { rows, missing } = P.resolveLabels(corpus, labels);
ok('the heading resolves to its canonical instance', rows[0].headingId === 0);
ok('the partner resolves through the alias', rows[0].partners.some(p => p.id === 1));
ok('a partner carries its role', rows[0].partners.every(p => p.role === 'body'));
ok('a dropped partner is honoured', !rows[1].partners.some(p => p.id === 2));
ok('a legacy list is accepted', rows.some(r => r.name === 'Beta'));
ok('an unresolvable heading is reported', missing.includes('Zeta'));
ok('a row with no resolvable partners is skipped', !rows.some(r => r.name === 'Zeta'));

// --- ranking math + pass/fail -------------------------------------------------
const scoreMap = new Map([[1, 5], [2, 1]]);  // Beta beats Gamma for heading Alpha
const p = (id, role = 'body') => ({ id, role, family: corpus.instances[id].family });
const rows2 = [{ name: 'Alpha', headingId: 0, partners: [p(2)], roles: ['body'], role: 'body' }];
const cands = [['fixed', { score: i => scoreMap.get(i) ?? 0 }]];
const res = P.evaluate(cands, corpus, rows2, 10);
const per = res.get('fixed');
ok('the better-scored partner ranks first', per[0].best === 2 && per[0].ranks[0].rank === 2);
ok('a partner inside k passes', per[0].pass === true);
const res2 = P.evaluate(cands, corpus, rows2, 1);
ok('a partner outside k fails', res2.get('fixed')[0].pass === false);
ok('nDCG is 1 when the relevant row is first', P.evaluate([['p', { score: i => (i === 1 ? 5 : 0) }]], corpus,
  [{ name: 'Alpha', headingId: 0, partners: [p(1)], roles: ['body'], role: 'body' }], 10)
  .get('p')[0].ndcgAt(10) === 1);
// A partner scored in its own slot: an accent partner ignores the body guard.
const twoRole = [{ name: 'Alpha', headingId: 0, partners: [p(1, 'body'), p(2, 'accent')], roles: ['accent', 'body'], role: 'accent/body' }];
const guarded = [['g', { score: i => scoreMap.get(i) ?? 0, keep: (inst, h, role) => role !== 'body' || inst.weight <= 400 }]];
ok('each partner is ranked in its own role',
   P.evaluate(guarded, corpus, twoRole, 10).get('g')[0].n === 2);
const s0 = P.summarise([], 10);
ok('an empty slice summarises without NaN', s0.n === 0 && Number.isFinite(s0.recall) && s0.median === null);

// --- the parametrized "shipped" candidate reproduces roleScore ----------------
const elig2 = P.eligibleIds(corpus);
const cmax = E.maxContrast(corpus, elig2);
const centroid = E.buildBodyCentroid(corpus);
const built = P.buildCandidates(corpus, cmax, { ...E, centroid }, {});
const shipped = built.find(([name]) => name === 'shipped (role+axis)')[1];
let parity = true;
for (const hi of elig2) for (const i of elig2) {
  const a = shipped.score(i, hi, 'body');
  const b = E.roleScore(corpus, cmax, hi, i, 0, 'body');
  if (!Object.is(a, b)) parity = false;
}
ok('the shipped candidate equals roleScore across the pool', parity);
const swept = P.buildCandidates(corpus, cmax, { ...E, centroid }, { sweeps: true });
ok('--sweep adds the constant grid', swept.length > built.length && swept.some(([n]) => n === 'role lambda 0.5'));

// --- the PCA-drift guard ------------------------------------------------------
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'typeorem-private-test-'));
const siteData = path.join(tmp, 'site', 'data');
fs.mkdirSync(siteData, { recursive: true });
fs.writeFileSync(path.join(siteData, 'pca.bin'), Buffer.from('canonical-bytes'));
const sha = createHash('sha256').update('canonical-bytes').digest('hex');
const versionPath = path.join(tmp, 'corpus.version.json');
fs.writeFileSync(versionPath, JSON.stringify({ pcaSha256: sha }));
ok('a matching PCA identity passes', P.checkPca(versionPath, siteData) === sha);
fs.writeFileSync(versionPath, JSON.stringify({ pcaSha256: '0'.repeat(64) }));
try { P.checkPca(versionPath, siteData); ok('a stale private corpus is refused', false); }
catch { ok('a stale private corpus is refused', true); }
fs.rmSync(tmp, { recursive: true, force: true });

console.log(`${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
