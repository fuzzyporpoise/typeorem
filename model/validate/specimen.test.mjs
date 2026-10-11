import path from 'node:path';
import * as S from './specimen.mjs';

// The specimen sheet's pieces, unit-tested without a private corpus: html
// escaping, the Google css2 url, the instance merge, a face's render spec, the
// two-checkout write guard, and the page it emits (an @font-face per unique
// local cut, a Google link per public one, the owner's label beside the resolved
// cut).

let pass = 0, fail = 0;
const ok = (label, cond) => { if (cond) pass++; else { fail++; console.log('FAIL ' + label); } };

const inst = (over) => ({ id: 0, family: 'F', weight: 400, style: 'normal', eligible: true,
  category: 'sans-serif', source: 'google', css2: 'F:wght@400', ...over });

// --- escaping and the css2 url -------------------------------------------------
ok('esc neutralizes the html metacharacters',
  S.esc('a & b < c > d "e"') === 'a &amp; b &lt; c &gt; d &quot;e&quot;');
ok('the css2 url matches the app font loader',
  S.css2Url('DM Mono:ital,wght@1,300') ===
  'https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@1,300&display=swap');

// --- the merge: public first, private rows renumbered --------------------------
const pub = { instances: [inst({ family: 'Pub0' }), inst({ family: 'Pub1' })] };
const priv = { instances: [inst({ family: 'Priv0', source: 'private', fontUrl: 'f/0.ttf' }),
  inst({ family: 'Priv1', source: 'private', fontUrl: 'f/1.ttf' })] };
const merged = S.mergeInstances(pub, priv);
ok('merge keeps the public rows first', merged.slice(0, 2).every(c => c.family.startsWith('Pub')));
ok('merge renumbers the private rows', merged[2].id === 2 && merged[3].id === 3);
ok('merge carries the private font url', merged[2].fontUrl === 'f/0.ttf');

// --- a face's render spec ------------------------------------------------------
const rootAbs = path.resolve('/tmp/typeorem-root');
const outDir = path.resolve('/tmp/typeorem-root');
const local = S.faceOf(inst({ family: 'Grenette Pro', source: 'private', fontUrl: 'fonts/g.ttf' }), rootAbs, outDir);
ok('a private face is a local @font-face url', local.kind === 'local' && local.url === 'fonts/g.ttf');
const nested = S.faceOf(inst({ source: 'private', fontUrl: 'fonts/g.ttf' }), rootAbs, path.join(rootAbs, 'sub'));
ok('a nested output path still resolves the url relatively', nested.url === '../fonts/g.ttf');
const google = S.faceOf(inst({ family: 'DM Mono', css2: 'DM Mono:wght@400' }), rootAbs, outDir);
ok('a public face keeps its css2 fragment', google.kind === 'google' && google.css2 === 'DM Mono:wght@400');

// --- the two-checkout write guard ----------------------------------------------
const guard = { rootAbs, siteDir: path.resolve('/tmp/app/site'), repoRoot: path.resolve('/tmp/science') };
ok('a path inside the private root is writable',
  S.assertWritable(path.join(rootAbs, 'specimen.html'), guard) === undefined);
try { S.assertWritable(path.resolve('/tmp/elsewhere/specimen.html'), guard); ok('a path outside the root is refused', false); }
catch { ok('a path outside the root is refused', true); }
try { S.assertWritable(path.join(guard.siteDir, 'specimen.html'), guard); ok('a path in the app checkout is refused', false); }
catch { ok('a path in the app checkout is refused', true); }
try { S.assertWritable(path.join(guard.repoRoot, 'specimen.html'), guard); ok('a path in the science repo is refused', false); }
catch { ok('a path in the science repo is refused', true); }

// --- the emitted page ----------------------------------------------------------
const instances = [
  inst({ id: 0, family: 'Grenette Pro', source: 'private', fontUrl: 'fonts/grenette.ttf' }),
  inst({ id: 1, family: 'Styrene A Trial', source: 'private', fontUrl: 'fonts/styrene.ttf' }),
  inst({ id: 2, family: 'DM Mono', css2: 'DM Mono:wght@400' }),
];
const rows = [{ name: 'Grenette', headingId: 0, headingFamily: 'Grenette Pro', role: 'body/accent',
  partners: [{ id: 1, family: 'Styrene A Trial', role: 'body', label: 'Styrene' },
    { id: 2, family: 'DM Mono', role: 'accent', label: 'DM Mono' }] }];
const html = S.renderSheet(rows, instances, { rootAbs, outDir });
ok('the sheet declares a local @font-face per private cut', (html.match(/@font-face/g) || []).length === 2);
ok('the sheet links the public partner through Google',
  html.includes('fonts.googleapis.com/css2?family=DM+Mono:wght@400&amp;display=swap'));
ok('the sheet carries the heading family name', html.includes("font-family:'Grenette Pro'"));
ok("the sheet shows the owner's label beside the resolved cut", html.includes('Styrene'));
ok('a body partner gets the reading sample', html.includes('font pairing is to select fonts'));
ok('an accent partner gets the accent sample', html.includes('a unified theory of type design'));
ok('the role is tagged', html.includes('>body</span>') && html.includes('>accent</span>'));

console.log(`${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
