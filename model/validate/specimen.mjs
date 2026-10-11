import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { SITE_DATA, SITE_DIR, requireSite } from './site.mjs';
import { defaultRoot, readJson, resolveLabels } from './private.mjs';

// A private specimen sheet for the owner's own pairings (TSK-002, plan B3).
//
// The harness (`private.mjs`) prints RANKS; this prints FACES. The glyph grid is
// a fingerprint, not a specimen, so a visual review needs the real faces set in
// real text: this emits a self-contained page into the private root
// (`~/.cache/typeorem/specimen.html` by default), with @font-face for every
// private cut it uses and the Google css2 links for any public partner. Open it
// from the private root, never from either checkout.
//
// Boundaries (plan section 3): it reads the labels + both catalogs and writes
// ONLY inside the private root. It refuses an output path inside the science
// repo or the app checkout, so a licensed face can never be rendered into a
// tree that could be committed or published.
//
//   node model/validate/specimen.mjs
//   node model/validate/specimen.mjs --private /some/where --out /some/where/sheet.html
//   python3 -m http.server -d ~/.cache/typeorem 8000   # then open /specimen.html

const HERE_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

const HEADING_SAMPLE = 'Handgloves & Hamburgefonstiv';
const BODY_SAMPLE = 'The goal of font pairing is to select fonts that share an overarching theme yet have a pleasing contrast.';
const ACCENT_SAMPLE = 'a unified theory of type design';

export function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function css2Url(css2) {
  return 'https://fonts.googleapis.com/css2?family=' + css2.replace(/ /g, '+') + '&display=swap';
}

// The merged corpus the app overlay reproduces, without vectors: the sheet needs
// instances only, so `resolveLabels` can run against the same rows the harness
// ranks (public first, then private, ids renumbered).
export function mergeInstances(publicCatalog, privateCatalog) {
  const base = publicCatalog.instances.length;
  return publicCatalog.instances.concat(privateCatalog.instances.map((c, i) => ({ ...c, id: base + i })));
}

// One face's render spec: a local @font-face url or a Google css2 fragment.
export function faceOf(inst, rootAbs, outDir) {
  const common = { family: inst.family, weight: inst.weight, style: inst.style === 'italic' ? 'italic' : 'normal', label: inst.label };
  if (inst.source === 'private') {
    const abs = path.join(rootAbs, inst.fontUrl);
    return { ...common, kind: 'local', url: path.relative(outDir, abs).split(path.sep).join('/') };
  }
  return { ...common, kind: 'google', css2: inst.css2 };
}

// Refuse an output path in either checkout. The private root already cannot be a
// git work tree (the writer enforces that), so this is the second belt: a sheet
// of licensed faces must never land where it could be committed or served.
export function assertWritable(outAbs, { rootAbs, siteDir = SITE_DIR, repoRoot = HERE_ROOT }) {
  const inRoot = outAbs === rootAbs || outAbs.startsWith(rootAbs + path.sep);
  if (!inRoot) throw new Error(`refusing to write outside the private root:\n  ${outAbs}\n  (root ${rootAbs})`);
  const inside = dir => outAbs === dir || outAbs.startsWith(dir + path.sep);
  if (inside(siteDir)) throw new Error(`refusing to write into the app checkout:\n  ${outAbs}`);
  if (inside(repoRoot)) throw new Error(`refusing to write into the science repo:\n  ${outAbs}`);
}

function fontFaceRule(face) {
  return `@font-face { font-family: "${esc(face.family)}"; font-weight: ${face.weight}; `
    + `font-style: ${face.style}; font-display: swap; src: url("${esc(face.url)}"); }`;
}

// The page. `rows` is `resolveLabels`'s output; `instances` is the merged list.
export function renderSheet(rows, instances, { rootAbs, outDir }) {
  const faces = new Map();   // key -> face, for @font-face dedupe
  const css2 = new Set();
  const spec = id => {
    const face = faceOf(instances[id], rootAbs, outDir);
    if (face.kind === 'local') faces.set(face.family + '|' + face.weight + '|' + face.style, face);
    else if (face.css2) css2.add(face.css2);
    return face;
  };

  const sections = rows.map(row => {
    const heading = spec(row.headingId);
    const partners = row.partners.map(p => ({ face: spec(p.id), role: p.role, owner: p.label }));
    return { row, heading, partners };
  });

  const fontStyle = [...faces.values()].map(fontFaceRule).join('\n');
  const links = [...css2].map(c => `<link rel="stylesheet" href="${esc(css2Url(c))}">`).join('\n');

  const blocks = sections.map(({ row, heading, partners }) => {
    const head = `<div class="spec spec--heading" style="font-family:'${esc(heading.family)}';`
      + `font-weight:${heading.weight};font-style:${heading.style}">${esc(HEADING_SAMPLE)}</div>`;
    const parts = partners.map(p => {
      const text = p.role === 'body' ? BODY_SAMPLE : ACCENT_SAMPLE;
      const cls = p.role === 'body' ? 'spec--body' : 'spec--accent';
      const size = p.role === 'body' ? '17px' : '14px';
      return `<div class="partner">
      <div class="meta">${esc(p.owner)} <span class="tag">${esc(p.role)}</span>`
        + ` <span class="cut">${esc(p.face.family)} ${p.face.weight}${p.face.style === 'italic' ? ' italic' : ''}</span></div>
      <div class="spec ${cls}" style="font-family:'${esc(p.face.family)}';font-weight:${p.face.weight};`
        + `font-style:${p.face.style};font-size:${size}">${esc(text)}</div>
    </div>`;
    }).join('\n');
    return `<section class="pairing">
  <h2>${esc(row.name)} <span class="cut">${esc(heading.family)} ${heading.weight}${heading.style === 'italic' ? ' italic' : ''}</span></h2>
  ${head}
  ${parts}
</section>`;
  }).join('\n');

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Typeorem private specimen sheet</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
${links}
<style>
${fontStyle}
:root { color-scheme: light dark; }
body { margin: 0 auto; max-width: 52rem; padding: 2rem 1.25rem 4rem;
  font: 15px/1.5 system-ui, sans-serif; background: #fafafa; color: #1a1a1a; }
h1 { font-size: 1.1rem; letter-spacing: .02em; text-transform: uppercase; opacity: .6; }
.note { opacity: .6; margin-top: -.5rem; }
.pairing { border-top: 1px solid #ddd; padding: 1.5rem 0; }
.pairing h2 { font-size: 1rem; margin: 0 0 1rem; display: flex; gap: .5rem; align-items: baseline; }
.partner { margin-top: 1rem; }
.meta { font-size: 12px; opacity: .65; margin-bottom: .25rem; }
.tag { border: 1px solid currentColor; border-radius: 3px; padding: 0 .3em; text-transform: uppercase; font-size: 10px; }
.cut { opacity: .55; font-weight: 400; }
.spec { white-space: pre-wrap; overflow-wrap: break-word; }
.spec--heading { font-size: 44px; line-height: 1.1; }
</style>
</head>
<body>
<h1>Private specimen sheet</h1>
<p class="note">Local only. These faces are licensed and never published.</p>
${blocks}
</body>
</html>
`;
}

function parseArgs(argv) {
  const out = { private: process.env.TYPEOREM_PRIVATE_DIR || null,
    labels: process.env.TYPEOREM_PRIVATE_LABELS || null, outPath: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--private') out.private = argv[++i];
    else if (a === '--labels') out.labels = argv[++i];
    else if (a === '--out') out.outPath = argv[++i];
    else throw new Error(`unknown argument ${a}`);
  }
  return out;
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (!args.private) {
    args.private = defaultRoot();
    console.log(`no private root given; using the default ${args.private}`);
  }
  const rootAbs = path.resolve(args.private.replace(/^~(?=$|\/)/, process.env.HOME || '~'));
  const corpusDir = path.join(rootAbs, 'corpus');
  if (!fs.existsSync(path.join(corpusDir, 'catalog.json'))) {
    throw new Error(`no private corpus at ${corpusDir}. Write one first:\n`
      + `    TYPEOREM_PRIVATE_DIR=${args.private} model/.venv/bin/python model/build_corpus.py --private`);
  }
  const labelsPath = args.labels
    || [path.join(rootAbs, 'pairings.json'), path.join(HERE_ROOT, 'model', 'validate', 'fixtures', 'user-pairings.json')]
      .find(p => fs.existsSync(p));
  if (!labelsPath || !fs.existsSync(labelsPath)) {
    throw new Error(`no labels file (pass --labels <path>, or keep the owner's pairings at <root>/pairings.json)`);
  }

  requireSite('the specimen sheet resolves catalog partners and their Google css2 links');
  const outAbs = path.resolve((args.outPath || path.join(rootAbs, 'specimen.html'))
    .replace(/^~(?=$|\/)/, process.env.HOME || '~'));
  assertWritable(outAbs, { rootAbs });

  const publicCatalog = readJson(path.join(SITE_DATA, 'catalog.json'));
  const privateCatalog = readJson(path.join(corpusDir, 'catalog.json'));
  const instances = mergeInstances(publicCatalog, privateCatalog);
  const labels = readJson(labelsPath);
  const { rows, missing } = resolveLabels({ dim: 0, count: instances.length, instances }, labels,
    m => console.error(`  ! ${m}`), { sameFamily: true });

  console.log(`private corpus ${corpusDir}`);
  console.log(`  shipped ${publicCatalog.instances.length} rows + private ${privateCatalog.instances.length} rows`);
  console.log(`labels ${labelsPath}: ${rows.length} headings`);
  if (missing.length) console.log(`  unresolved: ${[...new Set(missing)].join(', ')}`);
  if (!rows.length) {
    console.log('\nno labels resolved; nothing to render.');
    return null;
  }

  const html = renderSheet(rows, instances, { rootAbs, outDir: path.dirname(outAbs) });
  fs.writeFileSync(outAbs, html);
  console.log(`\nwrote ${outAbs} (${rows.length} pairings)`);
  console.log(`view it (a file:// open blocks the private font fetches):`);
  console.log(`    python3 -m http.server -d ${rootAbs} 8000   # then open http://127.0.0.1:8000/${path.basename(outAbs)}`);
  return outAbs;
}

const isMain = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  main().catch(e => { console.error(`\n${e.message}`); process.exit(1); });
}
