// Mine heading/partner font pairings from public sources and emit the
// multi-partner benchmark fixture (fixtures/fonts-pairings.json).
//
// Why: the old benchmark (fixtures/known-pairings.json) is 40 single-partner
// pairs, so a metric is judged on finding THE one arbitrary partner. This
// script builds the de-ambiguated yardstick: per heading, every partner the
// public sources endorse, so hit@k can ask "did we surface a good partner".
//
// Sources (all public; attribution also recorded in the fixture):
//   fontpair      Fontpair (fontpair.co), via the public Supabase REST API
//                 backing the site. Curated + community pairings with
//                 explicit heading/body roles. The anon key is read from
//                 their public JS bundle at runtime, never committed.
//   seed          fixtures/known-pairings.json: the 40 pairs the old
//                 benchmark used (Typewolf's five-fresh article, public
//                 design consensus, and the user's Google-Font picks).
//   elementor     Elementor, "The 30 Best Font Combinations for Web Design"
//                 (elementor.com/blog/font-pairing/), display font first.
//   google-type   Google Web Fonts Typographic Project (femmebot),
//                 femmebot.github.io/google-type: per-episode stylesheet,
//                 display font first.
//   pagecloud     Pagecloud, "Best Google Fonts Pairings"
//                 (pagecloud.com/blog/best-google-fonts-pairings),
//                 Title/Paragraph roles.
//   visme         Visme, "Font Combinations" (visme.co/blog/font-combinations/),
//                 headline font first; only pairs where both are catalog
//                 Google Fonts survive (many entries use system fonts).
//   typespiration Typespiration (Rafal Tomal), typespiration.com: per-design
//                 type-info-fonts links, display font first.
//   hundreddays   100 Days of Fonts (Do Hee Kim), 100daysoffonts.com: roles
//                 from the day-scoped CSS (.firstfont display, .secondfont
//                 text).
//   designyourway Design Your Way's pairing table
//                 (designyourway.net/blog/google-font-pairings/); direction
//                 from the table's roles column.
//   cssauthor     CSS Author, "Google Fonts Combinations"
//                 (cssauthor.com/google-fonts-combinations/), display font
//                 first.
//   256tools      256 Tools' font pairing tool (256-tools.com): curated pairs
//                 with explicit heading/body roles, read from the page's JS
//                 chunk (registry + pairs array).
//   tiny-online   tiny-online.tools' font pair generator: curated SSR cards,
//                 display font first.
//
// Every name is normalized (entity decode, Google renames such as
// Source Sans Pro -> Source Sans 3, trailing weight qualifiers stripped)
// and must resolve to an eligible family in site/data/catalog.json;
// anything else is dropped and reported. Headings with fewer than
// MIN_PARTNERS partners are excluded from the fixture.
//
// Re-run: node model/validate/mine-pairings.mjs
// The fixture is committed; re-running refreshes it from live sources, which
// may drift (articles edit, the fontpair table grows).

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SITE_DATA, requireSite } from './site.mjs';

requireSite('mined partners are validated against the shipped catalog');
const DIR = path.dirname(fileURLToPath(import.meta.url));
const CATALOG = path.join(SITE_DATA, 'catalog.json');
const SEED = path.join(DIR, 'fixtures', 'known-pairings.json');
const OUT = path.join(DIR, 'fixtures', 'fonts-pairings.json');
const MIN_PARTNERS = 2;
const MIN_HEADINGS = 100;

const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36';

async function getText(url) {
  const r = await fetch(url, { headers: { 'user-agent': UA } });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.text();
}

const catalog = JSON.parse(fs.readFileSync(CATALOG, 'utf8'));
const elig = new Set(catalog.instances.filter(i => i.eligible).map(i => i.family));

// Google renames (old name -> current catalog family).
const ALIAS = new Map([
  ['Source Sans Pro', 'Source Sans 3'],
  ['Source Serif Pro', 'Source Serif 4'],
]);
// Weight/style qualifiers sources append to family names ("Alegreya Sans
// Black", "EB Garamond Medium"). Width words (Condensed, Narrow, Expanded)
// and SC are never stripped: those are distinct families on Google Fonts.
const QUALIFIER = /^(?:Ultra[-\s]+|Extra[-\s]+|Semi[-\s]+)?(?:Thin|Light|Regular|Medium|Bold|Black|Heavy|Book|Demi|Italic|Roman)$/i;

function makeResolver() {
  const lower = new Map();
  for (const f of elig) if (!lower.has(f.toLowerCase())) lower.set(f.toLowerCase(), f);
  const hit = s => elig.has(s) ? s : lower.get(s.toLowerCase()) ?? null;
  return raw => {
    const s = raw.replace(/&amp;/g, '&').replace(/&nbsp;/g, ' ')
      .replace(/\s+/g, ' ').replace(/[.,;:!?]+$/, '').trim();
    if (!s) return null;
    if (ALIAS.has(s)) return ALIAS.get(s);
    const exact = hit(s);
    if (exact) return exact;
    let t = s;
    while (true) {
      const parts = t.split(' ');
      if (parts.length > 1 && QUALIFIER.test(parts[parts.length - 1])) {
        t = parts.slice(0, -1).join(' ');
        if (ALIAS.has(t)) return ALIAS.get(t);
        const h = hit(t);
        if (h) return h;
        continue;
      }
      break;
    }
    return null;
  };
}
const toFamily = makeResolver();

// ---- source extractors: each yields raw {heading, partner} name pairs ----

async function fromFontpair() {
  const home = await getText('https://www.fontpair.co/');
  const jsPath = home.match(/\/assets\/index-[^"]+\.js/)[0];
  const bundle = await getText('https://www.fontpair.co' + jsPath);
  const key = bundle.match(/eyJ[A-Za-z0-9_\-\.]{50,}/)[0];
  const api = 'https://kbbmlrtqzhhfopfsqgwd.supabase.co/rest/v1';
  const hdrs = { apikey: key, authorization: 'Bearer ' + key };
  async function all(table, select) {
    const out = [];
    for (let off = 0; ; off += 1000) {
      const r = await fetch(`${api}/${table}?select=${select}&offset=${off}&limit=1000`, { headers: hdrs });
      if (!r.ok) throw new Error(`${table}: HTTP ${r.status}`);
      const rows = await r.json();
      out.push(...rows);
      if (rows.length < 1000) return out;
    }
  }
  const fonts = await all('fonts', 'id,name,font_type');
  const byId = new Map(fonts.map(f => [f.id, f]));
  const out = [];
  for (const p of await all('font_pairings', 'heading_font_id,body_font_id,is_community_submitted')) {
    const hf = byId.get(p.heading_font_id), bf = byId.get(p.body_font_id);
    if (!hf || !bf || hf.font_type !== 'google' || bf.font_type !== 'google') continue;
    out.push({ heading: hf.name, partner: bf.name, tag: p.is_community_submitted ? 'fontpair-community' : 'fontpair' });
  }
  return out;
}

function fromSeed() {
  const seed = JSON.parse(fs.readFileSync(SEED, 'utf8'));
  return seed.pairs.map(([h, p]) => ({ heading: h, partner: p, tag: 'seed' }));
}

async function fromElementor() {
  const t = await getText('https://elementor.com/blog/font-pairing/');
  const out = [];
  for (const m of t.matchAll(/<h[23][^>]*>([\s\S]{0,160}?)<\/h[23]>/g)) {
    const mm = m[1].replace(/<[^>]+>/g, '').replace(/&amp;/g, '&').trim().match(/^\d+\.\s*(.+?)\s*&\s*(.+)$/);
    if (mm) out.push({ heading: mm[1], partner: mm[2], tag: 'elementor' });
  }
  return out;
}

async function fromGoogleType() {
  const t = await getText('https://femmebot.github.io/google-type/');
  const out = [];
  for (const m of t.matchAll(/href="(https:\/\/fonts\.googleapis\.com\/css2\?[^"]+)"/g)) {
    const fams = [...m[1].matchAll(/family=([^&"]+)/g)]
      .map(x => decodeURIComponent(x[1]).replace(/\+/g, ' ').replace(/:.*$/, ''));
    if (fams.length === 2) out.push({ heading: fams[0], partner: fams[1], tag: 'google-type' });
  }
  return out;
}

async function fromPagecloud() {
  const t = await getText('https://www.pagecloud.com/blog/best-google-fonts-pairings');
  const out = [];
  const re = /Title font:<\/b>[^<]*<a href="https:\/\/fonts\.google\.com\/specimen\/([^"]+)"[\s\S]{0,300}?Paragraph font:<\/b>[^<]*<a href="https:\/\/fonts\.google\.com\/specimen\/([^"]+)"/g;
  for (const m of t.matchAll(re)) {
    out.push({
      heading: decodeURIComponent(m[1]).replace(/\+/g, ' '),
      partner: decodeURIComponent(m[2]).replace(/\+/g, ' '),
      tag: 'pagecloud',
    });
  }
  return out;
}

async function fromVisme() {
  const t = await getText('https://visme.co/blog/font-combinations/');
  const out = [];
  for (const m of t.matchAll(/<h[23][^>]*>([\s\S]{0,160}?)<\/h[23]>/g)) {
    const txt = m[1].replace(/<[^>]+>/g, '').replace(/&amp;/g, '&').trim();
    const mm = txt.match(/^\d+\s+[^:]+:\s*(.+?)\s*(?:&|\+)\s*(.+)$/);
    if (mm) out.push({ heading: mm[1], partner: mm[2], tag: 'visme' });
  }
  return out;
}

async function fromTypespiration() {
  const t = await getText('https://typespiration.com/');
  const out = [];
  for (const m of t.matchAll(/<div class="type-info-fonts">([\s\S]{0,400}?)<\/div>/g)) {
    const fams = [...m[1].matchAll(/specimen\/([^"<>]+)"[^>]*>([^<]+)<\/a>/g)]
      .map(x => x[2]);
    if (fams.length >= 2) out.push({ heading: fams[0], partner: fams[1], tag: 'typespiration' });
  }
  return out;
}

async function fromDesignYourWay() {
  const t = await getText('https://www.designyourway.net/blog/google-font-pairings/');
  const out = [];
  // table rows: <td><strong>X and Y</strong></td><td>categories</td><td>roles</td>;
  // the roles cell carries the direction ("Body Text + Headings" => X is body).
  for (const m of t.matchAll(/<td><strong>([^<]+?) and ([^<]+?)<\/strong><\/td>\s*<td>([^<]*)<\/td>\s*<td>([^<]*)<\/td>/g)) {
    const roles = m[4];
    if (!/Headings/.test(roles) || !/Body Text/.test(roles)) continue;
    const firstIsHeading = roles.indexOf('Headings') < roles.indexOf('Body Text');
    const [h, b] = firstIsHeading ? [m[1], m[2]] : [m[2], m[1]];
    out.push({ heading: h, partner: b, tag: 'designyourway' });
  }
  return out;
}

async function fromCssauthor() {
  const t = await getText('https://cssauthor.com/google-fonts-combinations/');
  const out = [];
  // combos appear as "X + Y" in table cells and section headings; first is
  // the display face in the article's presentation.
  for (const m of t.matchAll(/<(?:td|h2|h3|h4|strong|li)[^>]*>([\s\S]{0,90}?)<\/(?:td|h2|h3|h4|strong|li)>/g)) {
    const txt = m[1].replace(/<[^>]+>/g, '').replace(/&amp;/g, '&').trim();
    const mm = txt.match(/^(?:\d+\.?\s*)?([A-Z][A-Za-z0-9 .'&-]{1,58}) \+ ([A-Z][A-Za-z0-9 .'&-]{1,58})$/);
    if (mm) out.push({ heading: mm[1], partner: mm[2], tag: 'cssauthor' });
  }
  return out;
}

async function fromThebrief() {
  // The Brief's "Google Font Pairings" article: numbered h3 combos,
  // display font first, weight qualifiers in the names ("Montserrat Black").
  const t = await getText('https://www.thebrief.ai/blog/google-font-pairings/');
  const out = [];
  for (const m of t.matchAll(/<h3[^>]*>([\s\S]{0,140}?)<\/h3>/g)) {
    const txt = m[1].replace(/<[^>]+>/g, '').replace(/&amp;/g, '&').trim();
    const mm = txt.match(/^\d+\.\s*(.+?)\s*&\s*(.+)$/);
    if (mm) out.push({ heading: mm[1], partner: mm[2], tag: 'thebrief' });
  }
  return out;
}

async function fromCodekithub() {
  // CodeKitHub's Google Fonts pairing finder: SSR pairing cards; the first
  // font-family is the heading preview (bold xl), the second the body.
  const t = await getText('https://codekithub.com/en/google-fonts-pairing-finder/');
  const out = [];
  for (const m of t.matchAll(/<div class="pairing-card[\s\S]{0,1400}?<\/div><\/div>/g)) {
    const fams = [...m[0].matchAll(/font-family:\s*'([^']+)'/g)].map(x => x[1]);
    if (fams.length >= 2) out.push({ heading: fams[0], partner: fams[1], tag: 'codekithub' });
  }
  return out;
}

async function fromFontfyi() {
  // FontFYI's open API: curated pairings with explicit heading/body roles.
  const r = await fetch('https://fontfyi.com/api/v1/pairings/?page_size=500', { headers: { 'user-agent': UA } });
  if (!r.ok) throw new Error(`fontfyi: HTTP ${r.status}`);
  const j = await r.json();
  return (j.results ?? []).map(p => ({ heading: p.heading_font_family, partner: p.body_font_family, tag: 'fontfyi' }));
}

async function fromLeadpages() {
  // Leadpages' best Google fonts article: combos as "X + Y" in list/heading
  // markup, display font first.
  const t = await getText('https://leadpages.com/blog/best-google-fonts');
  const out = [];
  for (const m of t.matchAll(/<(?:h2|h3|h4|td|strong|li)[^>]*>([\s\S]{0,90}?)<\/(?:h2|h3|h4|td|strong|li)>/g)) {
    const txt = m[1].replace(/<[^>]+>/g, '').replace(/&amp;/g, '&').trim();
    const mm = txt.match(/^\d*\.?\s*([A-Z][A-Za-z0-9 .'&-]{1,50}) \+ ([A-Z][A-Za-z0-9 .'&-]{1,50})$/);
    if (mm) out.push({ heading: mm[1], partner: mm[2], tag: 'leadpages' });
  }
  return out;
}

async function fromShapecat() {
  // Shapecat's font pairing tool: the tool page's JS chunk embeds the curated
  // list as objects with explicit heading/body roles.
  const page = await getText('https://shapecat.com/tools/font-pairing');
  const chunks = [...page.matchAll(/src="(\/_next\/static\/chunks\/[^"]+\.js)"/g)].map(m => m[1]);
  for (const c of chunks) {
    const t = await getText('https://shapecat.com' + c);
    const pairs = [...t.matchAll(/\{heading:\\?"([^"\\]+)\\?",body:\\?"([^"\\]+)\\?"/g)];
    if (pairs.length) return pairs.map(m => ({ heading: m[1], partner: m[2], tag: 'shapecat' }));
  }
  return [];
}

async function from256tools() {
  // 256 Tools font-pairing tool: the page chunk embeds a font registry and a
  // curated pairs array P with explicit headingFontId/bodyFontId roles.
  const page = await getText('https://256-tools.com/en/tools/font-pairing/');
  const chunk = page.match(/\/(_next\/static\/chunks\/app\/[^"]*font-pairing\/page-[^"]+\.js)/)[0];
  const t = await getText('https://256-tools.com/' + chunk.replace(/^\//, ''));
  const reg = new Map();
  for (const m of t.matchAll(/["']?([\w-]+)["']?:\{id:"[\w-]+",family:"([^"]+)"/g)) reg.set(m[1], m[2]);
  const out = [];
  for (const m of t.matchAll(/\{id:"[^"]+",headingFontId:"([^"]+)",bodyFontId:"([^"]+)"/g)) {
    const h = reg.get(m[1]), b = reg.get(m[2]);
    if (h && b) out.push({ heading: h, partner: b, tag: '256tools' });
  }
  return out;
}

async function fromTinyOnline() {
  // tiny-online.tools font pair generator: SSR pairing cards, display font
  // first in the "X + Y" label (heading weight 700, body 400 in the preview).
  const t = await getText('https://tiny-online.tools/design-tools/font-pair-generator');
  const out = [];
  for (const m of t.matchAll(/>([^<>]{2,50})<!-- --> \+ <!-- -->([^<>]{2,50})</g)) {
    out.push({ heading: m[1].trim(), partner: m[2].trim(), tag: 'tiny-online' });
  }
  return out;
}

async function fromHundredDays() {
  const css = await getText('http://100daysoffonts.com/stylesheets/styles.css');
  const first = new Map(), second = new Map();
  for (const m of css.matchAll(/#([a-z]+)\s+\.firstfont\s*\{[^}]*?font-family:\s*([^;}]+)/g)) {
    if (!first.has(m[1])) first.set(m[1], m[2].split(',')[0].replace(/['"]/g, '').trim());
  }
  for (const m of css.matchAll(/#([a-z]+)\s+\.secondfont\s*\{[^}]*?font-family:\s*([^;}]+)/g)) {
    if (!second.has(m[1])) second.set(m[1], m[2].split(',')[0].replace(/['"]/g, '').trim());
  }
  const out = [];
  for (const [day, h] of first) {
    const b = second.get(day);
    if (b) out.push({ heading: h, partner: b, tag: 'hundreddays' });
  }
  return out;
}

const SOURCES = [
  { run: fromFontpair, name: 'Fontpair (fontpair.co), curated + community pairings, via the site\'s public API', url: 'https://www.fontpair.co/' },
  { run: fromSeed, name: 'Seed pairs from fixtures/known-pairings.json (Typewolf five-fresh article, public design consensus)', url: null },
  { run: fromElementor, name: 'Elementor, "The 30 Best Font Combinations for Web Design"', url: 'https://elementor.com/blog/font-pairing/' },
  { run: fromGoogleType, name: 'Google Web Fonts Typographic Project (femmebot)', url: 'https://femmebot.github.io/google-type/' },
  { run: fromPagecloud, name: 'Pagecloud, "Best Google Fonts Pairings"', url: 'https://www.pagecloud.com/blog/best-google-fonts-pairings' },
  { run: fromVisme, name: 'Visme, "Font Combinations"', url: 'https://visme.co/blog/font-combinations/' },
  { run: fromTypespiration, name: 'Typespiration (Rafal Tomal)', url: 'https://typespiration.com/' },
  { run: fromHundredDays, name: '100 Days of Fonts (Do Hee Kim)', url: 'http://100daysoffonts.com/' },
  { run: fromDesignYourWay, name: 'Design Your Way, "Google Font Pairings" (roles from the article\'s roles column)', url: 'https://www.designyourway.net/blog/google-font-pairings/' },
  { run: fromCssauthor, name: 'CSS Author, "Google Fonts Combinations"', url: 'https://cssauthor.com/google-fonts-combinations/' },
  { run: from256tools, name: '256 Tools font pairing tool (curated pairs with heading/body roles)', url: 'https://256-tools.com/en/tools/font-pairing/' },
  { run: fromTinyOnline, name: 'tiny-online.tools font pair generator (curated pairs, display font first)', url: 'https://tiny-online.tools/design-tools/font-pair-generator' },
  { run: fromShapecat, name: 'Shapecat font pairing tool (curated pairs with heading/body roles)', url: 'https://shapecat.com/tools/font-pairing' },
  { run: fromLeadpages, name: 'Leadpages, "Best Google Fonts" (combos, display font first)', url: 'https://leadpages.com/blog/best-google-fonts' },
  { run: fromFontfyi, name: 'FontFYI curated pairings (open JSON API, heading/body roles)', url: 'https://fontfyi.com/tools/font-pairing/' },
  { run: fromCodekithub, name: 'CodeKitHub Google Fonts pairing finder (curated cards, heading/body roles from preview styles)', url: 'https://codekithub.com/en/google-fonts-pairing-finder/' },
  { run: fromThebrief, name: 'The Brief, "Google Font Pairings" (numbered combos, display font first)', url: 'https://www.thebrief.ai/blog/google-font-pairings/' },
];

const grouped = new Map(); // heading -> Map(partner -> Set(tags))
const drops = new Map();
for (const src of SOURCES) {
  let pairs;
  try {
    pairs = await src.run();
  } catch (e) {
    console.log(`WARN ${src.name}: ${e.message} (skipped)`);
    continue;
  }
  let added = 0;
  for (const { heading, partner, tag } of pairs) {
    const H = toFamily(heading), B = toFamily(partner);
    if (!H || !B || H === B) {
      if (H !== B) drops.set(`${src.name}: ${heading} & ${partner}`, true);
      continue;
    }
    if (!grouped.has(H)) grouped.set(H, new Map());
    const m = grouped.get(H);
    if (!m.has(B)) { m.set(B, new Set()); added++; }
    m.get(B).add(tag);
  }
  console.log(`${src.name}: ${pairs.length} pairs read, ${added} new eligible partners`);
}

const kept = [...grouped.entries()]
  .map(([heading, partners]) => ({
    heading,
    acceptable: [...partners.entries()]
      .map(([family, tags]) => ({ family, sources: [...tags].sort() }))
      .sort((a, b) => a.family.localeCompare(b.family)),
  }))
  .filter(e => e.acceptable.length >= MIN_PARTNERS)
  .sort((a, b) => a.heading.localeCompare(b.heading));

const singles = grouped.size - kept.length;
const totalPairs = kept.reduce((n, e) => n + e.acceptable.length, 0);
console.log(`\nheadings read: ${grouped.size}; kept ${kept.length} with >=${MIN_PARTNERS} partners (${singles} single-partner headings excluded)`);
console.log(`total heading/partner edges in fixture: ${totalPairs}`);
if (drops.size) {
  console.log(`dropped (not catalog-eligible): ${drops.size}`);
  for (const d of [...drops.keys()].sort()) console.log(`  - ${d}`);
}
if (kept.length < MIN_HEADINGS) {
  console.log(`FAIL: need at least ${MIN_HEADINGS} headings with >=${MIN_PARTNERS} partners, got ${kept.length}`);
  process.exit(1);
}

const fixture = {
  version: 1,
  generated: new Date().toISOString().slice(0, 10),
  note: 'Multi-partner pairing benchmark. For each heading, "acceptable" lists every body partner the public sources endorse, so hit@k answers "did the metric surface a good partner", not "did it surface the one partner". Scored under the strict rule: the partner family counts at its canonical 400-normal instance. See mine-pairings.mjs (sources, reproducibility).',
  minPartners: MIN_PARTNERS,
  sources: SOURCES.map(s => ({ name: s.name, url: s.url })),
  pairings: kept,
};
fs.writeFileSync(OUT, JSON.stringify(fixture, null, 1) + '\n');
console.log(`wrote ${path.relative(ROOT, OUT)}: ${kept.length} headings`);
