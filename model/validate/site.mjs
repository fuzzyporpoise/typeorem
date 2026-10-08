// Where the science finds the site checkout (TSK-016).
//
// The harnesses in this directory score the metric the browser actually runs,
// so they import it from the site's `site/js/` rather than keeping a copy that
// could drift. The corpus they read is the shipped one, also in the site tree.
//
// Resolution mirrors model/paths.py: TYPEOREM_SITE_DIR when set, else the first
// of <this repo>/site, <this repo>/../site, <this repo>/../typeorem-site that
// looks like a checkout.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');

function isCheckout(dir) {
  return fs.existsSync(path.join(dir, 'index.html'));
}

function resolveSiteDir() {
  if (process.env.TYPEOREM_SITE_DIR) return path.resolve(process.env.TYPEOREM_SITE_DIR);
  const candidates = [
    path.join(REPO, 'site'),
    path.resolve(REPO, '..', 'site'),
    path.resolve(REPO, '..', 'typeorem-site'),
  ];
  return candidates.find(isCheckout) || candidates[0];
}

export const SITE_DIR = resolveSiteDir();

export const SITE_DATA = path.join(SITE_DIR, 'data');

export function requireSite(purpose) {
  if (!isCheckout(SITE_DIR)) {
    console.error(
      `no site checkout at ${SITE_DIR} (${purpose}).\n` +
      `Clone the site next to this repo, or point TYPEOREM_SITE_DIR at it:\n` +
      `    TYPEOREM_SITE_DIR=/path/to/typeorem-site node model/validate/benchmark.mjs`);
    process.exit(2);
  }
  return SITE_DIR;
}

/* Import a site module by file name ("engine.js"), from wherever the checkout
   is. Dynamic, because the path is only known at run time. */
export function siteModule(name) {
  return import(pathToFileURL(path.join(SITE_DIR, 'js', name)).href);
}
