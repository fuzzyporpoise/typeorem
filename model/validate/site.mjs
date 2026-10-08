// Where the science finds the app checkout (TSK-016).
//
// The harnesses in this directory score the metric the browser actually runs,
// so they import it from the app's `site/js/` rather than keeping a copy that
// could drift. The corpus they read is the shipped one, also in the app tree.
//
// Resolution mirrors model/paths.py: TYPEOREM_SITE_DIR (the app checkout root)
// when set, else the first of <this repo>, <this repo>/.., and
// <this repo>/../typeorem-site that holds a site/index.html.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');

function isCheckout(dir) {
  return fs.existsSync(path.join(dir, 'site', 'index.html'));
}

function resolveAppRoot() {
  if (process.env.TYPEOREM_SITE_DIR) return path.resolve(process.env.TYPEOREM_SITE_DIR);
  const candidates = [REPO, path.resolve(REPO, '..'), path.resolve(REPO, '..', 'typeorem-site')];
  return candidates.find(isCheckout) || candidates[0];
}

const APP_ROOT = resolveAppRoot();

export const SITE_DIR = path.join(APP_ROOT, 'site');

export const SITE_DATA = path.join(SITE_DIR, 'data');

export function requireSite(purpose) {
  if (!isCheckout(APP_ROOT)) {
    console.error(
      `no app checkout at ${APP_ROOT} (${purpose}).\n` +
      `Clone the app next to this repo, or point TYPEOREM_SITE_DIR at it:\n` +
      `    TYPEOREM_SITE_DIR=/path/to/typeorem node model/validate/benchmark.mjs`);
    process.exit(2);
  }
  return SITE_DIR;
}

/* Import a site module by file name ("engine.js"), from wherever the checkout
   is. Dynamic, because the path is only known at run time. */
export function siteModule(name) {
  return import(pathToFileURL(path.join(SITE_DIR, 'js', name)).href);
}
