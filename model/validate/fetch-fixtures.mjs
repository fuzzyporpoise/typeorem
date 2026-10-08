import { createHash } from 'node:crypto';
import { mkdirSync, existsSync, writeFileSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(here, 'fixtures');
const BASE = 'https://raw.githubusercontent.com/Jack000/fontjoy/master';
const FILES = ['metadata.tsv', 'vectors-200.tsv'];

mkdirSync(FIXTURES, { recursive: true });

for (const name of FILES) {
  const dest = join(FIXTURES, name);
  if (existsSync(dest)) {
    console.log(`present ${name}`);
    continue;
  }
  const res = await fetch(`${BASE}/${name}`);
  if (!res.ok) throw new Error(`fetch ${name}: HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  writeFileSync(dest, buf);
  console.log(`fetched ${name} (${buf.length} bytes)`);
}

for (const line of readFileSync(join(FIXTURES, 'checksums.txt'), 'utf8').trim().split('\n')) {
  const [want, file] = line.trim().split(/\s+/);
  const got = createHash('sha256').update(readFileSync(join(FIXTURES, file))).digest('hex');
  if (got !== want) throw new Error(`checksum mismatch for ${file}`);
  console.log(`ok ${file}`);
}
