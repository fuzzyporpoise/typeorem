#!/usr/bin/env node
// Open a pull request on this repo and merge it.
//
// `main` is protected by a repository ruleset (no direct pushes, linear history),
// and GitHub's `gh` CLI is not installed in this environment, so this is the
// shortest path: it uses GITHUB_TOKEN and the REST API.
//
//     git push origin HEAD:refs/heads/some-branch
//     node scripts/pr.mjs some-branch "Title" "Body text or a path to a file"
//     node scripts/pr.mjs some-branch "Title" body.md --no-merge
//
// The branch is deleted after a successful merge (GitHub does it for a rebase
// merge too, so a 404 here is fine, not an error).

import fs from 'node:fs';

const args = process.argv.slice(2);
const noMerge = args.includes('--no-merge');
const [branch, title, bodyArg] = args.filter(a => a !== '--no-merge');

if (!branch || !title) {
  console.error('usage: node scripts/pr.mjs <branch> "<title>" ["<body>"|body-file] [--no-merge]');
  process.exit(2);
}

const token = process.env.GITHUB_TOKEN;
if (!token) {
  console.error('GITHUB_TOKEN is required (repo scope)');
  process.exit(2);
}

const body = bodyArg
  ? (fs.existsSync(bodyArg) ? fs.readFileSync(bodyArg, 'utf8') : bodyArg)
  : '';

const headers = {
  Authorization: `Bearer ${token}`,
  Accept: 'application/vnd.github+json',
  'Content-Type': 'application/json',
  'User-Agent': 'typeorem-pr',
};

const owner = 'fuzzyporpoise';
const repo = `${owner}/typeorem`;
const api = (path, method = 'GET', payload) =>
  fetch(`https://api.github.com${path}`, {
    method,
    headers,
    body: payload ? JSON.stringify(payload) : undefined,
  });

const opened = await api(`/repos/${repo}/pulls`, 'POST', { title, head: branch, base: 'main', body });
const pr = await opened.json();
if (!pr.number) {
  console.error(`opening the pull request failed (${opened.status}): ${JSON.stringify(pr).slice(0, 300)}`);
  process.exit(1);
}
console.log(`opened #${pr.number}: ${pr.html_url}`);
console.log('(CI runs on the pull request; waiting on it is on you)');

if (noMerge) process.exit(0);

const merged = await api(`/repos/${repo}/pulls/${pr.number}/merge`, 'PUT', { merge_method: 'rebase' });
const result = await merged.json();
if (merged.status !== 200) {
  console.error(`merge failed (${merged.status}): ${result.message || JSON.stringify(result).slice(0, 300)}`);
  process.exit(1);
}
console.log(`merged: ${result.sha}`);

const deleted = await api(`/repos/${repo}/git/refs/heads/${branch}`, 'DELETE');
if (deleted.status === 204) console.log(`deleted branch ${branch}`);
console.log('locally: git fetch origin && git checkout main && git reset --hard origin/main');
