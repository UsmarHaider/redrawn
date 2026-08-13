/*
 * maup.js -- the analysis, ported to run in the browser.
 *
 * The page does not display precomputed answers. It holds the same per-tract
 * crash counts the Python pipeline works from, and recomputes zone totals,
 * rates and the correlation every time the reader changes the map. These
 * functions are the port, and `tests/test_js_parity.py` runs this same file
 * under Node against the Python results so the two cannot drift apart.
 *
 * Loads as a plain script in the browser (window.MAUP) and via require() in Node.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.MAUP = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  /** Deterministic 32-bit PRNG, so a shared seed reproduces a map exactly. */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function pearson(x, y) {
    const n = x.length;
    if (n < 3) return NaN;
    let mx = 0, my = 0;
    for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < n; i++) {
      const dx = x[i] - mx, dy = y[i] - my;
      sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    const denom = Math.sqrt(sxx * syy);
    return denom > 0 ? sxy / denom : NaN;
  }

  function weightedPearson(x, y, w) {
    const n = x.length;
    let total = 0;
    for (let i = 0; i < n; i++) total += w[i];
    if (total <= 0 || n < 3) return NaN;
    let mx = 0, my = 0;
    for (let i = 0; i < n; i++) { mx += w[i] * x[i]; my += w[i] * y[i]; }
    mx /= total; my /= total;
    let cov = 0, vx = 0, vy = 0;
    for (let i = 0; i < n; i++) {
      const dx = x[i] - mx, dy = y[i] - my;
      cov += w[i] * dx * dy; vx += w[i] * dx * dx; vy += w[i] * dy * dy;
    }
    if (vx <= 0 || vy <= 0) return NaN;
    return cov / total / Math.sqrt((vx / total) * (vy / total));
  }

  /** Per-zone crash count and the two indicator rates; empty zones dropped. */
  function zoneRates(labels, n, x, y) {
    let k = 0;
    for (let i = 0; i < labels.length; i++) if (labels[i] + 1 > k) k = labels[i] + 1;
    const zn = new Float64Array(k), zx = new Float64Array(k), zy = new Float64Array(k);
    for (let i = 0; i < labels.length; i++) {
      const z = labels[i];
      zn[z] += n[i]; zx[z] += x[i]; zy[z] += y[i];
    }
    const outN = [], outX = [], outY = [];
    for (let z = 0; z < k; z++) {
      if (zn[z] > 0) { outN.push(zn[z]); outX.push(zx[z] / zn[z]); outY.push(zy[z] / zn[z]); }
    }
    return { n: outN, x: outX, y: outY, k: outN.length };
  }

  function partitionCorrelation(labels, n, x, y, weighted) {
    const z = zoneRates(labels, n, x, y);
    return weighted ? weightedPearson(z.x, z.y, z.n) : pearson(z.x, z.y);
  }

  /* ----------------------------------------------------------------------- *
   * Map generation. Both families mirror src/redrawn/partitions.py; the
   * random draws differ from Python's (different PRNG), so the maps are not
   * identical, but they come from the same procedure and the same constraints.
   * ----------------------------------------------------------------------- */

  /** Minimal binary heap keyed by a numeric score. */
  function Heap() {
    this.items = [];
  }
  Heap.prototype.push = function (score, value) {
    const a = this.items;
    a.push([score, value]);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[p][0] <= a[i][0]) break;
      const t = a[p]; a[p] = a[i]; a[i] = t;
      i = p;
    }
  };
  Heap.prototype.pop = function () {
    const a = this.items;
    if (!a.length) return null;
    const top = a[0], last = a.pop();
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let s = i;
        if (l < a.length && a[l][0] < a[s][0]) s = l;
        if (r < a.length && a[r][0] < a[s][0]) s = r;
        if (s === i) break;
        const t = a[s]; a[s] = a[i]; a[i] = t;
        i = s;
      }
    }
    return top;
  };
  Heap.prototype.size = function () { return this.items.length; };

  /**
   * Agglomerate the lightest adjacent pair of zones until K remain -- the same
   * balanced merge the Python pipeline uses to build neutral maps.
   */
  function mergeToK(adjStart, adj, weight, k, rand, jitter) {
    jitter = jitter === undefined ? 0.35 : jitter;
    const n = weight.length;
    const parent = new Int32Array(n);
    const zoneWeight = new Float64Array(n);
    const alive = new Uint8Array(n);
    const version = new Int32Array(n);
    const nbrs = [];
    for (let i = 0; i < n; i++) {
      parent[i] = i; zoneWeight[i] = weight[i]; alive[i] = 1;
      const s = new Set();
      for (let p = adjStart[i]; p < adjStart[i + 1]; p++) s.add(adj[p]);
      nbrs.push(s);
    }
    function find(i) {
      while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; }
      return i;
    }
    const key = (a, b) => (zoneWeight[a] + zoneWeight[b]) * (1 + jitter * rand());

    const heap = new Heap();
    for (let a = 0; a < n; a++) {
      nbrs[a].forEach(function (b) {
        if (a < b) heap.push(key(a, b), [a, b, 0, 0]);
      });
    }

    let live = n;
    while (live > k) {
      let a, b;
      if (!heap.size()) {
        const rest = [];
        for (let i = 0; i < n; i++) if (alive[i]) rest.push(i);
        if (rest.length < 2) break;
        rest.sort((p, q) => zoneWeight[p] - zoneWeight[q]);
        a = rest[0]; b = rest[1];
      } else {
        const top = heap.pop();
        a = top[1][0]; b = top[1][1];
        if (!alive[a] || !alive[b] || a === b) continue;
        if (version[a] !== top[1][2] || version[b] !== top[1][3]) continue;
      }
      parent[b] = a;
      alive[b] = 0;
      zoneWeight[a] += zoneWeight[b];
      const merged = new Set();
      nbrs[a].forEach((v) => { const f = find(v); if (f !== a) merged.add(f); });
      nbrs[b].forEach((v) => { const f = find(v); if (f !== a) merged.add(f); });
      nbrs[a] = merged;
      version[a]++;
      live--;
      merged.forEach(function (c) {
        if (alive[c]) {
          nbrs[c].add(a);
          const lo = Math.min(a, c), hi = Math.max(a, c);
          heap.push(key(a, c), [lo, hi, version[lo], version[hi]]);
        }
      });
    }
    return relabel(Array.from({ length: n }, (_, i) => find(i)));
  }

  /** Multi-seed region growing: K seeds annex random unclaimed neighbours. */
  function growToK(adjStart, adj, k, rand) {
    const n = adjStart.length - 1;
    const labels = new Int32Array(n).fill(-1);
    const order = shuffled(n, rand);
    const frontier = [];
    for (let z = 0; z < k; z++) {
      labels[order[z]] = z;
      const f = [];
      for (let p = adjStart[order[z]]; p < adjStart[order[z] + 1]; p++) {
        if (labels[adj[p]] < 0) f.push(adj[p]);
      }
      frontier.push(f);
    }
    let remaining = n - k;
    let active = [];
    for (let z = 0; z < k; z++) if (frontier[z].length) active.push(z);

    while (remaining > 0 && active.length) {
      const pos = Math.floor(rand() * active.length);
      const zone = active[pos];
      const pool = frontier[zone];
      let t = -1;
      while (pool.length) {
        const idx = Math.floor(rand() * pool.length);
        const cand = pool[idx];
        pool[idx] = pool[pool.length - 1];
        pool.pop();
        if (labels[cand] < 0) { t = cand; break; }
      }
      if (t < 0) { active.splice(pos, 1); continue; }
      labels[t] = zone;
      remaining--;
      for (let p = adjStart[t]; p < adjStart[t + 1]; p++) {
        if (labels[adj[p]] < 0) pool.push(adj[p]);
      }
      if (!pool.length) active.splice(pos, 1);
    }
    // Any pocket the growth could not reach joins an adjacent zone.
    for (let i = 0; i < n; i++) {
      if (labels[i] >= 0) continue;
      for (let p = adjStart[i]; p < adjStart[i + 1]; p++) {
        if (labels[adj[p]] >= 0) { labels[i] = labels[adj[p]]; break; }
      }
      if (labels[i] < 0) labels[i] = 0;
    }
    return relabel(Array.from(labels));
  }

  function shuffled(n, rand) {
    const a = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
  }

  function relabel(labels) {
    const map = new Map();
    const out = new Int32Array(labels.length);
    for (let i = 0; i < labels.length; i++) {
      if (!map.has(labels[i])) map.set(labels[i], map.size);
      out[i] = map.get(labels[i]);
    }
    return out;
  }

  /** Every zone one connected block? Used by the page's contiguity badge. */
  function isContiguous(labels, adjStart, adj) {
    const byZone = new Map();
    for (let i = 0; i < labels.length; i++) {
      if (!byZone.has(labels[i])) byZone.set(labels[i], []);
      byZone.get(labels[i]).push(i);
    }
    for (const members of byZone.values()) {
      const set = new Set(members);
      const stack = [members[0]];
      const seen = new Set(stack);
      while (stack.length) {
        const v = stack.pop();
        for (let p = adjStart[v]; p < adjStart[v + 1]; p++) {
          const w = adj[p];
          if (set.has(w) && !seen.has(w)) { seen.add(w); stack.push(w); }
        }
      }
      if (seen.size !== set.size) return false;
    }
    return true;
  }

  return {
    mulberry32, pearson, weightedPearson, zoneRates, partitionCorrelation,
    mergeToK, growToK, isContiguous, relabel,
  };
});
