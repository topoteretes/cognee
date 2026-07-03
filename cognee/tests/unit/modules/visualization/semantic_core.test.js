// Behavioral unit tests for the pure semantic-map core (semantic_core.js).
// Run via Node's built-in test runner: `node --test semantic_core.test.js`.
// The pytest wrapper (test_semantic_core.py) invokes this and skips when node
// is absent. No dependencies, no build step — the core is d3-free and DOM-free.

const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');

const Core = require(
  path.join(__dirname, '..', '..', '..', '..', 'modules', 'visualization', 'views', 'semantic_core.js'),
);

const CTX = {
  nodeCluster: { a: 0, b: 0, c: 1, d: 1 },
  typeById: { a: 'Entity', b: 'DocumentChunk', c: 'Entity', d: 'Entity' },
  colorById: { a: '#111111', b: '#222222', c: undefined, d: '#444444' },
  clusterColors: { 0: '#c0ffee', 1: '#bada55' },
};

function baseState(over) {
  return Object.assign(
    { colorBy: 'cluster', layoutMode: 'semantic', isolatedCluster: null, isolatedType: null, recall: null },
    over || {},
  );
}

test('screenPositions maps the [-1.2,1.2] box to padded pixels, y inverted', () => {
  const pos = { a: { x: 0, y: 0 }, b: { x: -1.2, y: 1.2 } };
  const s = Core.screenPositions(pos, 800, 600, 60);
  // center maps to the middle of the padded range
  assert.ok(Math.abs(s.a.x - 400) < 1e-9);
  assert.ok(Math.abs(s.a.y - 300) < 1e-9);
  // x=-1.2 -> left pad; y=+1.2 -> top pad (inverted)
  assert.ok(Math.abs(s.b.x - 60) < 1e-9);
  assert.ok(Math.abs(s.b.y - 60) < 1e-9);
});

test('styleFor: recall overlay rings members and dims the rest', () => {
  const state = baseState({ recall: new Set(['a']) });
  const hit = Core.styleFor('a', state, CTX);
  assert.deepStrictEqual(hit, { opacity: 1, stroke: '#ff3b3b', strokeWidth: 2.5, r: 7 });
  const miss = Core.styleFor('b', state, CTX);
  assert.strictEqual(miss.opacity, 0.06);
  assert.strictEqual(miss.stroke, 'rgba(0,0,0,0.25)');
  assert.strictEqual(miss.r, 5);
});

test('styleFor: cluster isolation dims non-members, no ring', () => {
  const state = baseState({ isolatedCluster: 0 });
  assert.strictEqual(Core.styleFor('a', state, CTX).opacity, 1); // cluster 0
  assert.strictEqual(Core.styleFor('c', state, CTX).opacity, 0.12); // cluster 1
  assert.strictEqual(Core.styleFor('a', state, CTX).strokeWidth, 0.5); // no recall ring
});

test('styleFor: type isolation keys off typeById', () => {
  const state = baseState({ colorBy: 'type', isolatedType: 'Entity' });
  assert.strictEqual(Core.styleFor('a', state, CTX).opacity, 1); // Entity
  assert.strictEqual(Core.styleFor('b', state, CTX).opacity, 0.12); // DocumentChunk
});

test('styleFor: no isolation -> everything full opacity', () => {
  const state = baseState();
  assert.strictEqual(Core.styleFor('a', state, CTX).opacity, 1);
  assert.strictEqual(Core.styleFor('c', state, CTX).opacity, 1);
});

test('fillFor: cluster mode uses palette, missing cluster -> grey', () => {
  const state = baseState();
  assert.strictEqual(Core.fillFor('a', state, CTX), '#c0ffee');
  assert.strictEqual(Core.fillFor('x', state, CTX), '#8a8a8a'); // unknown id, cid null
});

test('fillFor: type mode uses node color, missing color -> grey', () => {
  const state = baseState({ colorBy: 'type' });
  assert.strictEqual(Core.fillFor('a', state, CTX), '#111111');
  assert.strictEqual(Core.fillFor('c', state, CTX), '#8a8a8a'); // colorById undefined
});

test('clusterCentroids: mean of member screen points, label truncated at 34', () => {
  const clusters = [
    { id: 0, node_ids: ['a', 'b'], label: 'short' },
    { id: 9, node_ids: ['none'], label: 'x' }, // no positioned members -> skipped
  ];
  const screenPos = { a: { x: 10, y: 20 }, b: { x: 30, y: 40 } };
  const out = Core.clusterCentroids(clusters, screenPos);
  assert.strictEqual(out.length, 1);
  assert.deepStrictEqual({ x: out[0].x, y: out[0].y }, { x: 20, y: 30 });

  const long = 'x'.repeat(50);
  const [c] = Core.clusterCentroids([{ id: 1, node_ids: ['a'], label: long }], { a: { x: 0, y: 0 } });
  assert.strictEqual(c.label.length, 34); // 33 chars + ellipsis
  assert.ok(c.label.endsWith('…'));
});

test('legendModel: cluster mode maps clusters to rows', () => {
  const clusters = [{ id: 0, label: 'L0' }, { id: 1, label: 'L1' }];
  const rows = Core.legendModel(baseState(), clusters, ['a', 'c'], CTX);
  assert.deepStrictEqual(rows.map((r) => r.kind), ['cluster', 'cluster']);
  assert.deepStrictEqual(rows.map((r) => r.value), [0, 1]);
  assert.strictEqual(rows[0].color, '#c0ffee');
});

test('legendModel: type mode dedupes and sorts types', () => {
  const rows = Core.legendModel(baseState({ colorBy: 'type' }), [], ['a', 'b', 'c', 'd'], CTX);
  assert.deepStrictEqual(rows.map((r) => r.value), ['DocumentChunk', 'Entity']); // sorted, unique
  const entity = rows.find((r) => r.value === 'Entity');
  assert.strictEqual(entity.color, '#111111'); // first Entity's color (a)
});

test('recallQueries keeps only search events with node_ids; recallOnMap counts on-map', () => {
  const events = [
    { kind: 'search', question: 'q1', node_ids: ['a', 'b'] },
    { kind: 'search', question: 'empty', node_ids: [] }, // dropped
    { kind: 'improve', node_ids: ['a'] }, // dropped
    null, // dropped
  ];
  const qs = Core.recallQueries(events);
  assert.strictEqual(qs.length, 1);
  assert.strictEqual(qs[0].question, 'q1');

  const positions = { a: { x: 0, y: 0 } }; // only 'a' is on the map
  assert.strictEqual(Core.recallOnMap(qs[0], positions), 1);
});
