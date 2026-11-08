import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createCipheriv, hkdfSync } from 'node:crypto';

import { SeededPrng } from '../src/prng';

const KEY = Buffer.from(Array.from({ length: 32 }, (_, i) => i));
const OTHER_KEY = Buffer.from(Array.from({ length: 32 }, (_, i) => 255 - i));

const make = (label = 'test', key: Uint8Array = KEY) => SeededPrng.create(key, label);

/** Chi-square statistic of observed counts against a uniform expectation. */
function chiSquare(counts: number[]): number {
  const total = counts.reduce((a, b) => a + b, 0);
  const expected = total / counts.length;
  return counts.reduce((sum, c) => sum + (c - expected) ** 2 / expected, 0);
}

describe('stream correctness', () => {
  it('matches a reference AES-256-CTR keystream built from the spec', () => {
    // Independent re-derivation: spec = HKDF-SHA256(key, info = "<context>:<label>") then AES-CTR over zeros.
    const streamKey = Buffer.from(
      hkdfSync('sha256', KEY, Buffer.alloc(0), Buffer.from('cipher-engine/prng/v1:test'), 32),
    );
    const reference = createCipheriv('aes-256-ctr', streamKey, Buffer.alloc(16)).update(
      Buffer.alloc(10_000),
    );
    assert.deepEqual(make().nextBytes(10_000), reference);
  });

  it('pinned regression vector (first 32 bytes)', () => {
    assert.equal(
      make().nextBytes(32).toString('hex'),
      '31f324abbb8009cbcbab36aba8c91c584365f8ef3b4bf15eebe5705aac256ece',
    );
  });

  it('is deterministic for the same key and label', () => {
    assert.deepEqual(make().nextBytes(5000), make().nextBytes(5000));
  });

  it('differs for a different key or label', () => {
    const base = make().nextBytes(64);
    assert.notDeepEqual(make('test', OTHER_KEY).nextBytes(64), base);
    assert.notDeepEqual(make('other').nextBytes(64), base);
  });

  it('read sizes do not change the stream, including across chunk boundaries', () => {
    const whole = make().nextBytes(20_000);
    const prng = make();
    const parts: Buffer[] = [];
    let total = 0;
    for (const size of [1, 3, 7, 4093, 4096, 5, 1, 4099, 11, 2000, 1500]) {
      parts.push(prng.nextBytes(size));
      total += size;
    }
    parts.push(prng.nextBytes(20_000 - total));
    assert.deepEqual(Buffer.concat(parts), whole);
  });

  it('nextUint32 consumes exactly four stream bytes, even when straddling a chunk', () => {
    const reference = make().nextBytes(9000);
    const prng = make();
    prng.nextBytes(4094); // leave 2 bytes in the first chunk, so the next read straddles
    assert.equal(prng.nextUint32(), reference.readUInt32BE(4094));
    assert.equal(prng.nextUint32(), reference.readUInt32BE(4098));
  });

  it('nextBytes(0) returns an empty buffer and consumes nothing', () => {
    const prng = make();
    assert.equal(prng.nextBytes(0).length, 0);
    assert.deepEqual(prng.nextBytes(8), make().nextBytes(8));
  });

  it('passes basic statistical sanity checks (monobit and byte frequency)', () => {
    const data = make('stats').nextBytes(256_000);
    let ones = 0;
    const counts = new Array<number>(256).fill(0);
    for (const byte of data) {
      counts[byte] = (counts[byte] ?? 0) + 1;
      let x = byte;
      while (x) {
        ones += x & 1;
        x >>= 1;
      }
    }
    const ratio = ones / (data.length * 8);
    assert.ok(Math.abs(ratio - 0.5) < 0.005, `monobit ratio ${ratio}`);
    // df = 255, critical value at p = 0.001 is about 330.5
    assert.ok(chiSquare(counts) < 330.5, `byte chi-square ${chiSquare(counts)}`);
  });
});

describe('nextInt', () => {
  it('stays within [0, max) for assorted bounds', () => {
    const prng = make('range');
    for (const max of [1, 2, 3, 7, 255, 256, 257, 1000, 65_537, 2 ** 31, 2 ** 32 - 1, 2 ** 32]) {
      for (let i = 0; i < 200; i++) {
        const value = prng.nextInt(max);
        assert.ok(Number.isInteger(value) && value >= 0 && value < max, `max=${max} value=${value}`);
      }
    }
  });

  it('max = 1 always returns 0', () => {
    const prng = make('one');
    for (let i = 0; i < 20; i++) assert.equal(prng.nextInt(1), 0);
  });

  it('rejects raw values in the biased tail instead of reducing them with modulo', () => {
    // For max = 7: limit = floor(2^32 / 7) * 7 = 4294967292, so 4294967292..4294967295 must be redrawn.
    // (A statistical test cannot see this bias at 32 bits, so we script the raw draws.)
    const scripted = [4294967295, 4294967292, 4294967291];
    const prng = make('scripted');
    prng.nextUint32 = () => scripted.shift() as number;
    assert.equal(prng.nextInt(7), 4294967291 % 7);
    assert.equal(scripted.length, 0, 'both tail values should have been consumed and discarded');

    // For max = 2^32 - 1: only the single value 2^32 - 1 is in the tail.
    const second = [0xffffffff, 5];
    const prng2 = make('scripted-2');
    prng2.nextUint32 = () => second.shift() as number;
    assert.equal(prng2.nextInt(0xffffffff), 5);
  });

  it('is uniform for a non-power-of-two bound', () => {
    const prng = make('uniform-7');
    const counts = new Array<number>(7).fill(0);
    for (let i = 0; i < 70_000; i++) {
      const v = prng.nextInt(7);
      counts[v] = (counts[v] ?? 0) + 1;
    }
    // df = 6, critical value at p = 0.001 is about 22.46
    assert.ok(chiSquare(counts) < 22.46, `chi-square ${chiSquare(counts)}`);
  });

  it('is uniform for a power-of-two bound', () => {
    const prng = make('uniform-256');
    const counts = new Array<number>(256).fill(0);
    for (let i = 0; i < 256_000; i++) {
      const v = prng.nextInt(256);
      counts[v] = (counts[v] ?? 0) + 1;
    }
    assert.ok(chiSquare(counts) < 330.5, `chi-square ${chiSquare(counts)}`);
  });

  it('rejects invalid bounds', () => {
    const prng = make();
    for (const bad of [0, -1, 1.5, Number.NaN, Infinity, 2 ** 32 + 1]) {
      assert.throws(() => prng.nextInt(bad), RangeError, `bound ${bad}`);
    }
  });
});

describe('shuffle and permutation', () => {
  it('shuffle returns the same array, containing the same elements', () => {
    const items = Array.from({ length: 100 }, (_, i) => i);
    const result = make('shuffle').shuffle(items);
    assert.equal(result, items);
    assert.deepEqual([...items].sort((a, b) => a - b), Array.from({ length: 100 }, (_, i) => i));
  });

  it('is deterministic, and different keys give different orders', () => {
    const a = make('perm').permutation(256);
    const b = make('perm').permutation(256);
    const c = make('perm', OTHER_KEY).permutation(256);
    assert.deepEqual(a, b);
    assert.notDeepEqual(a, c);
  });

  it('permutation(n) is a bijection on 0..n-1', () => {
    for (const n of [0, 1, 2, 16, 255, 256, 1000]) {
      const perm = make(`bij-${n}`).permutation(n);
      assert.equal(perm.length, n);
      assert.deepEqual([...perm].sort((x, y) => x - y), Array.from({ length: n }, (_, i) => i));
    }
  });

  it('actually moves things (a 256-element shuffle is almost never the identity)', () => {
    const perm = make('moves').permutation(256);
    const fixedPoints = perm.filter((v, i) => v === i).length;
    assert.ok(fixedPoints < 16, `fixed points: ${fixedPoints}`); // expected is about 1
  });

  it('produces all 6 orderings of 3 elements with uniform frequency', () => {
    const prng = make('perm3');
    const tally = new Map<string, number>();
    for (let i = 0; i < 12_000; i++) {
      const key = prng.shuffle([0, 1, 2]).join('');
      tally.set(key, (tally.get(key) ?? 0) + 1);
    }
    assert.equal(tally.size, 6);
    // df = 5, critical value at p = 0.001 is about 20.52
    assert.ok(chiSquare([...tally.values()]) < 20.52, `chi-square ${chiSquare([...tally.values()])}`);
  });

  it('rejects invalid permutation sizes', () => {
    const prng = make();
    assert.throws(() => prng.permutation(-1), RangeError);
    assert.throws(() => prng.permutation(1.5), RangeError);
    assert.throws(() => prng.permutation(2 ** 25), RangeError);
  });
});

describe('fork', () => {
  it('is deterministic and independent of how much the parent has consumed', () => {
    const early = make().fork('child').nextBytes(64);
    const parent = make();
    parent.nextBytes(12_345);
    const late = parent.fork('child').nextBytes(64);
    assert.deepEqual(early, late);
  });

  it('children differ from the parent and from each other', () => {
    const parent = make();
    const a = parent.fork('a').nextBytes(64);
    const b = parent.fork('b').nextBytes(64);
    assert.notDeepEqual(a, b);
    assert.notDeepEqual(a, make().nextBytes(64));
  });

  it('forking does not disturb the parent stream', () => {
    const parent = make();
    parent.fork('x').nextBytes(100);
    assert.deepEqual(parent.nextBytes(32), make().nextBytes(32));
  });

  it('a fork is not the same as a root stream with the same label', () => {
    assert.notDeepEqual(make().fork('x').nextBytes(32), make('x').nextBytes(32));
  });
});

describe('input validation and lifecycle', () => {
  it('rejects keys that are not exactly 32 bytes', () => {
    assert.throws(() => SeededPrng.create(Buffer.alloc(31), 'x'), RangeError);
    assert.throws(() => SeededPrng.create(Buffer.alloc(33), 'x'), RangeError);
    assert.throws(() => SeededPrng.create('a'.repeat(32) as unknown as Uint8Array, 'x'), RangeError);
  });

  it('rejects empty labels', () => {
    assert.throws(() => SeededPrng.create(KEY, ''), TypeError);
    assert.throws(() => make().fork(''), TypeError);
  });

  it('rejects invalid byte counts', () => {
    const prng = make();
    assert.throws(() => prng.nextBytes(-1), RangeError);
    assert.throws(() => prng.nextBytes(1.5), RangeError);
    assert.throws(() => prng.nextBytes(2 ** 31), RangeError);
  });

  it('is unusable after destroy()', () => {
    const prng = make();
    prng.nextBytes(8);
    prng.destroy();
    assert.throws(() => prng.nextBytes(1), /destroyed/);
    assert.throws(() => prng.nextUint32(), /destroyed/);
    assert.throws(() => prng.nextInt(10), /destroyed/);
    assert.throws(() => prng.fork('x'), /destroyed/);
  });

  it('destroying a parent does not break an existing child', () => {
    const parent = make();
    const child = parent.fork('kid');
    const expected = make().fork('kid').nextBytes(16);
    parent.destroy();
    assert.deepEqual(child.nextBytes(16), expected);
  });
});
