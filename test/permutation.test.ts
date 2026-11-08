import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { SeededPrng } from '../src/prng';
import {
  PermutationError,
  generatePermutation,
  invertPermutation,
  isPermutation,
  permute,
  unpermute,
} from '../src/generators/permutation';

const KEY = Buffer.from(Array.from({ length: 32 }, (_, i) => i * 5 + 2));
const OTHER_KEY = Buffer.from(Array.from({ length: 32 }, (_, i) => 200 - i));
const prngFor = (label = 'perm', key: Uint8Array = KEY) => SeededPrng.create(key, label);

describe('isPermutation / invertPermutation', () => {
  it('accepts permutations and rejects everything else', () => {
    assert.ok(isPermutation([2, 0, 1]));
    assert.ok(isPermutation([]));
    assert.equal(isPermutation([0, 0, 1]), false);
    assert.equal(isPermutation([0, 1, 3]), false);
    assert.equal(isPermutation([0, 1.5, 2]), false);
    assert.equal(isPermutation([0, 1, 2], 4), false);
  });

  it('inverts correctly and refuses invalid input', () => {
    const inverse = invertPermutation([2, 0, 3, 1]);
    assert.deepEqual([...inverse], [1, 3, 0, 2]);
    assert.throws(() => invertPermutation([0, 0]), PermutationError);
  });
});

describe('generatePermutation', () => {
  it('returns a valid permutation with a matching inverse', () => {
    for (const size of [2, 3, 16, 32, 255, 256, 1000]) {
      const perm = generatePermutation(prngFor(`size-${size}`), size);
      assert.equal(perm.size, size);
      assert.ok(isPermutation(perm.forward));
      for (let i = 0; i < size; i++) {
        assert.equal(perm.inverse[perm.forward[i] as number], i);
        assert.equal(perm.forward[perm.inverse[i] as number], i);
      }
    }
  });

  it('is a derangement by default (no byte stays in place), even for tiny sizes', () => {
    for (let i = 0; i < 50; i++) {
      for (const size of [2, 3, 4, 16]) {
        const perm = generatePermutation(prngFor(`der-${size}-${i}`), size);
        assert.ok(perm.forward.every((v, idx) => v !== idx), `size ${size} has a fixed point`);
      }
    }
  });

  it('can allow fixed points when asked', () => {
    let sawFixedPoint = false;
    for (let i = 0; i < 100 && !sawFixedPoint; i++) {
      const perm = generatePermutation(prngFor(`fix-${i}`), 3, { derangement: false });
      sawFixedPoint = perm.forward.some((v, idx) => v === idx);
    }
    assert.ok(sawFixedPoint);
  });

  it('is deterministic, and varies with key and label', () => {
    const a = generatePermutation(prngFor(), 64);
    const b = generatePermutation(prngFor(), 64);
    const c = generatePermutation(prngFor('perm', OTHER_KEY), 64);
    const d = generatePermutation(prngFor('other'), 64);
    assert.deepEqual(a.forward, b.forward);
    assert.notDeepEqual(a.forward, c.forward);
    assert.notDeepEqual(a.forward, d.forward);
  });

  it('size 1: identity is allowed only without the derangement requirement', () => {
    assert.deepEqual([...generatePermutation(prngFor(), 1, { derangement: false }).forward], [0]);
    assert.throws(() => generatePermutation(prngFor(), 1), PermutationError);
  });

  it('rejects invalid sizes', () => {
    for (const bad of [0, -1, 1.5, 65_537, Number.NaN]) {
      assert.throws(() => generatePermutation(prngFor(), bad), PermutationError, `size ${bad}`);
    }
  });
});

describe('permute / unpermute', () => {
  it('round-trips random data of several block sizes', () => {
    for (const size of [2, 16, 32, 64, 1024]) {
      const perm = generatePermutation(prngFor(`rt-${size}`), size);
      const data = prngFor(`data-${size}`).nextBytes(size);
      const scrambled = permute(perm, data);
      assert.deepEqual(unpermute(perm, scrambled), data);
      assert.deepEqual(scrambled.slice().sort(), data.slice().sort(), 'same multiset of bytes');
    }
  });

  it('follows the documented gather convention: out[i] = data[forward[i]]', () => {
    const perm = generatePermutation(prngFor(), 8);
    const data = Buffer.from([10, 11, 12, 13, 14, 15, 16, 17]);
    const out = permute(perm, data);
    for (let i = 0; i < 8; i++) assert.equal(out[i], data[perm.forward[i] as number]);
  });

  it('moves every byte of a distinct-valued block (derangement)', () => {
    const perm = generatePermutation(prngFor(), 32);
    const data = Buffer.from(Array.from({ length: 32 }, (_, i) => i));
    const out = permute(perm, data);
    for (let i = 0; i < 32; i++) assert.notEqual(out[i], data[i]);
  });

  it('refuses data whose length differs from the permutation size', () => {
    const perm = generatePermutation(prngFor(), 16);
    assert.throws(() => permute(perm, Buffer.alloc(15)), PermutationError);
    assert.throws(() => unpermute(perm, Buffer.alloc(17)), PermutationError);
  });

  it('does not modify its input', () => {
    const perm = generatePermutation(prngFor(), 16);
    const data = Buffer.from(Array.from({ length: 16 }, (_, i) => i));
    const copy = Buffer.from(data);
    permute(perm, data);
    assert.deepEqual(data, copy);
  });
});
