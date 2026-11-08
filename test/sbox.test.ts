import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { SeededPrng } from '../src/prng';
import {
  DEFAULT_SBOX_POLICY,
  SboxError,
  differentialUniformity,
  generateSbox,
  hasFixedPoints,
  invertTable,
  isBijection,
  nonlinearity,
  substitute,
  validateSboxPolicy,
} from '../src/generators/sbox';

const KEY = Buffer.from(Array.from({ length: 32 }, (_, i) => i * 7 + 1));
const OTHER_KEY = Buffer.from(Array.from({ length: 32 }, (_, i) => 255 - i * 3));
const prngFor = (label = 'sbox', key: Uint8Array = KEY) => SeededPrng.create(key, label);

/** The real AES S-box, built from first principles (inverse in GF(2^8) + affine map). */
function buildAesSbox(): Uint8Array {
  const mul = (a: number, b: number): number => {
    let p = 0;
    for (let i = 0; i < 8; i++) {
      if (b & 1) p ^= a;
      const hi = a & 0x80;
      a = (a << 1) & 0xff;
      if (hi) a ^= 0x1b;
      b >>= 1;
    }
    return p;
  };
  const inv = (x: number): number => {
    if (x === 0) return 0;
    for (let y = 1; y < 256; y++) if (mul(x, y) === 1) return y;
    throw new Error('no inverse');
  };
  const rotl = (v: number, n: number) => ((v << n) | (v >> (8 - n))) & 0xff;
  const out = new Uint8Array(256);
  for (let x = 0; x < 256; x++) {
    const b = inv(x);
    out[x] = b ^ rotl(b, 1) ^ rotl(b, 2) ^ rotl(b, 3) ^ rotl(b, 4) ^ 0x63;
  }
  return out;
}

describe('analysis functions (validated against known S-boxes)', () => {
  const aes = buildAesSbox();

  it('the reconstructed AES S-box has the published values (sanity check of the reference)', () => {
    assert.equal(aes[0], 0x63);
    assert.equal(aes[1], 0x7c);
    assert.equal(aes[0x53], 0xed);
    assert.ok(isBijection(aes));
  });

  it('AES S-box: nonlinearity 112 and differential uniformity 4', () => {
    assert.equal(nonlinearity(aes), 112);
    assert.equal(differentialUniformity(aes), 4);
  });

  it('identity map is maximally weak: nonlinearity 0, differential uniformity 256', () => {
    const identity = Uint8Array.from({ length: 256 }, (_, i) => i);
    assert.equal(nonlinearity(identity), 0);
    assert.equal(differentialUniformity(identity), 256);
  });

  it('an affine map x -> 5x XOR 3 style permutation is detected as linear (nonlinearity 0)', () => {
    // x -> rotate-left by 1 is linear over GF(2).
    const rot = Uint8Array.from({ length: 256 }, (_, x) => ((x << 1) | (x >> 7)) & 0xff);
    assert.equal(nonlinearity(rot), 0);
  });

  it('hasFixedPoints detects fixed and opposite fixed points', () => {
    assert.equal(hasFixedPoints(aes), false);
    const a = Uint8Array.from(aes);
    a[10] = 10;
    assert.equal(hasFixedPoints(a), true);
    const b = Uint8Array.from(aes);
    b[10] = 10 ^ 0xff;
    assert.equal(hasFixedPoints(b), true);
  });

  it('isBijection rejects duplicates, wrong length and out-of-range values', () => {
    const dup = Uint8Array.from(aes);
    dup[1] = dup[0] as number;
    assert.equal(isBijection(dup), false);
    assert.equal(isBijection(new Uint8Array(255)), false);
    assert.equal(isBijection(Array.from({ length: 256 }, (_, i) => i + (i === 255 ? 1 : 0))), false);
  });

  it('invertTable inverts, and refuses non-permutations', () => {
    const inverse = invertTable(aes);
    for (let x = 0; x < 256; x++) {
      assert.equal(inverse[aes[x] as number], x);
      assert.equal(aes[inverse[x] as number], x);
    }
    assert.throws(() => invertTable(new Uint8Array(256)), SboxError);
  });
});

describe('generateSbox', () => {
  it('produces a bijection with a correct inverse', () => {
    const sbox = generateSbox(prngFor());
    assert.ok(isBijection(sbox.forward));
    assert.ok(isBijection(sbox.inverse));
    for (let x = 0; x < 256; x++) {
      assert.equal(sbox.inverse[sbox.forward[x] as number], x);
    }
  });

  it('meets the quality policy and reports honest metrics', () => {
    for (let i = 0; i < 20; i++) {
      const sbox = generateSbox(prngFor(`quality-${i}`));
      assert.equal(sbox.nonlinearity, nonlinearity(sbox.forward));
      assert.equal(sbox.differentialUniformity, differentialUniformity(sbox.forward));
      assert.ok(sbox.nonlinearity >= DEFAULT_SBOX_POLICY.minNonlinearity);
      assert.ok(sbox.differentialUniformity <= DEFAULT_SBOX_POLICY.maxDifferentialUniformity);
      assert.equal(hasFixedPoints(sbox.forward), false);
    }
  });

  it('the quality gate really rejects candidates (some seeds need more than one attempt)', () => {
    let rejectedSomewhere = false;
    let totalAttempts = 0;
    const runs = 40;
    for (let i = 0; i < runs; i++) {
      const sbox = generateSbox(prngFor(`gate-${i}`));
      totalAttempts += sbox.attempts;
      if (sbox.attempts > 1) rejectedSomewhere = true;
    }
    assert.ok(rejectedSomewhere, 'gate never rejected anything, so it may be disabled');
    assert.ok(totalAttempts / runs < 200, `average attempts too high: ${totalAttempts / runs}`);
  });

  it('is deterministic for the same stream', () => {
    const a = generateSbox(prngFor());
    const b = generateSbox(prngFor());
    assert.deepEqual(a.forward, b.forward);
    assert.equal(a.attempts, b.attempts);
  });

  it('a different key or label gives a completely different S-box', () => {
    const base = generateSbox(prngFor());
    for (const other of [generateSbox(prngFor('sbox', OTHER_KEY)), generateSbox(prngFor('other'))]) {
      const same = base.forward.filter((v, i) => v === other.forward[i]).length;
      // For independent random permutations, the expected number of equal positions is about 1.
      assert.ok(same < 12, `S-boxes share ${same} positions`);
    }
  });

  it('a strict-but-reachable policy works; an unreachable one throws SboxError', () => {
    assert.doesNotThrow(() =>
      generateSbox(prngFor(), { minNonlinearity: 90, maxDifferentialUniformity: 12, maxAttempts: 200 }),
    );
    assert.throws(
      () => generateSbox(prngFor(), { minNonlinearity: 112, maxDifferentialUniformity: 4, maxAttempts: 30 }),
      SboxError,
    );
  });

  it('validates the policy', () => {
    const ok = DEFAULT_SBOX_POLICY;
    assert.doesNotThrow(() => validateSboxPolicy(ok));
    assert.throws(() => validateSboxPolicy({ ...ok, minNonlinearity: -1 }), SboxError);
    assert.throws(() => validateSboxPolicy({ ...ok, minNonlinearity: 121 }), SboxError);
    assert.throws(() => validateSboxPolicy({ ...ok, maxDifferentialUniformity: 2 }), SboxError);
    assert.throws(() => validateSboxPolicy({ ...ok, maxAttempts: 0 }), SboxError);
    assert.throws(() => validateSboxPolicy({ ...ok, maxAttempts: 1.5 }), SboxError);
  });
});

describe('substitute', () => {
  it('round-trips arbitrary data through forward then inverse tables', () => {
    const sbox = generateSbox(prngFor());
    const data = prngFor('data').nextBytes(5000);
    const scrambled = substitute(sbox.forward, data);
    assert.notDeepEqual(scrambled, data);
    assert.deepEqual(substitute(sbox.inverse, scrambled), data);
  });

  it('handles empty input and maps single bytes per the table', () => {
    const sbox = generateSbox(prngFor());
    assert.equal(substitute(sbox.forward, new Uint8Array(0)).length, 0);
    assert.equal(substitute(sbox.forward, Uint8Array.of(0x41))[0], sbox.forward[0x41]);
  });
});
