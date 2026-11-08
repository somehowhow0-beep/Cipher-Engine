import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  DEFAULT_KDF_PARAMS,
  DEFAULT_KDF_POLICY,
  KdfError,
  SALT_LENGTH,
  SUBKEY_LENGTH,
  deriveKeySet,
  deriveMasterKey,
  deriveSubkey,
  destroyKeySet,
  generateSalt,
  validateKdfParams,
  zeroize,
} from '../src/kdf';
import type { KdfParams, KdfPolicy } from '../src/kdf';

// Fast parameters so the suite runs in milliseconds. Production defaults are far stronger.
const FAST_PARAMS: KdfParams = { N: 1024, r: 8, p: 1 };
const FAST_POLICY: KdfPolicy = { minN: 2, maxMemoryBytes: 64 * 1024 * 1024, maxP: 4 };
const SALT = Buffer.from('000102030405060708090a0b0c0d0e0f', 'hex');
const SEED = 'correct horse battery staple';

const derive = (seed: string | Uint8Array, salt: Uint8Array = SALT) =>
  deriveKeySet(seed, salt, FAST_PARAMS, FAST_POLICY);

describe('known-answer vectors', () => {
  // Cross-checked against an independent implementation (Python hashlib.scrypt + manual HKDF).
  it('derives the expected master key', async () => {
    const master = await deriveMasterKey(SEED, SALT, FAST_PARAMS, FAST_POLICY);
    assert.equal(
      master.toString('hex'),
      '9a9f74cc441de571a18c4bf8580ad51f86745d14b39065ad24ad92fc05c99515',
    );
  });

  it('derives the expected subkeys', async () => {
    const keys = await derive(SEED);
    assert.equal(
      keys.engineKey.toString('hex'),
      '357cfced87aeb7e304b83db6b5e06817faff9c3e77ea9246abb86cca9490ad66',
    );
    assert.equal(
      keys.aeadKey.toString('hex'),
      '74af4d45e340697eb0d761bffc2831be616c5547c8a476060c54a0c2593b3269',
    );
    assert.equal(
      keys.decoyKey.toString('hex'),
      '8cb6a130fd997532a255a7c9f091d860f7e7c2ad67c4a61ebe24192226b775ba',
    );
  });
});

describe('determinism and separation', () => {
  it('same seed and salt give identical keys', async () => {
    const a = await derive(SEED);
    const b = await derive(SEED);
    assert.deepEqual(a.engineKey, b.engineKey);
    assert.deepEqual(a.aeadKey, b.aeadKey);
    assert.deepEqual(a.decoyKey, b.decoyKey);
  });

  it('a different seed changes every key', async () => {
    const a = await derive(SEED);
    const b = await derive(SEED + '!');
    assert.notDeepEqual(a.engineKey, b.engineKey);
    assert.notDeepEqual(a.aeadKey, b.aeadKey);
    assert.notDeepEqual(a.decoyKey, b.decoyKey);
  });

  it('a different salt changes every key', async () => {
    const otherSalt = Buffer.from(SALT);
    otherSalt[15] = (otherSalt[15] ?? 0) ^ 1; // flip a single bit
    const a = await derive(SEED);
    const b = await derive(SEED, otherSalt);
    assert.notDeepEqual(a.engineKey, b.engineKey);
    assert.notDeepEqual(a.aeadKey, b.aeadKey);
    assert.notDeepEqual(a.decoyKey, b.decoyKey);
  });

  it('the three subkeys are 32 bytes and pairwise distinct', async () => {
    const { engineKey, aeadKey, decoyKey } = await derive(SEED);
    for (const key of [engineKey, aeadKey, decoyKey]) assert.equal(key.length, SUBKEY_LENGTH);
    assert.notDeepEqual(engineKey, aeadKey);
    assert.notDeepEqual(engineKey, decoyKey);
    assert.notDeepEqual(aeadKey, decoyKey);
  });

  it('deriveSubkey honors the requested length and purpose', async () => {
    const master = await deriveMasterKey(SEED, SALT, FAST_PARAMS, FAST_POLICY);
    assert.equal(deriveSubkey(master, 'engine', 64).length, 64);
    assert.notDeepEqual(deriveSubkey(master, 'engine'), deriveSubkey(master, 'aead'));
    // A longer output must extend, not change, the shorter one (HKDF prefix property).
    assert.deepEqual(deriveSubkey(master, 'aead', 64).subarray(0, 32), deriveSubkey(master, 'aead', 32));
  });

  it('a single-bit seed change flips about half of the key bits (avalanche)', async () => {
    const seedA = Buffer.alloc(32, 0x41);
    const seedB = Buffer.from(seedA);
    seedB[0] = (seedB[0] ?? 0) ^ 1;
    const a = await derive(seedA);
    const b = await derive(seedB);
    let differing = 0;
    for (let i = 0; i < 32; i++) {
      let x = (a.engineKey[i] ?? 0) ^ (b.engineKey[i] ?? 0);
      while (x) {
        differing += x & 1;
        x >>= 1;
      }
    }
    // 256 bits, expected 128, sigma = 8. Bounds are about +/- 6 sigma.
    assert.ok(differing > 80 && differing < 176, `unexpected bit difference: ${differing}`);
  });
});

describe('seed handling', () => {
  it('normalizes Unicode (NFKC): composed and decomposed forms match', async () => {
    const composed = await derive('café sécurité');
    const decomposed = await derive('café sécurité');
    assert.deepEqual(composed.engineKey, decomposed.engineKey);
  });

  it('a Uint8Array seed matches the same string as UTF-8 bytes', async () => {
    const fromString = await derive(SEED);
    const fromBytes = await derive(new TextEncoder().encode(SEED));
    assert.deepEqual(fromString.engineKey, fromBytes.engineKey);
  });

  it('does not mutate the caller seed bytes', async () => {
    const seed = new TextEncoder().encode(SEED);
    const copy = Uint8Array.from(seed);
    await derive(seed);
    assert.deepEqual(seed, copy);
  });

  it('rejects empty seeds', async () => {
    await assert.rejects(derive(''), { name: 'KdfError' });
    await assert.rejects(derive(new Uint8Array(0)), { name: 'KdfError' });
  });

  it('rejects seeds of the wrong type', async () => {
    await assert.rejects(derive(42 as unknown as string), { name: 'KdfError' });
  });
});

describe('salt', () => {
  it('generateSalt returns 16 fresh random bytes', () => {
    const a = generateSalt();
    const b = generateSalt();
    assert.equal(a.length, SALT_LENGTH);
    assert.notDeepEqual(a, b);
  });

  it('generateSalt refuses short lengths', () => {
    assert.throws(() => generateSalt(8), KdfError);
  });

  it('deriveKeySet refuses short salts', async () => {
    await assert.rejects(derive(SEED, Buffer.alloc(8)), { name: 'KdfError' });
  });
});

describe('parameter validation', () => {
  it('accepts the shipped defaults', () => {
    assert.doesNotThrow(() => validateKdfParams(DEFAULT_KDF_PARAMS));
  });

  it('rejects N that is not a power of two', () => {
    assert.throws(() => validateKdfParams({ N: 1000, r: 8, p: 1 }, FAST_POLICY), KdfError);
  });

  it('rejects non-integer or out-of-range r and p', () => {
    assert.throws(() => validateKdfParams({ N: 1024, r: 0, p: 1 }, FAST_POLICY), KdfError);
    assert.throws(() => validateKdfParams({ N: 1024, r: 8.5, p: 1 }, FAST_POLICY), KdfError);
    assert.throws(() => validateKdfParams({ N: 1024, r: 8, p: 0 }, FAST_POLICY), KdfError);
    assert.throws(() => validateKdfParams({ N: 1024, r: 8, p: 99 }, FAST_POLICY), KdfError);
  });

  it('rejects weak N under the default policy (downgrade protection)', () => {
    assert.throws(() => validateKdfParams({ N: 1024, r: 8, p: 1 }), KdfError);
  });

  it('rejects memory-exhaustion parameters (forged envelope DoS)', () => {
    assert.throws(() => validateKdfParams({ N: 2 ** 24, r: 8, p: 1 }), KdfError);
    assert.throws(() => validateKdfParams({ N: 2 ** 17, r: 64, p: 1 }), KdfError);
  });

  it('deriveKeySet enforces the policy before doing any work', async () => {
    await assert.rejects(
      deriveKeySet(SEED, SALT, { N: 2 ** 28, r: 8, p: 1 }, FAST_POLICY),
      { name: 'KdfError' },
    );
  });

  it('deriveSubkey validates its inputs', () => {
    assert.throws(() => deriveSubkey(Buffer.alloc(16), 'engine'), KdfError);
    assert.throws(() => deriveSubkey(Buffer.alloc(32), 'nope' as never), KdfError);
    assert.throws(() => deriveSubkey(Buffer.alloc(32), 'engine', 0), KdfError);
    assert.throws(() => deriveSubkey(Buffer.alloc(32), 'engine', 255 * 32 + 1), KdfError);
  });
});

describe('zeroization', () => {
  it('zeroize wipes a buffer', () => {
    const buffer = Buffer.alloc(16, 0xff);
    zeroize(buffer);
    assert.ok(buffer.every((byte) => byte === 0));
  });

  it('destroyKeySet wipes all three keys', async () => {
    const keys = await derive(SEED);
    destroyKeySet(keys);
    for (const key of [keys.engineKey, keys.aeadKey, keys.decoyKey]) {
      assert.ok(key.every((byte) => byte === 0));
    }
  });
});

describe('policy constants', () => {
  it('default policy admits the default params with headroom', () => {
    assert.ok(128 * DEFAULT_KDF_PARAMS.N * DEFAULT_KDF_PARAMS.r <= DEFAULT_KDF_POLICY.maxMemoryBytes);
    assert.ok(DEFAULT_KDF_PARAMS.N >= DEFAULT_KDF_POLICY.minN);
  });
});
