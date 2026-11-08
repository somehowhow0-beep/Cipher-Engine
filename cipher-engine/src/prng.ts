/**
 * prng.ts: deterministic, cryptographically strong pseudo-random stream.
 *
 * Construction: AES-256-CTR keystream (encrypting zeros) under a key derived
 * with HKDF from (parent key, label). Same key + same label always yields the
 * same byte stream; different labels yield independent streams.
 *
 * This is what turns a seed into "dynamic equations": the generators (S-box,
 * matrices, permutations, round constants) draw all their randomness from named
 * streams of this PRNG, so decryption can rebuild exactly the same math from the
 * same seed, and an attacker without the key cannot predict any of it.
 *
 * Deliberately NOT used: Math.random, LCGs, xorshift, Mersenne Twister. All are
 * predictable from a handful of outputs.
 */

import { createCipheriv, hkdfSync } from 'node:crypto';
import type { Cipheriv } from 'node:crypto';

/** Domain-separation string. Bump the version if the construction ever changes. */
export const PRNG_CONTEXT = 'cipher-engine/prng/v1';
export const PRNG_KEY_LENGTH = 32;

const CHUNK_SIZE = 4096; // keystream is generated in 4 KiB chunks
const ZERO_CHUNK = Buffer.alloc(CHUNK_SIZE);
const ZERO_IV = Buffer.alloc(16); // safe: every stream key is unique per (parent, label)
const MAX_REQUEST_BYTES = 1 << 30;
const MAX_PERMUTATION_SIZE = 1 << 24;
const TWO_POW_32 = 0x1_0000_0000;

function assertIntInRange(name: string, value: number, min: number, max: number): void {
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new RangeError(`${name} must be an integer in ${min}..${max}`);
  }
}

function assertLabel(label: string): void {
  if (typeof label !== 'string' || label.length === 0) {
    throw new TypeError('label must be a non-empty string');
  }
}

function deriveStreamKey(parentKey: Uint8Array, label: string): Buffer {
  const info = Buffer.from(`${PRNG_CONTEXT}:${label}`, 'utf8');
  return Buffer.from(hkdfSync('sha256', parentKey, Buffer.alloc(0), info, PRNG_KEY_LENGTH));
}

export class SeededPrng {
  private readonly streamKey: Buffer;
  private cipher: Cipheriv | null;
  private chunk: Buffer = Buffer.alloc(0);
  private offset = 0;

  private constructor(streamKey: Buffer) {
    this.streamKey = streamKey;
    this.cipher = createCipheriv('aes-256-ctr', streamKey, ZERO_IV);
  }

  /**
   * Create a stream from a 32-byte key (for example engineKey from kdf.ts) and a
   * label naming what the stream is for, such as "sbox/round-3".
   */
  static create(key: Uint8Array, label: string): SeededPrng {
    if (!(key instanceof Uint8Array) || key.byteLength !== PRNG_KEY_LENGTH) {
      throw new RangeError(`key must be exactly ${PRNG_KEY_LENGTH} bytes`);
    }
    assertLabel(label);
    return new SeededPrng(deriveStreamKey(key, label));
  }

  /**
   * Independent child stream. It depends only on this stream's key and the label,
   * never on how many bytes have been consumed, so generators can be built in any
   * order and still produce identical results on encrypt and decrypt.
   */
  fork(label: string): SeededPrng {
    this.assertAlive();
    assertLabel(label);
    return new SeededPrng(deriveStreamKey(this.streamKey, label));
  }

  /** Next `length` bytes of the stream. */
  nextBytes(length: number): Buffer {
    this.assertAlive();
    assertIntInRange('length', length, 0, MAX_REQUEST_BYTES);
    const out = Buffer.allocUnsafe(length);
    let written = 0;
    while (written < length) {
      if (this.offset >= this.chunk.length) this.refill();
      const take = Math.min(length - written, this.chunk.length - this.offset);
      this.chunk.copy(out, written, this.offset, this.offset + take);
      this.offset += take;
      written += take;
    }
    return out;
  }

  /** Uniform 32-bit unsigned integer. */
  nextUint32(): number {
    this.assertAlive();
    if (this.chunk.length - this.offset >= 4) {
      const value = this.chunk.readUInt32BE(this.offset);
      this.offset += 4;
      return value;
    }
    return this.nextBytes(4).readUInt32BE(0);
  }

  /**
   * Uniform integer in [0, maxExclusive), 1 <= maxExclusive <= 2^32.
   * Uses rejection sampling: a plain `x % max` would favor small values whenever
   * max does not divide 2^32, which biases shuffles and generated parameters.
   */
  nextInt(maxExclusive: number): number {
    assertIntInRange('maxExclusive', maxExclusive, 1, TWO_POW_32);
    const limit = Math.floor(TWO_POW_32 / maxExclusive) * maxExclusive;
    let value: number;
    do {
      value = this.nextUint32();
    } while (value >= limit);
    return value % maxExclusive;
  }

  /** In-place Fisher-Yates shuffle (unbiased). Returns the same array. */
  shuffle<T>(items: T[]): T[] {
    for (let i = items.length - 1; i > 0; i--) {
      const j = this.nextInt(i + 1);
      const held = items[i] as T;
      items[i] = items[j] as T;
      items[j] = held;
    }
    return items;
  }

  /** A random permutation of 0..n-1. */
  permutation(n: number): number[] {
    assertIntInRange('n', n, 0, MAX_PERMUTATION_SIZE);
    const values = Array.from({ length: n }, (_, i) => i);
    return this.shuffle(values);
  }

  /** Wipe internal key material. The instance is unusable afterwards. */
  destroy(): void {
    this.streamKey.fill(0);
    this.chunk.fill(0);
    this.chunk = Buffer.alloc(0);
    this.offset = 0;
    this.cipher = null;
  }

  private refill(): void {
    if (this.cipher === null) throw new Error('PRNG has been destroyed');
    this.chunk.fill(0);
    this.chunk = this.cipher.update(ZERO_CHUNK);
    this.offset = 0;
  }

  private assertAlive(): void {
    if (this.cipher === null) throw new Error('PRNG has been destroyed');
  }
}
