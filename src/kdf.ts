/**
 * kdf.ts: seed -> master key -> purpose-bound subkeys.
 *
 * Pipeline:
 *   seed --(NFKC + UTF-8)--> scrypt(salt, N, r, p) --> 32-byte master key
 *   master key --(HKDF-SHA256, one label per purpose)--> engineKey | aeadKey | decoyKey
 *
 * Why this module exists: every other module in the system is only as strong as
 * the key material it receives. A human-chosen seed has low entropy, so it must
 * pass through a memory-hard KDF (scrypt) before anything else touches it. The
 * HKDF step guarantees that no key is ever reused across purposes: the inner
 * engine, the AES-GCM layer and the decoy layer each get an independent key.
 *
 * Zeroization in JavaScript is best-effort: strings are immutable and the GC may
 * copy buffers. We wipe every Buffer we own as soon as we are done with it, but
 * this does not make the process memory-forensics proof.
 */

import { hkdfSync, randomBytes, scrypt } from 'node:crypto';

// ---------------------------------------------------------------------------
// Types and constants
// ---------------------------------------------------------------------------

/** A seed may be a passphrase string or raw bytes (for example a 256-bit random secret). */
export type SeedInput = string | Uint8Array;

/** Every subkey is bound to exactly one purpose. */
export const KEY_PURPOSES = ['engine', 'aead', 'decoy'] as const;
export type KeyPurpose = (typeof KEY_PURPOSES)[number];

/** scrypt cost parameters. These are stored (unencrypted) in the envelope. */
export interface KdfParams {
  readonly N: number; // CPU/memory cost, a power of two
  readonly r: number; // block size
  readonly p: number; // parallelism
}

/**
 * Bounds enforced on KdfParams. The envelope is attacker-controlled input on
 * decrypt, so without an upper bound a forged header could demand gigabytes of
 * RAM (denial of service). The lower bound stops accidental weak configs.
 */
export interface KdfPolicy {
  readonly minN: number;
  readonly maxMemoryBytes: number;
  readonly maxP: number;
}

/** The three independent keys produced from one seed. */
export interface KeySet {
  readonly engineKey: Buffer;
  readonly aeadKey: Buffer;
  readonly decoyKey: Buffer;
}

/** Bump when the derivation changes; it is mixed into every HKDF call. */
export const KDF_CONTEXT = 'cipher-engine/kdf/v1';

export const SALT_LENGTH = 16;
export const MASTER_KEY_LENGTH = 32;
export const SUBKEY_LENGTH = 32;

/** 128 MiB, 1 pass: the commonly recommended scrypt baseline for interactive-ish use. */
export const DEFAULT_KDF_PARAMS: KdfParams = Object.freeze({ N: 1 << 17, r: 8, p: 1 });

export const DEFAULT_KDF_POLICY: KdfPolicy = Object.freeze({
  minN: 1 << 15,
  maxMemoryBytes: 512 * 1024 * 1024,
  maxP: 4,
});

const MAX_HKDF_LENGTH = 255 * 32; // RFC 5869 limit for SHA-256

export class KdfError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'KdfError';
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Overwrite a buffer with zeros. Safe to call on any Uint8Array. */
export function zeroize(buffer: Uint8Array): void {
  buffer.fill(0);
}

/** Wipe all three keys of a KeySet. Call this when you are done encrypting/decrypting. */
export function destroyKeySet(keys: KeySet): void {
  zeroize(keys.engineKey);
  zeroize(keys.aeadKey);
  zeroize(keys.decoyKey);
}

/** Fresh random salt. A new salt per message means the same seed never yields the same keys twice. */
export function generateSalt(length: number = SALT_LENGTH): Buffer {
  if (!Number.isInteger(length) || length < SALT_LENGTH) {
    throw new KdfError(`salt must be at least ${SALT_LENGTH} bytes`);
  }
  return randomBytes(length);
}

/**
 * Copy the seed into a private Buffer we can wipe afterwards.
 * Strings are Unicode-normalized (NFKC) so that visually identical passphrases
 * typed on different systems ("é" as one code point vs "e" + combining accent)
 * derive the same keys. Changing this rule breaks old data, so it is versioned
 * through KDF_CONTEXT.
 */
function normalizeSeed(seed: SeedInput): Buffer {
  if (typeof seed === 'string') {
    const normalized = seed.normalize('NFKC');
    if (normalized.length === 0) throw new KdfError('seed must not be empty');
    return Buffer.from(normalized, 'utf8');
  }
  if (!(seed instanceof Uint8Array)) throw new KdfError('seed must be a string or Uint8Array');
  if (seed.byteLength === 0) throw new KdfError('seed must not be empty');
  return Buffer.from(seed); // Buffer.from(Uint8Array) copies
}

function isPowerOfTwo(n: number): boolean {
  return Number.isInteger(n) && n >= 2 && n <= 2 ** 30 && (n & (n - 1)) === 0;
}

/** Throws KdfError if the parameters are malformed or outside the policy. */
export function validateKdfParams(
  params: KdfParams,
  policy: KdfPolicy = DEFAULT_KDF_POLICY,
): void {
  const { N, r, p } = params;
  if (!isPowerOfTwo(N)) throw new KdfError('scrypt N must be a power of two');
  if (!Number.isInteger(r) || r < 1 || r > 64) throw new KdfError('scrypt r must be an integer in 1..64');
  if (!Number.isInteger(p) || p < 1) throw new KdfError('scrypt p must be a positive integer');
  if (N < policy.minN) throw new KdfError(`scrypt N is below the minimum of ${policy.minN}`);
  if (p > policy.maxP) throw new KdfError(`scrypt p exceeds the maximum of ${policy.maxP}`);
  if (128 * N * r > policy.maxMemoryBytes) {
    throw new KdfError(`scrypt memory cost exceeds the limit of ${policy.maxMemoryBytes} bytes`);
  }
}

// ---------------------------------------------------------------------------
// Derivation
// ---------------------------------------------------------------------------

/** Stretch the seed into a 32-byte master key. The caller owns (and should wipe) the result. */
export async function deriveMasterKey(
  seed: SeedInput,
  salt: Uint8Array,
  params: KdfParams = DEFAULT_KDF_PARAMS,
  policy: KdfPolicy = DEFAULT_KDF_POLICY,
): Promise<Buffer> {
  validateKdfParams(params, policy);
  if (!(salt instanceof Uint8Array) || salt.byteLength < SALT_LENGTH) {
    throw new KdfError(`salt must be at least ${SALT_LENGTH} bytes`);
  }

  const seedBytes = normalizeSeed(seed);
  const memory = 128 * params.N * params.r;

  return new Promise<Buffer>((resolve, reject) => {
    scrypt(
      seedBytes,
      salt,
      MASTER_KEY_LENGTH,
      { N: params.N, r: params.r, p: params.p, maxmem: memory * 2 },
      (error, key) => {
        zeroize(seedBytes);
        if (error) reject(new KdfError('key derivation failed'));
        else resolve(key);
      },
    );
  });
}

/**
 * Expand the master key into one independent subkey for a single purpose.
 * Distinct purposes give cryptographically independent outputs.
 */
export function deriveSubkey(
  masterKey: Uint8Array,
  purpose: KeyPurpose,
  length: number = SUBKEY_LENGTH,
): Buffer {
  if (!(masterKey instanceof Uint8Array) || masterKey.byteLength < MASTER_KEY_LENGTH) {
    throw new KdfError(`master key must be at least ${MASTER_KEY_LENGTH} bytes`);
  }
  if (!KEY_PURPOSES.includes(purpose)) throw new KdfError('unknown key purpose');
  if (!Number.isInteger(length) || length < 1 || length > MAX_HKDF_LENGTH) {
    throw new KdfError(`subkey length must be an integer in 1..${MAX_HKDF_LENGTH}`);
  }
  const info = Buffer.from(`${KDF_CONTEXT}:${purpose}`, 'utf8');
  // The master key is already uniformly random (scrypt output), so an empty HKDF salt is fine.
  return Buffer.from(hkdfSync('sha256', masterKey, Buffer.alloc(0), info, length));
}

/**
 * The main entry point: seed + salt -> { engineKey, aeadKey, decoyKey }.
 * The intermediate master key is wiped before returning.
 * Call destroyKeySet() on the result when finished.
 */
export async function deriveKeySet(
  seed: SeedInput,
  salt: Uint8Array,
  params: KdfParams = DEFAULT_KDF_PARAMS,
  policy: KdfPolicy = DEFAULT_KDF_POLICY,
): Promise<KeySet> {
  const masterKey = await deriveMasterKey(seed, salt, params, policy);
  try {
    return {
      engineKey: deriveSubkey(masterKey, 'engine'),
      aeadKey: deriveSubkey(masterKey, 'aead'),
      decoyKey: deriveSubkey(masterKey, 'decoy'),
    };
  } finally {
    zeroize(masterKey);
  }
}
