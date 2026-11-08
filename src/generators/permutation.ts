/**
 * generators/permutation.ts: seed-driven byte-position permutations.
 *
 * Diffusion building block: moves bytes to different positions inside a block.
 * The forward table is a seeded Fisher-Yates shuffle and the inverse table is
 * computed alongside it, so decryption can undo exactly what encryption did.
 *
 * Convention (gather): out[i] = data[forward[i]].
 * Undo:                out[forward[i]] = data[i].
 */

import type { SeededPrng } from '../prng';

export const MAX_PERMUTATION_SIZE = 65_536; // positions fit in Uint16Array

export interface BytePermutation {
  readonly size: number;
  readonly forward: Uint16Array;
  readonly inverse: Uint16Array;
}

export class PermutationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'PermutationError';
  }
}

export interface PermutationOptions {
  /** Reject permutations with any fixed point (position i mapped to itself). Default true. */
  readonly derangement?: boolean;
}

const MAX_ATTEMPTS = 1000;

export function isPermutation(table: ArrayLike<number>, size: number = table.length): boolean {
  if (table.length !== size) return false;
  const seen = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    const v = table[i] as number;
    if (!Number.isInteger(v) || v < 0 || v >= size || seen[v] === 1) return false;
    seen[v] = 1;
  }
  return true;
}

export function invertPermutation(table: ArrayLike<number>): Uint16Array {
  if (!isPermutation(table)) throw new PermutationError('table is not a permutation');
  const inverse = new Uint16Array(table.length);
  for (let i = 0; i < table.length; i++) inverse[table[i] as number] = i;
  return inverse;
}

/**
 * Generate a permutation of `size` positions. By default fixed points are
 * rejected (a derangement), so every byte actually moves; this needs size >= 2.
 */
export function generatePermutation(
  prng: SeededPrng,
  size: number,
  options: PermutationOptions = {},
): BytePermutation {
  if (!Number.isInteger(size) || size < 1 || size > MAX_PERMUTATION_SIZE) {
    throw new PermutationError(`size must be an integer in 1..${MAX_PERMUTATION_SIZE}`);
  }
  const derangement = options.derangement ?? true;
  if (derangement && size < 2) {
    throw new PermutationError('a derangement needs size >= 2');
  }
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    const forward = Uint16Array.from(prng.permutation(size));
    if (derangement && forward.some((v, i) => v === i)) continue;
    return { size, forward, inverse: invertPermutation(forward) };
  }
  throw new PermutationError('could not draw a valid permutation');
}

function assertLength(perm: BytePermutation, data: Uint8Array): void {
  if (data.length !== perm.size) {
    throw new PermutationError(`data length ${data.length} does not match permutation size ${perm.size}`);
  }
}

/** out[i] = data[forward[i]] */
export function permute(perm: BytePermutation, data: Uint8Array): Buffer {
  assertLength(perm, data);
  const out = Buffer.allocUnsafe(data.length);
  for (let i = 0; i < data.length; i++) out[i] = data[perm.forward[i] as number] as number;
  return out;
}

/** Exact inverse of permute(). */
export function unpermute(perm: BytePermutation, data: Uint8Array): Buffer {
  assertLength(perm, data);
  const out = Buffer.allocUnsafe(data.length);
  for (let i = 0; i < data.length; i++) out[perm.forward[i] as number] = data[i] as number;
  return out;
}
