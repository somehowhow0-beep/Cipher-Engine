/**
 * generators/matrix.ts: seed-driven invertible matrices over GF(2^8).
 *
 * Diffusion building block: multiplying a vector of bytes by an invertible
 * matrix makes every output byte depend on several input bytes, and the inverse
 * matrix undoes it exactly.
 *
 * Quality gate (default on): the matrix must be MDS (maximum distance
 * separable), meaning EVERY square submatrix is non-singular. For an n x n MDS
 * matrix, changing t >= 1 input bytes changes at least n + 1 - t output bytes,
 * the best possible diffusion (AES MixColumns is a 4x4 MDS matrix).
 *
 * Measured search cost over random draws (average attempts): n=4 about 1.4,
 * n=5 about 3, n=6 about 21, while n=7 failed every run at 20,000 attempts
 * (there are 3,431 square submatrices to keep non-singular). So MDS generation
 * is supported up to size 6; larger blocks should be handled as several small
 * MDS matrices combined with byte permutations.
 */

import { gfInv, gfMul } from './gf256';
import type { SeededPrng } from '../prng';

export const MIN_MATRIX_SIZE = 2;
export const MAX_MATRIX_SIZE = 8;
const MAX_MDS_CHECK_SIZE = 8;
/** Largest size for which a random search reliably finds an MDS matrix (see module comment). */
export const MAX_MDS_SEARCH_SIZE = 6;

export interface GfMatrix {
  readonly n: number;
  /** Row-major, n * n bytes. */
  readonly data: Uint8Array;
}

export interface MatrixPair {
  readonly forward: GfMatrix;
  readonly inverse: GfMatrix;
  /** How many candidates were drawn before one passed. */
  readonly attempts: number;
}

export interface MatrixOptions {
  /** Require the MDS property (default true). */
  readonly mds?: boolean;
  readonly maxAttempts?: number;
}

export class MatrixError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'MatrixError';
  }
}

export function makeMatrix(n: number, entries: ArrayLike<number>): GfMatrix {
  if (!Number.isInteger(n) || n < 1 || n > 16) throw new MatrixError('invalid matrix size');
  if (entries.length !== n * n) throw new MatrixError(`expected ${n * n} entries`);
  const data = new Uint8Array(n * n);
  for (let i = 0; i < data.length; i++) {
    const v = entries[i] as number;
    if (!Number.isInteger(v) || v < 0 || v > 255) throw new MatrixError('entries must be bytes');
    data[i] = v;
  }
  return { n, data };
}

export function identityMatrix(n: number): GfMatrix {
  const data = new Uint8Array(n * n);
  for (let i = 0; i < n; i++) data[i * n + i] = 1;
  return { n, data };
}

export function matricesEqual(a: GfMatrix, b: GfMatrix): boolean {
  return a.n === b.n && a.data.every((v, i) => v === b.data[i]);
}

/** Matrix product a * b. */
export function multiplyMatrices(a: GfMatrix, b: GfMatrix): GfMatrix {
  if (a.n !== b.n) throw new MatrixError('matrix sizes differ');
  const n = a.n;
  const out = new Uint8Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let acc = 0;
      for (let k = 0; k < n; k++) {
        acc ^= gfMul(a.data[i * n + k] as number, b.data[k * n + j] as number);
      }
      out[i * n + j] = acc;
    }
  }
  return { n, data: out };
}

/** Matrix-vector product m * v. */
export function multiplyVector(m: GfMatrix, v: Uint8Array): Uint8Array {
  if (v.length !== m.n) throw new MatrixError(`vector length ${v.length} does not match matrix size ${m.n}`);
  const out = new Uint8Array(m.n);
  for (let i = 0; i < m.n; i++) {
    let acc = 0;
    for (let k = 0; k < m.n; k++) acc ^= gfMul(m.data[i * m.n + k] as number, v[k] as number);
    out[i] = acc;
  }
  return out;
}

/** Gauss-Jordan inverse. Returns null if the matrix is singular. */
export function invertMatrix(m: GfMatrix): GfMatrix | null {
  const n = m.n;
  const a = Uint8Array.from(m.data);
  const inv = identityMatrix(n).data;
  for (let col = 0; col < n; col++) {
    let pivot = -1;
    for (let r = col; r < n; r++) {
      if (a[r * n + col] !== 0) {
        pivot = r;
        break;
      }
    }
    if (pivot < 0) return null;
    if (pivot !== col) {
      for (let c = 0; c < n; c++) {
        const t1 = a[col * n + c] as number;
        a[col * n + c] = a[pivot * n + c] as number;
        a[pivot * n + c] = t1;
        const t2 = inv[col * n + c] as number;
        inv[col * n + c] = inv[pivot * n + c] as number;
        inv[pivot * n + c] = t2;
      }
    }
    const scale = gfInv(a[col * n + col] as number);
    for (let c = 0; c < n; c++) {
      a[col * n + c] = gfMul(a[col * n + c] as number, scale);
      inv[col * n + c] = gfMul(inv[col * n + c] as number, scale);
    }
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = a[r * n + col] as number;
      if (factor === 0) continue;
      for (let c = 0; c < n; c++) {
        a[r * n + c] = (a[r * n + c] as number) ^ gfMul(factor, a[col * n + c] as number);
        inv[r * n + c] = (inv[r * n + c] as number) ^ gfMul(factor, inv[col * n + c] as number);
      }
    }
  }
  return { n, data: inv };
}

function* subsets(n: number, k: number): Generator<number[]> {
  const idx = Array.from({ length: k }, (_, i) => i);
  while (true) {
    yield [...idx];
    let i = k - 1;
    while (i >= 0 && idx[i] === n - k + i) i--;
    if (i < 0) return;
    idx[i] = (idx[i] as number) + 1;
    for (let j = i + 1; j < k; j++) idx[j] = (idx[j - 1] as number) + 1;
  }
}

function submatrixInvertible(m: GfMatrix, rows: number[], cols: number[]): boolean {
  const k = rows.length;
  const sub = new Uint8Array(k * k);
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      sub[i * k + j] = m.data[(rows[i] as number) * m.n + (cols[j] as number)] as number;
    }
  }
  return invertMatrix({ n: k, data: sub }) !== null;
}

/** True when every square submatrix of every size is non-singular (the MDS property). */
export function isMds(m: GfMatrix): boolean {
  if (m.n > MAX_MDS_CHECK_SIZE) throw new MatrixError(`MDS check supports sizes up to ${MAX_MDS_CHECK_SIZE}`);
  for (let k = 1; k <= m.n; k++) {
    for (const rows of subsets(m.n, k)) {
      for (const cols of subsets(m.n, k)) {
        if (!submatrixInvertible(m, rows, cols)) return false;
      }
    }
  }
  return true;
}

/** Draw matrices from `prng` until one is invertible (and MDS unless disabled). */
export function generateMatrix(prng: SeededPrng, n: number, options: MatrixOptions = {}): MatrixPair {
  if (!Number.isInteger(n) || n < MIN_MATRIX_SIZE || n > MAX_MATRIX_SIZE) {
    throw new MatrixError(`size must be an integer in ${MIN_MATRIX_SIZE}..${MAX_MATRIX_SIZE}`);
  }
  const requireMds = options.mds ?? true;
  if (requireMds && n > MAX_MDS_SEARCH_SIZE) {
    throw new MatrixError(
      `MDS search is only practical up to size ${MAX_MDS_SEARCH_SIZE}; use smaller blocks or pass { mds: false }`,
    );
  }
  const maxAttempts = options.maxAttempts ?? 20_000;
  if (!Number.isInteger(maxAttempts) || maxAttempts < 1 || maxAttempts > 10_000_000) {
    throw new MatrixError('maxAttempts must be an integer in 1..10000000');
  }
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const forward: GfMatrix = { n, data: Uint8Array.from(prng.nextBytes(n * n)) };
    if (requireMds ? !isMds(forward) : forward.data.includes(0)) continue;
    const inverse = invertMatrix(forward);
    if (inverse === null) continue;
    return { forward, inverse, attempts: attempt };
  }
  throw new MatrixError(`no matrix met the requirements within ${maxAttempts} attempts`);
}
