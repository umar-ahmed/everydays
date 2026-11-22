import { sdf, type Shape } from './shapes.ts'

/** Grid segments per bubble cell. The GPU kernels loop over exactly this many quads per cell. */
export const CELL = 12
export const MAX_COLS = 20
export const MAX_ROWS = 14
export const MAX_W = MAX_COLS * CELL + 1
export const MAX_H = MAX_ROWS * CELL + 1
export const MAX_VERTS = MAX_W * MAX_H
export const MAX_BUBBLES = MAX_COLS * MAX_ROWS

export type SheetConfig = { shape: Shape; size: number; spacing: number; cols: number; pinholeChance: number }

export type Sheet = {
  cols: number
  rows: number
  W: number
  H: number
  /** Rest spacing between neighbouring particles (cm). */
  h: number
  /** Bubble footprint diameter and dome height (cm). */
  size: number
  domeHeight: number
  width: number
  depth: number
  /** vec4 per vertex: rest xyz, domeness 0..1 (thin-film and bending use it). */
  rest: Float32Array
  /** vec4 per vertex: bubble id, inside footprint (0/1). */
  meta: Float32Array
  /** vec4 per bubble: popped, burst gauge, burst time, rest volume. */
  bubbles: Float32Array
  /** 1 for bubbles born with a pinhole. Rolled again for every new sheet. */
  pinholes: Float32Array
  index: Uint32Array
}

export const rowsFor = (cols: number) => Math.max(2, Math.round(cols * 0.7))

export function buildSheet({ shape, size, spacing, cols, pinholeChance }: SheetConfig): Sheet {
  cols = Math.min(Math.max(Math.round(cols), 2), MAX_COLS)
  const rows = Math.min(rowsFor(cols), MAX_ROWS)
  const W = cols * CELL + 1
  const H = rows * CELL + 1
  const cellSize = size * (1 + spacing)
  const h = cellSize / CELL
  const f = sdf[shape]

  // Deepest inside distance over the footprint, so every shape's dome peaks at t = 1.
  let deepest = 0
  for (let j = 0; j <= CELL; j++) {
    for (let i = 0; i <= CELL; i++) {
      const u = ((i - CELL / 2) * h) / (size / 2)
      const v = ((j - CELL / 2) * h) / (size / 2)
      deepest = Math.max(deepest, -f(u, -v))
    }
  }
  // Real bubbles stand about 0.4x their diameter; narrow shapes come out a little lower.
  const domeHeight = size * 0.38 * Math.min(1, 0.55 + 0.45 * deepest)

  const rest = new Float32Array(W * H * 4)
  const meta = new Float32Array(W * H * 4)
  for (let iy = 0; iy < H; iy++) {
    for (let ix = 0; ix < W; ix++) {
      const k = (iy * W + ix) * 4
      const cx = Math.min(Math.floor(ix / CELL), cols - 1)
      const cy = Math.min(Math.floor(iy / CELL), rows - 1)
      const u = ((ix - cx * CELL - CELL / 2) * h) / (size / 2)
      const v = ((iy - cy * CELL - CELL / 2) * h) / (size / 2)
      const s = f(u, -v)
      const t = Math.min(Math.max(-s / Math.max(deepest, 1e-3), 0), 1)
      // Spherical-cap-like profile with steep walls at the rim, like a thermoformed bubble.
      const y = s < 0 ? domeHeight * Math.sqrt(1 - (1 - t) * (1 - t)) : 0
      rest[k] = (ix - (W - 1) / 2) * h
      rest[k + 1] = y
      rest[k + 2] = (iy - (H - 1) / 2) * h
      rest[k + 3] = t
      meta[k] = cy * cols + cx
      meta[k + 1] = s < 0 ? 1 : 0
    }
  }

  const bubbles = new Float32Array(cols * rows * 4)
  for (let b = 0; b < cols * rows; b++) {
    const cx = b % cols
    const cy = Math.floor(b / cols)
    let volume = 0
    for (let q = 0; q < CELL * CELL; q++) {
      const ix = cx * CELL + (q % CELL)
      const iy = cy * CELL + Math.floor(q / CELL)
      const i0 = (iy * W + ix) * 4
      const i1 = i0 + 4
      const i2 = i0 + (W + 1) * 4
      const i3 = i0 + W * 4
      const ax = rest[i2] - rest[i0]
      const az = rest[i2 + 2] - rest[i0 + 2]
      const bx = rest[i3] - rest[i1]
      const bz = rest[i3 + 2] - rest[i1 + 2]
      const area = 0.5 * (ax * bz - az * bx)
      volume += (area * (rest[i0 + 1] + rest[i1 + 1] + rest[i2 + 1] + rest[i3 + 1])) / 4
    }
    bubbles[b * 4 + 3] = volume
  }

  const pinholes = new Float32Array(cols * rows).map(() => (Math.random() < pinholeChance ? 1 : 0))

  const index = new Uint32Array((W - 1) * (H - 1) * 6)
  let n = 0
  for (let iy = 0; iy < H - 1; iy++) {
    for (let ix = 0; ix < W - 1; ix++) {
      const a = iy * W + ix
      const b = a + 1
      const c = a + W + 1
      const d = a + W
      // Alternate the diagonal so domes triangulate symmetrically.
      if ((ix + iy) % 2 === 0) index.set([a, d, c, a, c, b], n)
      else index.set([a, d, b, b, d, c], n)
      n += 6
    }
  }

  return {
    cols,
    rows,
    W,
    H,
    h,
    size,
    domeHeight,
    width: cols * cellSize,
    depth: rows * cellSize,
    rest,
    meta,
    bubbles,
    pinholes,
    index,
  }
}
