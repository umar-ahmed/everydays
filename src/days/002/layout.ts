/**
 * Sheet rectangle, cloth grid and bubble-map grid. Nothing here knows about shapes; the lattice
 * only contributes its row geometry and site count. A generator fills the map, and the sheet is
 * built from whatever the map says.
 */

/** Cloth grid segments per bubble pitch (center-to-center distance of neighbouring bubbles). */
export const CELL = 12
/**
 * Capacity: every buffer is sized for a 20 x 14 pitch grid. Sheets may take any shape whose
 * totals fit (buffers are linear), so a tall phone sheet uses the same storage as a wide one.
 */
const CAP_W = 20 * CELL + 1
const CAP_H = 14 * CELL + 1
export const MAX_VERTS = CAP_W * CAP_H
export const MAX_QUADS = (CAP_W - 1) * (CAP_H - 1)
/** Triangular packing fits ~15% more bubbles than square in the same rectangle; leave headroom. */
export const MAX_BUBBLES = 512
/** Map texels per cloth segment. Texel (MAP_SCALE * ix, MAP_SCALE * iy) sits exactly on vertex (ix, iy). */
export const MAP_SCALE = 2
const mapSize = (W: number, H: number) => ((W - 1) * MAP_SCALE + 1) * ((H - 1) * MAP_SCALE + 1)
export const MAX_MAP = mapSize(CAP_W, CAP_H)

/** Render vertices per sim segment, the Render detail dial. */
export const DETAILS = [1, 2, 4] as const
export const MAX_DETAIL = 4
export const renderSize = (W: number, H: number, detail: number) => ({ RW: (W - 1) * detail + 1, RH: (H - 1) * detail + 1 })
const CAP_RENDER = renderSize(CAP_W, CAP_H, MAX_DETAIL)
export const MAX_RENDER_VERTS = CAP_RENDER.RW * CAP_RENDER.RH

/** Per-device limits on top of the buffer capacity. */
export type Budget = { verts: number; bubbles: number }
export const FULL_BUDGET: Budget = { verts: MAX_VERTS, bubbles: MAX_BUBBLES }
/** Phones: half the cloth and a third of the bubbles keep the solver and readbacks cheap. */
export const PHONE_BUDGET: Budget = { verts: 20000, bubbles: 180 }

/** Row geometry of a lattice, in pitches: rows rowStep apart, odd rows shifted by rowShift. */
export type Lattice = { rowStep: number; rowShift: number; count: (cols: number, rows: number) => number }

export type LayoutConfig = {
  size: number
  spacing: number
  /** Bubbles along the sheet's shorter side. */
  across: number
  /** Sheet width / depth. */
  aspect: number
  lattice: Lattice
  budget?: Budget
}

export type Layout = {
  /** Lattice sites: cols per row and rows; `across` is how many of the requested bubbles across the budget allowed. */
  cols: number
  rows: number
  across: number
  /** Lattice sites (bubbles) the layout holds. */
  sites: number
  /** Bubble footprint diameter and center-to-center pitch (cm); edge gap = size * spacing. */
  size: number
  pitch: number
  rowStep: number
  rowShift: number
  /** Cloth grid: W x H vertices, h cm apart. */
  W: number
  H: number
  h: number
  width: number
  depth: number
  /** Bubble map: mapW x mapH texels, texel cm apart, texel (0, 0) at the sheet's (-width/2, -depth/2) corner. */
  mapW: number
  mapH: number
  texel: number
}

export const fitsCapacity = (W: number, H: number) =>
  W * H <= MAX_VERTS &&
  (W - 1) * (H - 1) <= MAX_QUADS &&
  mapSize(W, H) <= MAX_MAP &&
  renderSize(W, H, MAX_DETAIL).RW * renderSize(W, H, MAX_DETAIL).RH <= MAX_RENDER_VERTS

/**
 * Sizes the sheet to `aspect` with `across` whole bubbles on the shorter side, keeping bubble
 * size and spacing in world units. The longer side takes as many whole columns or rows as fit.
 * If the result exceeds the budget or any buffer, the density drops until it fits.
 */
export function layoutFor({ size, spacing, across, aspect, lattice, budget = FULL_BUDGET }: LayoutConfig): Layout {
  const pitch = size * (1 + spacing)
  const h = pitch / CELL
  const texel = h / MAP_SCALE
  // Outermost centers sit half a pitch in from the edge (at least a footprint radius + 2 texels).
  const edge = Math.max(pitch / 2, size / 2 + 2 * texel)
  const rowStep = lattice.rowStep * pitch
  const rowShift = lattice.rowShift * pitch
  const spanX = (cols: number, rows: number) => (cols - 1) * pitch + (rows > 1 ? rowShift : 0) + 2 * edge
  const spanZ = (rows: number) => (rows - 1) * rowStep + 2 * edge
  let fallback: Layout | null = null
  for (let n = Math.max(2, Math.round(across)); n >= 2; n--) {
    let cols: number
    let rows: number
    let width: number
    let depth: number
    if (aspect >= 1) {
      rows = n
      depth = spanZ(rows)
      cols = Math.max(2, Math.floor((depth * aspect - spanX(1, rows)) / pitch) + 1)
      width = Math.max(depth * aspect, spanX(cols, rows))
    } else {
      cols = n
      width = spanX(cols, 2)
      rows = Math.max(2, Math.floor((width / aspect - spanZ(1)) / rowStep) + 1)
      depth = Math.max(width / aspect, spanZ(rows))
    }
    const W = Math.ceil(width / h - 1e-6) + 1
    const H = Math.ceil(depth / h - 1e-6) + 1
    const layout: Layout = {
      cols,
      rows,
      across: n,
      sites: lattice.count(cols, rows),
      size,
      pitch,
      rowStep,
      rowShift,
      W,
      H,
      h,
      width: (W - 1) * h,
      depth: (H - 1) * h,
      mapW: (W - 1) * MAP_SCALE + 1,
      mapH: (H - 1) * MAP_SCALE + 1,
      texel,
    }
    fallback = layout
    if (fitsCapacity(W, H) && W * H <= budget.verts && layout.sites <= Math.min(budget.bubbles, MAX_BUBBLES)) return layout
  }
  // Even two bubbles across overflow (an extreme aspect): shorten the long side instead.
  const l = fallback!
  let { cols, rows } = l
  while (cols * rows > 4) {
    if (cols >= rows) cols--
    else rows--
    const W = Math.ceil(spanX(cols, rows) / h - 1e-6) + 1
    const H = Math.ceil(spanZ(rows) / h - 1e-6) + 1
    if (fitsCapacity(W, H) && W * H <= budget.verts && lattice.count(cols, rows) <= budget.bubbles) {
      return {
        ...l,
        cols,
        rows,
        sites: lattice.count(cols, rows),
        W,
        H,
        width: (W - 1) * h,
        depth: (H - 1) * h,
        mapW: (W - 1) * MAP_SCALE + 1,
        mapH: (H - 1) * MAP_SCALE + 1,
      }
    }
  }
  return l
}
