/**
 * Sheet rectangle, cloth grid and bubble-map grid. Nothing here knows about lattices or shapes:
 * a generator fills the map, and the sheet is built from whatever the map says.
 */

/** Cloth grid segments per bubble pitch (center-to-center distance of neighbouring bubbles). */
export const CELL = 12
export const MAX_COLS = 20
export const MAX_ROWS = 14
export const MAX_W = MAX_COLS * CELL + 1
export const MAX_H = MAX_ROWS * CELL + 1
export const MAX_VERTS = MAX_W * MAX_H
export const MAX_QUADS = (MAX_W - 1) * (MAX_H - 1)
/** Triangular packing fits ~15% more bubbles than square in the same rectangle; leave headroom. */
export const MAX_BUBBLES = 512
/** Map texels per cloth segment. Texel (MAP_SCALE * ix, MAP_SCALE * iy) sits exactly on vertex (ix, iy). */
export const MAP_SCALE = 2
export const MAX_MAP_W = (MAX_W - 1) * MAP_SCALE + 1
export const MAX_MAP_H = (MAX_H - 1) * MAP_SCALE + 1
export const MAX_MAP = MAX_MAP_W * MAX_MAP_H

export type LayoutConfig = { size: number; spacing: number; cols: number }

export type Layout = {
  cols: number
  rows: number
  /** Bubble footprint diameter and center-to-center pitch (cm); edge gap = size * spacing. */
  size: number
  pitch: number
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

export const rowsFor = (cols: number) => Math.max(2, Math.round(cols * 0.7))

export function layoutFor({ size, spacing, cols }: LayoutConfig): Layout {
  cols = Math.min(Math.max(Math.round(cols), 2), MAX_COLS)
  const rows = Math.min(rowsFor(cols), MAX_ROWS)
  const pitch = size * (1 + spacing)
  const W = cols * CELL + 1
  const H = rows * CELL + 1
  const h = pitch / CELL
  return {
    cols,
    rows,
    size,
    pitch,
    W,
    H,
    h,
    width: cols * pitch,
    depth: rows * pitch,
    mapW: (W - 1) * MAP_SCALE + 1,
    mapH: (H - 1) * MAP_SCALE + 1,
    texel: h / MAP_SCALE,
  }
}
