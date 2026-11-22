import type { Layout } from './layout.ts'

export * from './layout.ts'

/** Everything the sim and renderer need, derived from a bubble map alone. */
export type Sheet = Layout & {
  /** Number of bubbles; ids are dense, 0..count-1. */
  count: number
  /** Tallest dome (cm), for picking and the lifted finger. */
  domeHeight: number
  /** vec4 per vertex: rest xyz, domeness 0..1 (thin-film and bending use it). */
  rest: Float32Array
  /** vec4 per vertex: rest normal xyz, w = 0 (no compression). */
  restNormal: Float32Array
  /** vec4 per vertex: bubble id, inside footprint (0/1). */
  meta: Float32Array
  /** vec4 per bubble: popped, burst gauge, burst time, rest volume. */
  bubbles: Float32Array
  /** vec4 per bubble: center x, center z, inside depth (inradius), equivalent diameter (cm). */
  centers: Float32Array
  /** 1 for bubbles born with a pinhole. Rolled again for every new sheet. */
  pinholes: Float32Array
  /** vec2 per bubble: (first, count) into quadList. The volume kernel sums exactly these quads. */
  quadRange: Float32Array
  /** Quad indices (iy * (W - 1) + ix), grouped by bubble. */
  quadList: Float32Array
  /** The canonical map: vec4 per texel with dense id (or -1), signed distance, seed, coverage. */
  map: Float32Array
  index: Uint32Array
}

export type SheetBuild = { layout: Layout; raw: Float32Array; pinholeChance: number }

/**
 * Builds the cloth from a generator's raw map (see generators/index.ts). Runs once per rebuild
 * on the readback; nothing here assumes a lattice.
 */
export function buildSheetFromMap({ layout, raw, pinholeChance }: SheetBuild): Sheet {
  const { W, H, h, mapW, mapH, texel, width, depth } = layout
  const texels = mapW * mapH

  // Dense ids in order of first appearance; only bubbles that actually cover a texel count.
  const dense = new Map<number, number>()
  for (let t = 0; t < texels; t++) {
    const id = raw[t * 4]
    if (id >= 0 && raw[t * 4 + 1] < 0 && !dense.has(id)) dense.set(id, dense.size)
  }
  const count = dense.size
  const map = new Float32Array(texels * 4)
  const area = new Float64Array(count)
  const cx = new Float64Array(count)
  const cz = new Float64Array(count)
  const deepest = new Float64Array(count)
  for (let t = 0; t < texels; t++) {
    const id = raw[t * 4]
    const sd = raw[t * 4 + 1]
    const b = id >= 0 ? (dense.get(id) ?? -1) : -1
    map[t * 4] = b
    map[t * 4 + 1] = b >= 0 ? sd : Math.max(sd, 0)
    map[t * 4 + 2] = raw[t * 4 + 2]
    map[t * 4 + 3] = raw[t * 4 + 3]
    if (b >= 0 && sd < 0) {
      area[b]++
      cx[b] += (t % mapW) * texel - width / 2
      cz[b] += Math.floor(t / mapW) * texel - depth / 2
      deepest[b] = Math.max(deepest[b], -sd)
    }
  }

  const centers = new Float32Array(count * 4)
  const domeH = new Float64Array(count)
  let domeHeight = 0
  for (let b = 0; b < count; b++) {
    const rEq = Math.sqrt((area[b] * texel * texel) / Math.PI)
    centers[b * 4] = cx[b] / area[b]
    centers[b * 4 + 1] = cz[b] / area[b]
    centers[b * 4 + 2] = deepest[b]
    centers[b * 4 + 3] = 2 * rEq
    // Real bubbles stand about 0.4x their diameter; narrow footprints come out a little lower.
    domeH[b] = 0.76 * rEq * Math.min(1, 0.55 + (0.45 * deepest[b]) / Math.max(rEq, 1e-6))
    domeHeight = Math.max(domeHeight, domeH[b])
  }

  // Dome height from the distance to the edge: a spherical cap with steep walls at the rim.
  // Easing the rim in (a soft foot) makes the base a shallow skirt that flattens sideways
  // instead of compressing the gas, and non-circular bubbles stop popping; rim smoothing is a
  // render-side job.
  const rest = new Float32Array(W * H * 4)
  const meta = new Float32Array(W * H * 4)
  for (let iy = 0; iy < H; iy++) {
    for (let ix = 0; ix < W; ix++) {
      const k = (iy * W + ix) * 4
      const t = (iy * 2 * mapW + ix * 2) * 4
      const b = map[t]
      const sd = map[t + 1]
      const inside = b >= 0 && sd < 0
      let y = 0
      let dome = 0
      if (inside) {
        const d = -sd
        const u = Math.min(d / Math.max(deepest[b], 1e-6), 1)
        y = domeH[b] * Math.sqrt(1 - (1 - u) * (1 - u))
        dome = u
      }
      rest[k] = ix * h - width / 2
      rest[k + 1] = y
      rest[k + 2] = iy * h - depth / 2
      rest[k + 3] = dome
      meta[k] = inside ? b : 0
      meta[k + 1] = inside ? 1 : 0
    }
  }

  // Each quad belongs to the bubble of its first inside corner; flat quads belong to none.
  const owner = new Int32Array((W - 1) * (H - 1)).fill(-1)
  const perBubble = new Int32Array(count)
  for (let iy = 0; iy < H - 1; iy++) {
    for (let ix = 0; ix < W - 1; ix++) {
      const corners = [iy * W + ix, iy * W + ix + 1, (iy + 1) * W + ix + 1, (iy + 1) * W + ix]
      const c = corners.find((v) => meta[v * 4 + 1] > 0)
      if (c === undefined) continue
      const b = meta[c * 4]
      owner[iy * (W - 1) + ix] = b
      perBubble[b]++
    }
  }
  const quadRange = new Float32Array(count * 2)
  let offset = 0
  for (let b = 0; b < count; b++) {
    quadRange[b * 2] = offset
    offset += perBubble[b]
  }
  const quadList = new Float32Array(offset)
  const fill = new Int32Array(count)
  for (let q = 0; q < owner.length; q++) {
    const b = owner[q]
    if (b < 0) continue
    quadList[quadRange[b * 2] + fill[b]++] = q
  }
  for (let b = 0; b < count; b++) quadRange[b * 2 + 1] = fill[b]

  // Rest volume over each bubble's quads: the same sum the GPU kernel does every substep.
  const bubbles = new Float32Array(count * 4)
  for (let b = 0; b < count; b++) {
    let volume = 0
    for (let n = 0; n < quadRange[b * 2 + 1]; n++) {
      const q = quadList[quadRange[b * 2] + n]
      volume += quadVolume(rest, W, q % (W - 1), Math.floor(q / (W - 1)))
    }
    bubbles[b * 4 + 3] = volume
  }

  const pinholes = new Float32Array(count).map(() => (Math.random() < pinholeChance ? 1 : 0))

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
    ...layout,
    count,
    domeHeight,
    rest,
    restNormal: restNormals(rest, W, H),
    meta,
    bubbles,
    centers,
    pinholes,
    quadRange,
    quadList,
    map,
    index,
  }
}

/** Projected quad area times its mean height; matches the GPU bubble kernel. */
export function quadVolume(rest: Float32Array, W: number, ix: number, iy: number) {
  const i0 = (iy * W + ix) * 4
  const i1 = i0 + 4
  const i2 = i0 + (W + 1) * 4
  const i3 = i0 + W * 4
  const ax = rest[i2] - rest[i0]
  const az = rest[i2 + 2] - rest[i0 + 2]
  const bx = rest[i3] - rest[i1]
  const bz = rest[i3 + 2] - rest[i1 + 2]
  const area = 0.5 * (ax * bz - az * bx)
  return (area * (rest[i0 + 1] + rest[i1 + 1] + rest[i2 + 1] + rest[i3 + 1])) / 4
}

/** CPU twin of the normals kernel, run on the rest positions. */
function restNormals(rest: Float32Array, W: number, H: number) {
  const out = new Float32Array(W * H * 4)
  const at = (ix: number, iy: number, c: number) => rest[(iy * W + ix) * 4 + c]
  for (let iy = 0; iy < H; iy++) {
    for (let ix = 0; ix < W; ix++) {
      const l = Math.max(ix - 1, 0)
      const r = Math.min(ix + 1, W - 1)
      const u = Math.max(iy - 1, 0)
      const d = Math.min(iy + 1, H - 1)
      const ax = at(ix, d, 0) - at(ix, u, 0)
      const ay = at(ix, d, 1) - at(ix, u, 1)
      const az = at(ix, d, 2) - at(ix, u, 2)
      const bx = at(r, iy, 0) - at(l, iy, 0)
      const by = at(r, iy, 1) - at(l, iy, 1)
      const bz = at(r, iy, 2) - at(l, iy, 2)
      const nx = ay * bz - az * by
      const ny = az * bx - ax * bz
      const nz = ax * by - ay * bx
      const len = Math.hypot(nx, ny, nz) || 1
      const k = (iy * W + ix) * 4
      out[k] = nx / len
      out[k + 1] = ny / len
      out[k + 2] = nz / len
    }
  }
  return out
}
