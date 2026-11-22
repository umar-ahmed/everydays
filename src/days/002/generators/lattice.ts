import {
  Fn,
  If,
  abs,
  clamp,
  cos,
  dot,
  float,
  floor,
  fract,
  instanceIndex,
  int,
  length,
  max,
  min,
  select,
  sign,
  sin,
  sqrt,
  uniform,
  vec2,
  vec4,
} from 'three/tsl'
import type { Node, StorageBufferNode } from 'three/webgpu'
import type { Lattice, Layout } from '../layout.ts'
import type { MapGenerator } from './index.ts'

export const SHAPES = ['circle', 'square', 'hexagon', 'star', 'heart', 'triangle'] as const
export type Shape = (typeof SHAPES)[number]
export const TILINGS = ['square', 'triangular', 'hexagonal'] as const
export type Tiling = (typeof TILINGS)[number]

export type LatticeOptions = { tiling: Tiling; shape: Shape }

type V2 = Node<'vec2'>
type F = Node<'float'>

// 2D signed distance functions after Inigo Quilez, each scaled to fit the unit circle so
// Bubble size reads the same across shapes. +y points away from the viewer.
const circle = (p: V2) => length(p).sub(1)

const square = (p: V2) => {
  const q = abs(p).sub(0.8 - 0.14)
  return length(max(q, 0)).add(min(max(q.x, q.y), 0)).sub(0.14)
}

const hexagon = (p: V2) => {
  const r = 0.86
  const k = vec2(-0.866025404, 0.5)
  const kz = 0.577350269
  const a = abs(p).toVar()
  a.subAssign(k.mul(min(dot(k, a), 0).mul(2)))
  a.subAssign(vec2(clamp(a.x, -kz * r, kz * r), r))
  return length(a).mul(sign(a.y)).sub(0.06)
}

const star = (p: V2) => {
  const r = 1.02
  const rf = 0.52
  const k1 = vec2(0.809016994375, -0.587785252292)
  const k2 = vec2(-0.809016994375, -0.587785252292)
  const a = vec2(abs(p.x), p.y.add(0.06)).toVar()
  a.subAssign(k1.mul(max(dot(k1, a), 0).mul(2)))
  a.subAssign(k2.mul(max(dot(k2, a), 0).mul(2)))
  a.assign(vec2(abs(a.x), a.y.sub(r)))
  const ba = vec2(rf * 0.587785252292, rf * 0.809016994375 - 1)
  const t = clamp(dot(a, ba).div(dot(ba, ba)), 0, r)
  return length(a.sub(ba.mul(t))).mul(sign(a.y.mul(ba.x).sub(a.x.mul(ba.y)))).sub(0.05)
}

const heart = (p: V2) => {
  const s = 0.6
  const a = vec2(abs(p.x).mul(s), p.y.add(0.98).mul(s))
  const lobe = length(a.sub(vec2(0.25, 0.75))).sub(Math.SQRT2 / 4)
  const m = max(a.x.add(a.y), 0).mul(0.5)
  const tip = sqrt(min(dot(a.sub(vec2(0, 1)), a.sub(vec2(0, 1))), dot(a.sub(vec2(m, m)), a.sub(vec2(m, m))))).mul(
    sign(a.x.sub(a.y)),
  )
  return select(a.y.add(a.x).greaterThan(1), lobe, tip).div(s).sub(0.04)
}

const triangle = (p: V2) => {
  // Equilateral, centered on its centroid: circumradius 0.88 plus 0.12 rounding keeps the
  // corners inside the unit circle like every other shape.
  const r = 0.88 * (Math.sqrt(3) / 2)
  const k = Math.sqrt(3)
  const a0 = vec2(abs(p.x).sub(r), p.y.add(r / k))
  const folded = vec2(a0.x.sub(a0.y.mul(k)), a0.x.mul(-k).sub(a0.y)).mul(0.5)
  const a = select(a0.x.add(a0.y.mul(k)).greaterThan(0), folded, a0).toVar()
  a.assign(vec2(a.x.sub(clamp(a.x, -2 * r, 0)), a.y))
  return length(a).mul(sign(a.y)).negate().sub(0.12)
}

const SDFS = [circle, square, hexagon, star, heart, triangle]

/** Honeycomb holes: axial (c - r/2, r) sites whose axial difference is a multiple of 3. */
const isHole = (c: number, r: number) => (((c - Math.floor(r / 2) - r) % 3) + 3) % 3 === 0

/** Row geometry and exact site count per tiling, for sizing the sheet before the map exists. */
export function latticeFor(tiling: Tiling): Lattice {
  if (tiling === 'square') return { rowStep: 1, rowShift: 0, count: (cols, rows) => cols * rows }
  const rowStep = Math.sqrt(3) / 2
  if (tiling === 'triangular') return { rowStep, rowShift: 0.5, count: (cols, rows) => cols * rows }
  return {
    rowStep,
    rowShift: 0.5,
    count: (cols, rows) => {
      let n = 0
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) if (!isHole(c, r)) n++
      return n
    },
  }
}

/**
 * Lattice generator: bubble centers on a square, triangular (offset rows at pitch * sqrt(3)/2,
 * six equidistant neighbours) or honeycomb lattice (the triangular lattice minus every third
 * site, which becomes the hexagonal hole; three neighbours each), clipped to the sheet so only
 * whole bubbles remain. Neighbouring centers are always one pitch apart, so the edge-to-edge
 * gap is size * spacing on every tiling.
 */
export class LatticeGenerator implements MapGenerator<LatticeOptions> {
  readonly compute

  private readonly tiling = uniform(0, 'int')
  private readonly shape = uniform(0, 'int')
  private readonly mapW = uniform(1, 'int')
  private readonly texel = uniform(0.05)
  private readonly half = uniform(vec2(1, 1))
  private readonly radius = uniform(0.5)
  private readonly pitch = uniform(1)
  private readonly rowStep = uniform(1)
  private readonly origin = uniform(vec2(0, 0))
  private readonly counts = uniform(vec2(1, 1))

  constructor(mapRaw: StorageBufferNode<'vec4'>) {
    this.compute = Fn(() => {
      const i = int(instanceIndex)
      const p = vec2(float(i.mod(this.mapW)), float(i.div(this.mapW))).mul(this.texel).sub(this.half)
      const best = float(1e4).toVar()
      const bestId = float(-1).toVar()
      const cover = float(0).toVar()
      const r0 = int(floor(p.y.sub(this.origin.y).div(this.rowStep).add(0.5)))
      for (let dr = -1; dr <= 1; dr++) {
        const r = r0.add(dr)
        const odd = r.sub(r.div(2).mul(2)).equal(1)
        const shift = select(this.tiling.greaterThan(0).and(odd), this.pitch.mul(0.5), float(0))
        const c0 = int(floor(p.x.sub(this.origin.x).sub(shift).div(this.pitch).add(0.5)))
        for (let dc = -1; dc <= 1; dc++) {
          const c = c0.add(dc)
          const inRange = r
            .greaterThanEqual(0)
            .and(r.lessThan(int(this.counts.y)))
            .and(c.greaterThanEqual(0))
            .and(c.lessThan(int(this.counts.x)))
          // Honeycomb: axial (c - r/2, r); sites with (axial difference) % 3 == 0 are the holes.
          const diff = c.sub(r.div(2)).sub(r)
          const sub = diff.sub(diff.div(3).mul(3)).add(3).mod(3)
          const valid = inRange.and(this.tiling.notEqual(2).or(sub.notEqual(0)))
          const center = vec2(this.origin.x.add(float(c).mul(this.pitch)).add(shift), this.origin.y.add(float(r).mul(this.rowStep)))
          // Hexagons sit flat-to-flat on the triangular lattice; honeycomb triangles alternate.
          const angle = select(
            this.shape.equal(2).and(this.tiling.equal(1)),
            float(Math.PI / 6),
            select(this.shape.equal(5).and(this.tiling.equal(2)).and(sub.equal(2)), float(Math.PI), float(0)),
          )
          const d = p.sub(center)
          const cs = cos(angle)
          const sn = sin(angle)
          const local = vec2(d.x.mul(cs).sub(d.y.mul(sn)), d.x.mul(sn).add(d.y.mul(cs)).negate()).div(this.radius)
          const sd = float(1e4).toVar()
          SDFS.forEach((f, k) => {
            If(this.shape.equal(k), () => {
              sd.assign((f as (p: V2) => F)(local).mul(this.radius))
            })
          })
          If(valid, () => {
            If(sd.lessThan(0), () => {
              cover.addAssign(1)
            })
            If(sd.lessThan(best), () => {
              best.assign(sd)
              bestId.assign(float(r.mul(1024).add(c)))
            })
          })
        }
      }
      const seed = fract(sin(bestId.mul(12.9898).add(1.7)).mul(43758.5453))
      mapRaw.element(i).assign(vec4(bestId, best, seed, cover))
    })().compute(1)
  }

  configure(layout: Layout, { tiling, shape }: LatticeOptions) {
    const { pitch, size, width, depth, cols, rows, rowStep, rowShift } = layout
    // Center the lattice on the sheet; the layout already sized the sheet to fit cols x rows.
    const spanX = (cols - 1) * pitch + (rows > 1 ? rowShift : 0)
    const spanZ = (rows - 1) * rowStep
    this.tiling.value = TILINGS.indexOf(tiling)
    this.shape.value = SHAPES.indexOf(shape)
    this.mapW.value = layout.mapW
    this.texel.value = layout.texel
    this.half.value.set(width / 2, depth / 2)
    this.radius.value = size / 2
    this.pitch.value = pitch
    this.rowStep.value = rowStep
    this.origin.value.set(-spanX / 2, -spanZ / 2)
    this.counts.value.set(cols, rows)
    this.compute.count = layout.mapW * layout.mapH
  }
}
