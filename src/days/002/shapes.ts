// 2D signed distance functions (after Inigo Quilez) for the bubble footprints.
// Each shape is scaled to fit a unit circle, so `bubble size` reads the same across shapes.
// p.y points away from the viewer (screen up).

export const SHAPES = ['circle', 'square', 'hexagon', 'star', 'heart', 'triangle'] as const
export type Shape = (typeof SHAPES)[number]

type Sdf = (x: number, y: number) => number

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi)
const len = (x: number, y: number) => Math.hypot(x, y)

const circle: Sdf = (x, y) => len(x, y) - 1

const square: Sdf = (x, y) => {
  const b = 0.8
  const r = 0.14
  const qx = Math.abs(x) - b + r
  const qy = Math.abs(y) - b + r
  return len(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - r
}

const hexagon: Sdf = (x, y) => {
  const r = 0.86
  const kx = -0.866025404
  const ky = 0.5
  const kz = 0.577350269
  let px = Math.abs(x)
  let py = Math.abs(y)
  const d = 2 * Math.min(kx * px + ky * py, 0)
  px -= d * kx
  py -= d * ky
  px -= clamp(px, -kz * r, kz * r)
  py -= r
  return len(px, py) * Math.sign(py) - 0.06
}

const star: Sdf = (x, y) => {
  const r = 1.02
  const rf = 0.52
  const k1x = 0.809016994375
  const k1y = -0.587785252292
  const k2x = -k1x
  const k2y = k1y
  let px = Math.abs(x)
  let py = y + 0.06
  let d = 2 * Math.max(k1x * px + k1y * py, 0)
  px -= d * k1x
  py -= d * k1y
  d = 2 * Math.max(k2x * px + k2y * py, 0)
  px -= d * k2x
  py -= d * k2y
  px = Math.abs(px)
  py -= r
  const bax = rf * -k1y
  const bay = rf * k1x - 1
  const h = clamp((px * bax + py * bay) / (bax * bax + bay * bay), 0, r)
  return len(px - bax * h, py - bay * h) * Math.sign(py * bax - px * bay) - 0.05
}

const heart: Sdf = (x, y) => {
  const s = 0.6
  const px = Math.abs(x) * s
  const py = (y + 0.98) * s
  const dot2 = (a: number, b: number) => a * a + b * b
  let d: number
  if (py + px > 1) d = Math.sqrt(dot2(px - 0.25, py - 0.75)) - Math.SQRT2 / 4
  else {
    const m = 0.5 * Math.max(px + py, 0)
    d = Math.sqrt(Math.min(dot2(px, py - 1), dot2(px - m, py - m))) * Math.sign(px - py)
  }
  return d / s - 0.04
}

const triangle: Sdf = (x, y) => {
  const r = 0.98
  const k = Math.sqrt(3)
  let px = Math.abs(x) - r
  let py = y + r / k + 0.12
  if (px + k * py > 0) {
    const nx = (px - k * py) / 2
    const ny = (-k * px - py) / 2
    px = nx
    py = ny
  }
  px -= clamp(px, -2 * r, 0)
  return -len(px, py) * Math.sign(py) - 0.1
}

export const sdf: Record<Shape, Sdf> = { circle, square, hexagon, star, heart, triangle }
