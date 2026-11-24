/**
 * Shared 2D optics: the scene, its geometry, the spectral colour of a wavelength and a CPU
 * reference tracer. The GPU kernel in `gpu.ts` mirrors `tracePath` line for line; the canvas
 * fallback and the tests run this one.
 *
 * World units: the shorter side of the view spans [-1, 1]; y points down, like the screen.
 */

export type ShapeKind = 'triangle' | 'circle' | 'hexagon'
export type Material = 'glass' | 'mirror'

export type Shape = {
  id: number
  kind: ShapeKind
  material: Material
  x: number
  y: number
  /** Circumradius. */
  r: number
  /** Radians, clockwise on screen. */
  angle: number
}

export type Flashlight = { x: number; y: number }

export type Scene = { flashlight: Flashlight; shapes: Shape[] }

export type Optics = {
  /** Total emitted power, arbitrary units (1 is a comfortable default). */
  power: number
  /** Full cone angle, degrees. */
  cone: number
  /** Beam direction, degrees clockwise from +x. */
  rotation: number
  /** Width of the flashlight's lens, world units. */
  aperture: number
  /** Scattering events per path (segments drawn = bounces + 1). */
  bounces: number
  /** Cauchy A: the index of refraction far in the red. */
  ior: number
  /** Cauchy B in µm²: how much more blue bends than red. BK7 is ~0.0042, dense flint ~0.014. */
  dispersion: number
  /** Scattering coefficient of the air, per world unit. */
  haze: number
  /** Albedo of the diffuse walls. */
  walls: number
}

export const LAMBDA_MIN = 380
export const LAMBDA_MAX = 720
/** Mirrors keep this much of the light per bounce (a good silvered mirror). */
export const MIRROR_REFLECTANCE = 0.94
/** Distance from the flashlight's centre to its lens. */
export const LENS_OFFSET = 0.11
export const MAX_SHAPES = 16
/** A hexagon has the most edges. */
export const MAX_EDGES = MAX_SHAPES * 6
export const EPS = 1e-4

export const SIDES: Record<ShapeKind, number> = { triangle: 3, circle: 0, hexagon: 6 }

/** Vertices of a regular polygon, counter-clockwise in math orientation; triangles point up at angle 0. */
export function polygon(s: Shape): [number, number][] {
  const n = SIDES[s.kind]
  return Array.from({ length: n }, (_, i) => {
    const a = s.angle - Math.PI / 2 + (2 * Math.PI * i) / n
    return [s.x + s.r * Math.cos(a), s.y + s.r * Math.sin(a)]
  })
}

/**
 * The scene flattened for tracing: edges (x0, y0, x1, y1) and circles (cx, cy, r, 0), glass
 * first, then mirrors, so a hit's material is just whether its index is below the glass count.
 * Each polygon's edges run so that (ey, -ex) is the outward normal.
 */
export type Compiled = {
  edges: Float32Array
  edgeCount: number
  glassEdges: number
  circles: Float32Array
  circleCount: number
  glassCircles: number
}

export function compile(shapes: Shape[]): Compiled {
  const edges = new Float32Array(MAX_EDGES * 4)
  const circles = new Float32Array(MAX_SHAPES * 4)
  let edgeCount = 0
  let circleCount = 0
  let glassEdges = 0
  let glassCircles = 0
  for (const material of ['glass', 'mirror'] as const) {
    for (const s of shapes.slice(0, MAX_SHAPES)) {
      if (s.material !== material) continue
      if (s.kind === 'circle') {
        circles.set([s.x, s.y, s.r, 0], circleCount++ * 4)
        continue
      }
      const v = polygon(s)
      for (let i = 0; i < v.length; i++) {
        const [a, b] = [v[i], v[(i + 1) % v.length]]
        edges.set([a[0], a[1], b[0], b[1]], edgeCount++ * 4)
      }
    }
    if (material === 'glass') {
      glassEdges = edgeCount
      glassCircles = circleCount
    }
  }
  return { edges, edgeCount, glassEdges, circles, circleCount, glassCircles }
}

/** Cauchy's equation, with the wavelength in nm. */
export const cauchy = (lambda: number, a: number, b: number) => a + b / (lambda / 1000) ** 2

/**
 * CIE 1931 2° colour matching functions, as the multi-lobe Gaussian fit of Wyman, Sloan and
 * Shirley, "Simple Analytic Approximations to the CIE XYZ Color Matching Functions" (2013).
 * Each lobe is (weight, mean, sigma below the mean, sigma above).
 */
export const CMF = {
  x: [
    [1.056, 599.8, 37.9, 31.0],
    [0.362, 442.0, 16.0, 26.7],
    [-0.065, 501.1, 20.4, 26.2],
  ],
  y: [
    [0.821, 568.8, 46.9, 40.5],
    [0.286, 530.9, 16.3, 31.1],
  ],
  z: [
    [1.217, 437.0, 11.8, 36.0],
    [0.681, 459.0, 26.0, 13.8],
  ],
} as const

const lobes = (l: number, fit: readonly (readonly number[])[]) =>
  fit.reduce((sum, [w, mu, s1, s2]) => {
    const t = (l - mu) / (l < mu ? s1 : s2)
    return sum + w * Math.exp(-0.5 * t * t)
  }, 0)

export const xyz = (l: number): [number, number, number] => [lobes(l, CMF.x), lobes(l, CMF.y), lobes(l, CMF.z)]

/** CIE XYZ to linear sRGB (D65), row-major. */
export const XYZ_TO_SRGB = [3.2406, -1.5372, -0.4986, -0.9689, 1.8758, 0.0415, 0.0557, -0.204, 1.057] as const

const toSrgb = ([x, y, z]: [number, number, number]): [number, number, number] => {
  const m = XYZ_TO_SRGB
  return [m[0] * x + m[1] * y + m[2] * z, m[3] * x + m[4] * y + m[5] * z, m[6] * x + m[7] * y + m[8] * z]
}

/**
 * Per-channel gains that make a flat spectrum, sampled uniformly over [LAMBDA_MIN, LAMBDA_MAX],
 * average to linear (1, 1, 1): the flashlight's white comes out white.
 */
export const WHITE_BALANCE: [number, number, number] = (() => {
  const sum = [0, 0, 0]
  const n = 3400
  for (let i = 0; i < n; i++) {
    const c = toSrgb(xyz(LAMBDA_MIN + ((i + 0.5) / n) * (LAMBDA_MAX - LAMBDA_MIN)))
    for (let k = 0; k < 3; k++) sum[k] += c[k] / n
  }
  return [1 / sum[0], 1 / sum[1], 1 / sum[2]]
})()

/**
 * Linear sRGB of one wavelength, white balanced so a uniform sample of wavelengths averages to
 * white. Deep reds, cyans and violets fall outside sRGB and come out with a negative channel;
 * they are kept, so mixtures stay right, and clipped only for display.
 */
export const spectralRgb = (l: number): [number, number, number] => {
  const c = toSrgb(xyz(l))
  return [c[0] * WHITE_BALANCE[0], c[1] * WHITE_BALANCE[1], c[2] * WHITE_BALANCE[2]]
}

/** Unpolarised Fresnel reflectance from medium n1 into n2; 1 under total internal reflection. */
export function fresnel(cosi: number, n1: number, n2: number) {
  const eta = n1 / n2
  const k = 1 - eta * eta * (1 - cosi * cosi)
  if (k < 0) return 1
  const cost = Math.sqrt(k)
  const rs = (n1 * cosi - n2 * cost) / (n1 * cosi + n2 * cost)
  const rp = (n1 * cost - n2 * cosi) / (n1 * cost + n2 * cosi)
  return (rs * rs + rp * rp) / 2
}

export type View = { hw: number; hh: number }

/** Half extents of the view for an aspect ratio (width / height): the shorter side spans [-1, 1]. */
export const viewFor = (aspect: number): View => (aspect >= 1 ? { hw: aspect, hh: 1 } : { hw: 1, hh: 1 / aspect })

export type Hit = { t: number; nx: number; ny: number; kind: 'wall' | 'glass' | 'mirror' }

/** Nearest surface along a ray inside the view; the normal is outward for shapes, inward for walls. */
export function intersect(c: Compiled, view: View, ox: number, oy: number, dx: number, dy: number): Hit {
  const tx = dx > 0 ? (view.hw - ox) / dx : dx < 0 ? (-view.hw - ox) / dx : Infinity
  const ty = dy > 0 ? (view.hh - oy) / dy : dy < 0 ? (-view.hh - oy) / dy : Infinity
  const hit: Hit =
    tx < ty
      ? { t: Math.max(tx, 0), nx: -Math.sign(dx), ny: 0, kind: 'wall' }
      : { t: Math.max(ty, 0), nx: 0, ny: -Math.sign(dy), kind: 'wall' }
  for (let i = 0; i < c.edgeCount; i++) {
    const [x0, y0, x1, y1] = c.edges.subarray(i * 4, i * 4 + 4)
    const ex = x1 - x0
    const ey = y1 - y0
    const den = dx * ey - dy * ex
    if (Math.abs(den) < 1e-12) continue
    const px = x0 - ox
    const py = y0 - oy
    const t = (px * ey - py * ex) / den
    const s = (px * dy - py * dx) / den
    if (t > EPS && t < hit.t && s >= 0 && s <= 1) {
      const len = Math.hypot(ex, ey)
      Object.assign(hit, { t, nx: ey / len, ny: -ex / len, kind: i < c.glassEdges ? 'glass' : 'mirror' })
    }
  }
  for (let i = 0; i < c.circleCount; i++) {
    const [cx, cy, r] = c.circles.subarray(i * 4, i * 4 + 3)
    const qx = ox - cx
    const qy = oy - cy
    const b = qx * dx + qy * dy
    const disc = b * b - (qx * qx + qy * qy - r * r)
    if (disc < 0) continue
    const sq = Math.sqrt(disc)
    const t = -b - sq > EPS ? -b - sq : -b + sq
    if (t > EPS && t < hit.t) {
      const nx = (qx + t * dx) / r
      const ny = (qy + t * dy) / r
      Object.assign(hit, { t, nx, ny, kind: i < c.glassCircles ? 'glass' : 'mirror' })
    }
  }
  return hit
}

export type Segment = { x0: number; y0: number; x1: number; y1: number; throughput: number }

/**
 * One light path: emitted from the flashlight's lens at wavelength `lambda`, then up to
 * `bounces` scattering events (air, walls, mirrors, glass). Returns the segments it travels;
 * each carries the fraction of the path's power still in it.
 */
export function tracePath(
  c: Compiled,
  view: View,
  scene: Scene,
  o: Optics,
  lambda: number,
  rand: () => number,
): Segment[] {
  const rot = (o.rotation * Math.PI) / 180
  const a = rot + (rand() - 0.5) * ((o.cone * Math.PI) / 180)
  const lens = (rand() - 0.5) * o.aperture
  let ox = scene.flashlight.x + Math.cos(rot) * LENS_OFFSET - Math.sin(rot) * lens
  let oy = scene.flashlight.y + Math.sin(rot) * LENS_OFFSET + Math.cos(rot) * lens
  let dx = Math.cos(a)
  let dy = Math.sin(a)
  let throughput = 1
  const n = cauchy(lambda, o.ior, o.dispersion)
  const out: Segment[] = []
  for (let b = 0; b <= o.bounces; b++) {
    const hit = intersect(c, view, ox, oy, dx, dy)
    const free = o.haze > 0 ? -Math.log(1 - rand()) / o.haze : Infinity
    const scatter = free < hit.t
    const t = scatter ? free : hit.t
    const ex = ox + dx * t
    const ey = oy + dy * t
    out.push({ x0: ox, y0: oy, x1: ex, y1: ey, throughput })
    if (scatter) {
      const phi = rand() * 2 * Math.PI
      dx = Math.cos(phi)
      dy = Math.sin(phi)
    } else if (hit.kind === 'wall') {
      // Russian roulette on the albedo keeps every drawn segment at full strength.
      if (rand() >= o.walls) break
      const s = 2 * rand() - 1
      const k = Math.sqrt(1 - s * s)
      dx = hit.nx * k - hit.ny * s
      dy = hit.ny * k + hit.nx * s
    } else if (hit.kind === 'mirror') {
      const d = dx * hit.nx + dy * hit.ny
      dx -= 2 * d * hit.nx
      dy -= 2 * d * hit.ny
      throughput *= MIRROR_REFLECTANCE
    } else {
      const entering = dx * hit.nx + dy * hit.ny < 0
      const nx = entering ? hit.nx : -hit.nx
      const ny = entering ? hit.ny : -hit.ny
      const [n1, n2] = entering ? [1, n] : [n, 1]
      const cosi = -(dx * nx + dy * ny)
      const eta = n1 / n2
      const k = 1 - eta * eta * (1 - cosi * cosi)
      if (k < 0 || rand() < fresnel(cosi, n1, n2)) {
        dx += 2 * cosi * nx
        dy += 2 * cosi * ny
      } else {
        const m = eta * cosi - Math.sqrt(k)
        dx = eta * dx + m * nx
        dy = eta * dy + m * ny
      }
    }
    ox = ex + dx * EPS
    oy = ey + dy * EPS
  }
  return out
}

let nextId = 1
export const makeShape = (s: Omit<Shape, 'id'>): Shape => ({ ...s, id: nextId++ })

/**
 * The opening layout: a flashlight aimed at an equilateral glass prism that fans its beam into
 * a rainbow, which falls across a round lens and a hexagonal mirror. Portrait views get the same
 * layout turned a quarter turn, shining upward.
 */
export function defaultScene(aspect: number): { scene: Scene; rotation: number } {
  const portrait = aspect < 1
  const place = (x: number, y: number): [number, number] => (portrait ? [y, -x] : [x, y])
  const turn = portrait ? -Math.PI / 2 : 0
  const at = (x: number, y: number) => {
    const [px, py] = place(x, y)
    return { x: px, y: py }
  }
  // The beam meets the prism's left face at about minimum deviation for n ≈ 1.55.
  const shapes = [
    makeShape({ kind: 'triangle', material: 'glass', ...at(-0.5, 0.02), r: 0.34, angle: turn }),
    makeShape({ kind: 'circle', material: 'glass', ...at(0.27, 0.25), r: 0.19, angle: 0 }),
    makeShape({ kind: 'hexagon', material: 'mirror', ...at(0.8, 0.55), r: 0.15, angle: 0.1 + turn }),
  ]
  const [fx, fy] = place(-1.5, 0.26)
  return { scene: { flashlight: { x: fx, y: fy }, shapes }, rotation: portrait ? -111 : -21 }
}
