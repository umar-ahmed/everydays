import { describe, expect, it } from 'vite-plus/test'
import {
  LAMBDA_MAX,
  LAMBDA_MIN,
  cauchy,
  compile,
  fresnel,
  intersect,
  makeShape,
  polygon,
  spectralRgb,
  tracePath,
  type Optics,
} from './optics.ts'

const optics: Optics = {
  power: 1,
  cone: 0,
  rotation: 0,
  aperture: 0,
  bounces: 8,
  ior: 1.5,
  dispersion: 0.02,
  haze: 0,
  walls: 0,
}
const view = { hw: 2, hh: 1 }

describe('spectral colour', () => {
  it('averages a flat spectrum to white', () => {
    const sum = [0, 0, 0]
    const n = 2000
    for (let i = 0; i < n; i++) {
      const c = spectralRgb(LAMBDA_MIN + ((i + 0.5) / n) * (LAMBDA_MAX - LAMBDA_MIN))
      for (let k = 0; k < 3; k++) sum[k] += c[k] / n
    }
    for (const s of sum) expect(s).toBeCloseTo(1, 2)
  })

  it('maps wavelengths to the right hues', () => {
    const red = spectralRgb(650)
    const green = spectralRgb(530)
    const blue = spectralRgb(450)
    expect(red[0]).toBeGreaterThan(Math.max(red[1], red[2]))
    expect(green[1]).toBeGreaterThan(Math.max(green[0], green[2]))
    expect(blue[2]).toBeGreaterThan(Math.max(blue[0], blue[1]))
  })
})

describe('dispersion and Fresnel', () => {
  it('bends violet more than red', () => {
    expect(cauchy(400, 1.5, 0.02)).toBeGreaterThan(cauchy(700, 1.5, 0.02))
    expect(cauchy(550, 1.5, 0)).toBe(1.5)
  })

  it('reflects 4% of light at normal incidence on glass and all of it past the critical angle', () => {
    expect(fresnel(1, 1, 1.5)).toBeCloseTo(0.04, 3)
    const critical = Math.asin(1 / 1.5)
    expect(fresnel(Math.cos(critical + 0.01), 1.5, 1)).toBe(1)
    expect(fresnel(Math.cos(critical - 0.05), 1.5, 1)).toBeLessThan(1)
  })
})

describe('geometry', () => {
  it('points triangles up and gives polygon edges outward normals', () => {
    const tri = makeShape({ kind: 'triangle', material: 'glass', x: 0, y: 0, r: 1, angle: 0 })
    const v = polygon(tri)
    expect(v[0][0]).toBeCloseTo(0)
    expect(v[0][1]).toBeCloseTo(-1)
    const c = compile([tri])
    for (let i = 0; i < c.edgeCount; i++) {
      const [x0, y0, x1, y1] = c.edges.subarray(i * 4, i * 4 + 4)
      const mid = [(x0 + x1) / 2, (y0 + y1) / 2]
      const normal = [y1 - y0, -(x1 - x0)]
      expect(mid[0] * normal[0] + mid[1] * normal[1]).toBeGreaterThan(0)
    }
  })

  it('finds the nearest surface and sorts glass before mirrors', () => {
    const shapes = [
      makeShape({ kind: 'circle', material: 'mirror', x: 1, y: 0, r: 0.2, angle: 0 }),
      makeShape({ kind: 'hexagon', material: 'glass', x: 0.2, y: 0, r: 0.1, angle: 0 }),
    ]
    const c = compile(shapes)
    expect(c.glassEdges).toBe(6)
    expect(c.circleCount - c.glassCircles).toBe(1)
    const hit = intersect(c, view, -1, 0, 1, 0)
    expect(hit.kind).toBe('glass')
    expect(hit.nx).toBeLessThan(0)
    const past = intersect(c, view, 0.5, 0, 1, 0)
    expect(past.kind).toBe('mirror')
    expect(past.t).toBeCloseTo(0.3)
    expect(intersect(c, view, 1.5, 0, 1, 0)).toMatchObject({ kind: 'wall', nx: -1 })
  })
})

describe('tracing', () => {
  const scene = (material: 'glass' | 'mirror') => ({
    flashlight: { x: -1.5, y: 0 },
    shapes: [makeShape({ kind: 'circle', material, x: 0, y: 0, r: 0.3, angle: 0 })],
  })

  it('sends a mirror hit straight back', () => {
    const s = scene('mirror')
    const segs = tracePath(compile(s.shapes), view, s, optics, 550, () => 0.5)
    expect(segs[0].x1).toBeCloseTo(-0.3)
    expect(segs[1].x1).toBeLessThan(segs[1].x0)
    expect(segs[1].throughput).toBeLessThan(1)
  })

  it('refracts a ray through a glass ball, bending blue more than red', () => {
    const s = { ...scene('glass'), flashlight: { x: -1.5, y: -0.15 } }
    const c = compile(s.shapes)
    // rand 0.99 never takes the Fresnel reflection, so both rays refract twice.
    const exitAngle = (l: number) => {
      const segs = tracePath(c, view, s, optics, l, () => 0.99)
      const out = segs[2]
      return Math.atan2(out.y1 - out.y0, out.x1 - out.x0)
    }
    expect(exitAngle(450)).toBeGreaterThan(exitAngle(650))
    expect(exitAngle(650)).toBeGreaterThan(0)
  })

  it('ends a path at an absorbing wall', () => {
    const s = { flashlight: { x: 0, y: 0 }, shapes: [] }
    const segs = tracePath(compile([]), view, s, optics, 550, () => 0.5)
    expect(segs).toHaveLength(1)
    expect(segs[0].x1).toBeCloseTo(2)
  })
})
