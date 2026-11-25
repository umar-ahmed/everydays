import { describe, expect, it } from 'vite-plus/test'
import {
  LAMBDA_MAX,
  LAMBDA_MIN,
  cauchy,
  compile,
  defaultScene,
  fresnel,
  intersect,
  MATERIAL_CODE,
  reflectance,
  surfaceReflectance,
  TINTS,
  makeShape,
  polygon,
  refit,
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
  albedo: 0.8,
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

  it('finds the nearest surface and its material', () => {
    const shapes = [
      makeShape({ kind: 'circle', material: 'mirror', x: 1, y: 0, r: 0.2, angle: 0 }),
      makeShape({ kind: 'hexagon', material: 'glass', x: 0.2, y: 0, r: 0.1, angle: 0 }),
    ]
    const c = compile(shapes)
    expect(c.edgeCount).toBe(6)
    expect(c.circleSurfaces[0]).toBe(MATERIAL_CODE.mirror)
    expect(c.edgeSurfaces[0]).toBe(MATERIAL_CODE.glass)
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

  it('bounces off the view edges diffusely, with a cosine-weighted lobe', () => {
    const s = { flashlight: { x: 0, y: 0 }, shapes: [] }
    const walls = { ...optics, walls: 1, bounces: 16, cone: 360 }
    let rng = 7
    const rand = () => ((rng = (rng * 16807) % 2147483647) - 1) / 2147483646
    let cosSum = 0
    let n = 0
    for (let p = 0; p < 400; p++) {
      const segs = tracePath(compile([]), view, s, walls, 550, rand)
      expect(segs).toHaveLength(17)
      for (let i = 0; i < segs.length; i++) {
        const g = segs[i]
        expect(Math.abs(g.x1)).toBeLessThanOrEqual(view.hw + 1e-6)
        expect(Math.abs(g.y1)).toBeLessThanOrEqual(view.hh + 1e-6)
        const onWall = Math.abs(Math.abs(g.x1) - view.hw) < 1e-6 || Math.abs(Math.abs(g.y1) - view.hh) < 1e-6
        expect(onWall).toBe(true)
        if (i + 1 < segs.length) {
          const next = segs[i + 1]
          const len = Math.hypot(next.x1 - next.x0, next.y1 - next.y0)
          const normal = Math.abs(Math.abs(g.x1) - view.hw) < 1e-6 ? [-Math.sign(g.x1), 0] : [0, -Math.sign(g.y1)]
          cosSum += ((next.x1 - next.x0) * normal[0] + (next.y1 - next.y0) * normal[1]) / len
          n++
        }
      }
    }
    // A 2D Lambertian lobe (pdf cos θ / 2) has a mean cosine of π/4.
    expect(cosSum / n).toBeCloseTo(Math.PI / 4, 1)
  })

  it('ends a path at an absorbing wall', () => {
    const s = { flashlight: { x: 0, y: 0 }, shapes: [] }
    const segs = tracePath(compile([]), view, s, optics, 550, () => 0.5)
    expect(segs).toHaveLength(1)
    expect(segs[0].x1).toBeCloseTo(2)
  })
})

describe('refit', () => {
  it('turns the layout a quarter turn into portrait and keeps everything inside the walls', () => {
    const land = { hw: 1.78, hh: 1 }
    const port = { hw: 1, hh: 2.16 }
    const { scene, rotation } = defaultScene(land.hw)
    const out = refit(scene, rotation, land, port)
    expect(out.rotation).toBeCloseTo(rotation - 90)
    for (const p of [out.scene.flashlight, ...out.scene.shapes]) {
      expect(Math.abs(p.x)).toBeLessThan(port.hw)
      expect(Math.abs(p.y)).toBeLessThan(port.hh)
    }
    const back = refit(out.scene, out.rotation, port, land)
    expect(back.rotation).toBeCloseTo(rotation)
    expect(back.scene.shapes[0].angle).toBeCloseTo(scene.shapes[0].angle)
    expect(refit(scene, rotation, land, { hw: 1.6, hh: 1 }).rotation).toBe(rotation)
  })
})

describe('diffuse shapes', () => {
  const lcg = (seed: number) => () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646

  it('reflect by their tint: flat for white, long wavelengths for red', () => {
    for (const l of [400, 550, 700]) expect(reflectance(l, 'white')).toBe(1)
    expect(reflectance(660, 'red')).toBeGreaterThan(0.8)
    expect(reflectance(450, 'red')).toBeLessThan(0.06)
    expect(reflectance(535, 'green')).toBeCloseTo(1)
    expect(reflectance(650, 'green')).toBeLessThan(0.06)
    expect(reflectance(440, 'blue')).toBeCloseTo(1)
    expect(reflectance(600, 'blue')).toBeLessThan(0.06)
    const red = TINTS.red
    expect(surfaceReflectance(620, [2, red.center, red.width, red.floor])).toBeCloseTo(reflectance(620, 'red'))
  })

  it('bounce cosine-weighted off the lit side, surviving in proportion to albedo times reflectance', () => {
    // A big matte hexagon right in front of the flashlight; light hits its left face head-on.
    const s = {
      flashlight: { x: -1.5, y: 0 },
      shapes: [makeShape({ kind: 'hexagon', material: 'diffuse', x: 0.5, y: 0, r: 0.6, angle: 0 })],
    }
    const run = (tint: 'white' | 'red', lambda: number) => {
      s.shapes[0].tint = tint
      const rand = lcg(11)
      let survived = 0
      let cosSum = 0
      const n = 4000
      for (let p = 0; p < n; p++) {
        const segs = tracePath(compile(s.shapes), view, s, { ...optics, bounces: 1, albedo: 0.8 }, lambda, rand)
        expect(segs[0].x1).toBeCloseTo(0.5 - 0.6 * Math.cos(Math.PI / 6), 5)
        if (segs.length < 2) continue
        survived++
        const g = segs[1]
        const dx = g.x1 - g.x0
        const dy = g.y1 - g.y0
        // The lit face's normal is -x: every bounce leaves back toward the light.
        expect(dx).toBeLessThan(0)
        cosSum += -dx / Math.hypot(dx, dy)
      }
      return { survival: survived / n, meanCos: cosSum / survived }
    }
    const white = run('white', 550)
    expect(white.survival).toBeCloseTo(0.8, 1)
    expect(white.meanCos).toBeCloseTo(Math.PI / 4, 1)
    expect(run('red', 660).survival).toBeGreaterThan(0.65)
    expect(run('red', 450).survival).toBeLessThan(0.06)
  })
})
