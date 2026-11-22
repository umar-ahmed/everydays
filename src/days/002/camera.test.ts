import { describe, expect, it } from 'vite-plus/test'
import {
  MIN_GAP,
  clipPlanes,
  depthStep,
  fitView,
  rayHit,
  safeRect,
  sheetAspect,
  sheetBounds,
  viewDepth,
  zoomRange,
} from './camera.ts'

const VIEWS = {
  iphonePortrait: { width: 390, height: 844 },
  iphoneLandscape: { width: 844, height: 390 },
  ipad: { width: 820, height: 1180 },
  desktop: { width: 1920, height: 1080 },
  ultrawide: { width: 3440, height: 1440 },
}

describe('sheetAspect', () => {
  it('gives portrait screens a tall sheet and landscape screens a wide one', () => {
    expect(sheetAspect(VIEWS.iphonePortrait)).toBeLessThan(1)
    expect(sheetAspect(VIEWS.ipad)).toBeLessThan(1)
    expect(sheetAspect(VIEWS.iphoneLandscape)).toBeGreaterThan(1)
    expect(sheetAspect(VIEWS.desktop)).toBeGreaterThan(1)
    expect(sheetAspect(VIEWS.ultrawide)).toBeGreaterThan(sheetAspect(VIEWS.desktop))
  })
})

describe('fitView', () => {
  for (const [name, view] of Object.entries(VIEWS)) {
    it(`frames a ${name} sheet inside the safe area and fills it`, () => {
      const aspect = sheetAspect(view)
      const box = { width: 10 * aspect, depth: 10, top: 0.4 }
      const fit = fitView(box, view)
      const b = sheetBounds(fit, fit.offset, box, view)!
      const r = safeRect(view)
      expect(b.x0).toBeGreaterThanOrEqual(r.x0 - 1e-3)
      expect(b.x1).toBeLessThanOrEqual(r.x1 + 1e-3)
      expect(b.y0).toBeGreaterThanOrEqual(r.y0 - 1e-3)
      expect(b.y1).toBeLessThanOrEqual(r.y1 + 1e-3)
      const fill = Math.min((b.x1 - b.x0) / (r.x1 - r.x0), (b.y1 - b.y0) / (r.y1 - r.y0))
      expect(fill).toBeGreaterThan(0.95)
    })
  }
})

describe('clipPlanes', () => {
  it('keeps everything visible between the planes at the fit view and at macro zoom', () => {
    const view = VIEWS.desktop
    const box = { width: 14, depth: 11, top: 0.42 }
    const fit = fitView(box, view)
    const macro = { ...fit, dist: zoomRange(fit, 1.1, 0.42, view).min }
    for (const pose of [fit, macro]) {
      const { near, far } = clipPlanes(pose, fit.offset, box, view, fit.far)
      for (const ndc of [
        [0, 0],
        [-1, -1],
        [1, -1],
        [0, 1],
      ] as [number, number][]) {
        for (const y of [0, box.top]) {
          const p = rayHit(pose, fit.offset, view, ndc, y)
          if (!p) continue
          const z = viewDepth(pose, p)
          expect(z).toBeGreaterThan(near)
          expect(z).toBeLessThan(far)
          expect(depthStep(z, near, far) * 8).toBeLessThan(MIN_GAP)
        }
      }
    }
  })

  it('pulls the near plane in with the zoom', () => {
    const view = VIEWS.iphonePortrait
    const box = { width: 8, depth: 25, top: 0.42 }
    const fit = fitView(box, view)
    const close = clipPlanes({ ...fit, dist: fit.dist * 0.2 }, fit.offset, box, view)
    expect(close.near).toBeLessThan(fit.near * 0.5)
  })
})
