import { describe, expect, it } from 'vite-plus/test'
import { AXIS_RANGE, axesAt, ends, fitLines, influence, THIN } from './field.ts'

describe('influence', () => {
  it('is a smooth bell: full at the pointer, falling monotonically to ~0 at the radius', () => {
    expect(influence(0, 100)).toBe(1)
    let prev = 1
    for (let d = 5; d <= 100; d += 5) {
      const i = influence(d, 100)
      expect(i).toBeLessThan(prev)
      prev = i
    }
    expect(influence(100, 100)).toBeLessThan(0.02)
    expect(influence(50, 100)).toBeGreaterThan(0.3)
    expect(influence(10, 0)).toBe(0)
  })
})

describe('axes', () => {
  it('bulge swells thin type to the max; dent thins fat type', () => {
    const bulge = ends('bulge', 900, 140)
    expect(axesAt(0, bulge)).toEqual(THIN)
    expect(axesAt(1, bulge)).toEqual({ wght: 900, wdth: 140 })
    const dent = ends('dent', 900, 140)
    expect(axesAt(0, dent)).toEqual({ wght: 900, wdth: 140 })
    const dented = axesAt(1, dent)
    expect(dented.wght).toBeLessThan(900)
    expect(dented.wdth).toBeLessThan(140)
  })

  it('clamps spring overshoot to the font design space', () => {
    const a = axesAt(1.6, ends('bulge', 1000, 151))
    expect(a.wght).toBe(AXIS_RANGE.wght[1])
    expect(a.wdth).toBe(AXIS_RANGE.wdth[1])
    const b = axesAt(-1, ends('bulge', 1000, 151))
    expect(b.wght).toBeGreaterThanOrEqual(AXIS_RANGE.wght[0])
    expect(b.wdth).toBeGreaterThanOrEqual(AXIS_RANGE.wdth[0])
  })
})

describe('fitLines', () => {
  const widthOf = (t: string) => t.length * 0.5

  it('keeps one word on one line, sized by width or height', () => {
    const wide = fitLines(['magnetic'], widthOf, { width: 800, height: 600, lineHeight: 1 })
    expect(wide.lines).toEqual(['magnetic'])
    expect(wide.fontSize).toBe(200)
    const short = fitLines(['hi'], widthOf, { width: 800, height: 300, lineHeight: 1 })
    expect(short.fontSize).toBe(300)
  })

  it('breaks long phrases into balanced lines when that makes the type bigger', () => {
    const r = fitLines('type is a material'.split(' '), widthOf, { width: 400, height: 800, lineHeight: 1 })
    expect(r.lines.length).toBeGreaterThan(1)
    expect(r.lines.join(' ')).toBe('type is a material')
    const one = 400 / widthOf('type is a material')
    expect(r.fontSize).toBeGreaterThan(one)
    for (const l of r.lines) expect(widthOf(l) * r.fontSize).toBeLessThanOrEqual(400 + 1e-9)
  })
})
