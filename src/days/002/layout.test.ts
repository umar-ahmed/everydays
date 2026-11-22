import { describe, expect, it } from 'vite-plus/test'
import { latticeFor, TILINGS } from './generators/lattice.ts'
import { FULL_BUDGET, MAX_BUBBLES, PHONE_BUDGET, fitsCapacity, layoutFor } from './layout.ts'

const ASPECTS = [0.34, 0.5, 1, 1.29, 1.73, 2.17]

describe('layoutFor', () => {
  for (const tiling of TILINGS) {
    const lattice = latticeFor(tiling)

    it(`keeps ${tiling} sheets inside every buffer and the budget`, () => {
      for (const aspect of ASPECTS) {
        for (const budget of [FULL_BUDGET, PHONE_BUDGET]) {
          for (const across of [3, 8, 16]) {
            const l = layoutFor({ size: 1.1, spacing: 0.28, across, aspect, lattice, budget })
            expect(fitsCapacity(l.W, l.H)).toBe(true)
            expect(l.W * l.H).toBeLessThanOrEqual(budget.verts)
            expect(l.sites).toBeLessThanOrEqual(Math.min(budget.bubbles, MAX_BUBBLES))
            expect(l.sites).toBe(lattice.count(l.cols, l.rows))
          }
        }
      }
    })

    it(`puts the requested ${tiling} count on the shorter side when it fits`, () => {
      const wide = layoutFor({ size: 1.1, spacing: 0.28, across: 6, aspect: 1.6, lattice })
      expect(wide.rows).toBe(6)
      expect(wide.width).toBeGreaterThan(wide.depth)
      const tall = layoutFor({ size: 1.1, spacing: 0.28, across: 6, aspect: 0.5, lattice })
      expect(tall.cols).toBe(6)
      expect(tall.depth).toBeGreaterThan(tall.width)
    })
  }

  it('keeps bubble size and spacing physical: the pitch never depends on the aspect', () => {
    const lattice = latticeFor('square')
    const pitches = ASPECTS.map((aspect) => layoutFor({ size: 1.35, spacing: 0.5, across: 8, aspect, lattice }).pitch)
    expect(new Set(pitches).size).toBe(1)
  })

  it('lowers the density instead of overflowing the phone budget', () => {
    const lattice = latticeFor('triangular')
    const l = layoutFor({ size: 0.6, spacing: 0.1, across: 16, aspect: 0.34, lattice, budget: PHONE_BUDGET })
    expect(l.across).toBeLessThan(16)
    expect(l.W * l.H).toBeLessThanOrEqual(PHONE_BUDGET.verts)
  })
})

describe('latticeFor', () => {
  it('uses the triangular row pitch and drops every third honeycomb site', () => {
    expect(latticeFor('square').rowStep).toBe(1)
    expect(latticeFor('triangular').rowStep).toBeCloseTo(Math.sqrt(3) / 2)
    const honeycomb = latticeFor('hexagonal').count(12, 12)
    expect(honeycomb).toBe(96)
  })
})
