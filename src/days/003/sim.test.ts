import { describe, expect, it } from 'vite-plus/test'
import { ErosionSim, SIZE, gridFor } from './sim.ts'

describe('ErosionSim', () => {
  it('sizes the grid for the device and measures heights in cells', () => {
    expect(gridFor(false)).toBe(512)
    expect(gridFor(true)).toBe(256)
    const sim = new ErosionSim(64)
    expect(sim.cell).toBeCloseTo(SIZE / 63)
    sim.setTerrain({ seed: 3, octaves: 6, roughness: 0.5, ridge: 0.4, heightScale: 2 })
    expect(sim.heightCells.value).toBeCloseTo(2 / sim.cell)
    expect(sim.state.value.array.length).toBe(64 * 64 * 4)
  })
})
