import { describe, expect, it } from 'vite-plus/test'
import { ErosionSim, SIZE, defaultResolution, maxResolution } from './sim.ts'

describe('ErosionSim', () => {
  it('sizes the grid for the device and measures heights in cells', () => {
    expect(defaultResolution(false)).toBe(512)
    expect(defaultResolution(true)).toBe(256)
    expect(maxResolution(true)).toBe(512)
    const sim = new ErosionSim(64)
    expect(sim.cell).toBeCloseTo(SIZE / 63)
    sim.setTerrain({ seed: 3, octaves: 6, roughness: 0.5, ridge: 0.4, heightScale: 2 })
    expect(sim.heightCells.value).toBeCloseTo(2 / sim.cell)
    expect(sim.state.value.array.length).toBe(64 * 64 * 4)
  })

  it('scales the erosion that is measured in cells with the resolution', () => {
    const coarse = new ErosionSim(128)
    const fine = new ErosionSim(1024)
    const p = { rain: 0.01, capacity: 1, erosion: 0.05, deposition: 0.05, evaporation: 0.01, thermal: 0.2 }
    coarse.setErosion(p)
    fine.setErosion(p)
    expect(fine.rain.value / coarse.rain.value).toBeCloseTo(1023 / 127)
    expect(fine.capacity.value).toBe(coarse.capacity.value)
    // Same world height in both: the fine grid counts more cells.
    coarse.setTerrain({ seed: 1, octaves: 7, roughness: 0.5, ridge: 0.5, heightScale: 2.4 })
    fine.setTerrain({ seed: 1, octaves: 7, roughness: 0.5, ridge: 0.5, heightScale: 2.4 })
    expect(fine.heightCells.value * fine.cell).toBeCloseTo(coarse.heightCells.value * coarse.cell)
  })
})
