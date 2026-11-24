import { describe, expect, it } from 'vite-plus/test'
import { ErosionSim, SIZE, bufferBytes, defaultResolution, fitResolution, simBytes, type ErosionParams } from './sim.ts'

const erosion: ErosionParams = {
  rain: 0.01,
  rainVariation: 0.6,
  capacity: 1,
  erosion: 0.05,
  deposition: 0.05,
  evaporation: 0.01,
  thermal: 0.2,
  talusVariation: 0.6,
  hardness: 0.6,
  strataScale: 0.3,
  strataTilt: 12,
  joints: 0.4,
}

describe('ErosionSim', () => {
  it('sizes the grid for the device and measures heights in cells', () => {
    expect(defaultResolution(false)).toBe(512)
    expect(defaultResolution(true)).toBe(256)
    const sim = new ErosionSim(64)
    expect(sim.cell).toBeCloseTo(SIZE / 63)
    sim.setTerrain({ seed: 3, octaves: 6, roughness: 0.5, ridge: 0.4, heightScale: 2 })
    expect(sim.heightCells.value).toBeCloseTo(2 / sim.cell)
    expect(sim.state.value.array.length).toBe(64 * 64 * 4)
    // All buffers start from one shared zeroed upload array.
    expect(sim.rock.value.array).toBe(sim.state.value.array)
  })

  it('scales the erosion that is measured in cells with the resolution', () => {
    const coarse = new ErosionSim(128)
    const fine = new ErosionSim(1024)
    coarse.setErosion(erosion)
    fine.setErosion(erosion)
    expect(fine.rain.value / coarse.rain.value).toBeCloseTo(1023 / 127)
    expect(fine.capacity.value).toBe(coarse.capacity.value)
    // Strata are a world-space thickness, so a finer grid needs more cells of height per band.
    expect(coarse.strataK.value / fine.strataK.value).toBeCloseTo(1023 / 127)
    // Same world height in both: the fine grid counts more cells.
    coarse.setTerrain({ seed: 1, octaves: 7, roughness: 0.5, ridge: 0.5, heightScale: 2.4 })
    fine.setTerrain({ seed: 1, octaves: 7, roughness: 0.5, ridge: 0.5, heightScale: 2.4 })
    expect(fine.heightCells.value * fine.cell).toBeCloseTo(coarse.heightCells.value * coarse.cell)
  })

  it('falls back to the largest grid whose buffers fit the device limits', () => {
    expect(bufferBytes(4096)).toBe(256 * 1024 ** 2)
    expect(simBytes(4096)).toBe(1.5 * 1024 ** 3)
    // WebGPU's default limits: 256 MB buffers, 128 MB storage bindings.
    const defaults = { maxBufferSize: 256 * 1024 ** 2, maxStorageBufferBindingSize: 128 * 1024 ** 2 }
    expect(fitResolution(4096, defaults)).toBe(2048)
    expect(fitResolution(1024, defaults)).toBe(1024)
    const big = { maxBufferSize: 4 * 1024 ** 3, maxStorageBufferBindingSize: 4 * 1024 ** 3 }
    expect(fitResolution(4096, big)).toBe(4096)
    expect(fitResolution(256, { maxBufferSize: 1, maxStorageBufferBindingSize: 1 })).toBe(128)
  })
})
