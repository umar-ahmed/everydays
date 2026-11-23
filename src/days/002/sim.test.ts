import { describe, expect, it } from 'vite-plus/test'
import { FixedStep } from './sim.ts'

const run = (frameDt: number, frames: number, speed = 1) => {
  const clock = new FixedStep()
  let ticks = 0
  for (let f = 0; f < frames; f++) ticks += clock.ticks(frameDt, speed)
  return ticks
}

describe('FixedStep', () => {
  it('runs one tick per 60 Hz frame and one every other 120 Hz frame', () => {
    expect(run(1 / 60, 600)).toBe(600)
    expect(run(1 / 120, 1200)).toBe(600)
    expect(run(1 / 144, 1440)).toBeGreaterThanOrEqual(599)
    expect(run(1 / 144, 1440)).toBeLessThanOrEqual(600)
  })

  it('slows down below 60 fps instead of stacking ticks', () => {
    expect(run(1 / 30, 300)).toBe(300)
    expect(run(0.5, 10)).toBe(10)
  })

  it('runs slow motion as fewer ticks of the same length', () => {
    expect(run(1 / 60, 600, 0.25)).toBe(150)
    expect(run(1 / 60, 600, 0.1)).toBe(60)
  })
})
