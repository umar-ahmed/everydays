import { describe, expect, it } from 'vite-plus/test'
import { midiName, midiToHz, stringMidi, Voice, type Pluck } from './ks.ts'

const SR = 48000

/** Seeded so renders are repeatable. */
function lcg(seed: number) {
  return () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646
}

function render(voice: Voice, seconds: number) {
  const out = new Float32Array(Math.round(seconds * SR))
  for (let i = 0; i < out.length; i++) out[i] = voice.next()
  return out
}

/** Fundamental from the autocorrelation peak near the expected period, parabolically refined. */
function measurePitch(x: Float32Array, expected: number) {
  const p = SR / expected
  const lo = Math.floor(p * 0.8)
  const hi = Math.ceil(p * 1.25)
  const win = x.length - hi
  const r = (lag: number) => {
    let s = 0
    for (let i = 0; i < win; i++) s += x[i] * x[i + lag]
    return s
  }
  let best = lo
  let bestR = -Infinity
  const rs = new Map<number, number>()
  for (let lag = lo - 1; lag <= hi + 1; lag++) rs.set(lag, r(lag))
  for (let lag = lo; lag <= hi; lag++) {
    if (rs.get(lag)! > bestR) {
      bestR = rs.get(lag)!
      best = lag
    }
  }
  const [a, b, c] = [rs.get(best - 1)!, rs.get(best)!, rs.get(best + 1)!]
  const lag = best + (0.5 * (a - c)) / (a - 2 * b + c)
  return SR / lag
}

const cents = (f: number, ref: number) => 1200 * Math.log2(f / ref)

const pluck = (freq: number, extra: Partial<Pluck> = {}): Pluck => ({
  freq,
  velocity: 0.8,
  position: 0.2,
  brightness: 0.6,
  decay: 4,
  ...extra,
})

describe('Karplus-Strong voice', () => {
  // Includes periods far from whole samples (48000 / 987.77 ≈ 48.59) where truncating the
  // delay line would be off by tens of cents.
  for (const freq of [82.41, 146.83, 261.63, 440, 987.77, 1567.98]) {
    it(`sounds at ${freq} Hz to within a cent`, () => {
      const v = new Voice(SR, lcg(7))
      v.pluck(pluck(freq))
      const x = render(v, 0.6).subarray(Math.round(0.1 * SR))
      expect(Math.abs(cents(measurePitch(x, freq), freq))).toBeLessThan(1)
    })
  }

  it('stays in tune across the brightness range', () => {
    for (const brightness of [0, 0.5, 1]) {
      const v = new Voice(SR, lcg(3))
      v.pluck(pluck(220, { brightness }))
      const x = render(v, 0.5).subarray(Math.round(0.1 * SR))
      expect(Math.abs(cents(measurePitch(x, 220), 220))).toBeLessThan(1.5)
    }
  })

  it('bends by retuning the delay: up a whole step is +200 cents', () => {
    const v = new Voice(SR, lcg(5))
    v.pluck(pluck(196))
    v.setBend(2 ** (2 / 12), true)
    const x = render(v, 0.5).subarray(Math.round(0.1 * SR))
    expect(cents(measurePitch(x, 220), 196)).toBeCloseTo(200, 0)
  })

  it('decays: the decay dial sets roughly when the string falls 60 dB', () => {
    const v = new Voice(SR, lcg(9))
    v.pluck(pluck(220, { decay: 1, brightness: 1 }))
    const x = render(v, 1.2)
    const rms = (a: number, b: number) => {
      let s = 0
      for (let i = Math.round(a * SR); i < Math.round(b * SR); i++) s += x[i] * x[i]
      return Math.sqrt(s / ((b - a) * SR))
    }
    const drop = 20 * Math.log10(rms(0.9, 1.0) / rms(0.05, 0.15))
    // −60 dB per second for the fundamental, a little faster overall as harmonics die first.
    expect(drop).toBeLessThan(-40)
    expect(drop).toBeGreaterThan(-75)
  })

  it('goes silent and reports its level for the visuals', () => {
    const v = new Voice(SR, lcg(1))
    v.pluck(pluck(440, { decay: 0.2, velocity: 0.5 }))
    render(v, 0.01)
    expect(v.level).toBeGreaterThan(0.2)
    expect(v.level).toBeLessThanOrEqual(0.5 + 1e-6)
    const tail = render(v, 2)
    expect(tail.at(-1)).toBe(0)
    expect(v.level).toBeLessThan(1e-4)
  })
})

describe('tuning', () => {
  it('maps strings onto the scale from the root', () => {
    expect(midiToHz(69)).toBe(440)
    expect(midiName(stringMidi(0, 'pentatonic', 'D'))).toBe('D3')
    expect([0, 1, 2, 3, 4, 5].map((i) => midiName(stringMidi(i, 'pentatonic', 'D')))).toEqual([
      'D3',
      'E3',
      'F#3',
      'A3',
      'B3',
      'D4',
    ])
    expect(midiName(stringMidi(2, 'minor', 'A'))).toBe('C4')
    expect(midiName(stringMidi(6, 'whole tone', 'C'))).toBe('C4')
  })
})
