/**
 * Karplus-Strong plucked string, one sample at a time. The AudioWorklet runs this, and so do
 * the tests, which render it offline and measure its pitch.
 *
 * The loop is a delay line of N samples, a two-tap loss filter g·((1 − s) + s·z⁻¹) that dulls
 * high harmonics (phase delay ≈ s samples), and a first-order allpass that supplies the
 * fractional delay Δ (Jaffe & Smith 1983). Their sum N + s + Δ is exactly one period, sr / f,
 * so tuning is sample-accurate at any pitch. Bends retune the loop smoothly.
 */

export type Pluck = {
  /** Hz. */
  freq: number
  /** 0..1: loudness, and a little extra brightness. */
  velocity: number
  /** Fraction of the string from its end where it is plucked, 0..0.5; shapes the harmonics. */
  position: number
  /** 0..1: the loop filter and the colour of the noise burst. */
  brightness: number
  /** Seconds for the fundamental to fall 60 dB. */
  decay: number
}

/** Room for one period of a 20 Hz string at 192 kHz. */
const MAX_DELAY = 1 << 14
/** Below this the string is silent and skipped. */
const SILENT = 1e-5

export class Voice {
  private readonly buf = new Float32Array(MAX_DELAY)
  private write = 0
  private n = 100
  private c = 0
  private s = 0.5
  private g = 0.999
  private freq = 220
  private bend = 1
  private bendTarget = 1
  private xPrev = 0
  private apX = 0
  private apY = 0
  private active = false
  private quiet = 0
  /** Peak-following envelope of the output, for the visuals. */
  level = 0
  private readonly levelFall: number
  private readonly bendRate: number

  readonly sampleRate: number
  private readonly rand: () => number

  constructor(sampleRate: number, rand: () => number = Math.random) {
    this.sampleRate = sampleRate
    this.rand = rand
    this.levelFall = Math.exp(-1 / (0.03 * sampleRate))
    this.bendRate = 1 - Math.exp(-1 / (0.012 * sampleRate))
  }

  /** Total loop delay in samples for the current pitch and bend. */
  get period() {
    return this.sampleRate / (this.freq * this.bend)
  }

  private tune() {
    const p = this.period
    this.n = Math.min(MAX_DELAY - 1, Math.max(2, Math.floor(p - this.s - 0.1)))
    // Δ in [0.1, 1.1): the allpass is well-behaved there and its delay ≈ Δ at low frequencies.
    const delta = p - this.s - this.n
    this.c = (1 - delta) / (1 + delta)
  }

  pluck(p: Pluck) {
    this.freq = p.freq
    this.s = 0.5 - 0.47 * Math.min(1, Math.max(0, p.brightness))
    this.g = 10 ** (-3 / (p.freq * Math.max(0.05, p.decay)))
    this.tune()
    const n = this.n
    // A noise burst one period long, coloured by a one-pole lowpass, then a comb that removes
    // the harmonics with a node at the pluck point, then the DC taken out.
    const open = Math.min(1, 0.08 + 0.75 * p.brightness + 0.25 * p.velocity)
    const burst = new Float32Array(n)
    let lp = 0
    for (let k = 0; k < n; k++) {
      lp += open * (this.rand() * 2 - 1 - lp)
      burst[k] = lp
    }
    const m = Math.max(1, Math.round(Math.min(0.5, Math.max(0.02, p.position)) * n))
    const shaped = burst.map((v, k) => v - (k >= m ? burst[k - m] : 0))
    let mean = 0
    for (const v of shaped) mean += v / n
    let peak = 1e-9
    for (let k = 0; k < n; k++) peak = Math.max(peak, Math.abs((shaped[k] -= mean)))
    const amp = p.velocity / peak
    // Retriggering damps what is still ringing instead of stacking on top of it.
    for (let k = 0; k < n; k++) {
      const i = (this.write - n + k + MAX_DELAY) & (MAX_DELAY - 1)
      this.buf[i] = this.buf[i] * 0.25 + shaped[k] * amp
    }
    this.active = true
    this.quiet = 0
  }

  /** Pitch multiplier; glides there over ~12 ms. `snap` jumps instead (for tests). */
  setBend(ratio: number, snap = false) {
    this.bendTarget = ratio
    if (snap) {
      this.bend = ratio
      this.tune()
    }
  }

  next(): number {
    if (!this.active) {
      this.level *= this.levelFall
      return 0
    }
    if (this.bend !== this.bendTarget) {
      this.bend += (this.bendTarget - this.bend) * this.bendRate
      if (Math.abs(this.bend - this.bendTarget) < 1e-7) this.bend = this.bendTarget
      this.tune()
    }
    const mask = MAX_DELAY - 1
    const x = this.buf[(this.write - this.n + MAX_DELAY) & mask]
    const f = this.g * ((1 - this.s) * x + this.s * this.xPrev)
    this.xPrev = x
    const y = this.c * f + this.apX - this.c * this.apY
    this.apX = f
    this.apY = y
    this.buf[this.write] = y
    this.write = (this.write + 1) & mask
    const a = Math.abs(y)
    this.level = a > this.level ? a : this.level * this.levelFall
    // Stop once a whole period has been inaudible.
    this.quiet = a < SILENT ? this.quiet + 1 : 0
    if (this.quiet > this.n + 2) {
      this.active = false
      this.buf.fill(0)
      this.xPrev = this.apX = this.apY = 0
    }
    return y
  }
}

/** One voice per string, up to the most strings the dial allows. */
export const VOICES = 16

export const SCALES = {
  pentatonic: [0, 2, 4, 7, 9],
  major: [0, 2, 4, 5, 7, 9, 11],
  minor: [0, 2, 3, 5, 7, 8, 10],
  'whole tone': [0, 2, 4, 6, 8, 10],
} as const
export type ScaleName = keyof typeof SCALES

export const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] as const
export type NoteName = (typeof NOTE_NAMES)[number]

/** MIDI note of string `i` (0 = lowest), rooted in octave 3. */
export function stringMidi(i: number, scale: ScaleName, root: NoteName) {
  const steps = SCALES[scale]
  return 48 + NOTE_NAMES.indexOf(root) + steps[i % steps.length] + 12 * Math.floor(i / steps.length)
}

export const midiToHz = (m: number) => 440 * 2 ** ((m - 69) / 12)
export const midiName = (m: number) => `${NOTE_NAMES[((m % 12) + 12) % 12]}${Math.floor(m / 12) - 1}`
