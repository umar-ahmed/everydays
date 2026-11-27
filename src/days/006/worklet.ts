import { Voice, VOICES, type Pluck } from './ks.ts'

// The AudioWorkletGlobalScope isn't in TypeScript's libs.
declare const sampleRate: number
declare class AudioWorkletProcessor {
  readonly port: MessagePort
}
declare function registerProcessor(name: string, ctor: new () => AudioWorkletProcessor): void

/** What the main thread sends (see `audio.ts`). */
export type Message =
  | { type: 'pluck'; voice: number; pan: number; delay: number; pluck: Pluck }
  | { type: 'bend'; voice: number; ratio: number }

/** Posts each voice's level about 60 times a second. */
const LEVEL_EVERY = 6

class PluckLab extends AudioWorkletProcessor {
  private readonly voices = Array.from({ length: VOICES }, () => new Voice(sampleRate))
  private readonly pans = Array.from({ length: VOICES }, () => ({ l: Math.SQRT1_2, r: Math.SQRT1_2 }))
  /** Plucks a strum crossed partway through the last pointer event wait their turn, in samples. */
  private pending: { at: number; voice: number; pan: number; pluck: Pluck }[] = []
  private clock = 0
  private blocks = 0

  constructor() {
    super()
    this.port.onmessage = (e: MessageEvent<Message>) => {
      const m = e.data
      if (m.type === 'pluck')
        this.pending.push({ at: this.clock + m.delay, voice: m.voice, pan: m.pan, pluck: m.pluck })
      else this.voices[m.voice]?.setBend(m.ratio)
    }
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]) {
    const [left, right = left] = outputs[0]
    const frames = left.length
    left.fill(0)
    if (right !== left) right.fill(0)
    for (let i = 0; i < frames; i++, this.clock++) {
      if (this.pending.length) {
        this.pending = this.pending.filter((p) => {
          if (p.at > this.clock) return true
          const a = ((Math.max(-1, Math.min(1, p.pan)) + 1) * Math.PI) / 4
          this.pans[p.voice] = { l: Math.cos(a), r: Math.sin(a) }
          this.voices[p.voice]?.pluck(p.pluck)
          return false
        })
      }
      let l = 0
      let r = 0
      for (let v = 0; v < VOICES; v++) {
        const y = this.voices[v].next()
        if (y === 0) continue
        l += y * this.pans[v].l
        r += y * this.pans[v].r
      }
      left[i] = l * 0.5
      if (right !== left) right[i] = r * 0.5
    }
    if (++this.blocks % LEVEL_EVERY === 0) {
      this.port.postMessage(Float32Array.from(this.voices, (v) => v.level))
    }
    return true
  }
}

registerProcessor('pluck-lab', PluckLab)
