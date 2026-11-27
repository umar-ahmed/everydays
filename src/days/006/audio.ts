import { VOICES, type Pluck } from './ks.ts'
import type { Message } from './worklet.ts'
import workletUrl from './worklet.ts?worker&url'

type AudioSessionNavigator = Navigator & { audioSession?: { type: string } }

/**
 * The Web Audio graph: the Karplus-Strong worklet, then a dry path and a convolution reverb
 * with a generated impulse response, then a gentle compressor so a sixteen-string strum
 * doesn't clip.
 */
export class Engine {
  /** Each voice's output level, updated by the worklet ~60 times a second. */
  readonly levels = new Float32Array(VOICES)
  private node: AudioWorkletNode | null = null
  private readonly dry: GainNode
  private readonly wet: GainNode
  readonly ctx: AudioContext
  private readonly reverbIn: ConvolverNode

  private constructor(ctx: AudioContext) {
    this.ctx = ctx
    const out = new DynamicsCompressorNode(ctx, { threshold: -10, knee: 8, ratio: 4, attack: 0.003, release: 0.2 })
    out.connect(ctx.destination)
    this.dry = new GainNode(ctx)
    this.wet = new GainNode(ctx)
    this.dry.connect(out)
    const reverb = new ConvolverNode(ctx, { buffer: impulse(ctx, 2.6) })
    reverb.connect(this.wet).connect(out)
    this.reverbIn = reverb
  }

  /**
   * Call from inside a user gesture: the context is created and resumed synchronously there, as
   * iOS requires, and the worklet loads after.
   */
  static start(): Engine {
    // iOS 17+: play through the ringer switch where the browser allows it.
    const session = (navigator as AudioSessionNavigator).audioSession
    if (session) session.type = 'playback'
    const engine = new Engine(new AudioContext({ latencyHint: 'interactive' }))
    void engine.ctx.resume()
    void engine.load()
    return engine
  }

  private async load() {
    await this.ctx.audioWorklet.addModule(workletUrl)
    const node = new AudioWorkletNode(this.ctx, 'pluck-lab', { numberOfInputs: 0, outputChannelCount: [2] })
    node.port.onmessage = (e: MessageEvent<Float32Array>) => this.levels.set(e.data)
    node.connect(this.dry)
    node.connect(this.reverbIn)
    this.node = node
  }

  get ready() {
    return this.node !== null && this.ctx.state === 'running'
  }

  /** `delayMs` staggers the strings of one fast strum by when the pointer actually crossed them. */
  pluck(voice: number, pluck: Pluck, pan: number, delayMs = 0) {
    if (this.ctx.state !== 'running') void this.ctx.resume()
    this.send({ type: 'pluck', voice, pan, delay: Math.round((delayMs / 1000) * this.ctx.sampleRate), pluck })
  }

  bend(voice: number, ratio: number) {
    this.send({ type: 'bend', voice, ratio })
  }

  setReverb(mix: number) {
    const t = this.ctx.currentTime
    this.dry.gain.setTargetAtTime(1 - 0.55 * mix, t, 0.03)
    this.wet.gain.setTargetAtTime(1.1 * mix, t, 0.03)
  }

  private send(m: Message) {
    this.node?.port.postMessage(m)
  }
}

/** A stereo room: decaying noise that darkens as it fades, with a short pre-delay. */
function impulse(ctx: BaseAudioContext, seconds: number) {
  const sr = ctx.sampleRate
  const length = Math.round(seconds * sr)
  const buffer = new AudioBuffer({ numberOfChannels: 2, length, sampleRate: sr })
  const pre = Math.round(0.012 * sr)
  for (let ch = 0; ch < 2; ch++) {
    const data = buffer.getChannelData(ch)
    let lp = 0
    for (let i = pre; i < length; i++) {
      const t = (i - pre) / sr
      const k = 0.9 * Math.exp(-t * 1.6) + 0.05
      lp += k * (Math.random() * 2 - 1 - lp)
      data[i] = lp * Math.exp((-6.9 * t) / seconds) * 0.6
    }
  }
  return buffer
}
