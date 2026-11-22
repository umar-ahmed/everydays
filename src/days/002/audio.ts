export type Pop = {
  /** Gauge pressure at burst, atm. */
  pressure: number
  /** Bubble footprint diameter, cm. */
  size: number
  /** -1 (left) .. 1 (right). */
  pan: number
}

/**
 * Synthesised pops, no samples. A pop is the stored overpressure escaping in one go:
 * an N-shaped pressure pulse about 2R/c long (so effectively broadband), scaled by the
 * burst pressure; a short band-passed ring whose centre falls with bubble size (the
 * cavity and film ringing); and a small low thump from the film snapping flat.
 * Pinholed bubbles hiss instead: looped noise whose level follows the venting flow.
 */
export class PopSynth {
  private ctx: AudioContext | null = null
  private out: GainNode | null = null
  private noise: AudioBuffer | null = null
  private hissGain: GainNode | null = null
  private recent = 0
  private recentAt = 0
  volume = 0.7
  pitch = 1

  /** Must run inside a user gesture (browsers keep audio locked until then). */
  unlock() {
    if (!this.ctx) {
      const Ctor = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      const ctx = new Ctor({ latencyHint: 'interactive' })
      const limiter = ctx.createDynamicsCompressor()
      limiter.threshold.value = -10
      limiter.knee.value = 6
      limiter.ratio.value = 12
      limiter.attack.value = 0.001
      limiter.release.value = 0.08
      const out = ctx.createGain()
      out.connect(limiter).connect(ctx.destination)
      const noise = ctx.createBuffer(1, Math.round(ctx.sampleRate * 0.25), ctx.sampleRate)
      const data = noise.getChannelData(0)
      for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1
      this.ctx = ctx
      this.out = out
      this.noise = noise

      const hiss = ctx.createBufferSource()
      hiss.buffer = noise
      hiss.loop = true
      const hp = ctx.createBiquadFilter()
      hp.type = 'highpass'
      hp.frequency.value = 3500
      const peak = ctx.createBiquadFilter()
      peak.type = 'peaking'
      peak.frequency.value = 7000
      peak.gain.value = 8
      this.hissGain = ctx.createGain()
      this.hissGain.gain.value = 0
      hiss.connect(hp).connect(peak).connect(this.hissGain).connect(out)
      hiss.start()
    }
    if (this.ctx.state !== 'running') void this.ctx.resume()
  }

  get ready() {
    return this.ctx?.state === 'running'
  }

  play({ pressure, size, pan }: Pop) {
    const { ctx, out, noise } = this
    if (!ctx || !out || !noise || ctx.state !== 'running' || this.volume <= 0) return
    const now = ctx.currentTime
    // A rolling pin can pop a dozen bubbles in a frame; thin the crowd instead of clipping.
    if (now - this.recentAt > 0.03) this.recent = 0
    this.recentAt = now
    if (++this.recent > 6) return

    const p = Math.min(Math.max(pressure, 0.05), 4)
    const jitter = () => 1 + (Math.random() - 0.5) * 0.16
    const gain = this.volume * Math.min(1, 0.25 + 0.5 * Math.pow(p, 0.7)) * jitter()
    // Smaller cavities ring higher; higher pressure releases faster, so the click is brighter.
    const ring = (2300 / Math.max(size, 0.3)) * this.pitch * jitter()
    const decay = (0.012 + 0.03 * Math.min(size, 2.5) / 2.5) / (0.8 + 0.4 * p)
    const start = now + 0.002

    const panner = ctx.createStereoPanner()
    panner.pan.value = Math.max(-1, Math.min(1, pan)) * 0.7
    panner.connect(out)
    const offset = Math.random() * (noise.duration - 0.1)

    // N-wave: compression then rarefaction over ~2R/c (tens of microseconds for a 1 cm bubble),
    // so at audio rates it is a few-sample doublet with energy right up the spectrum.
    const halfSamples = Math.max(1, Math.round(((size / 100) / 343) * ctx.sampleRate))
    const pulse = ctx.createBuffer(1, halfSamples * 2 + 2, ctx.sampleRate)
    const pd = pulse.getChannelData(0)
    for (let k = 0; k < halfSamples * 2; k++) pd[k] = 1 - k / halfSamples
    const click = ctx.createBufferSource()
    click.buffer = pulse
    const clickGain = ctx.createGain()
    clickGain.gain.value = gain * (0.6 + 0.5 * Math.min(p, 2))
    click.connect(clickGain).connect(panner)
    click.start(start)

    const body = ctx.createBufferSource()
    body.buffer = noise
    const band = ctx.createBiquadFilter()
    band.type = 'bandpass'
    band.frequency.value = ring
    band.Q.value = 3 + 3 * Math.min(p, 2)
    const bodyGain = ctx.createGain()
    bodyGain.gain.setValueAtTime(gain * 1.6, start)
    bodyGain.gain.exponentialRampToValueAtTime(0.0001, start + decay * 5)
    body.connect(band).connect(bodyGain).connect(panner)
    body.start(start, offset, decay * 5 + 0.01)

    const thump = ctx.createOscillator()
    thump.type = 'sine'
    thump.frequency.setValueAtTime(ring * 0.18, start)
    thump.frequency.exponentialRampToValueAtTime(ring * 0.07, start + 0.03)
    const thumpGain = ctx.createGain()
    thumpGain.gain.setValueAtTime(gain * 0.35, start)
    thumpGain.gain.exponentialRampToValueAtTime(0.0001, start + 0.04)
    thump.connect(thumpGain).connect(panner)
    thump.start(start)
    thump.stop(start + 0.05)
  }

  /** 0..1 venting level from pinholed bubbles; smoothed so per-frame readbacks don't zipper. */
  hiss(level: number) {
    const { ctx, hissGain } = this
    if (!ctx || !hissGain) return
    const target = Math.min(1, Math.max(0, level)) * this.volume * 0.35
    hissGain.gain.setTargetAtTime(target, ctx.currentTime, target > hissGain.gain.value ? 0.015 : 0.06)
  }

  dispose() {
    void this.ctx?.close()
    this.ctx = null
  }
}
