/**
 * The magnetic field and the layout, kept free of the DOM so they can be tested.
 *
 * Each letter's influence is a smooth bell over its distance from the pointer; a spring then
 * chases it, and the font's axes are a straight blend from the resting end to the peak end.
 */

export type Mode = 'bulge' | 'dent'
export type Axes = { wght: number; wdth: number }

/** Roboto Flex's design space. */
export const AXIS_RANGE = { wght: [100, 1000], wdth: [25, 151] } as const

/** The thin, condensed end of the range: where bulging letters rest and denting letters go. */
export const THIN: Axes = { wght: 180, wdth: 40 }

/** Bell falloff: 1 at the pointer, about 1% by `radius` (a Gaussian with σ = radius / 3). */
export function influence(distance: number, radius: number) {
  if (radius <= 0) return 0
  const t = (3 * distance) / radius
  return Math.exp(-0.5 * t * t)
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

/** Where letters rest and where full influence takes them. Bulge swells thin type; dent thins fat type. */
export function ends(mode: Mode, maxWeight: number, maxWidth: number): { rest: Axes; peak: Axes } {
  const fat = {
    wght: clamp(maxWeight, THIN.wght, AXIS_RANGE.wght[1]),
    wdth: clamp(maxWidth, THIN.wdth, AXIS_RANGE.wdth[1]),
  }
  return mode === 'bulge' ? { rest: THIN, peak: fat } : { rest: fat, peak: THIN }
}

/** Axes at influence `i` (springs overshoot, so `i` may stray outside [0, 1]; the range clamps). */
export function axesAt(i: number, { rest, peak }: { rest: Axes; peak: Axes }): Axes {
  return {
    wght: clamp(rest.wght + (peak.wght - rest.wght) * i, ...AXIS_RANGE.wght),
    wdth: clamp(rest.wdth + (peak.wdth - rest.wdth) * i, ...AXIS_RANGE.wdth),
  }
}

/**
 * Breaks `words` into at most `maxLines` lines and picks the biggest font size that fits.
 * `widthOf(text)` is the width of `text` at a font size of 1.
 */
export function fitLines(
  words: string[],
  widthOf: (text: string) => number,
  box: { width: number; height: number; lineHeight: number; maxLines?: number },
): { lines: string[]; fontSize: number } {
  if (words.length === 0) return { lines: [], fontSize: 0 }
  const maxLines = Math.min(box.maxLines ?? 4, words.length)
  let best = { lines: [words.join(' ')], fontSize: 0 }
  for (let n = 1; n <= maxLines; n++) {
    const lines = balance(words, n, widthOf)
    const widest = Math.max(...lines.map(widthOf))
    const fontSize = Math.min(box.width / widest, box.height / (n * box.lineHeight))
    // Strictly bigger only: fewer lines win ties.
    if (fontSize > best.fontSize * 1.0001) best = { lines, fontSize }
  }
  return best
}

/** Splits `words` into `n` contiguous lines with the narrowest widest line. */
function balance(words: string[], n: number, widthOf: (text: string) => number): string[] {
  const line = (a: number, b: number) => words.slice(a, b).join(' ')
  // cost[k][i]: the widest line when the first i words make k lines.
  const cost = Array.from({ length: n + 1 }, () => Array<number>(words.length + 1).fill(Infinity))
  const cut = Array.from({ length: n + 1 }, () => Array<number>(words.length + 1).fill(0))
  cost[0][0] = 0
  for (let k = 1; k <= n; k++) {
    for (let i = k; i <= words.length; i++) {
      for (let j = k - 1; j < i; j++) {
        const c = Math.max(cost[k - 1][j], widthOf(line(j, i)))
        if (c < cost[k][i]) {
          cost[k][i] = c
          cut[k][i] = j
        }
      }
    }
  }
  const lines: string[] = []
  for (let k = n, i = words.length; k > 0; k--) {
    const j = cut[k][i]
    lines.unshift(line(j, i))
    i = j
  }
  return lines
}
