/**
 * Bubble map generators.
 *
 * A generator is a GPU pass that fills `mapRaw`, one vec4<f32> per map texel (see layout.ts for
 * the grid), row-major from the sheet's (-width/2, -depth/2) corner:
 *
 *   x  raw bubble id of the nearest bubble, any non-negative integer (or -1: no bubble near)
 *   y  signed distance to that bubble's edge in cm, negative inside
 *   z  per-bubble seed in [0, 1) (free for variation)
 *   w  coverage: how many bubble footprints contain this texel (> 1 means footprints overlap)
 *
 * Raw ids need not be dense or ordered; sheet.ts compacts them to 0..n-1 after a one-time
 * readback, writes the canonical `map`, and derives everything else (dome heights from the
 * distance, per-bubble volumes, centers, the pop counter total) from that data. Nothing
 * downstream knows how the map was made.
 *
 * Plugging in an image generator later: upload the image as a texture, threshold it, label
 * connected components on the GPU (label propagation or a jump-flood pass seeded with texel
 * indices, iterated until stable), then run a jump flood from the component boundaries to get a
 * signed distance. Write label -> x, distance -> y, hash(label) -> z and 1 or 0 -> w. Components
 * touching the sheet edge can be dropped (x = -1) to keep the "whole bubbles only" rule.
 */
import type { ComputeNode } from 'three/webgpu'
import type { Layout } from '../layout.ts'

export interface MapGenerator<P> {
  /** One dispatch fills mapRaw for the current uniforms. */
  readonly compute: ComputeNode
  /** Point the generator at a layout and its own options before dispatching. */
  configure(layout: Layout, options: P): void
}
