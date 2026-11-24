import { uniform } from 'three/tsl'
import * as THREE from 'three/webgpu'

type RGB = [number, number, number]

/**
 * A colorization: a four-stop height ramp for gentle ground (stops at 0, 0.3, 0.6 and 1 of the
 * peak height), rock for steep ground (warmer where water cut into it), sediment, and the snow
 * line the palette starts with.
 */
export type Palette = {
  ramp: [RGB, RGB, RGB, RGB]
  rock: RGB
  rockCut: RGB
  sand: RGB
  /** Strata banding in exposed rock, 0..~0.2. */
  strata: number
  /** How much steep slopes turn to rock (0: the ramp alone, as on a hypsometric map). */
  rockiness: number
  /** Contour line strength (topographic maps). */
  contours: number
  water: RGB
  /** Snow line the preset starts with (fraction of the peak height; > 1 means no snow). */
  snowLine: number
}

export const PALETTES = {
  natural: {
    ramp: [
      [0.3, 0.35, 0.16],
      [0.33, 0.37, 0.18],
      [0.42, 0.4, 0.26],
      [0.45, 0.42, 0.34],
    ],
    rock: [0.38, 0.35, 0.32],
    rockCut: [0.5, 0.36, 0.27],
    sand: [0.74, 0.65, 0.47],
    strata: 0.05,
    rockiness: 1,
    contours: 0,
    water: [0.16, 0.32, 0.36],
    snowLine: 0.88,
  },
  alpine: {
    ramp: [
      [0.13, 0.22, 0.11],
      [0.2, 0.3, 0.14],
      [0.36, 0.38, 0.24],
      [0.44, 0.44, 0.42],
    ],
    rock: [0.4, 0.41, 0.43],
    rockCut: [0.46, 0.42, 0.38],
    sand: [0.6, 0.58, 0.52],
    strata: 0.04,
    rockiness: 1,
    contours: 0,
    water: [0.12, 0.34, 0.42],
    snowLine: 0.36,
  },
  desert: {
    ramp: [
      [0.8, 0.62, 0.42],
      [0.76, 0.52, 0.32],
      [0.66, 0.38, 0.22],
      [0.58, 0.32, 0.2],
    ],
    rock: [0.64, 0.33, 0.2],
    rockCut: [0.78, 0.5, 0.3],
    sand: [0.86, 0.72, 0.5],
    strata: 0.16,
    rockiness: 1,
    contours: 0,
    water: [0.2, 0.36, 0.34],
    snowLine: 1.5,
  },
  volcanic: {
    ramp: [
      [0.1, 0.1, 0.1],
      [0.16, 0.15, 0.14],
      [0.22, 0.19, 0.17],
      [0.3, 0.26, 0.24],
    ],
    rock: [0.12, 0.11, 0.11],
    rockCut: [0.36, 0.14, 0.07],
    sand: [0.14, 0.13, 0.13],
    strata: 0.03,
    rockiness: 1,
    contours: 0,
    water: [0.1, 0.22, 0.24],
    snowLine: 0.92,
  },
  arctic: {
    ramp: [
      [0.62, 0.66, 0.68],
      [0.78, 0.82, 0.86],
      [0.88, 0.91, 0.95],
      [0.93, 0.95, 0.98],
    ],
    rock: [0.32, 0.34, 0.38],
    rockCut: [0.4, 0.38, 0.38],
    sand: [0.55, 0.56, 0.58],
    strata: 0.03,
    rockiness: 1,
    contours: 0,
    water: [0.2, 0.4, 0.5],
    snowLine: 0.12,
  },
  topographic: {
    ramp: [
      [0.36, 0.58, 0.34],
      [0.86, 0.82, 0.52],
      [0.66, 0.46, 0.3],
      [0.9, 0.9, 0.9],
    ],
    rock: [0.6, 0.45, 0.32],
    rockCut: [0.6, 0.45, 0.32],
    sand: [0.86, 0.82, 0.52],
    strata: 0,
    rockiness: 0,
    contours: 0.35,
    water: [0.3, 0.55, 0.85],
    snowLine: 1.5,
  },
  monochrome: {
    ramp: [
      [0.24, 0.24, 0.24],
      [0.4, 0.4, 0.4],
      [0.58, 0.58, 0.58],
      [0.78, 0.78, 0.78],
    ],
    rock: [0.42, 0.42, 0.42],
    rockCut: [0.5, 0.5, 0.5],
    sand: [0.62, 0.62, 0.62],
    strata: 0.04,
    rockiness: 0.8,
    contours: 0,
    water: [0.2, 0.2, 0.22],
    snowLine: 1.5,
  },
} satisfies Record<string, Palette>

export type PaletteName = keyof typeof PALETTES
export const PALETTE_NAMES = Object.keys(PALETTES) as PaletteName[]

const v3 = (c: RGB) => new THREE.Vector3(...c)

/** The palette and snow settings as shader uniforms, so presets and dials swap live. */
export class Look {
  readonly ramp = [0, 1, 2, 3].map(() => uniform(new THREE.Vector3()))
  readonly rock = uniform(new THREE.Vector3())
  readonly rockCut = uniform(new THREE.Vector3())
  readonly sand = uniform(new THREE.Vector3())
  readonly water = uniform(new THREE.Vector3())
  readonly strata = uniform(0)
  readonly rockiness = uniform(1)
  readonly contours = uniform(0)
  /** Snow line (fraction of peak height), softness, slope limit (cos of the angle), aspect bias, sparkle. */
  readonly snowLine = uniform(0.88)
  readonly snowSoftness = uniform(0.08)
  readonly snowSlope = uniform(Math.cos((55 * Math.PI) / 180))
  readonly snowAspect = uniform(0.3)
  readonly sparkle = uniform(0.5)
  /** Colour transport shown (0/1), and its vividness boost. */
  readonly colorOn = uniform(1)
  readonly vivid = uniform(0.5)

  setPalette(p: Palette) {
    p.ramp.forEach((c, k) => this.ramp[k].value.copy(v3(c)))
    this.rock.value.copy(v3(p.rock))
    this.rockCut.value.copy(v3(p.rockCut))
    this.sand.value.copy(v3(p.sand))
    this.water.value.copy(v3(p.water))
    this.strata.value = p.strata
    this.rockiness.value = p.rockiness
    this.contours.value = p.contours
  }
}
