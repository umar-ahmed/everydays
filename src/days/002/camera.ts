/** The table sits this far under the flat film (cm); the depth buffer has to resolve it. */
export const TABLE_GAP = 0.004
/** The back film renders this far under the cap layer where they are sealed, to avoid z-fighting. */
export const BACK_OFFSET = 0.002
/** Smallest depth separation the renderer relies on: film/back and back/table. */
export const MIN_GAP = Math.min(BACK_OFFSET, TABLE_GAP - BACK_OFFSET)

export const FOV = 30
/** Default camera elevation (degrees); the Camera > Elevation dial ranges over ELEVATIONS. */
export const ELEVATION = 56
export const ELEVATIONS = [12, 85] as const

export type CameraFit = {
  dist: number
  elevation: number
  near: number
  far: number
  /** Look-at point (x, y, z). */
  target: [number, number, number]
}

/**
 * Frames the sheet and sizes the clip range to it. Near and far scale with the camera
 * distance: a fixed near plane of 0.1 cm left too little depth precision to keep the film
 * in front of the table once a large sheet pushed the camera ~60 cm back.
 */
export function fitCamera(width: number, depth: number, aspect: number, elevationDeg = ELEVATION): CameraFit {
  const vfov = (FOV * Math.PI) / 180
  const hfov = 2 * Math.atan(Math.tan(vfov / 2) * aspect)
  // Leave room for the bottom chrome; the receding far edge needs less height than the near edge.
  const dist = Math.max((width * 0.58) / Math.tan(hfov / 2), (depth * 0.5) / Math.tan(vfov / 2)) * 1.05
  // The nearest thing in view (the table under the bottom of the frustum) is ~0.85 dist away
  // and the farthest visible table ~1.3 dist, with room for the idle sway.
  const elevation = (elevationDeg * Math.PI) / 180
  // Low, grazing views see the far side of the sheet much deeper, so far grows as elevation drops.
  const far = dist * (3 + 4 * Math.max(0, 1 - elevationDeg / 45))
  return { dist, elevation, near: dist * 0.3, far, target: [0, 0, depth * 0.04] }
}

/** Smallest view-depth step (cm) a 24-bit-precision depth buffer resolves at view depth z. */
export function depthStep(z: number, near: number, far: number) {
  // depth(z) = far (z - near) / (z (far - near)), so d depth / dz = far near / ((far - near) z^2).
  return (2 ** -24 * (far - near) * z * z) / (far * near)
}
