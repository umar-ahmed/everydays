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

type V3 = [number, number, number]

/** A camera orbiting `target`: elevation and azimuth in radians, distance in cm. */
export type Pose = { target: V3; dist: number; elevation: number; azimuth: number }

/** The sheet's bounding box: width x depth centered on the origin, from y = 0 up to `top`. */
export type SheetBox = { width: number; depth: number; top: number }

/** Viewport in CSS pixels. */
export type View = { width: number; height: number }

export type CameraFit = Pose & {
  /** NDC shift (off-axis projection) that centers the sheet in the space the page chrome leaves. */
  offset: [number, number]
  /** Clip planes for the fit pose; zoomed poses get theirs from clipPlanes. */
  near: number
  far: number
}

/** Pixels kept clear around the sheet: ~9% per side, more for the dials button and bottom bar. */
export function chromeMargins({ width, height }: View) {
  const side = 0.09 * width
  return { left: side, right: side, top: Math.max(0.09 * height, 68), bottom: Math.max(0.09 * height, 84) }
}

export function cameraPosition({ target, dist, elevation, azimuth }: Pose): V3 {
  const ce = Math.cos(elevation)
  return [
    target[0] + Math.sin(azimuth) * ce * dist,
    target[1] + Math.sin(elevation) * dist,
    target[2] + Math.cos(azimuth) * ce * dist,
  ]
}

/** Camera basis: right, up, forward. */
function basis({ elevation, azimuth }: Pose) {
  const ce = Math.cos(elevation)
  const se = Math.sin(elevation)
  const f: V3 = [-Math.sin(azimuth) * ce, -se, -Math.cos(azimuth) * ce]
  const r: V3 = [Math.cos(azimuth), 0, -Math.sin(azimuth)]
  const u: V3 = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
  return { r, u, f }
}

const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

function tangents(view: View) {
  const ty = Math.tan((FOV * Math.PI) / 360)
  return { tx: ty * (view.width / view.height), ty }
}

function corners({ width, depth, top }: SheetBox): V3[] {
  const out: V3[] = []
  for (const x of [-width / 2, width / 2]) for (const z of [-depth / 2, depth / 2]) for (const y of [0, top]) out.push([x, y, z])
  return out
}

/** View depth of a world point. */
export function viewDepth(pose: Pose, p: V3) {
  const c = cameraPosition(pose)
  return dot([p[0] - c[0], p[1] - c[1], p[2] - c[2]], basis(pose).f)
}

/** NDC bounds of the box seen from `pose` (symmetric frustum), or null if it reaches behind the camera. */
function bounds(pose: Pose, box: SheetBox, view: View) {
  const c = cameraPosition(pose)
  const { r, u, f } = basis(pose)
  const { tx, ty } = tangents(view)
  let x0 = Infinity
  let x1 = -Infinity
  let y0 = Infinity
  let y1 = -Infinity
  for (const p of corners(box)) {
    const d: V3 = [p[0] - c[0], p[1] - c[1], p[2] - c[2]]
    const z = dot(d, f)
    if (z <= 1e-6) return null
    const x = dot(d, r) / (z * tx)
    const y = dot(d, u) / (z * ty)
    x0 = Math.min(x0, x)
    x1 = Math.max(x1, x)
    y0 = Math.min(y0, y)
    y1 = Math.max(y1, y)
  }
  return { x0, x1, y0, y1 }
}

/** Projected NDC bounds of the sheet in the off-axis frame (as drawn), or null if it reaches behind the camera. */
export function sheetBounds(pose: Pose, offset: [number, number], box: SheetBox, view: View) {
  const b = bounds(pose, box, view)
  return b && { x0: b.x0 + offset[0], x1: b.x1 + offset[0], y0: b.y0 + offset[1], y1: b.y1 + offset[1] }
}

/** The NDC rectangle the sheet may fill. */
export function safeRect(view: View) {
  const m = chromeMargins(view)
  return {
    x0: -1 + (2 * m.left) / view.width,
    x1: 1 - (2 * m.right) / view.width,
    y0: -1 + (2 * m.bottom) / view.height,
    y1: 1 - (2 * m.top) / view.height,
  }
}

/**
 * Frames the sheet exactly: the camera distance at which the projected sheet just fills the
 * safe rectangle, then an off-axis shift that centers it there (so the receding far edge and
 * the bottom chrome don't push it off-center).
 */
export function fitView(box: SheetBox, view: View, elevationDeg = ELEVATION): CameraFit {
  const elevation = (elevationDeg * Math.PI) / 180
  const rect = safeRect(view)
  const pose = (dist: number): Pose => ({ target: [0, 0, 0], dist, elevation, azimuth: 0 })
  const overflow = (dist: number) => {
    const b = bounds(pose(dist), box, view)
    if (!b) return Infinity
    return Math.max((b.x1 - b.x0) / (rect.x1 - rect.x0), (b.y1 - b.y0) / (rect.y1 - rect.y0))
  }
  const size = Math.max(box.width, box.depth, 1e-3)
  let lo = size * 0.05
  let hi = size * 200
  for (let i = 0; i < 60; i++) {
    const mid = Math.sqrt(lo * hi)
    if (overflow(mid) > 1) lo = mid
    else hi = mid
  }
  const fit = pose(hi)
  const b = bounds(fit, box, view)!
  const offset: [number, number] = [(rect.x0 + rect.x1) / 2 - (b.x0 + b.x1) / 2, (rect.y0 + rect.y1) / 2 - (b.y0 + b.y1) / 2]
  const { near, far } = clipPlanes(fit, offset, box, view)
  return { ...fit, offset, near, far }
}

/** Width / depth of the sheet that fills `view` at the default elevation. */
export function sheetAspect(view: View) {
  const rect = safeRect(view)
  const want = ((rect.x1 - rect.x0) * view.width) / ((rect.y1 - rect.y0) * view.height)
  const got = (aspect: number) => {
    const box = { width: aspect, depth: 1, top: 0 }
    const b = bounds(fitView(box, view), box, view)!
    return ((b.x1 - b.x0) * view.width) / ((b.y1 - b.y0) * view.height)
  }
  let lo = 0.1
  let hi = 10
  for (let i = 0; i < 40; i++) {
    const mid = Math.sqrt(lo * hi)
    if (got(mid) < want) lo = mid
    else hi = mid
  }
  return Math.sqrt(lo * hi)
}

/**
 * Clip planes for any pose. Near is a lower bound on the depth of everything visible: the
 * frustum's corner rays meet the plane over the dome tops no closer than this, and nothing in
 * view rises above it. Far covers the sheet's farthest corner and, at the fit framing, the
 * table out to where it fades (grazing views see much further across it).
 */
export function clipPlanes(pose: Pose, offset: [number, number], box: SheetBox, view: View, minFar = 0) {
  const c = cameraPosition(pose)
  const { r, u, f } = basis(pose)
  const { tx, ty } = tangents(view)
  const ceiling = box.top * 1.25 + 0.05
  let nearest = Infinity
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      const x = (sx - offset[0]) * tx
      const y = (sy - offset[1]) * ty
      const dir: V3 = [f[0] + r[0] * x + u[0] * y, f[1] + r[1] * x + u[1] * y, f[2] + r[2] * x + u[2] * y]
      // dir has unit forward component, so the ray parameter is the view depth.
      if (dir[1] < 0) nearest = Math.min(nearest, (ceiling - c[1]) / dir[1])
    }
  }
  if (!Number.isFinite(nearest)) nearest = pose.dist * 0.3
  const near = Math.max(nearest * 0.8, 1e-3)
  let farthest = 0
  for (const p of corners(box)) farthest = Math.max(farthest, viewDepth(pose, p))
  const elevationDeg = (pose.elevation * 180) / Math.PI
  const table = pose.dist * (3 + 4 * Math.max(0, 1 - elevationDeg / 45))
  return { near, far: Math.max(farthest * 1.25, table, minFar, near * 2) }
}

/** Where the ray through an NDC point (off-axis by `offset`) meets the plane y = planeY, or null. */
export function rayHit(pose: Pose, offset: [number, number], view: View, ndc: [number, number], planeY: number): V3 | null {
  const c = cameraPosition(pose)
  const { r, u, f } = basis(pose)
  const { tx, ty } = tangents(view)
  const x = (ndc[0] - offset[0]) * tx
  const y = (ndc[1] - offset[1]) * ty
  const dir: V3 = [f[0] + r[0] * x + u[0] * y, f[1] + r[1] * x + u[1] * y, f[2] + r[2] * x + u[2] * y]
  if (dir[1] > -1e-4) return null
  const t = (planeY - c[1]) / dir[1]
  return [c[0] + dir[0] * t, planeY, c[2] + dir[2] * t]
}

/**
 * Zoom limits as camera distances: in close enough that one bubble nearly fills the screen
 * (but never down into the domes), out to a little beyond the fit framing.
 */
export function zoomRange(fit: CameraFit, size: number, domeHeight: number, view: View) {
  const { tx, ty } = tangents(view)
  const macro = size / (1.7 * Math.min(tx, ty))
  const clear = (domeHeight * 1.6 + size * 0.15) / Math.sin(fit.elevation)
  return { min: Math.min(Math.max(macro, clear), fit.dist), max: fit.dist * 1.12 }
}

/** Smallest view-depth step (cm) the reversed-Z float32 depth buffer resolves at view depth z. */
export function depthStep(z: number, near: number, far: number) {
  // Reversed: depth(z) = near (far - z) / (z (far - near)), |d depth / dz| = near far / ((far - near) z^2),
  // and a float32 resolves depth to ~2^-23 of its value, so precision is relative to z, not near.
  const depth = (near * (far - z)) / (z * (far - near))
  return (Math.max(depth, 2 ** -126) * 2 ** -23 * (far - near) * z * z) / (near * far)
}
