import type { MotionValue } from 'motion/react'

export type Body = {
  x: number
  y: number
  vx: number
  vy: number
  r: number
  mx: MotionValue<number>
  my: MotionValue<number>
}

export type World = {
  /** Screen-space gravity per axis in [-1, 1], from tilt. */
  gx: number
  gy: number
  /** Acceleration in px/s² at full tilt. */
  gravity: number
  /** Exponential velocity decay rate, 1/s. */
  friction: number
  /** Edge restitution, 0..1. */
  bounce: number
  width: number
  height: number
}

const MAX_SPEED = 5000
// Blobs may overlap this far before pushing apart, so touching drops still goo together.
const CONTACT = 0.8
const REPEL = 70

export function step(bodies: Body[], world: World, dt: number) {
  const { gx, gy, gravity, friction, bounce, width, height } = world
  const decay = Math.exp(-friction * dt)

  for (let i = 0; i < bodies.length; i++) {
    const a = bodies[i]
    let ax = gx * gravity
    let ay = gy * gravity
    for (let j = 0; j < bodies.length; j++) {
      if (i === j) continue
      const b = bodies[j]
      const dx = a.x - b.x
      const dy = a.y - b.y
      const dist = Math.hypot(dx, dy)
      const min = (a.r + b.r) * CONTACT
      if (dist >= min) continue
      if (dist < 0.01) {
        ax += (i < j ? -1 : 1) * REPEL * min
        continue
      }
      const push = (REPEL * (min - dist)) / dist
      ax += dx * push
      ay += dy * push
    }
    a.vx = (a.vx + ax * dt) * decay
    a.vy = (a.vy + ay * dt) * decay
    const speed = Math.hypot(a.vx, a.vy)
    if (speed > MAX_SPEED) {
      a.vx *= MAX_SPEED / speed
      a.vy *= MAX_SPEED / speed
    }
  }

  for (const b of bodies) {
    b.x += b.vx * dt
    b.y += b.vy * dt
    const right = Math.max(b.r, width - b.r)
    const bottom = Math.max(b.r, height - b.r)
    if (b.x < b.r) {
      b.x = b.r
      b.vx = Math.abs(b.vx) * bounce
    } else if (b.x > right) {
      b.x = right
      b.vx = -Math.abs(b.vx) * bounce
    }
    if (b.y < b.r) {
      b.y = b.r
      b.vy = Math.abs(b.vy) * bounce
    } else if (b.y > bottom) {
      b.y = bottom
      b.vy = -Math.abs(b.vy) * bounce
    }
    b.mx.set(b.x)
    b.my.set(b.y)
  }
}
