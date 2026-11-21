import { useCallback, useEffect, useRef, useState } from 'react'

// prompt: iOS-style permission API present and not yet granted.
// listening: may have a sensor; waiting for a real reading (desktops never send one).
export type TiltStatus = 'unsupported' | 'prompt' | 'listening' | 'active' | 'denied'

type RequestPermission = () => Promise<'granted' | 'denied'>

function permissionApi(): RequestPermission | undefined {
  if (typeof DeviceOrientationEvent === 'undefined') return undefined
  const ctor = DeviceOrientationEvent as unknown as { requestPermission?: RequestPermission }
  return typeof ctor.requestPermission === 'function' ? ctor.requestPermission.bind(ctor) : undefined
}

// Desktop Chrome also exposes requestPermission, so only offer the prompt where a
// motion sensor is plausible: touch devices (phones, tablets, iPads reporting as Mac).
function initialStatus(): TiltStatus {
  if (typeof window === 'undefined' || !('DeviceOrientationEvent' in window)) return 'unsupported'
  const touch = navigator.maxTouchPoints > 0 || window.matchMedia('(any-pointer: coarse)').matches
  return permissionApi() && touch ? 'prompt' : 'listening'
}

/** Calls `onGravity(x, y)` with screen-space gravity per axis in [-1, 1] (+x right, +y down). */
export function useTilt(enabled: boolean, onGravity: (x: number, y: number) => void) {
  const [status, setStatus] = useState<TiltStatus>(initialStatus)
  const callback = useRef(onGravity)
  useEffect(() => {
    callback.current = onGravity
  })

  // Also listens while in `prompt`: Safari may already have permission from earlier in the session.
  const listen = enabled && status !== 'unsupported' && status !== 'denied'
  useEffect(() => {
    if (!listen) return
    const onOrient = (e: DeviceOrientationEvent) => {
      if (e.beta == null || e.gamma == null) return
      const rad = Math.PI / 180
      const dx = Math.sin(e.gamma * rad)
      const dy = Math.sin(e.beta * rad)
      const a = (screen.orientation?.angle ?? 0) * rad
      callback.current(dx * Math.cos(a) + dy * Math.sin(a), -dx * Math.sin(a) + dy * Math.cos(a))
      setStatus('active')
    }
    window.addEventListener('deviceorientation', onOrient)
    return () => window.removeEventListener('deviceorientation', onOrient)
  }, [listen])

  const request = useCallback(async () => {
    const ask = permissionApi()
    if (!ask) return
    try {
      const result: string = await ask()
      setStatus(result === 'granted' ? 'listening' : result === 'denied' ? 'denied' : 'prompt')
    } catch {
      setStatus('denied')
    }
  }, [])

  return { status, request }
}
