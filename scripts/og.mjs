// Regenerates the 1200x630 Open Graph images in public/.
// Usage: start `npm run dev`, then `npm run og` (OG_BASE overrides the dev URL,
// OG_FRAME picks which frame of the day 001 sweep becomes the image).
import { mkdirSync, copyFileSync } from 'node:fs'
import { chromium } from 'playwright'

const base = process.env.OG_BASE || 'http://localhost:43117'
const pick = Number(process.env.OG_FRAME ?? 6)
const size = { width: 1200, height: 630 }
const framesDir = '/tmp/everydays-og-frames'
mkdirSync('public/001', { recursive: true })
mkdirSync(framesDir, { recursive: true })

const browser = await chromium.launch()

async function indexCard() {
  const page = await browser.newPage({ viewport: size })
  const blobs = [
    [905, 300, 210],
    [1040, 205, 120],
    [1010, 430, 150],
    [790, 455, 90],
    [1110, 340, 80],
    [760, 190, 70],
  ]
  await page.setContent(`<!doctype html><html><head><style>
    body{margin:0;width:1200px;height:630px;overflow:hidden;font-family:Inter,ui-sans-serif,system-ui,sans-serif;
      background:radial-gradient(90% 120% at 80% 45%,#2a0d24 0%,#0b0a12 55%,#06050a 100%);color:#ece9f5}
    .goo{position:absolute;inset:0;filter:url(#goo)}
    .b{position:absolute;border-radius:50%;background:hsl(322 96% 62%);transform:translate(-50%,-50%)}
    .copy{position:absolute;left:84px;top:150px}
    .kicker{font:500 18px ui-monospace,Menlo,monospace;letter-spacing:.2em;color:#f0abfcb0;text-transform:uppercase}
    h1{margin:18px 0 0;font-size:132px;line-height:1;font-weight:650;letter-spacing:-.04em}
    h1 span{color:#e879f9}
    p{margin:26px 0 0;font-size:30px;color:#ffffff99;max-width:560px;line-height:1.35}
    .foot{position:absolute;left:84px;bottom:64px;font:500 20px ui-monospace,Menlo,monospace;color:#ffffff59}
  </style></head><body>
    <svg width="0" height="0" style="position:absolute"><filter id="goo" color-interpolation-filters="sRGB">
      <feGaussianBlur in="SourceGraphic" stdDeviation="16" result="b"/>
      <feColorMatrix in="b" values="1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 38 -16" result="g"/>
      <feGaussianBlur in="g" stdDeviation="5" result="s"/>
      <feSpecularLighting in="s" surfaceScale="5" specularConstant=".8" specularExponent="24" lighting-color="#fff" result="sp">
        <feDistantLight azimuth="235" elevation="52"/></feSpecularLighting>
      <feComposite in="sp" in2="g" operator="in" result="si"/>
      <feComposite in="g" in2="si" operator="arithmetic" k2="1" k3=".7"/>
    </filter></svg>
    <div class="goo">${blobs.map(([x, y, d]) => `<div class="b" style="left:${x}px;top:${y}px;width:${d}px;height:${d}px"></div>`).join('')}</div>
    <div class="copy"><div class="kicker">daily interactive toys</div><h1>every<span>days</span></h1>
    <p>One small 30-minute toy a day, numbered and kept.</p></div>
    <div class="foot">001 · gooey cursor blobs</div>
  </body></html>`)
  await page.waitForTimeout(200)
  await page.screenshot({ path: 'public/og.png' })
  await page.close()
}

async function day001() {
  const page = await browser.newPage({ viewport: size })
  // A fake clock lets us step the springs frame by frame, so the capture is deterministic.
  await page.clock.install()
  await page.goto(`${base}/001/`)
  await page.addStyleTag({ content: '.dialkit-root,[data-chrome]{display:none!important}' })
  const tick = (ms) => page.clock.runFor(ms)
  await page.mouse.move(330, 390)
  await page.mouse.click(330, 390)
  await page.mouse.click(905, 235)
  await page.mouse.move(330, 390)
  await tick(2500)
  // Sweep away from the first drop so the trail stretches out of it like taffy.
  const frames = 60
  for (let i = 1; i <= frames; i++) {
    const t = Math.min(i / 24, 1)
    await page.mouse.move(330 + (800 - 330) * t, 390 - Math.sin(t * Math.PI) * 70 - 120 * t)
    await tick(16)
    if (i % 4 === 0) await page.screenshot({ path: `${framesDir}/${String(i / 4 - 1).padStart(2, '0')}.png` })
  }
  copyFileSync(`${framesDir}/${String(pick).padStart(2, '0')}.png`, 'public/001/og.png')
  await page.close()
}

await indexCard()
await day001()
await browser.close()
console.log(`wrote public/og.png and public/001/og.png (frame ${pick}; all frames in ${framesDir})`)
