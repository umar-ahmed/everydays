# everydays

A home for daily 30-minute interactive toys, one per numbered day. Each day is its own page (`/001/`, `/002/`, …) and the root `/` lists them all.

Built with Vite + React + TypeScript, [Motion](https://motion.dev) (`motion/react`), [DialKit](https://github.com/joshpuckett/dialkit) for live tuning panels, and Tailwind. Day 002 uses [three.js](https://threejs.org) with its WebGPU renderer and TSL compute shaders.

## Run it

```bash
npm install
npm run dev      # http://localhost:43117
npm run build    # static site in dist/
npm run preview  # serve dist/ at http://localhost:43118
```

## Days

| Day | Title | Concept |
| --- | --- | --- |
| [001](./001/) | gooey cursor blobs | Spring-chained metaballs (SVG blur + alpha-threshold filter) that stretch on fast moves and melt back together. Click to drop blobs, then drag the goo into them. On a phone, tilt sets gravity: drops roll like marbles of goo, coast when you level out and bounce off the edges (iOS asks via an "enable tilt" button). |
| [002](./002/) | bubble wrap | An XPBD cloth on the GPU (WebGPU compute, many small substeps of Jacobi passes) whose domes hold gas: squeezing one raises its pressure adiabatically until it bursts, collapses and crinkles. Pops are read back from the GPU and synthesised with Web Audio; some bubbles have pinholes and hiss flat instead. Six dome shapes, a finger or rolling pin, thin-film iridescence and a procedural crease shader. Needs WebGPU; other browsers get a message. |

## Adding a day

1. Append an entry to `src/days.ts` (slug, title, one-line concept, date).
2. Copy `001/index.html` to `NNN/index.html` and point its script at `src/days/NNN/main.tsx`.
3. Build the toy in `src/days/NNN/`.
4. Add a 1200x630 `public/NNN/og.png`.

The Vite config reads `src/days.ts` to register each page as a build input and to fill in the page's title and Open Graph tags. The build fails if a listed day has no `index.html`.

## Open Graph and the base URL

Every page gets `og:title`, `og:description`, `og:image`, `og:url` and `twitter:card=summary_large_image`, injected at build time in place of the `<!-- head:meta -->` comment. The absolute URLs use the first base URL found:

1. `SITE_URL` or `VITE_SITE_URL` (from the environment or a `.env` file), e.g. `SITE_URL=https://everydays.dev npm run build`
2. On Vercel, `VERCEL_PROJECT_PRODUCTION_URL` (or `VERCEL_URL`), prefixed with `https://`
3. Otherwise `http://localhost:43118`

On Vercel, set `SITE_URL` in the project's environment variables once you have the domain. Until then, the Vercel production URL is used automatically.

`npm run og` regenerates `public/og.png` (the index card), `public/001/og.png` and `public/002/og.png`. Run it while `npm run dev` is running (set `OG_BASE` to the dev URL including any base path, e.g. `http://localhost:43117/everydays`). Day 001 is driven on a fake clock so its screenshot comes out the same every run; `OG_FRAME` picks a different frame of the sweep. Day 002 presses a few bubbles in the real WebGPU sim (headless Chromium with SwiftShader, so it takes a few minutes). `OG_ONLY=index,002` rebuilds only the listed cards.

If the base URL has a path (e.g. `https://umar-ahmed.github.io/everydays`), that path also becomes Vite's `base`, so assets and links are served from `/everydays/`. In code, build in-site links from `import.meta.env.BASE_URL` rather than a leading `/`.

## Deploying

The site is live at <https://umar-ahmed.github.io/everydays/>. `.github/workflows/pages.yml` builds and deploys it to GitHub Pages on every push to `main`, with `SITE_URL` set to the Pages URL.

`vercel.json` sets the build to `npm run build` with output in `dist/`. Any static host that serves `dist/` works.
