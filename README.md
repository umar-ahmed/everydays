# everydays

A home for daily 30-minute interactive toys, one per numbered day. Each day is its own page (`/001/`, `/002/`, …) and the root `/` lists them all.

Built with Vite + React + TypeScript, [Motion](https://motion.dev) (`motion/react`), [DialKit](https://github.com/joshpuckett/dialkit) for live tuning panels, and Tailwind.

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
| [001](./001/) | gooey cursor blobs | Spring-chained metaballs (SVG blur + alpha-threshold filter) that stretch on fast moves and melt back together. Click to drop blobs, then drag the goo into them. |

## Adding a day

1. Append an entry to `src/days.ts` (slug, title, one-line concept, date).
2. Copy `001/index.html` to `002/index.html` and point its script at `src/days/002/main.tsx`.
3. Build the toy in `src/days/002/`.
4. Add a 1200x630 `public/002/og.png`.

The Vite config reads `src/days.ts` to register each page as a build input and to fill in the page's title and Open Graph tags. The build fails if a listed day has no `index.html`.

## Open Graph and the base URL

Every page gets `og:title`, `og:description`, `og:image`, `og:url` and `twitter:card=summary_large_image`, injected at build time in place of the `<!-- head:meta -->` comment. The absolute URLs use the first base URL found:

1. `SITE_URL` or `VITE_SITE_URL` (from the environment or a `.env` file), e.g. `SITE_URL=https://everydays.dev npm run build`
2. On Vercel, `VERCEL_PROJECT_PRODUCTION_URL` (or `VERCEL_URL`), prefixed with `https://`
3. Otherwise `http://localhost:43118`

On Vercel, set `SITE_URL` in the project's environment variables once you have the domain. Until then, the Vercel production URL is used automatically.

`npm run og` regenerates `public/og.png` (the index card) and `public/001/og.png`. Run it while `npm run dev` is running. It drives the real day 001 page on a fake clock so the screenshot comes out the same every run. Set `OG_FRAME` to pick a different frame of the sweep.

## Deploying

`vercel.json` sets the build to `npm run build` with output in `dist/`. Any static host that serves `dist/` works.
