export type Day = {
  slug: string
  title: string
  concept: string
  date: string
}

// Add a day: append an entry here and create `/<slug>/index.html`.
export const days: Day[] = [
  {
    slug: '001',
    title: 'gooey cursor blobs',
    concept: 'A spring-chained trail of metaballs that stretches on fast moves and melts back into one blob.',
    date: '2026-11-21',
  },
  {
    slug: '002',
    title: 'bubble wrap',
    concept:
      'Physically based bubble wrap: an XPBD cloth with gas-filled domes on the GPU that pop, with synthesized sound, when you squeeze them.',
    date: '2026-11-22',
  },
  {
    slug: '003',
    title: 'terrain',
    concept:
      'A procedural mountain range eroded on the GPU: fractal noise terrain, rain carving valleys and river channels, sediment fans and cliffs, live.',
    date: '2026-11-23',
  },
  {
    slug: '004',
    title: 'spectral pathtracer',
    concept:
      'A 2D spectral pathtracer on the GPU: a flashlight, glass prisms that split white light into rainbows, and mirrors in triangle, circle and hexagon shapes you can drag and turn.',
    date: '2026-11-24',
  },
]

export const site = {
  name: 'everydays',
  description: 'Daily 30-minute interactive toys. One numbered day at a time.',
}
