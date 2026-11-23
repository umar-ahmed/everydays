import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import 'dialkit/styles.css'
import '../../styles.css'
import { Terrain } from './Terrain.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Terrain />
  </StrictMode>,
)
