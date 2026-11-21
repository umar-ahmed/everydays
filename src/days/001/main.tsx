import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import 'dialkit/styles.css'
import '../../styles.css'
import { GooeyBlobs } from './GooeyBlobs.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <GooeyBlobs />
  </StrictMode>,
)
