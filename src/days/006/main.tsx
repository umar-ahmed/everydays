import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import 'dialkit/styles.css'
import '../../styles.css'
import { PluckLab } from './PluckLab.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <PluckLab />
  </StrictMode>,
)
