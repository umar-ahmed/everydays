import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import 'dialkit/styles.css'
import '../../styles.css'
import './type.css'
import { MagneticType } from './MagneticType.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <MagneticType />
  </StrictMode>,
)
