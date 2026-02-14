'use client'

import { Challenge } from '@/types'
import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'

interface ChallengeDisplayProps {
  challenge: Challenge | null
  timeRemaining?: number
}

const challengeIcons: Record<string, string> = {
  nod_up: '⬆️',
  nod_down: '⬇️',
  turn_left: '⬅️',
  turn_right: '➡️',
  tilt_left: '↖️',
  tilt_right: '↗️',
  open_mouth: '😮',
  close_eyes: '😑',
  raise_eyebrows: '😲',
  blink: '😉',
  smile: '😊',
  frown: '☹️',
  surprised: '😲',
  neutral: '😐',
  angry: '😠'
}

export default function ChallengeDisplay({ 
  challenge, 
  timeRemaining = 10 
}: ChallengeDisplayProps) {
  const [progress, setProgress] = useState(100)

  useEffect(() => {
    if (!challenge) return
    
    const percentage = (timeRemaining / challenge.timeoutSeconds) * 100
    setProgress(percentage)
  }, [timeRemaining, challenge])

  if (!challenge) {
    return (
      <div className="bg-void-200/60 border border-white/[0.06] p-8 text-center" style={{ borderRadius: '2px' }}>
        <motion.p className="text-ink-400 font-mono text-sm tracking-widest uppercase"
          animate={{ opacity: [0.4, 0.8, 0.4] }} transition={{ duration: 2, repeat: Infinity }}>
          Awaiting challenge...
        </motion.p>
      </div>
    )
  }

  const instructionKey = challenge.instruction?.toLowerCase().replace(/\s+/g, '_') ?? ''
  const icon = challengeIcons[instructionKey] || '👤'

  return (
    <div className="bg-void-200/80 border border-neon-cyan/20 p-8 shadow-glow-cyan" style={{ borderRadius: '2px' }}>
      {/* Top accent */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-neon-cyan/40 to-transparent" />
      
      <div className="text-center mb-6">
        <motion.div className="text-5xl mb-4"
          animate={{ scale: [1, 1.1, 1] }} transition={{ duration: 1.5, repeat: Infinity }}>
          {icon}
        </motion.div>
        <h2 className="font-display text-2xl font-bold text-ink-100 mb-2">
          {challenge.instruction}
        </h2>
        <p className="text-ink-400 font-mono text-xs tracking-[0.2em] uppercase">
          {challenge.type === 'gesture' ? 'Perform gesture' : 'Show expression'}
        </p>
      </div>

      <div className="relative w-full h-px bg-white/[0.06] overflow-hidden">
        <motion.div 
          className={`absolute left-0 top-0 h-full transition-all duration-1000 ${
            progress > 50 ? 'bg-neon-green' : progress > 25 ? 'bg-neon-amber' : 'bg-neon-red'
          }`}
          style={{ width: `${progress}%` }}
          initial={{ width: '100%' }}
        />
      </div>
      
      <div className="text-center mt-4">
        <span className={`font-mono text-2xl font-bold ${
          progress > 50 ? 'text-ink-200' : progress > 25 ? 'text-neon-amber' : 'text-neon-red'
        }`}>
          {Math.ceil(timeRemaining)}s
        </span>
      </div>
    </div>
  )
}
