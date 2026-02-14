"use client";

import { motion } from "framer-motion";
import { ReactNode } from "react";

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  hover?: boolean;
  glow?: 'cyan' | 'amber' | 'green' | 'red' | 'none';
}

const glowMap = {
  cyan: 'shadow-glow-cyan border-neon-cyan/20',
  amber: 'shadow-glow-amber border-neon-amber/20',
  green: 'shadow-glow-green border-neon-green/20',
  red: 'shadow-glow-red border-neon-red/20',
  none: 'border-white/[0.06]',
}

export default function GlassCard({ children, className = "", hover = false, glow = 'none' }: GlassCardProps) {
  return (
    <motion.div
      className={`
        relative overflow-hidden
        bg-void-100/80 backdrop-blur-xl
        border
        ${glowMap[glow]}
        ${className}
      `}
      style={{ borderRadius: '2px' }}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={hover ? { y: -3, borderColor: 'rgba(0, 240, 255, 0.25)' } : {}}
      transition={{ duration: 0.3 }}
    >
      {/* Top accent line */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-neon-cyan/30 to-transparent" />
      
      {/* Content */}
      <div className="relative z-10">{children}</div>
    </motion.div>
  );
}
