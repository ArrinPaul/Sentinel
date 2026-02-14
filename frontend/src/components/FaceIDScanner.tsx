"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useState } from "react";

interface FaceIDScannerProps {
  isScanning: boolean;
  progress: number;
  status: "idle" | "scanning" | "success" | "error";
  scores?: {
    liveness: number;
    emotion: number;
    deepfake: number;
  };
  currentChallenge?: string;
}

const statusColor = (status: string) =>
  status === "success" ? "#00ff88" : status === "error" ? "#ff3366" : "#00f0ff";

export default function FaceIDScanner({ isScanning, progress, status, scores, currentChallenge }: FaceIDScannerProps) {
  const [scanLines, setScanLines] = useState<number[]>([]);

  useEffect(() => {
    if (isScanning) {
      const interval = setInterval(() => {
        setScanLines((prev) => {
          const newLines = [...prev, Math.random()];
          return newLines.slice(-20);
        });
      }, 100);
      return () => clearInterval(interval);
    } else {
      setScanLines([]);
    }
  }, [isScanning]);

  const color = statusColor(status);

  return (
    <div className="relative w-full h-full flex items-center justify-center">
      {/* Dark scanner background */}
      <div className="absolute inset-0 bg-void-100/60 backdrop-blur-xl border border-white/[0.06]" style={{ borderRadius: '2px' }} />

      {/* Face outline */}
      <motion.div
        className="relative z-10"
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        {/* Outer glow ring */}
        <motion.div
          className="absolute inset-0"
          style={{
            width: "280px",
            height: "360px",
            left: "50%",
            top: "50%",
            transform: "translate(-50%, -50%)",
          }}
          animate={{
            boxShadow: isScanning
              ? [
                  `0 0 20px ${color}33`,
                  `0 0 50px ${color}66`,
                  `0 0 20px ${color}33`,
                ]
              : `0 0 0px ${color}00`,
          }}
          transition={{ duration: 2, repeat: Infinity }}
        />

        {/* Face mesh grid */}
        <svg width="280" height="360" viewBox="0 0 280 360" className="relative z-20">
          {[...Array(15)].map((_, i) => (
            <motion.line
              key={`v-${i}`}
              x1={i * 20} y1="0" x2={i * 20} y2="360"
              stroke={color} strokeWidth="1" opacity="0.15"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: isScanning ? 1 : 0 }}
              transition={{ duration: 0.5, delay: i * 0.05 }}
            />
          ))}
          {[...Array(19)].map((_, i) => (
            <motion.line
              key={`h-${i}`}
              x1="0" y1={i * 20} x2="280" y2={i * 20}
              stroke={color} strokeWidth="1" opacity="0.15"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: isScanning ? 1 : 0 }}
              transition={{ duration: 0.5, delay: i * 0.05 }}
            />
          ))}

          {/* Face outline ellipse */}
          <motion.ellipse
            cx="140" cy="180" rx="120" ry="160"
            fill="none" stroke={color} strokeWidth="2"
            strokeDasharray="4 6"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: isScanning ? 0.7 : 0.2 }}
            transition={{ duration: 1 }}
          />

          {/* Scanning line */}
          <AnimatePresence>
            {isScanning && (
              <motion.line
                x1="0" y1="0" x2="280" y2="0"
                stroke={color} strokeWidth="2"
                initial={{ y: 0, opacity: 0 }}
                animate={{ y: [0, 360, 0], opacity: [0, 0.8, 0] }}
                exit={{ opacity: 0 }}
                transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
                style={{ filter: `drop-shadow(0 0 6px ${color})` }}
              />
            )}
          </AnimatePresence>

          {/* Landmark dots */}
          {isScanning &&
            scanLines.map((_, i) => (
              <motion.circle
                key={i}
                cx={Math.random() * 280} cy={Math.random() * 360} r="1.5"
                fill={color}
                initial={{ opacity: 0, scale: 0 }}
                animate={{ opacity: [0, 0.8, 0], scale: [0, 1, 0] }}
                transition={{ duration: 1 }}
              />
            ))}
        </svg>

        {/* Corner brackets — neon cyan */}
        <div className="absolute inset-0 pointer-events-none">
          <motion.div className="absolute top-0 left-0 w-10 h-10 border-t border-l border-neon-cyan/70"
            initial={{ opacity: 0, x: -8, y: -8 }} animate={{ opacity: 1, x: 0, y: 0 }} transition={{ delay: 0.2 }} />
          <motion.div className="absolute top-0 right-0 w-10 h-10 border-t border-r border-neon-cyan/70"
            initial={{ opacity: 0, x: 8, y: -8 }} animate={{ opacity: 1, x: 0, y: 0 }} transition={{ delay: 0.3 }} />
          <motion.div className="absolute bottom-0 left-0 w-10 h-10 border-b border-l border-neon-cyan/70"
            initial={{ opacity: 0, x: -8, y: 8 }} animate={{ opacity: 1, x: 0, y: 0 }} transition={{ delay: 0.4 }} />
          <motion.div className="absolute bottom-0 right-0 w-10 h-10 border-b border-r border-neon-cyan/70"
            initial={{ opacity: 0, x: 8, y: 8 }} animate={{ opacity: 1, x: 0, y: 0 }} transition={{ delay: 0.5 }} />
        </div>
      </motion.div>

      {/* Status indicator — success */}
      <AnimatePresence>
        {status === "success" && (
          <motion.div className="absolute inset-0 flex items-center justify-center z-30"
            initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0, opacity: 0 }}>
            <motion.div className="w-24 h-24 rounded-full bg-neon-green/10 backdrop-blur-sm flex items-center justify-center border border-neon-green/30"
              animate={{ scale: [1, 1.15, 1] }} transition={{ duration: 0.5 }}>
              <svg className="w-12 h-12 text-neon-green" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <motion.path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7"
                  initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.5 }} />
              </svg>
            </motion.div>
          </motion.div>
        )}
        {status === "error" && (
          <motion.div className="absolute inset-0 flex items-center justify-center z-30"
            initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0, opacity: 0 }}>
            <motion.div className="w-24 h-24 rounded-full bg-neon-red/10 backdrop-blur-sm flex items-center justify-center border border-neon-red/30"
              animate={{ scale: [1, 1.15, 1] }} transition={{ duration: 0.5 }}>
              <svg className="w-12 h-12 text-neon-red" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <motion.path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12"
                  initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.5 }} />
              </svg>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Progress bar */}
      {isScanning && (
        <motion.div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 w-64"
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <div className="h-px bg-white/[0.06] overflow-hidden">
            <motion.div className="h-full bg-neon-cyan shadow-glow-cyan"
              initial={{ width: "0%" }} animate={{ width: `${progress}%` }} transition={{ duration: 0.3 }} />
          </div>
          <motion.p className="text-center text-xs font-mono text-ink-400 mt-3 tracking-widest uppercase"
            animate={{ opacity: [0.4, 1, 0.4] }} transition={{ duration: 2, repeat: Infinity }}>
            Biometric scan in progress
          </motion.p>
        </motion.div>
      )}

      {/* Challenge Display overlay */}
      <AnimatePresence>
        {currentChallenge && isScanning && (
          <motion.div className="absolute top-6 left-1/2 transform -translate-x-1/2 z-30"
            initial={{ opacity: 0, y: -16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -16 }} transition={{ duration: 0.3 }}>
            <div className="bg-void-200/90 backdrop-blur-md border border-neon-cyan/20 px-5 py-3 shadow-glow-cyan">
              <div className="flex items-center gap-3">
                <motion.div className="w-2 h-2 bg-neon-cyan" style={{ clipPath: 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)' }}
                  animate={{ scale: [1, 1.4, 1], opacity: [0.5, 1, 0.5] }} transition={{ duration: 1.5, repeat: Infinity }} />
                <p className="text-ink-100 font-mono text-sm tracking-wide">{currentChallenge}</p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Scores Display */}
      {scores && isScanning && (
        <motion.div className="absolute top-6 right-6 z-30 space-y-2"
          initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.5 }}>
          {[
            { label: 'LIV', value: scores.liveness, color: 'neon-green' },
            { label: 'EMO', value: scores.emotion, color: 'neon-cyan' },
            { label: 'DPF', value: scores.deepfake, color: 'neon-purple' },
          ].map((s) => (
            <div key={s.label} className="bg-void-200/80 backdrop-blur-md border border-white/[0.06] px-3 py-2 min-w-[140px]">
              <div className="flex justify-between items-center mb-1.5">
                <span className="text-ink-400 text-[10px] font-mono tracking-[0.2em] uppercase">{s.label}</span>
                <span className="text-ink-100 font-mono text-xs font-bold">{Math.round(s.value * 100)}%</span>
              </div>
              <div className="h-px bg-white/[0.06] overflow-hidden">
                <motion.div className={`h-full bg-${s.color}`}
                  initial={{ width: "0%" }} animate={{ width: `${s.value * 100}%` }} transition={{ duration: 0.5 }} />
              </div>
            </div>
          ))}
        </motion.div>
      )}
    </div>
  );
}
