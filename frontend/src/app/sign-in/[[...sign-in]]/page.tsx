import { SignIn } from '@clerk/nextjs'

export default function SignInPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-grid relative overflow-hidden">
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-neon-cyan/[0.04] rounded-full blur-[100px]" />
      </div>
      <div className="relative z-10">
        <div className="text-center mb-8">
          <h1 className="font-display text-3xl font-bold text-ink-100 mb-2">SENTINEL</h1>
          <p className="font-mono text-xs tracking-[0.3em] uppercase text-neon-cyan/60">Identity Verification</p>
        </div>
        <SignIn
          appearance={{
            elements: {
              rootBox: "mx-auto",
              card: "bg-void-100 border border-white/10 shadow-glow-cyan rounded-lg"
            }
          }}
          redirectUrl="/verify"
        />
      </div>
    </div>
  )
}
