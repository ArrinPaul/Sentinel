import { UserProfile } from '@clerk/nextjs'
import { auth } from '@clerk/nextjs'
import { redirect } from 'next/navigation'

export default async function ProfilePage() {
  const { userId } = auth()
  
  if (!userId) {
    redirect('/sign-in')
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-grid relative overflow-hidden p-4">
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/4 right-1/4 w-[500px] h-[500px] bg-neon-amber/[0.03] rounded-full blur-[100px]" />
      </div>
      <div className="relative z-10">
        <UserProfile
          appearance={{
            elements: {
              rootBox: "mx-auto",
              card: "bg-void-100 border border-white/10 shadow-glow-cyan rounded-lg"
            }
          }}
        />
      </div>
    </div>
  )
}
