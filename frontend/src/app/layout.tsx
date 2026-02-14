import type { Metadata } from 'next'
import { ClerkProvider } from '@clerk/nextjs'
import { dark } from '@clerk/themes'
import './globals.css'

export const metadata: Metadata = {
  title: 'SENTINEL — Proof of Life',
  description: 'Neural biometric authentication with ML-powered liveness detection',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <ClerkProvider
      appearance={{
        baseTheme: dark,
        variables: {
          colorPrimary: '#00f0ff',
          colorBackground: '#0e1117',
          colorInputBackground: '#161b26',
          colorInputText: '#e8ecf4',
          colorText: '#e8ecf4',
          colorTextSecondary: '#8896b0',
          borderRadius: '8px',
          fontFamily: 'Outfit, sans-serif',
        },
      }}
    >
      <html lang="en" className="dark">
        <body className="bg-void-50 text-ink-100 font-body antialiased">{children}</body>
      </html>
    </ClerkProvider>
  )
}
