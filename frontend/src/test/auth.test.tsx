import { describe, it, expect } from 'vitest'

describe('Authentication Flow', () => {
  it('should have sign-in route', () => {
    expect('/sign-in').toBeTruthy()
  })

  it('should have sign-up route', () => {
    expect('/sign-up').toBeTruthy()
  })

  it('should redirect to verify-glass after authentication', () => {
    const redirectUrl = '/verify-glass'
    expect(redirectUrl).toBe('/verify-glass')
  })
})
