import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    include: ['bug-bash/results/**/*.test.ts'],
    environment: 'node',
  },
})
