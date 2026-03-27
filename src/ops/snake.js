// smith/src/ops/snake.js
// Snake activation: x + sin²(αx) / α
// α is a per-channel learnable parameter.
// Input: [channels, length], alpha: [channels]

import * as T from '../tensor.js'
import { run, k } from '../dispatch.js'

function snake(input, alpha) {
  const [channels, length] = input.shape
  const output = T.create(input.shape, input.dtype)
  const params = new Uint32Array([channels, length])

  run(k('snake_forward', input.dtype), [
    { buffer: input.buffer, index: 0 },
    { buffer: alpha.buffer, index: 1 },
    { buffer: output.buffer, index: 2 },
  ], { x: channels * length },
  { x: Math.min(channels * length, 256) },
  { data: params, index: 3 })

  return output
}

export { snake }
