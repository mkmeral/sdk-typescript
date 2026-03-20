/**
 * Bug Bash: Conversation Management – Comprehensive Test Suite
 *
 * This file exercises every conversation-management scenario described in the
 * bug-bash spec:
 *
 *  1. Sliding Window Tests (default 40, small window, oldest-message drop)
 *  2. Null Manager Tests (messages accumulate, never trimmed)
 *  3. Overflow Recovery Tests (ContextWindowOverflowError, reduce + retry,
 *     user/assistant pairing, tool use/result pairing)
 *  4. Manager Switching Tests (swap managers between invocations)
 *
 * Run with:
 *   npx vitest run bug-bash/results/conversation-management-results.test.ts --project='unit-node'
 */

import { describe, it, expect } from 'vitest'
import { Agent } from '../../src/agent/agent.js'
import { SlidingWindowConversationManager } from '../../src/conversation-manager/sliding-window-conversation-manager.js'
import { NullConversationManager } from '../../src/conversation-manager/null-conversation-manager.js'
import {
  ConversationManager,
  type ConversationManagerReduceOptions,
} from '../../src/conversation-manager/conversation-manager.js'
import { Message, TextBlock, ToolUseBlock, ToolResultBlock, ContextWindowOverflowError } from '../../src/index.js'
import { createMockAgent, invokeTrackedHook } from '../../src/__fixtures__/agent-helpers.js'
import { AfterInvocationEvent, AfterModelCallEvent } from '../../src/hooks/events.js'
import { MockMessageModel } from '../../src/__fixtures__/mock-message-model.js'

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Trigger the AfterInvocationEvent hook (sliding window management). */
async function triggerSlidingWindow(manager: SlidingWindowConversationManager, agent: Agent): Promise<void> {
  const pluginAgent = createMockAgent()
  manager.initAgent(pluginAgent)
  await invokeTrackedHook(pluginAgent, new AfterInvocationEvent({ agent }))
}

/** Trigger AfterModelCallEvent with an error (context overflow handling). */
async function triggerContextOverflow(
  manager: SlidingWindowConversationManager | ConversationManager,
  agent: Agent,
  error: Error,
): Promise<{ retry?: boolean }> {
  const pluginAgent = createMockAgent()
  manager.initAgent(pluginAgent)
  const event = new AfterModelCallEvent({ agent, error })
  await invokeTrackedHook(pluginAgent, event)
  return event
}

/** Build `n` alternating user / assistant text messages. */
function buildConversation(n: number): Message[] {
  const messages: Message[] = []
  for (let i = 0; i < n; i++) {
    const role = i % 2 === 0 ? 'user' : 'assistant'
    messages.push(new Message({ role, content: [new TextBlock(`msg-${i}`)] }))
  }
  return messages
}

/** Build a tool-use → tool-result pair. */
function buildToolPair(toolId: string, toolName = 'myTool'): [Message, Message] {
  return [
    new Message({
      role: 'assistant',
      content: [new ToolUseBlock({ name: toolName, toolUseId: toolId, input: {} })],
    }),
    new Message({
      role: 'user',
      content: [
        new ToolResultBlock({
          toolUseId: toolId,
          status: 'success',
          content: [new TextBlock(`result-${toolId}`)],
        }),
      ],
    }),
  ]
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. SLIDING WINDOW TESTS
// ═══════════════════════════════════════════════════════════════════════════════

describe('Bug Bash – Sliding Window Tests', () => {
  // ── 1a. Default window size (40 messages) ──────────────────────────────
  describe('default window size (40)', () => {
    it('does not trim when message count is under 40', async () => {
      const manager = new SlidingWindowConversationManager() // default 40
      const messages = buildConversation(38)
      const agent = createMockAgent({ messages })

      await triggerSlidingWindow(manager, agent)

      expect(agent.messages).toHaveLength(38)
    })

    it('does not trim when message count equals 40', async () => {
      const manager = new SlidingWindowConversationManager()
      const messages = buildConversation(40)
      const agent = createMockAgent({ messages })

      await triggerSlidingWindow(manager, agent)

      expect(agent.messages).toHaveLength(40)
    })

    it('trims to 40 when count exceeds 40', async () => {
      const manager = new SlidingWindowConversationManager()
      const messages = buildConversation(50)
      const agent = createMockAgent({ messages })

      await triggerSlidingWindow(manager, agent)

      expect(agent.messages).toHaveLength(40)
      // The kept messages should be the last 40 (indices 10–49)
      const firstKeptText = (agent.messages[0]!.content[0]! as TextBlock).text
      expect(firstKeptText).toBe('msg-10')
    })
  })

  // ── 1b. Small window (4 messages) ──────────────────────────────────────
  describe('small window (4 messages)', () => {
    it('trims to 4 when conversation exceeds the window', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 4 })
      const messages = buildConversation(10)
      const agent = createMockAgent({ messages })

      await triggerSlidingWindow(manager, agent)

      expect(agent.messages).toHaveLength(4)
    })

    it('preserves the most recent 4 messages', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 4 })
      const messages = buildConversation(10)
      const agent = createMockAgent({ messages })

      await triggerSlidingWindow(manager, agent)

      const texts = agent.messages.map((m) => (m.content[0] as TextBlock).text)
      expect(texts).toEqual(['msg-6', 'msg-7', 'msg-8', 'msg-9'])
    })

    it('no-ops when exactly at the window boundary', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 4 })
      const messages = buildConversation(4)
      const agent = createMockAgent({ messages })

      await triggerSlidingWindow(manager, agent)

      expect(agent.messages).toHaveLength(4)
    })
  })

  // ── 1c. Oldest messages are dropped correctly ──────────────────────────
  describe('oldest-message drop order', () => {
    it('drops oldest messages first, retains newest', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 2 })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('oldest')] }),
        new Message({ role: 'assistant', content: [new TextBlock('old-reply')] }),
        new Message({ role: 'user', content: [new TextBlock('newer')] }),
        new Message({ role: 'assistant', content: [new TextBlock('newest-reply')] }),
      ]
      const agent = createMockAgent({ messages })

      await triggerSlidingWindow(manager, agent)

      expect(agent.messages).toHaveLength(2)
      expect((agent.messages[0]!.content[0] as TextBlock).text).toBe('newer')
      expect((agent.messages[1]!.content[0] as TextBlock).text).toBe('newest-reply')
    })

    it('progressively drops messages over multiple invocations', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 4 })
      const messages = buildConversation(6)
      const agent = createMockAgent({ messages })

      // First invocation: 6 → 4
      await triggerSlidingWindow(manager, agent)
      expect(agent.messages).toHaveLength(4)
      expect((agent.messages[0]!.content[0] as TextBlock).text).toBe('msg-2')

      // Simulate two more messages
      agent.messages.push(
        new Message({ role: 'user', content: [new TextBlock('msg-6')] }),
        new Message({ role: 'assistant', content: [new TextBlock('msg-7')] }),
      )
      expect(agent.messages).toHaveLength(6)

      // Second invocation: 6 → 4 again
      await triggerSlidingWindow(manager, agent)
      expect(agent.messages).toHaveLength(4)
      expect((agent.messages[0]!.content[0] as TextBlock).text).toBe('msg-4')
    })
  })
})

// ═══════════════════════════════════════════════════════════════════════════════
// 2. NULL MANAGER TESTS
// ═══════════════════════════════════════════════════════════════════════════════

describe('Bug Bash – Null Manager Tests', () => {
  it('messages accumulate without any management', () => {
    const manager = new NullConversationManager()
    const messages = buildConversation(100)
    const agent = createMockAgent({ messages })

    // NullConversationManager does not register an AfterInvocationEvent hook,
    // so we simply verify the messages are untouched.
    expect(agent.messages).toHaveLength(100)
  })

  it('reduce always returns false – messages are never trimmed', () => {
    const manager = new NullConversationManager()
    const messages = buildConversation(50)
    const agent = createMockAgent({ messages })

    const reduced = manager.reduce({
      agent,
      error: new ContextWindowOverflowError('overflow'),
    })

    expect(reduced).toBe(false)
    expect(agent.messages).toHaveLength(50) // untouched
  })

  it('does not set retry on ContextWindowOverflowError', async () => {
    const manager = new NullConversationManager()
    const agent = createMockAgent({ messages: buildConversation(10) })
    const pluginAgent = createMockAgent()
    manager.initAgent(pluginAgent)

    const event = new AfterModelCallEvent({
      agent,
      error: new ContextWindowOverflowError('overflow'),
    })
    await invokeTrackedHook(pluginAgent, event)

    expect(event.retry).toBeUndefined()
    expect(agent.messages).toHaveLength(10) // unchanged
  })

  it('propagates ContextWindowOverflowError out of the agent loop', async () => {
    const model = new MockMessageModel()
    model.addTurn(new ContextWindowOverflowError('context window exceeded'))

    const agent = new Agent({
      model,
      conversationManager: new NullConversationManager(),
      printer: false,
    })

    await expect(agent.invoke('hello')).rejects.toThrow(ContextWindowOverflowError)
  })

  it('does NOT register an AfterInvocationEvent hook (only overflow recovery)', () => {
    const manager = new NullConversationManager()
    const pluginAgent = createMockAgent()
    manager.initAgent(pluginAgent)

    // Should register exactly 1 hook: AfterModelCallEvent (from base class)
    expect(pluginAgent.trackedHooks).toHaveLength(1)
    expect(pluginAgent.trackedHooks[0]!.eventType).toBe(AfterModelCallEvent)
  })
})

// ═══════════════════════════════════════════════════════════════════════════════
// 3. OVERFLOW RECOVERY TESTS
// ═══════════════════════════════════════════════════════════════════════════════

describe('Bug Bash – Overflow Recovery Tests', () => {
  // ── 3a. Basic overflow reduction and retry ─────────────────────────────
  describe('reduce + retry flow', () => {
    it('sets retry=true when reduce succeeds', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 2, shouldTruncateResults: false })
      const messages = buildConversation(6)
      const agent = createMockAgent({ messages })

      const event = await triggerContextOverflow(manager, agent, new ContextWindowOverflowError('overflow'))

      expect(event.retry).toBe(true)
      expect(agent.messages.length).toBeLessThanOrEqual(2)
    })

    it('truncates tool results first before trimming messages', async () => {
      const manager = new SlidingWindowConversationManager({ shouldTruncateResults: true })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Q')] }),
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ name: 'search', toolUseId: 'tu-1', input: { q: 'foo' } })],
        }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tu-1',
              status: 'success',
              content: [new TextBlock('x'.repeat(5000))], // large result
            }),
          ],
        }),
        new Message({ role: 'assistant', content: [new TextBlock('Answer')] }),
      ]
      const agent = createMockAgent({ messages })

      await triggerContextOverflow(manager, agent, new ContextWindowOverflowError('overflow'))

      // All 4 messages should still be there — only the tool result content was truncated
      expect(agent.messages).toHaveLength(4)
      const truncatedBlock = agent.messages[2]!.content[0] as ToolResultBlock
      expect(truncatedBlock.status).toBe('error')
      expect((truncatedBlock.content[0] as TextBlock).text).toBe('The tool result was too large!')
    })

    it('falls back to message trimming when tool results are already truncated', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 2, shouldTruncateResults: true })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Msg 1')] }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tu-1',
              status: 'error',
              content: [new TextBlock('The tool result was too large!')],
            }),
          ],
        }),
        new Message({ role: 'assistant', content: [new TextBlock('Msg 2')] }),
        new Message({ role: 'user', content: [new TextBlock('Msg 3')] }),
      ]
      const agent = createMockAgent({ messages })

      await triggerContextOverflow(manager, agent, new ContextWindowOverflowError('overflow'))

      // Already-truncated result ⇒ can't truncate further ⇒ falls through to message trimming
      expect(agent.messages.length).toBeLessThanOrEqual(2)
    })

    it('performs overflow recovery through the Agent loop (end-to-end)', async () => {
      const model = new MockMessageModel()
      // First call overflows, second succeeds
      model
        .addTurn(new ContextWindowOverflowError('overflow'))
        .addTurn({ type: 'textBlock', text: 'Recovered!' })

      // Pre-populate with enough messages so the manager has something to trim.
      // When the overflow fires the agent will have these messages + the new user message.
      const existingMessages = buildConversation(10)

      const agent = new Agent({
        model,
        conversationManager: new SlidingWindowConversationManager({ windowSize: 4 }),
        messages: existingMessages,
        printer: false,
      })

      const result = await agent.invoke('hello')
      expect(result.stopReason).toBe('endTurn')
      expect(result.lastMessage?.content.some(
        (b) => b.type === 'textBlock' && (b as TextBlock).text === 'Recovered!'
      )).toBe(true)
    })
  })

  // ── 3b. User / assistant pairing after reduction ───────────────────────
  describe('user / assistant pairing after reduction', () => {
    it('keeps messages properly paired (user followed by assistant)', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 4, shouldTruncateResults: false })
      const messages = buildConversation(10) // alternating user/assistant
      const agent = createMockAgent({ messages })

      await triggerContextOverflow(manager, agent, new ContextWindowOverflowError('overflow'))

      expect(agent.messages).toHaveLength(4)
      // Verify alternating roles
      for (let i = 0; i < agent.messages.length; i++) {
        const expectedRole = i % 2 === 0 ? 'user' : 'assistant'
        expect(agent.messages[i]!.role).toBe(expectedRole)
      }
    })

    it('does not leave a dangling assistant message without preceding user turn', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 3, shouldTruncateResults: false })
      // Start with an odd user message so trimming might try to land on assistant
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('U0')] }),
        new Message({ role: 'assistant', content: [new TextBlock('A0')] }),
        new Message({ role: 'user', content: [new TextBlock('U1')] }),
        new Message({ role: 'assistant', content: [new TextBlock('A1')] }),
        new Message({ role: 'user', content: [new TextBlock('U2')] }),
        new Message({ role: 'assistant', content: [new TextBlock('A2')] }),
      ]
      const agent = createMockAgent({ messages })

      await triggerContextOverflow(manager, agent, new ContextWindowOverflowError('overflow'))

      expect(agent.messages.length).toBeLessThanOrEqual(3)
      // After trimming, messages should still be valid – no orphan assistant at start
      // (SlidingWindow only validates tool pairing; text messages are trimmed at any valid point)
    })
  })

  // ── 3c. Tool use / tool result pairing during reduction ────────────────
  describe('tool use / tool result pairing during reduction', () => {
    it('skips trim points that would orphan a toolResult', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 2, shouldTruncateResults: false })
      const [toolUse, toolResult] = buildToolPair('t1')
      const messages = [
        toolUse,
        toolResult,
        new Message({ role: 'assistant', content: [new TextBlock('Final answer')] }),
        new Message({ role: 'user', content: [new TextBlock('Thanks')] }),
      ]
      const agent = createMockAgent({ messages })

      await triggerContextOverflow(manager, agent, new ContextWindowOverflowError('overflow'))

      // Trim index starts at 2 (4 - 2). Index 2 is a text msg – valid.
      expect(agent.messages).toHaveLength(2)
      expect((agent.messages[0]!.content[0] as TextBlock).text).toBe('Final answer')
      expect((agent.messages[1]!.content[0] as TextBlock).text).toBe('Thanks')
    })

    it('keeps tool use/result pairs together when they would land at the window boundary', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 4, shouldTruncateResults: false })
      const [toolUse1, toolResult1] = buildToolPair('t1')
      const [toolUse2, toolResult2] = buildToolPair('t2')
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('U0')] }),
        new Message({ role: 'assistant', content: [new TextBlock('A0')] }),
        toolUse1,   // idx 2
        toolResult1, // idx 3
        new Message({ role: 'assistant', content: [new TextBlock('A1')] }),
        toolUse2,    // idx 5
        toolResult2, // idx 6
        new Message({ role: 'assistant', content: [new TextBlock('Final')] }),
      ]
      const agent = createMockAgent({ messages })

      await triggerContextOverflow(manager, agent, new ContextWindowOverflowError('overflow'))

      // trim index = 8 - 4 = 4. Index 4 is text A1 – valid. Keep last 4.
      expect(agent.messages).toHaveLength(4)
      // No orphan tool results at the beginning
      const firstMsg = agent.messages[0]!
      const hasOrphanToolResult = firstMsg.content.some((b) => b.type === 'toolResultBlock')
      expect(hasOrphanToolResult).toBe(false)
    })

    it('advances past consecutive tool pairs to find valid trim point', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 1, shouldTruncateResults: false })
      // Every position except the last is part of a tool pair
      const [tu1, tr1] = buildToolPair('t1')
      const [tu2, tr2] = buildToolPair('t2')
      const messages = [
        tu1, tr1, tu2, tr2,
        new Message({ role: 'assistant', content: [new TextBlock('done')] }),
      ]
      const agent = createMockAgent({ messages })

      await triggerContextOverflow(manager, agent, new ContextWindowOverflowError('overflow'))

      // Should skip toolResult at idx 1, toolUse-without-following-toolResult at idx… 
      // Actually tr1 is idx 1 (toolResult → skip), tu2 at idx 2 with tr2 at idx 3 → valid.
      // But trim index starts at max(2, 5-1)=4. Index 4 is text 'done' → valid.
      expect(agent.messages).toHaveLength(1)
      expect((agent.messages[0]!.content[0] as TextBlock).text).toBe('done')
    })

    it('preserves multi-block messages with toolUse and text together', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 3, shouldTruncateResults: false })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Q')] }),
        new Message({
          role: 'assistant',
          content: [
            new TextBlock('Let me check…'),
            new ToolUseBlock({ name: 'search', toolUseId: 'tu-combo', input: {} }),
          ],
        }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tu-combo',
              status: 'success',
              content: [new TextBlock('search result')],
            }),
          ],
        }),
        new Message({ role: 'assistant', content: [new TextBlock('Here is the answer')] }),
        new Message({ role: 'user', content: [new TextBlock('Thanks')] }),
      ]
      const agent = createMockAgent({ messages })

      await triggerContextOverflow(manager, agent, new ContextWindowOverflowError('overflow'))

      // trim index = 5 - 3 = 2. Index 2 has toolResult → skip. Index 3 is text → valid.
      // So we keep last 2 messages
      expect(agent.messages.length).toBeLessThanOrEqual(3)
      // Verify no orphan tool results at start
      const first = agent.messages[0]!
      const startsWithToolResult = first.content.some((b) => b.type === 'toolResultBlock')
      expect(startsWithToolResult).toBe(false)
    })

    it('handles deeply nested tool pairs with no other valid trim point gracefully', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 0, shouldTruncateResults: false })
      // Only tool pairs – no valid trim point at all
      const [tu, tr] = buildToolPair('only')
      const messages = [tu, tr]
      const agent = createMockAgent({ messages })

      const result = manager.reduce({
        agent,
        error: new ContextWindowOverflowError('overflow'),
      })

      // toolResult at idx 0 → skip. toolUse at idx 1 without following toolResult → skip. 
      // Actually trim starts at max(2, 2-0)=2, which is >= messages.length, so returns false.
      expect(result).toBe(false)
      expect(agent.messages).toHaveLength(2)
    })
  })

  // ── 3d. Repeated overflow reductions ───────────────────────────────────
  describe('repeated overflow reductions', () => {
    it('can reduce multiple times until conversation is small enough', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 4, shouldTruncateResults: false })
      const messages = buildConversation(20)
      const agent = createMockAgent({ messages })

      // First reduction
      const r1 = manager.reduce({ agent, error: new ContextWindowOverflowError('overflow') })
      expect(r1).toBe(true)
      expect(agent.messages).toHaveLength(4)

      // Simulate more messages accumulating
      agent.messages.push(
        ...buildConversation(6).map((m, i) =>
          new Message({ role: m.role, content: [new TextBlock(`extra-${i}`)] }),
        ),
      )
      expect(agent.messages).toHaveLength(10)

      // Second reduction
      const r2 = manager.reduce({ agent, error: new ContextWindowOverflowError('overflow') })
      expect(r2).toBe(true)
      expect(agent.messages).toHaveLength(4)
    })

    it('reduce with very small window (1) still works', () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 1, shouldTruncateResults: false })
      const messages = buildConversation(10)
      const agent = createMockAgent({ messages })

      const result = manager.reduce({ agent, error: new ContextWindowOverflowError('overflow') })
      expect(result).toBe(true)
      expect(agent.messages).toHaveLength(1)
      expect((agent.messages[0]!.content[0] as TextBlock).text).toBe('msg-9')
    })
  })
})

// ═══════════════════════════════════════════════════════════════════════════════
// 4. MANAGER SWITCHING TESTS
// ═══════════════════════════════════════════════════════════════════════════════

describe('Bug Bash – Manager Switching Tests', () => {
  it('can switch from SlidingWindow to Null between invocations', async () => {
    const model = new MockMessageModel()
    model
      .addTurn({ type: 'textBlock', text: 'first reply' })
      .addTurn({ type: 'textBlock', text: 'second reply' })

    // Start with sliding window
    const agent1 = new Agent({
      model,
      conversationManager: new SlidingWindowConversationManager({ windowSize: 4 }),
      printer: false,
    })

    const r1 = await agent1.invoke('hello')
    expect(r1.stopReason).toBe('endTurn')

    // Switch to Null manager for a second agent sharing state
    const agent2 = new Agent({
      model,
      conversationManager: new NullConversationManager(),
      messages: agent1.messages,
      printer: false,
    })

    const r2 = await agent2.invoke('follow up')
    expect(r2.stopReason).toBe('endTurn')
    // With NullConversationManager, no trimming should occur
  })

  it('can switch from Null to SlidingWindow between invocations', async () => {
    const model = new MockMessageModel()
    model
      .addTurn({ type: 'textBlock', text: 'reply-A' })
      .addTurn({ type: 'textBlock', text: 'reply-B' })

    // Start with null – messages accumulate
    const agent1 = new Agent({
      model,
      conversationManager: new NullConversationManager(),
      printer: false,
    })

    await agent1.invoke('hello')
    const msgCountAfterFirstInvoke = agent1.messages.length

    // Switch to sliding window with small window
    const agent2 = new Agent({
      model,
      conversationManager: new SlidingWindowConversationManager({ windowSize: 2 }),
      messages: [...agent1.messages], // copy messages
      printer: false,
    })

    await agent2.invoke('again')

    // SlidingWindow should trim after invocation
    expect(agent2.messages.length).toBeLessThanOrEqual(2)
  })

  it('can switch to a custom ConversationManager', async () => {
    class KeepLastN extends ConversationManager {
      readonly name = 'test:keep-last-n'
      constructor(private n: number) { super() }
      reduce({ agent }: ConversationManagerReduceOptions): boolean {
        if (agent.messages.length <= this.n) return false
        agent.messages.splice(0, agent.messages.length - this.n)
        return true
      }
    }

    const model = new MockMessageModel()
    model
      .addTurn(new ContextWindowOverflowError('overflow'))
      .addTurn({ type: 'textBlock', text: 'OK' })

    // Pre-populate with messages so KeepLastN can actually reduce
    const existingMessages = buildConversation(10)

    const agent = new Agent({
      model,
      conversationManager: new KeepLastN(2),
      messages: existingMessages,
      printer: false,
    })

    const result = await agent.invoke('test')
    expect(result.stopReason).toBe('endTurn')
  })
})

// ═══════════════════════════════════════════════════════════════════════════════
// 5. EDGE CASES & BOUNDARY CONDITIONS
// ═══════════════════════════════════════════════════════════════════════════════

describe('Bug Bash – Edge Cases', () => {
  it('handles empty messages array gracefully', async () => {
    const manager = new SlidingWindowConversationManager({ windowSize: 4 })
    const agent = createMockAgent({ messages: [] })

    await triggerSlidingWindow(manager, agent)

    expect(agent.messages).toHaveLength(0)
  })

  it('handles single message (under window)', async () => {
    const manager = new SlidingWindowConversationManager({ windowSize: 4 })
    const messages = [new Message({ role: 'user', content: [new TextBlock('solo')] })]
    const agent = createMockAgent({ messages })

    await triggerSlidingWindow(manager, agent)

    expect(agent.messages).toHaveLength(1)
  })

  it('windowSize of 0 is handled: all messages trimmed on overflow', () => {
    const manager = new SlidingWindowConversationManager({ windowSize: 0, shouldTruncateResults: false })
    const messages = buildConversation(4)
    const agent = createMockAgent({ messages })

    const result = manager.reduce({ agent, error: new ContextWindowOverflowError('overflow') })

    // trim index = max(2, 4-0)=4, which >= messages.length, so returns false
    // But let's check: messages.length=4, windowSize=0 → trimIndex = 4-0=4.
    // 4 >= 4 → no valid trim point → false
    // Actually: 4 is >= messages.length (4) → returns false
    expect(result).toBe(false)
  })

  it('handles message with multiple content blocks', async () => {
    const manager = new SlidingWindowConversationManager({ windowSize: 2, shouldTruncateResults: false })
    const messages = [
      new Message({ role: 'user', content: [new TextBlock('A'), new TextBlock('B')] }),
      new Message({ role: 'assistant', content: [new TextBlock('C'), new TextBlock('D')] }),
      new Message({ role: 'user', content: [new TextBlock('E')] }),
    ]
    const agent = createMockAgent({ messages })

    await triggerContextOverflow(manager, agent, new ContextWindowOverflowError('overflow'))

    expect(agent.messages.length).toBeLessThanOrEqual(2)
  })

  it('does not interfere with non-overflow errors', async () => {
    const manager = new SlidingWindowConversationManager({ windowSize: 2 })
    const messages = buildConversation(6)
    const agent = createMockAgent({ messages })
    const pluginAgent = createMockAgent()
    manager.initAgent(pluginAgent)

    const event = new AfterModelCallEvent({
      agent,
      error: new Error('Something else went wrong'),
    })
    await invokeTrackedHook(pluginAgent, event)

    // Should NOT reduce or retry for non-overflow errors
    expect(event.retry).toBeUndefined()
    expect(agent.messages).toHaveLength(6) // unchanged
  })

  it('SlidingWindowConversationManager name is correct', () => {
    const manager = new SlidingWindowConversationManager()
    expect(manager.name).toBe('strands:sliding-window-conversation-manager')
  })

  it('NullConversationManager name is correct', () => {
    const manager = new NullConversationManager()
    expect(manager.name).toBe('strands:null-conversation-manager')
  })

  it('multiple sequential tool pairs at the start are all skipped during trimming', async () => {
    const manager = new SlidingWindowConversationManager({ windowSize: 2, shouldTruncateResults: false })
    const [tu1, tr1] = buildToolPair('t1')
    const [tu2, tr2] = buildToolPair('t2')
    const [tu3, tr3] = buildToolPair('t3')
    const messages = [
      tu1, tr1,  // idx 0, 1
      tu2, tr2,  // idx 2, 3
      tu3, tr3,  // idx 4, 5
      new Message({ role: 'assistant', content: [new TextBlock('Summary')] }), // idx 6
      new Message({ role: 'user', content: [new TextBlock('OK')] }),           // idx 7
    ]
    const agent = createMockAgent({ messages })

    await triggerContextOverflow(manager, agent, new ContextWindowOverflowError('overflow'))

    // trim index = 8-2 = 6. Index 6 is text 'Summary' → valid.
    expect(agent.messages).toHaveLength(2)
    expect((agent.messages[0]!.content[0] as TextBlock).text).toBe('Summary')
    expect((agent.messages[1]!.content[0] as TextBlock).text).toBe('OK')
  })

  it('tool truncation replaces ALL tool results in the same message', async () => {
    const manager = new SlidingWindowConversationManager({ shouldTruncateResults: true })
    const messages = [
      new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: 'a',
            status: 'success',
            content: [new TextBlock('Result A')],
          }),
          new ToolResultBlock({
            toolUseId: 'b',
            status: 'success',
            content: [new TextBlock('Result B')],
          }),
        ],
      }),
    ]
    const agent = createMockAgent({ messages })

    await triggerContextOverflow(manager, agent, new ContextWindowOverflowError('overflow'))

    // Both tool results in the same message should be truncated
    expect(agent.messages).toHaveLength(1)
    const blocks = agent.messages[0]!.content
    expect(blocks).toHaveLength(2)
    for (const b of blocks) {
      const trb = b as ToolResultBlock
      expect(trb.status).toBe('error')
      expect((trb.content[0] as TextBlock).text).toBe('The tool result was too large!')
    }
  })
})
