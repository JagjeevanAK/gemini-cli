/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { act } from 'react';
import type { Config } from '@google/gemini-cli-core';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { renderHook } from '../../test-utils/render.js';
import { useVoiceAssistantController } from './useVoiceAssistantController.js';

interface MockAudioState {
  rms: number;
  smoothedRms: number;
  level: number;
  probability: number;
  isTalking: boolean;
  timestamp: number;
  error: string | null;
  permissionRequired: boolean;
}

const voiceAssistantTestMocks = vi.hoisted(() => ({
  audioEngine: {
    subscribe: vi.fn(),
    subscribePcm: vi.fn(),
  },
  audioPlayback: {
    start: vi.fn(),
    stop: vi.fn(),
    playChunk: vi.fn(),
    getPendingPlaybackMs: vi.fn(() => 0),
  },
  liveVoiceSession: {
    instances: [] as Array<{
      emit: (event: string, ...args: unknown[]) => boolean;
    }>,
    start: vi.fn(async () => 'gemini-live-test'),
    isConnected: vi.fn(() => true),
    sendTextTurn: vi.fn(),
    sendAudioChunk: vi.fn(),
    sendAudioStreamEnd: vi.fn(),
    sendToolResponses: vi.fn(),
    close: vi.fn(),
  },
}));

vi.mock('../../services/audioEngine.js', () => ({
  audioEngine: voiceAssistantTestMocks.audioEngine,
}));

vi.mock('../../services/audioPlayback.js', () => ({
  AudioPlayback: class {
    start = voiceAssistantTestMocks.audioPlayback.start;
    stop = voiceAssistantTestMocks.audioPlayback.stop;
    playChunk = voiceAssistantTestMocks.audioPlayback.playChunk;
    getPendingPlaybackMs =
      voiceAssistantTestMocks.audioPlayback.getPendingPlaybackMs;
  },
}));

vi.mock('../../services/liveVoiceSession.js', () => ({
  LiveVoiceSession: class {
    private handlers = new Map<string, Set<(...args: unknown[]) => void>>();

    start = voiceAssistantTestMocks.liveVoiceSession.start;
    isConnected = voiceAssistantTestMocks.liveVoiceSession.isConnected;
    sendTextTurn = voiceAssistantTestMocks.liveVoiceSession.sendTextTurn;
    sendAudioChunk = voiceAssistantTestMocks.liveVoiceSession.sendAudioChunk;
    sendAudioStreamEnd =
      voiceAssistantTestMocks.liveVoiceSession.sendAudioStreamEnd;
    sendToolResponses =
      voiceAssistantTestMocks.liveVoiceSession.sendToolResponses;
    close = voiceAssistantTestMocks.liveVoiceSession.close;

    constructor(_config: unknown) {
      voiceAssistantTestMocks.liveVoiceSession.instances.push(this);
    }

    on(event: string, handler: (...args: unknown[]) => void) {
      const handlers = this.handlers.get(event) ?? new Set();
      handlers.add(handler);
      this.handlers.set(event, handlers);
      return this;
    }

    off(event: string, handler: (...args: unknown[]) => void) {
      this.handlers.get(event)?.delete(handler);
      return this;
    }

    emit(event: string, ...args: unknown[]) {
      const handlers = this.handlers.get(event);
      if (!handlers || handlers.size === 0) {
        return false;
      }
      for (const handler of [...handlers]) {
        handler(...args);
      }
      return true;
    }
  },
}));

vi.mock('../../services/voiceDebugLogger.js', () => ({
  voiceDebugLog: vi.fn(),
}));

describe('useVoiceAssistantController', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    voiceAssistantTestMocks.liveVoiceSession.instances.length = 0;
    voiceAssistantTestMocks.audioEngine.subscribe.mockImplementation(
      () => () => {},
    );
    voiceAssistantTestMocks.audioEngine.subscribePcm.mockReturnValue(() => {});
    voiceAssistantTestMocks.audioPlayback.start.mockResolvedValue({
      available: true,
    });
    voiceAssistantTestMocks.audioPlayback.stop.mockReturnValue(undefined);
    voiceAssistantTestMocks.audioPlayback.playChunk.mockReturnValue(undefined);
    voiceAssistantTestMocks.audioPlayback.getPendingPlaybackMs.mockReturnValue(
      0,
    );
    voiceAssistantTestMocks.liveVoiceSession.start.mockResolvedValue(
      'gemini-live-test',
    );
    voiceAssistantTestMocks.liveVoiceSession.isConnected.mockReturnValue(true);
  });

  it('commits a completed voice reply into output history and clears the live transcript', async () => {
    let resolveStart: ((model: string) => void) | undefined;
    voiceAssistantTestMocks.liveVoiceSession.start.mockImplementation(
      () =>
        new Promise((resolve: (model: string) => void) => {
          resolveStart = resolve;
        }),
    );
    const params = {
      config: {} as Config,
      enabled: true,
      captureAudio: false,
      runtimeConfig: {},
      isAgentBusy: () => false,
      onDisableRequested: vi.fn(),
      getRuntimeStatus: () => 'idle',
      getLatestHistoryId: () => 42,
      getPendingActions: () => [],
      submitUserRequest: vi.fn(async () => undefined),
      submitUserHint: vi.fn(),
      resolvePendingAction: vi.fn(async () => ''),
      cancelCurrentRun: vi.fn(),
    };
    const { result, waitUntilReady, unmount } = renderHook(() =>
      useVoiceAssistantController(params),
    );

    await waitUntilReady();

    const session = voiceAssistantTestMocks.liveVoiceSession.instances[0];
    if (!session) {
      throw new Error('Expected a live voice session instance.');
    }

    await act(async () => {
      resolveStart?.('gemini-live-test');
      await Promise.resolve();
    });

    await act(async () => {
      session.emit('outputTranscript', 'First line\nSecond line');
    });

    expect(result.current.outputTranscript).toBe('First line\nSecond line');
    expect(result.current.outputHistory).toEqual([]);

    await act(async () => {
      session.emit('turnComplete', null);
    });

    expect(result.current.outputTranscript).toBe('');
    expect(result.current.outputHistory).toEqual([
      {
        id: 1,
        anchorHistoryId: 42,
        text: 'First line\nSecond line',
      },
    ]);

    unmount();
  });

  it('queues notification speech until the current model turn completes', async () => {
    let resolveStart: ((model: string) => void) | undefined;
    voiceAssistantTestMocks.liveVoiceSession.start.mockImplementation(
      () =>
        new Promise((resolve: (model: string) => void) => {
          resolveStart = resolve;
        }),
    );
    const params = {
      config: {} as Config,
      enabled: true,
      captureAudio: false,
      runtimeConfig: {},
      isAgentBusy: () => false,
      onDisableRequested: vi.fn(),
      getRuntimeStatus: () => 'idle',
      getLatestHistoryId: () => 7,
      getPendingActions: () => [],
      submitUserRequest: vi.fn(async () => undefined),
      submitUserHint: vi.fn(),
      resolvePendingAction: vi.fn(async () => ''),
      cancelCurrentRun: vi.fn(),
    };

    const { result, waitUntilReady, unmount } = renderHook(() =>
      useVoiceAssistantController(params),
    );

    await waitUntilReady();

    const session = voiceAssistantTestMocks.liveVoiceSession.instances[0];
    if (!session) {
      throw new Error('Expected a live voice session instance.');
    }

    await act(async () => {
      resolveStart?.('gemini-live-test');
      await Promise.resolve();
    });

    await act(async () => {
      session.emit('outputTranscript', 'Already speaking');
    });

    act(() => {
      expect(result.current.speak('Queued follow up')).toBe(true);
    });

    expect(
      voiceAssistantTestMocks.liveVoiceSession.sendTextTurn,
    ).not.toHaveBeenCalled();

    await act(async () => {
      session.emit('turnComplete', null);
    });

    expect(
      voiceAssistantTestMocks.liveVoiceSession.sendTextTurn,
    ).toHaveBeenCalledWith(
      expect.stringContaining('Notification: Queued follow up'),
    );

    unmount();
  });

  it('does not mute the real spoken reply after suppressing a meta preamble', async () => {
    let resolveStart: ((model: string) => void) | undefined;
    voiceAssistantTestMocks.liveVoiceSession.start.mockImplementation(
      () =>
        new Promise((resolve: (model: string) => void) => {
          resolveStart = resolve;
        }),
    );
    const params = {
      config: {} as Config,
      enabled: true,
      captureAudio: true,
      runtimeConfig: {},
      isAgentBusy: () => false,
      onDisableRequested: vi.fn(),
      getRuntimeStatus: () => 'idle',
      getLatestHistoryId: () => 7,
      getPendingActions: () => [],
      submitUserRequest: vi.fn(async () => undefined),
      submitUserHint: vi.fn(),
      resolvePendingAction: vi.fn(async () => ''),
      cancelCurrentRun: vi.fn(),
    };

    const { result, waitUntilReady, unmount } = renderHook(() =>
      useVoiceAssistantController(params),
    );

    await waitUntilReady();

    const session = voiceAssistantTestMocks.liveVoiceSession.instances[0];
    if (!session) {
      throw new Error('Expected a live voice session instance.');
    }

    await act(async () => {
      resolveStart?.('gemini-live-test');
      await Promise.resolve();
    });

    act(() => {
      expect(result.current.speak('Started working on your request.')).toBe(
        true,
      );
    });

    await act(async () => {
      session.emit('outputTranscript', '**Delegating Git Status Inquiry**');
      session.emit('outputTranscript', 'Sure, I can help with that.');
      session.emit('outputAudioChunk', {
        chunk: Buffer.from([0, 1, 2, 3]),
        mimeType: 'audio/pcm;rate=24000',
      });
    });

    expect(result.current.outputTranscript).toContain(
      'Sure, I can help with that.',
    );
    expect(voiceAssistantTestMocks.audioPlayback.playChunk).toHaveBeenCalled();

    unmount();
  });

  it('waits for playback drain before sending the next queued spoken turn', async () => {
    vi.useFakeTimers();
    let resolveStart: ((model: string) => void) | undefined;
    voiceAssistantTestMocks.liveVoiceSession.start.mockImplementation(
      () =>
        new Promise((resolve: (model: string) => void) => {
          resolveStart = resolve;
        }),
    );
    voiceAssistantTestMocks.audioPlayback.getPendingPlaybackMs.mockReturnValue(
      240,
    );

    const params = {
      config: {} as Config,
      enabled: true,
      captureAudio: false,
      runtimeConfig: {},
      isAgentBusy: () => false,
      onDisableRequested: vi.fn(),
      getRuntimeStatus: () => 'idle',
      getLatestHistoryId: () => 9,
      getPendingActions: () => [],
      submitUserRequest: vi.fn(async () => undefined),
      submitUserHint: vi.fn(),
      resolvePendingAction: vi.fn(async () => ''),
      cancelCurrentRun: vi.fn(),
    };

    try {
      const { result, waitUntilReady, unmount } = renderHook(() =>
        useVoiceAssistantController(params),
      );

      await waitUntilReady();

      const session = voiceAssistantTestMocks.liveVoiceSession.instances[0];
      if (!session) {
        throw new Error('Expected a live voice session instance.');
      }

      await act(async () => {
        resolveStart?.('gemini-live-test');
        await Promise.resolve();
      });

      await act(async () => {
        session.emit('outputTranscript', 'Wrapping up');
        session.emit('outputAudioChunk', {
          chunk: Buffer.from([0, 1, 2, 3]),
          mimeType: 'audio/pcm;rate=24000',
        });
      });

      act(() => {
        expect(result.current.speak('Next sentence')).toBe(true);
      });

      await act(async () => {
        session.emit('turnComplete', null);
      });

      expect(
        voiceAssistantTestMocks.liveVoiceSession.sendTextTurn,
      ).not.toHaveBeenCalled();

      voiceAssistantTestMocks.audioPlayback.getPendingPlaybackMs.mockReturnValue(
        0,
      );
      await act(async () => {
        await vi.advanceTimersByTimeAsync(450);
      });

      expect(
        voiceAssistantTestMocks.liveVoiceSession.sendTextTurn,
      ).toHaveBeenCalledWith(
        expect.stringContaining('Notification: Next sentence'),
      );

      unmount();
    } finally {
      vi.useRealTimers();
    }
  });

  it('does not run client silence turn detection when advanced VAD is enabled', async () => {
    let resolveStart: ((model: string) => void) | undefined;
    let audioStateHandler: ((audioState: MockAudioState) => void) | undefined;
    voiceAssistantTestMocks.liveVoiceSession.start.mockImplementation(
      () =>
        new Promise((resolve: (model: string) => void) => {
          resolveStart = resolve;
        }),
    );
    voiceAssistantTestMocks.audioEngine.subscribe.mockImplementation(
      (handler: (audioState: MockAudioState) => void) => {
        audioStateHandler = handler;
        return () => {};
      },
    );

    const params = {
      config: {} as Config,
      enabled: true,
      captureAudio: true,
      runtimeConfig: {
        advancedVad: true,
        forceTurnEndOnSilence: true,
      },
      isAgentBusy: () => false,
      onDisableRequested: vi.fn(),
      getRuntimeStatus: () => 'idle',
      getLatestHistoryId: () => 0,
      getPendingActions: () => [],
      submitUserRequest: vi.fn(async () => undefined),
      submitUserHint: vi.fn(),
      resolvePendingAction: vi.fn(async () => ''),
      cancelCurrentRun: vi.fn(),
    };

    const { waitUntilReady, unmount } = renderHook(() =>
      useVoiceAssistantController(params),
    );

    await waitUntilReady();
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    const session = voiceAssistantTestMocks.liveVoiceSession.instances[0];
    if (!session) {
      throw new Error('Expected a live voice session instance.');
    }
    if (!audioStateHandler) {
      throw new Error('Expected audio state handler to be subscribed.');
    }

    await act(async () => {
      resolveStart?.('gemini-live-test');
      await Promise.resolve();
    });

    await act(async () => {
      audioStateHandler?.({
        rms: 0,
        smoothedRms: 0,
        level: 0,
        probability: 0,
        isTalking: true,
        timestamp: Date.now(),
        error: null,
        permissionRequired: false,
      });
      audioStateHandler?.({
        rms: 0,
        smoothedRms: 0,
        level: 0,
        probability: 0,
        isTalking: false,
        timestamp: Date.now(),
        error: null,
        permissionRequired: false,
      });
      await Promise.resolve();
    });

    expect(
      voiceAssistantTestMocks.liveVoiceSession.sendAudioStreamEnd,
    ).not.toHaveBeenCalled();

    unmount();
  });

  it('submits non-Latin request transcripts directly without speaking a redundant handoff ack', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-11T15:30:00.000Z'));
    let resolveStart: ((model: string) => void) | undefined;
    voiceAssistantTestMocks.liveVoiceSession.start.mockImplementation(
      () =>
        new Promise((resolve: (model: string) => void) => {
          resolveStart = resolve;
        }),
    );
    const submitUserRequest = vi.fn(
      async () => "On it. I'll check that and report back.",
    );

    const params = {
      config: {} as Config,
      enabled: true,
      captureAudio: false,
      runtimeConfig: {},
      isAgentBusy: () => false,
      onDisableRequested: vi.fn(),
      getRuntimeStatus: () => 'idle',
      getLatestHistoryId: () => 0,
      getPendingActions: () => [],
      submitUserRequest,
      submitUserHint: vi.fn(),
      resolvePendingAction: vi.fn(async () => ''),
      cancelCurrentRun: vi.fn(),
    };

    try {
      const { waitUntilReady, unmount } = renderHook(() =>
        useVoiceAssistantController(params),
      );

      await waitUntilReady();

      const session = voiceAssistantTestMocks.liveVoiceSession.instances[0];
      if (!session) {
        throw new Error('Expected a live voice session instance.');
      }

      await act(async () => {
        resolveStart?.('gemini-live-test');
        await Promise.resolve();
      });

      await act(async () => {
        session.emit(
          'inputTranscript',
          'हे कैन यू टेल मी व्हाट आर द अन कमिटेड फाइल्स इन द कोड बेस',
        );
        await vi.advanceTimersByTimeAsync(1005);
      });

      expect(submitUserRequest).toHaveBeenCalledWith(
        'हे कैन यू टेल मी व्हाट आर द अन कमिटेड फाइल्स इन द कोड बेस',
      );
      expect(
        voiceAssistantTestMocks.liveVoiceSession.sendTextTurn,
      ).not.toHaveBeenCalled();

      unmount();
    } finally {
      vi.useRealTimers();
    }
  });

  it('keeps internal interpreter turns silent while still resolving decision fallback from their transcript', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-11T15:31:00.000Z'));
    let resolveStart: ((model: string) => void) | undefined;
    voiceAssistantTestMocks.liveVoiceSession.start.mockImplementation(
      () =>
        new Promise((resolve: (model: string) => void) => {
          resolveStart = resolve;
        }),
    );
    const resolvePendingAction = vi.fn(async () => 'Okay, running it now.');
    const params = {
      config: {} as Config,
      enabled: true,
      captureAudio: false,
      runtimeConfig: {},
      isAgentBusy: () => false,
      onDisableRequested: vi.fn(),
      getRuntimeStatus: () => 'idle',
      getLatestHistoryId: () => 0,
      getPendingActions: () => [
        {
          id: 'tool:1',
          type: 'tool' as const,
          title: 'Run git status',
          detail: 'Run git status',
          allowedDecisions: [
            'allow_once',
            'allow_session',
            'allow_always',
            'cancel',
          ],
        },
      ],
      submitUserRequest: vi.fn(async () => undefined),
      submitUserHint: vi.fn(),
      resolvePendingAction,
      cancelCurrentRun: vi.fn(),
    };

    try {
      const { result, waitUntilReady, unmount } = renderHook(() =>
        useVoiceAssistantController(params),
      );

      await waitUntilReady();

      const session = voiceAssistantTestMocks.liveVoiceSession.instances[0];
      if (!session) {
        throw new Error('Expected a live voice session instance.');
      }

      await act(async () => {
        resolveStart?.('gemini-live-test');
        await Promise.resolve();
      });

      await act(async () => {
        session.emit('inputTranscript', 'हे बरोबर');
        await vi.advanceTimersByTimeAsync(1005);
      });

      expect(
        voiceAssistantTestMocks.liveVoiceSession.sendTextTurn,
      ).toHaveBeenCalledWith(
        expect.stringContaining(
          'Internal control turn. Do not address the user directly.',
        ),
      );

      await act(async () => {
        session.emit('outputTranscript', 'Allowing `git status` for once.');
        session.emit('outputAudioChunk', {
          chunk: Buffer.from([0, 1, 2, 3]),
          mimeType: 'audio/pcm;rate=24000',
        });
        session.emit('turnComplete', null);
      });

      expect(result.current.outputTranscript).toBe('');
      expect(result.current.outputHistory).toEqual([]);
      expect(
        voiceAssistantTestMocks.audioPlayback.playChunk,
      ).not.toHaveBeenCalled();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(2205);
      });

      expect(resolvePendingAction).toHaveBeenCalledWith({
        actionId: 'tool:1',
        decision: 'allow_once',
      });

      unmount();
    } finally {
      vi.useRealTimers();
    }
  });

  it('restarts the live voice session with carry-over memory when the voice context budget is exceeded', async () => {
    const longReply = 'status report '.repeat(5000).trim();
    const getCompressionThreshold = vi.fn(async () => 0.0001);
    const startResolvers: Array<(model: string) => void> = [];
    voiceAssistantTestMocks.liveVoiceSession.start.mockImplementation(
      () =>
        new Promise((resolve: (model: string) => void) => {
          startResolvers.push(resolve);
        }),
    );
    const params = {
      config: {
        getCompressionThreshold,
      } as unknown as Config,
      enabled: true,
      captureAudio: false,
      runtimeConfig: {},
      isAgentBusy: () => false,
      onDisableRequested: vi.fn(),
      getRuntimeStatus: () => 'idle',
      getLatestHistoryId: () => 12,
      getPendingActions: () => [],
      submitUserRequest: vi.fn(async () => undefined),
      submitUserHint: vi.fn(),
      resolvePendingAction: vi.fn(async () => ''),
      cancelCurrentRun: vi.fn(),
    };

    const { waitUntilReady, unmount } = renderHook(() =>
      useVoiceAssistantController(params),
    );

    await waitUntilReady();
    await act(async () => {
      while (getCompressionThreshold.mock.results.length === 0) {
        await Promise.resolve();
      }
      await getCompressionThreshold.mock.results[0]?.value;
      await Promise.resolve();
      await Promise.resolve();
    });
    await act(async () => {
      startResolvers.shift()?.('gemini-live-test');
      await Promise.resolve();
    });

    const session = voiceAssistantTestMocks.liveVoiceSession.instances[0];
    if (!session) {
      throw new Error('Expected a live voice session instance.');
    }

    await act(async () => {
      session.emit('outputTranscript', longReply);
      session.emit('turnComplete', null);
      await Promise.resolve();
      await Promise.resolve();
    });
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(startResolvers.length).toBe(1);
    await act(async () => {
      startResolvers.shift()?.('gemini-live-test');
      await Promise.resolve();
    });

    expect(
      voiceAssistantTestMocks.liveVoiceSession.start,
    ).toHaveBeenCalledTimes(2);
    const startCalls = Array.from(
      voiceAssistantTestMocks.liveVoiceSession.start.mock.calls,
    ) as unknown as Array<[{ systemInstruction?: string }]>;
    const secondStartArgs = startCalls[1]?.[0];
    expect(secondStartArgs?.systemInstruction).toContain(
      '<voice_session_memory>',
    );
    expect(secondStartArgs?.systemInstruction).toContain(
      'Recent exchanges: Assistant: status report status report',
    );

    unmount();
  });

  it('restarts the live voice session when coding-agent context sync changes', async () => {
    const startResolvers: Array<(model: string) => void> = [];
    voiceAssistantTestMocks.liveVoiceSession.start.mockImplementation(
      () =>
        new Promise((resolve: (model: string) => void) => {
          startResolvers.push(resolve);
        }),
    );
    const initialProps = {
      config: {} as Config,
      enabled: true,
      captureAudio: false,
      runtimeConfig: {},
      contextSyncGeneration: 0,
      contextSyncMessage: null as string | null,
      isAgentBusy: () => false,
      onDisableRequested: vi.fn(),
      getRuntimeStatus: () => 'idle',
      getLatestHistoryId: () => 0,
      getPendingActions: () => [],
      submitUserRequest: vi.fn(async () => undefined),
      submitUserHint: vi.fn(),
      resolvePendingAction: vi.fn(async () => ''),
      cancelCurrentRun: vi.fn(),
    };

    const { rerender, waitUntilReady, unmount } = renderHook(
      (props: typeof initialProps) => useVoiceAssistantController(props),
      {
        initialProps,
      },
    );

    await waitUntilReady();
    await act(async () => {
      startResolvers.shift()?.('gemini-live-test');
      await Promise.resolve();
    });

    await act(async () => {
      rerender({
        ...initialProps,
        contextSyncGeneration: 1,
        contextSyncMessage: 'Conversation context has been cleared.',
      });
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(startResolvers.length).toBe(1);
    await act(async () => {
      startResolvers.shift()?.('gemini-live-test');
      await Promise.resolve();
    });

    expect(
      voiceAssistantTestMocks.liveVoiceSession.start,
    ).toHaveBeenCalledTimes(2);
    const startCalls = Array.from(
      voiceAssistantTestMocks.liveVoiceSession.start.mock.calls,
    ) as unknown as Array<[{ systemInstruction?: string }]>;
    const secondStartArgs = startCalls[1]?.[0];
    expect(secondStartArgs?.systemInstruction).toContain(
      '<voice_session_memory>',
    );
    expect(secondStartArgs?.systemInstruction).toContain(
      'Conversation context has been cleared.',
    );

    unmount();
  });
});
