/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { Config } from '@google/gemini-cli-core';
import { LiveVoiceSession } from './liveVoiceSession.js';
import { createLiveGoogleGenAI } from './liveAuth.js';

vi.mock('./liveAuth.js', () => ({
  createLiveGoogleGenAI: vi.fn(),
}));

vi.mock('./voiceDebugLogger.js', () => ({
  getVoiceDebugLogPath: vi.fn(),
  isVoiceDebugEnabled: vi.fn(() => false),
  voiceDebugLog: vi.fn(),
}));

describe('LiveVoiceSession', () => {
  const connect = vi.fn();
  const config = {} as Config;
  const mockSession = {
    sendRealtimeInput: vi.fn(),
    sendClientContent: vi.fn(),
    sendToolResponse: vi.fn(),
    close: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    connect.mockResolvedValue(mockSession);
    vi.mocked(createLiveGoogleGenAI).mockResolvedValue({
      live: {
        connect,
      },
    } as never);
  });

  it('passes the selected prebuilt voice to the Live API speech config', async () => {
    const session = new LiveVoiceSession(config);

    await session.start({
      model: 'gemini-live-test',
      voiceName: 'Zephyr',
    });

    expect(connect).toHaveBeenCalledWith(
      expect.objectContaining({
        model: 'gemini-live-test',
        config: expect.objectContaining({
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: {
                voiceName: 'Zephyr',
              },
            },
          },
        }),
      }),
    );
  });

  it('omits speech config when no persona voice is provided', async () => {
    const session = new LiveVoiceSession(config);

    await session.start({
      model: 'gemini-live-test',
    });

    expect(connect).toHaveBeenCalledWith(
      expect.objectContaining({
        config: expect.not.objectContaining({
          speechConfig: expect.anything(),
        }),
      }),
    );
  });

  it('does not send unsupported transcription languageCode hints', async () => {
    const session = new LiveVoiceSession(config);

    await session.start({
      model: 'gemini-live-test',
      inputTranscriptionLanguageCode: 'en-US',
    });

    expect(connect).toHaveBeenCalledWith(
      expect.objectContaining({
        config: expect.objectContaining({
          inputAudioTranscription: {},
        }),
      }),
    );
    expect(connect).not.toHaveBeenCalledWith(
      expect.objectContaining({
        config: expect.objectContaining({
          inputAudioTranscription: expect.objectContaining({
            languageCode: 'en-US',
          }),
        }),
      }),
    );
  });

  it('emits final output transcript chunks before turnComplete for the same message', () => {
    const session = new LiveVoiceSession(config);
    const events: string[] = [];

    session.on('outputTranscript', (text) => {
      events.push(`output:${text}`);
    });
    session.on('turnComplete', (reason) => {
      events.push(`turn:${reason ?? 'null'}`);
    });

    (
      session as unknown as { handleServerMessage: (message: unknown) => void }
    ).handleServerMessage({
      serverContent: {
        outputTranscription: {
          text: 'All set.',
        },
        turnComplete: true,
      },
    });

    expect(events).toEqual(['output:All set.', 'turn:null']);
  });
});
