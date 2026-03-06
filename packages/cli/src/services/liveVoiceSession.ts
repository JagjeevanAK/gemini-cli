/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { EventEmitter } from 'node:events';
import type { Config } from '@google/gemini-cli-core';
import {
  ActivityHandling,
  type AudioTranscriptionConfig,
  EndSensitivity,
  Modality,
  type FunctionCall,
  type FunctionResponse,
  type LiveServerMessage,
  type ToolListUnion,
  type Session,
} from '@google/genai';
import { createLiveGoogleGenAI } from './liveAuth.js';
import {
  getVoiceDebugLogPath,
  isVoiceDebugEnabled,
  voiceDebugLog,
} from './voiceDebugLogger.js';

// Temporary hardcoded testing default for voice sessions.
const DEFAULT_MODEL = 'gemini-2.5-flash-native-audio-preview-12-2025';
const QUOTA_FALLBACK_MODEL = 'gemini-live-2.5-flash-preview';
const DEFAULT_INPUT_AUDIO_MIME = 'audio/pcm;rate=16000';
const DEFAULT_SERVER_SILENCE_DURATION_MS = 1600;
const DEFAULT_SETUP_COMPLETE_WAIT_MS = 1500;

export interface VoiceOutputAudioChunk {
  chunk: Buffer;
  mimeType: string;
}

export type LiveVoiceEvents = {
  open: [];
  close: [];
  turnComplete: [string | null];
  inputTranscript: [string];
  outputTranscript: [string];
  outputAudioChunk: [VoiceOutputAudioChunk];
  toolCall: [FunctionCall[]];
  error: [Error];
};

export interface StartVoiceSessionParams {
  model?: string;
  inputTranscriptionLanguageCode?: string;
  advancedVad?: boolean;
  serverSilenceMs?: number;
  setupWaitMs?: number;
  tools?: ToolListUnion;
  systemInstruction?: string;
}

const normalizeError = (error: unknown) =>
  error instanceof Error
    ? error
    : new Error(typeof error === 'string' ? error : String(error));

const toClampedNumber = (
  value: number | undefined,
  fallback: number,
  minValue: number,
) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return fallback;
  }
  return Math.max(minValue, Math.floor(value));
};

const isQuotaExceededError = (message: string) => {
  const lower = message.toLowerCase();
  return (
    lower.includes('exhausted your daily quota') ||
    lower.includes('resource_exhausted') ||
    (lower.includes('quota') && lower.includes('exhaust'))
  );
};

const resolveModel = (model?: string) => {
  if (model) {
    return model;
  }

  return DEFAULT_MODEL;
};

export class LiveVoiceSession extends EventEmitter<LiveVoiceEvents> {
  private session?: Session;
  private model?: string;
  private closing = false;
  private setupComplete = false;
  private sentAudioChunks = 0;
  private sentAudioBytes = 0;
  private droppedAudioChunksBeforeSetup = 0;
  private recvAudioChunks = 0;
  private recvAudioBytes = 0;
  private waitingForInput: boolean | null = null;
  private sessionOpenedAtMs = 0;
  private setupCompleteWaitMs = DEFAULT_SETUP_COMPLETE_WAIT_MS;
  private allowAudioWithoutSetup = false;

  constructor(private readonly config: Config) {
    super();
    this.on('error', (_error) => {});

    if (isVoiceDebugEnabled()) {
      voiceDebugLog('voice_debug.enabled', {
        logPath: getVoiceDebugLogPath(),
      });
    }
  }

  getModel() {
    return this.model;
  }

  isConnected() {
    return this.session !== undefined;
  }

  async start(params: StartVoiceSessionParams = {}) {
    if (this.session) {
      return this.model || resolveModel(params.model);
    }

    const ai = await createLiveGoogleGenAI(this.config);
    const requestedModel = resolveModel(params.model);
    const candidateModels = params.model
      ? [requestedModel]
      : [...new Set([requestedModel, QUOTA_FALLBACK_MODEL])];
    const advancedVad = params.advancedVad ?? false;
    const serverSilenceMs = toClampedNumber(
      params.serverSilenceMs,
      DEFAULT_SERVER_SILENCE_DURATION_MS,
      500,
    );
    const setupWaitMs = toClampedNumber(
      params.setupWaitMs,
      DEFAULT_SETUP_COMPLETE_WAIT_MS,
      400,
    );
    const inputTranscriptionLanguageCode =
      typeof params.inputTranscriptionLanguageCode === 'string' &&
      params.inputTranscriptionLanguageCode.trim().length > 0
        ? params.inputTranscriptionLanguageCode.trim()
        : undefined;
    this.setupCompleteWaitMs = setupWaitMs;
    voiceDebugLog('session.start.request', {
      model: requestedModel,
      candidateModels,
      inputTranscriptionLanguageCode: inputTranscriptionLanguageCode || null,
      advancedVad,
      serverSilenceMs,
      setupWaitMs,
      hasTools: Boolean(params.tools),
      hasSystemInstruction: Boolean(params.systemInstruction),
    });

    let lastError: Error | undefined;
    const transcriptionConfigs: Array<{
      inputAudioTranscription: AudioTranscriptionConfig;
      usedLanguageHint: boolean;
      requestedLanguageCode: string | null;
    }> = [];
    if (inputTranscriptionLanguageCode) {
      // Try honoring the requested transcription language first, then
      // automatically fall back to an empty transcription config for
      // compatibility with Live backends that reject this hint.
      transcriptionConfigs.push({
        inputAudioTranscription: {
          languageCode: inputTranscriptionLanguageCode,
        } as AudioTranscriptionConfig,
        usedLanguageHint: true,
        requestedLanguageCode: inputTranscriptionLanguageCode,
      });
      voiceDebugLog('session.start.language_hint_requested', {
        requestedLanguageCode: inputTranscriptionLanguageCode,
      });
    }
    transcriptionConfigs.push({
      inputAudioTranscription: {},
      usedLanguageHint: false,
      requestedLanguageCode: null,
    });

    modelLoop: for (let index = 0; index < candidateModels.length; index += 1) {
      const candidateModel = candidateModels[index];
      for (
        let transcriptionIndex = 0;
        transcriptionIndex < transcriptionConfigs.length;
        transcriptionIndex += 1
      ) {
        const transcriptionConfig = transcriptionConfigs[transcriptionIndex];
        try {
          const session = await ai.live.connect({
            model: candidateModel,
            config: {
              ...(params.systemInstruction
                ? { systemInstruction: params.systemInstruction }
                : {}),
              responseModalities: [Modality.AUDIO],
              inputAudioTranscription:
                transcriptionConfig.inputAudioTranscription,
              outputAudioTranscription: {},
              ...(advancedVad
                ? {
                    proactivity: {
                      proactiveAudio: false,
                    },
                    realtimeInputConfig: {
                      activityHandling:
                        ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
                      automaticActivityDetection: {
                        endOfSpeechSensitivity:
                          EndSensitivity.END_SENSITIVITY_LOW,
                        silenceDurationMs: serverSilenceMs,
                      },
                    },
                  }
                : {}),
              tools: params.tools,
            },
            callbacks: {
              onopen: () => {
                this.sessionOpenedAtMs = Date.now();
                this.allowAudioWithoutSetup = false;
                voiceDebugLog('session.open', {
                  model: candidateModel,
                  usedLanguageHint: transcriptionConfig.usedLanguageHint,
                  languageCode: transcriptionConfig.requestedLanguageCode,
                });
                this.emit('open');
              },
              onmessage: (message: LiveServerMessage) => {
                this.handleServerMessage(message);
              },
              onerror: (error) => {
                voiceDebugLog('session.error', {
                  message: normalizeError(error.error).message,
                });
                this.emit('error', normalizeError(error.error));
              },
              onclose: (event) => {
                voiceDebugLog('session.close', {
                  model: this.model,
                  closing: this.closing,
                  code: event.code,
                  reason: event.reason || null,
                  wasClean: event.wasClean,
                  setupComplete: this.setupComplete,
                  sentAudioChunks: this.sentAudioChunks,
                  sentAudioBytes: this.sentAudioBytes,
                  droppedAudioChunksBeforeSetup:
                    this.droppedAudioChunksBeforeSetup,
                  recvAudioChunks: this.recvAudioChunks,
                  recvAudioBytes: this.recvAudioBytes,
                });
                this.session = undefined;
                this.sessionOpenedAtMs = 0;
                this.allowAudioWithoutSetup = false;
                if (!this.closing) {
                  this.emit('close');
                }
                this.closing = false;
              },
            },
          });

          this.session = session;
          this.model = candidateModel;
          this.sentAudioChunks = 0;
          this.sentAudioBytes = 0;
          this.droppedAudioChunksBeforeSetup = 0;
          this.recvAudioChunks = 0;
          this.recvAudioBytes = 0;
          this.waitingForInput = null;
          this.setupComplete = false;
          return candidateModel;
        } catch (error) {
          const normalizedError = normalizeError(error);
          lastError = normalizedError;
          const hasFallbackWithoutLanguageHint =
            transcriptionIndex < transcriptionConfigs.length - 1;
          const hasNextCandidate = index < candidateModels.length - 1;
          const shouldFallback =
            !hasFallbackWithoutLanguageHint &&
            hasNextCandidate &&
            isQuotaExceededError(normalizedError.message);
          voiceDebugLog('session.start.connect_failed', {
            model: candidateModel,
            message: normalizedError.message,
            usedLanguageHint: transcriptionConfig.usedLanguageHint,
            languageCode: transcriptionConfig.requestedLanguageCode,
            hasFallbackWithoutLanguageHint,
            shouldFallback,
            nextModel: shouldFallback ? candidateModels[index + 1] : undefined,
          });
          if (hasFallbackWithoutLanguageHint) {
            continue;
          }
          if (shouldFallback) {
            continue modelLoop;
          }
          break modelLoop;
        }
      }
    }

    const finalError =
      lastError || new Error('Failed to start live voice session.');
    voiceDebugLog('session.start.failed', {
      message: finalError.message,
    });
    throw finalError;
  }

  sendAudioChunk(
    pcmBytes: Buffer,
    mimeType: string = DEFAULT_INPUT_AUDIO_MIME,
  ) {
    if (!this.session) {
      return;
    }

    if (!this.setupComplete) {
      const setupTimedOut =
        this.sessionOpenedAtMs > 0 &&
        Date.now() - this.sessionOpenedAtMs >= this.setupCompleteWaitMs;
      if (setupTimedOut) {
        if (!this.allowAudioWithoutSetup) {
          this.allowAudioWithoutSetup = true;
          voiceDebugLog('audio.send.setup_timeout_bypass', {
            waitMs: this.setupCompleteWaitMs,
          });
        }
      } else {
        this.droppedAudioChunksBeforeSetup += 1;
        if (this.droppedAudioChunksBeforeSetup === 1) {
          voiceDebugLog('audio.send.blocked_until_setup_complete');
        }
        return;
      }
    }

    try {
      this.sentAudioChunks += 1;
      this.sentAudioBytes += pcmBytes.length;
      if (this.sentAudioChunks % 50 === 0) {
        voiceDebugLog('audio.send.stats', {
          chunks: this.sentAudioChunks,
          bytes: this.sentAudioBytes,
          mimeType,
        });
      }
      this.session.sendRealtimeInput({
        audio: {
          mimeType,
          data: pcmBytes.toString('base64'),
        },
      });
    } catch (error) {
      voiceDebugLog('audio.send.failed', {
        message: normalizeError(error).message,
      });
      this.emit('error', normalizeError(error));
    }
  }

  sendAudioStreamEnd(reason: string = 'manual') {
    if (!this.session) {
      return;
    }

    try {
      voiceDebugLog('audio.stream_end.send', { reason });
      this.session.sendRealtimeInput({ audioStreamEnd: true });
    } catch (error) {
      voiceDebugLog('audio.stream_end.failed', {
        message: normalizeError(error).message,
      });
      this.emit('error', normalizeError(error));
    }
  }

  sendTextTurn(text: string) {
    if (!this.session || !text.trim()) {
      return;
    }

    try {
      voiceDebugLog('text.turn.send', {
        length: text.length,
        preview: text.slice(0, 160),
      });
      this.session.sendClientContent({
        turns: [{ role: 'user', parts: [{ text }] }],
        turnComplete: true,
      });
    } catch (error) {
      voiceDebugLog('text.turn.failed', {
        message: normalizeError(error).message,
      });
      this.emit('error', normalizeError(error));
    }
  }

  sendToolResponses(functionResponses: FunctionResponse[] | FunctionResponse) {
    if (!this.session) {
      return;
    }

    try {
      const responses = Array.isArray(functionResponses)
        ? functionResponses
        : [functionResponses];
      voiceDebugLog('tool.responses.send', {
        count: responses.length,
        names: responses.map((response) => response.name || 'unknown'),
      });
      this.session.sendToolResponse({
        functionResponses,
      });
    } catch (error) {
      voiceDebugLog('tool.responses.failed', {
        message: normalizeError(error).message,
      });
      this.emit('error', normalizeError(error));
    }
  }

  close() {
    if (!this.session) {
      return;
    }

    this.closing = true;
    voiceDebugLog('session.close.requested', {
      sentAudioChunks: this.sentAudioChunks,
      sentAudioBytes: this.sentAudioBytes,
      recvAudioChunks: this.recvAudioChunks,
      recvAudioBytes: this.recvAudioBytes,
    });

    try {
      this.sendAudioStreamEnd('session_close');
    } catch {
      // Best-effort end-of-stream signaling.
    }

    try {
      this.session.close();
    } catch (error) {
      voiceDebugLog('session.close.failed', {
        message: normalizeError(error).message,
      });
      this.emit('error', normalizeError(error));
    } finally {
      this.session = undefined;
      this.closing = false;
      this.setupComplete = false;
      this.sessionOpenedAtMs = 0;
      this.setupCompleteWaitMs = DEFAULT_SETUP_COMPLETE_WAIT_MS;
      this.allowAudioWithoutSetup = false;
      this.emit('close');
    }
  }

  private handleServerMessage(message: LiveServerMessage) {
    if (message.setupComplete) {
      this.setupComplete = true;
      this.allowAudioWithoutSetup = false;
      voiceDebugLog('session.setup_complete');
    }

    if (message.goAway) {
      voiceDebugLog('session.go_away', {
        timeLeft: message.goAway.timeLeft || null,
      });
    }

    if (message.serverContent) {
      const { inputTranscription, outputTranscription, modelTurn } =
        message.serverContent;
      const waitingForInput = message.serverContent.waitingForInput;
      const turnComplete = message.serverContent.turnComplete;
      const turnCompleteReason = message.serverContent.turnCompleteReason;

      if (typeof waitingForInput === 'boolean') {
        if (waitingForInput !== this.waitingForInput) {
          this.waitingForInput = waitingForInput;
          voiceDebugLog('server.waiting_for_input', {
            waitingForInput,
          });
        }
      }

      if (turnComplete || turnCompleteReason) {
        voiceDebugLog('server.turn_complete', {
          turnComplete: Boolean(turnComplete),
          turnCompleteReason: turnCompleteReason || null,
        });
        this.emit('turnComplete', turnCompleteReason || null);
      }

      if (inputTranscription?.text) {
        voiceDebugLog('server.input_transcript', {
          length: inputTranscription.text.length,
          text: inputTranscription.text,
        });
        this.emit('inputTranscript', inputTranscription.text);
      }

      if (outputTranscription?.text) {
        voiceDebugLog('server.output_transcript', {
          length: outputTranscription.text.length,
          text: outputTranscription.text,
        });
        this.emit('outputTranscript', outputTranscription.text);
      }

      const parts = modelTurn?.parts ?? [];
      const textFromParts = parts
        .map((part) => part.text || '')
        .join(' ')
        .trim();
      if (!outputTranscription?.text && textFromParts) {
        voiceDebugLog('server.output_text_part', {
          length: textFromParts.length,
          text: textFromParts,
        });
        this.emit('outputTranscript', textFromParts);
      }
      for (const part of parts) {
        const inlineData = part.inlineData;
        const mimeType = inlineData?.mimeType?.toLowerCase() || '';
        if (inlineData?.data && mimeType.startsWith('audio/pcm')) {
          this.recvAudioChunks += 1;
          const chunk = Buffer.from(inlineData.data, 'base64');
          this.recvAudioBytes += chunk.length;
          if (this.recvAudioChunks % 10 === 0) {
            voiceDebugLog('audio.recv.stats', {
              chunks: this.recvAudioChunks,
              bytes: this.recvAudioBytes,
              mimeType,
            });
          }
          this.emit('outputAudioChunk', {
            chunk,
            mimeType,
          });
        }
      }
    }

    const toolCallFunctionCalls = message.toolCall?.functionCalls ?? [];
    const allFunctionCalls = [...toolCallFunctionCalls];
    if (message.serverContent?.modelTurn?.parts) {
      for (const part of message.serverContent.modelTurn.parts) {
        if (part.functionCall) {
          allFunctionCalls.push(part.functionCall);
        }
      }
    }

    if (allFunctionCalls.length > 0) {
      const seen = new Set<string>();
      const deduped: FunctionCall[] = [];
      for (const call of allFunctionCalls) {
        const key = `${call.id || ''}:${call.name || ''}:${JSON.stringify(call.args || {})}`;
        if (seen.has(key)) {
          continue;
        }
        seen.add(key);
        deduped.push(call);
      }

      voiceDebugLog('server.tool_call', {
        count: deduped.length,
        names: deduped.map((call) => call.name || 'unknown'),
        fromTopLevel: toolCallFunctionCalls.length,
      });
      this.emit('toolCall', deduped);
    }
  }
}
