/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { execFile } from 'node:child_process';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { promisify } from 'node:util';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { getErrorMessage, type Config } from '@google/gemini-cli-core';
import {
  type FunctionCall,
  type FunctionResponse,
  type ToolListUnion,
} from '@google/genai';
import { audioEngine } from '../../services/audioEngine.js';
import { AudioPlayback } from '../../services/audioPlayback.js';
import { LiveVoiceSession } from '../../services/liveVoiceSession.js';
import { voiceDebugLog } from '../../services/voiceDebugLogger.js';
import type { VoicePendingAction } from '../utils/voiceAssistantState.js';

export type VoiceConnectionState = 'idle' | 'connecting' | 'connected';

export interface VoiceAssistantControllerState {
  connectionState: VoiceConnectionState;
  inputTranscript: string;
  outputTranscript: string;
  assistantSpeaking: boolean;
  outputLevel: number;
  model: string | null;
  error: string | null;
  playbackWarning: string | null;
}

export interface ResolvePendingActionRequest {
  actionId?: string;
  decision: string;
  answers?: { [questionIndex: string]: string };
  feedback?: string;
  approvalMode?: string;
}

export interface VoiceAssistantRuntimeConfig {
  model?: string;
  inputTranscriptionLanguageCode?: string;
  forceTurnEndOnSilence?: boolean;
  turnEndSilenceMs?: number;
  maxSpeechSegmentMs?: number;
  transcriptTurnFallback?: boolean;
  transcriptTurnCooldownMs?: number;
  localAssistantFallback?: boolean;
  localAssistantFallbackMs?: number;
  advancedVad?: boolean;
  serverSilenceMs?: number;
  setupWaitMs?: number;
}

interface VoiceAssistantControllerParams {
  config: Config | undefined;
  enabled: boolean;
  captureAudio: boolean;
  runtimeConfig?: VoiceAssistantRuntimeConfig;
  isAgentBusy: () => boolean;
  onDisableRequested: () => void;
  getRuntimeStatus: () => string;
  getPendingActions: () => VoicePendingAction[];
  submitUserRequest: (text: string) => Promise<string | void>;
  submitUserHint: (text: string) => Promise<string | void> | string | void;
  resolvePendingAction: (
    request: ResolvePendingActionRequest,
  ) => Promise<string>;
  cancelCurrentRun: () => void;
}

const INITIAL_STATE: VoiceAssistantControllerState = {
  connectionState: 'idle',
  inputTranscript: '',
  outputTranscript: '',
  assistantSpeaking: false,
  outputLevel: 0,
  model: null,
  error: null,
  playbackWarning: null,
};

const DEFAULT_FORCE_TURN_END_ON_SILENCE = true;
const DEFAULT_INPUT_TRANSCRIPTION_LANGUAGE_CODE = 'en-US';
const DEFAULT_TURN_END_SILENCE_MS = 1600;
const DEFAULT_MAX_SPEECH_SEGMENT_MS = 9000;
const DEFAULT_TRANSCRIPT_TURN_FALLBACK = false;
const DEFAULT_TRANSCRIPT_TURN_COOLDOWN_MS = 4000;
const DEFAULT_LOCAL_ASSISTANT_FALLBACK = false;
const DEFAULT_LOCAL_ASSISTANT_FALLBACK_MS = 3200;
const DEFAULT_ADVANCED_VAD = false;
const DEFAULT_SERVER_SILENCE_MS = 1600;
const DEFAULT_SETUP_WAIT_MS = 1500;
const PUSH_TO_TALK_RELEASE_DRAIN_MS = 140;
const AUTO_HANDOFF_FALLBACK_MS = 1000;
const AUTO_HANDOFF_BUSY_FALLBACK_MS = 450;
const SILENCE_HANDOFF_FALLBACK_MS = 700;
const SILENCE_HANDOFF_BUSY_FALLBACK_MS = 350;
const MODEL_INTERPRET_SUBMIT_FALLBACK_MS = 2200;
const DIRECT_DECISION_TAIL_IGNORE_MS = 1800;
const INPUT_UTTERANCE_WINDOW_MS = 7000;
const OUTPUT_ECHO_WINDOW_MS = 12000;
const POST_TOOL_MODEL_MUTE_MS = 2400;
const REQUEST_DEDUP_WINDOW_MS = 12000;
const APPROVAL_REPLY_WINDOW_MS = 20000;
const SESSION_RECONNECT_RESET_WINDOW_MS = 10000;
const SESSION_MAX_RECONNECT_ATTEMPTS = 3;
const LOCAL_ANNOUNCEMENT_SPEECH_RATE = '182';
const LOCAL_ANNOUNCEMENT_TIMEOUT_MS = 20000;
const LOCAL_ANNOUNCEMENT_MAX_CHARS = 260;
const LOCAL_SPEECH_INPUT_GUARD_MS = 900;
const LOCAL_SPEECH_OUTPUT_MUTE_BUFFER_MS = 800;
const OUTPUT_AUDIO_ACTIVITY_THRESHOLD = 0.008;
const OUTPUT_LEVEL_DECAY_MS = 220;
const OUTPUT_SPEAKING_HOLD_MS = 1200;
const OUTPUT_TRANSCRIPT_RESET_GAP_MS = 1400;
const OUTPUT_TRANSCRIPT_MAX_LINES = 14;
const ENABLE_LOCAL_SYSTEM_ANNOUNCEMENTS =
  process.platform === 'darwin' &&
  process.env['GEMINI_VOICE_LOCAL_TTS'] === '1';
const execFileAsync = promisify(execFile);
const MAX_LOCAL_QUERY_LIST_LIMIT = 200;
const MAX_LOCAL_QUERY_FILE_LINES = 200;
const FRAGMENT_LEADS = new Set([
  'in',
  'on',
  'at',
  'for',
  'to',
  'from',
  'with',
  'about',
  'regarding',
  'inside',
  'within',
]);
const QUESTION_LEADS = new Set([
  'how',
  'what',
  'when',
  'where',
  'why',
  'who',
  'can',
  'could',
  'would',
  'will',
  'is',
  'are',
  'do',
  'does',
  'did',
]);
const COMMAND_LEADS = new Set([
  'run',
  'check',
  'show',
  'tell',
  'count',
  'list',
  'open',
  'create',
  'fix',
  'update',
  'implement',
  'build',
  'test',
  'explain',
  'search',
  'find',
  'git',
]);
const AFFIRMATIVE_TERMS = new Set([
  'yes',
  'yeah',
  'yep',
  'sure',
  'ok',
  'okay',
  'go ahead',
  'proceed',
  'allow',
  'run it',
  'do it',
  'go on',
  'carry on',
  'continue',
  'alow',
  'allo',
]);
const NEGATIVE_TERMS = new Set([
  'no',
  'nope',
  'nah',
  'deny',
  'cancel',
  'stop',
  "don't",
  'do not',
]);
const CONVERSATIONAL_PREFIX_TERMS = new Set([
  'hey',
  'hi',
  'hello',
  'yo',
  'ok',
  'okay',
  'please',
  'assistant',
  'gemini',
  'buddy',
  'bro',
]);
const AFFIRMATIVE_MARKERS = [
  'yes',
  'yeah',
  'yep',
  'sure',
  'okay',
  'ok',
  'go ahead',
  'do it',
  'run it',
  'allow',
  'approve',
  'grant',
] as const;
const NEGATIVE_MARKERS = [
  'no',
  'nope',
  'nah',
  'stop',
  'cancel',
  'deny',
  'reject',
  "don't",
  'do not',
] as const;
const NEGATIVE_FUZZY_MARKERS = ['cancel', 'deny', 'reject'] as const;
const NON_DECISION_NOISE_TOKENS = new Set([
  'noise',
  'silence',
  'background',
  'static',
  'inaudible',
]);
const NON_DECISION_LEADING_PREPOSITIONS = new Set([
  'for',
  'in',
  'on',
  'at',
  'to',
  'from',
  'with',
  'about',
  'regarding',
  'inside',
  'within',
]);
const INTERPRET_PENDING_ACKS = [
  "Got it. I'll interpret that and handle the pending choice.",
  "Nice, understood. I'll sort out that pending action now.",
  "Perfect, I heard you. I'll resolve that approval now.",
] as const;
const SUBMITTED_TO_AGENT_ACKS = [
  'Done. I passed that to the coding agent.',
  'All set. The coding agent has your request.',
  "Done. I've handed that to the coding agent.",
] as const;
const SUBMITTED_GENERIC_ACKS = [
  "Done. I've submitted your request.",
  'All set. Your request is submitted.',
  'Done. I sent that through.',
] as const;
const HINT_ADDED_ACKS = [
  'Nice, I added that as a live hint.',
  'Done. I passed that update to the running task.',
  'Perfect, that guidance is now in the run.',
] as const;

function isTooShortDecisionTranscript(transcript: string): boolean {
  const normalized = normalizeTranscriptText(transcript);
  if (!normalized) {
    return true;
  }
  if (/^\d+$/.test(normalized)) {
    return false;
  }
  const words = normalized.split(/\s+/).filter(Boolean);
  if (words.length === 0) {
    return true;
  }
  if (words.length >= 2) {
    return false;
  }
  return words[0].length < 2;
}
const DECISION_ORDINAL_TO_INDEX: Readonly<Record<string, number>> = {
  first: 0,
  second: 1,
  third: 2,
  fourth: 3,
  fifth: 4,
};
const DECISION_PHRASE_HINTS: Readonly<Record<string, readonly string[]>> = {
  allow_once: ['allow once', 'just once', 'only once', 'one time', 'this time'],
  allow_session: [
    'allow session',
    'for this session',
    'for the session',
    'in this session',
    'this session',
    'current session',
    'session only',
    'session',
  ],
  allow_always: [
    'allow always',
    'always allow',
    'allow permanently',
    'every time',
    'all the time',
    'permanently',
    'forever',
  ],
  allow_tool_session: [
    'allow tool session',
    'for this tool',
    'this tool session',
    'tool only',
  ],
  allow_server_session: [
    'allow server session',
    'for this server',
    'this server session',
    'server only',
  ],
  implement_auto_edit: [
    'auto edit',
    'automatic edit',
    'apply edits',
    'edit automatically',
    'implement auto',
  ],
  implement_manual: [
    'manual',
    'manually',
    'implement manually',
    'manual edit',
    'manual mode',
    'do it manually',
  ],
  stay_in_plan: [
    'stay in plan',
    'stay in plan mode',
    'keep plan',
    'continue planning',
    'do not implement',
    'dont implement',
  ],
  keep: ['keep', 'keep enabled', 'leave enabled', 'do not disable'],
  disable: ['disable', 'turn off', 'switch off'],
  modify: ['modify', 'suggest changes', 'change it', 'edit command'],
  cancel: ['cancel', 'stop', 'abort'],
  deny: ['deny', 'reject', 'disallow'],
};
const DECISION_PRIORITY: readonly string[] = [
  'allow_tool_session',
  'allow_server_session',
  'allow_session',
  'allow_always',
  'allow_once',
  'implement_auto_edit',
  'implement_manual',
  'stay_in_plan',
  'keep',
  'disable',
  'modify',
  'answer',
  'allow',
  'deny',
  'cancel',
];

function hasMarkerSequenceMatch(
  normalizedText: string,
  markers: readonly string[],
): boolean {
  const words = normalizedText.split(/\s+/).filter(Boolean);
  if (words.length === 0) {
    return false;
  }

  for (const marker of markers) {
    const markerWords = marker.split(/\s+/).filter(Boolean);
    if (markerWords.length === 0 || markerWords.length > words.length) {
      continue;
    }

    const limit = words.length - markerWords.length;
    for (let start = 0; start <= limit; start += 1) {
      let matched = true;
      for (let index = 0; index < markerWords.length; index += 1) {
        if (words[start + index] !== markerWords[index]) {
          matched = false;
          break;
        }
      }
      if (matched) {
        return true;
      }
    }
  }

  return false;
}

function toClampedNumber(
  value: number | undefined,
  fallback: number,
  minValue: number,
) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return fallback;
  }
  return Math.max(minValue, Math.floor(value));
}

function resolveVoiceAssistantRuntimeConfig(
  config: VoiceAssistantRuntimeConfig | undefined,
) {
  const model =
    typeof config?.model === 'string' && config.model.trim().length > 0
      ? config.model.trim()
      : undefined;
  const inputTranscriptionLanguageCode =
    typeof config?.inputTranscriptionLanguageCode === 'string' &&
    config.inputTranscriptionLanguageCode.trim().length > 0
      ? config.inputTranscriptionLanguageCode.trim()
      : DEFAULT_INPUT_TRANSCRIPTION_LANGUAGE_CODE;

  return {
    model,
    inputTranscriptionLanguageCode,
    forceTurnEndOnSilence:
      config?.forceTurnEndOnSilence ?? DEFAULT_FORCE_TURN_END_ON_SILENCE,
    turnEndSilenceMs: toClampedNumber(
      config?.turnEndSilenceMs,
      DEFAULT_TURN_END_SILENCE_MS,
      300,
    ),
    maxSpeechSegmentMs: toClampedNumber(
      config?.maxSpeechSegmentMs,
      DEFAULT_MAX_SPEECH_SEGMENT_MS,
      1200,
    ),
    transcriptTurnFallback:
      config?.transcriptTurnFallback ?? DEFAULT_TRANSCRIPT_TURN_FALLBACK,
    transcriptTurnCooldownMs: toClampedNumber(
      config?.transcriptTurnCooldownMs,
      DEFAULT_TRANSCRIPT_TURN_COOLDOWN_MS,
      1200,
    ),
    localAssistantFallback:
      config?.localAssistantFallback ?? DEFAULT_LOCAL_ASSISTANT_FALLBACK,
    localAssistantFallbackMs: toClampedNumber(
      config?.localAssistantFallbackMs,
      DEFAULT_LOCAL_ASSISTANT_FALLBACK_MS,
      1500,
    ),
    advancedVad: config?.advancedVad ?? DEFAULT_ADVANCED_VAD,
    serverSilenceMs: toClampedNumber(
      config?.serverSilenceMs,
      DEFAULT_SERVER_SILENCE_MS,
      500,
    ),
    setupWaitMs: toClampedNumber(
      config?.setupWaitMs,
      DEFAULT_SETUP_WAIT_MS,
      400,
    ),
  };
}

function normalizePathTokenForSpeech(token: string): string {
  const match = token.match(/^([("'`[]*)(.*?)([)"'`\],.;:!?]*)$/);
  const leading = match?.[1] ?? '';
  const core = match?.[2] ?? token;
  const trailing = match?.[3] ?? '';

  if (!core.includes('/') && !core.includes('\\')) {
    return token;
  }
  if (!/[a-z]/i.test(core)) {
    return token;
  }

  const segments = core.split(/[\\/]+/).filter(Boolean);
  if (segments.length < 2) {
    return token;
  }

  const spokenSegments = segments
    .map((segment) =>
      segment
        .replace(/^[~.]+/, '')
        .replace(/[._:-]+/g, ' ')
        .replace(/\s+/g, ' ')
        .trim(),
    )
    .filter(Boolean);
  if (spokenSegments.length < 2) {
    return token;
  }

  return `${leading}${spokenSegments.join(' ')}${trailing}`;
}

function normalizePathsForSpeech(text: string): string {
  return text
    .split(/\s+/)
    .map((token) => normalizePathTokenForSpeech(token))
    .join(' ')
    .trim();
}

function sanitizeAnnouncementText(text: string) {
  const cleaned = text
    .replace(
      /\[VOICE_AGENT_HANDOFF_V1\][\s\S]*?\[\/VOICE_AGENT_HANDOFF_V1\]/g,
      '',
    )
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/^\s*[-*]\s+/gm, '')
    .replace(/\s+/g, ' ')
    .trim();
  const speechSafe = normalizePathsForSpeech(cleaned);
  if (!speechSafe) {
    return '';
  }
  if (speechSafe.length <= LOCAL_ANNOUNCEMENT_MAX_CHARS) {
    return speechSafe;
  }

  const sentenceMatches = speechSafe.match(/[^.!?]+[.!?]?/g) || [];
  let sentenceSummary = '';
  for (const sentence of sentenceMatches) {
    const candidate = `${sentenceSummary}${sentence}`.trim();
    if (candidate.length > LOCAL_ANNOUNCEMENT_MAX_CHARS) {
      break;
    }
    sentenceSummary = candidate.endsWith(' ') ? candidate : `${candidate} `;
  }

  const sentenceResult = sentenceSummary.trim();
  if (
    sentenceResult.length >= Math.floor(LOCAL_ANNOUNCEMENT_MAX_CHARS * 0.55)
  ) {
    return sentenceResult;
  }

  const hardLimit = LOCAL_ANNOUNCEMENT_MAX_CHARS - 1;
  let clipped = speechSafe.slice(0, hardLimit).trim();
  const lastSpace = clipped.lastIndexOf(' ');
  if (lastSpace >= 48) {
    clipped = clipped.slice(0, lastSpace).trim();
  }
  clipped = clipped.replace(/[,:;.-]+$/g, '').trim();
  if (!clipped) {
    return speechSafe.slice(0, LOCAL_ANNOUNCEMENT_MAX_CHARS);
  }
  return `${clipped}.`;
}

const VOICE_ASSISTANT_SYSTEM_INSTRUCTION = `
You are Gemini CLI Voice Assistant, a concise personal assistant controlling a coding agent.
Hard rules:
- Sound like a friendly, confident coding partner and keep replies concise.
- Use natural conversational phrasing with contractions and varied wording; avoid robotic status narration.
- Keep the vibe human and lightly playful while staying professional and clear.
- Avoid repeating the same acknowledgement opener across turns.
- Respond in plain text using one to two short sentences unless you need one clarification question.
- Never narrate internal reasoning, tool usage, or phrases like "initiating analysis".
- Do not use markdown headings or status banners.
- Do not dump long raw file-path lists; summarize counts and mention at most two examples.
- If paths must be spoken, mention key path segments naturally and avoid reading slash characters one by one.
- If the transcript sounds partial, wait for additional speech instead of immediately asking for restatement.
- Never invent workspace facts, file names, git counts, paths, or command outputs.
- Default to submit_user_request for user work requests so the coding agent does the real execution.
- Use query_local_context only for lightweight read-only checks when delegation is unnecessary.
- For status, approvals, permissions, or control, use tools instead of guessing.
- For git status, changed files, or uncommitted file questions, delegate via submit_user_request.
- Support speech in any language or dialect supported by the model, including European and Asian languages.
- Translate user intent internally to clear English before delegating to coding tools.
- If the meaning is ambiguous, ask one short clarification question before taking action.
- Never auto-approve actions without an explicit user decision.
- If the user says "stop" ambiguously, ask whether they mean stop listening, stop the active run, or both.
- For mid-run changes while the agent is working, submit a steering hint so work continues.
`;

const VOICE_ASSISTANT_TOOLS: ToolListUnion = [
  {
    functionDeclarations: [
      {
        name: 'get_runtime_status',
        description:
          'Get exact current coding-agent progress and current intent summary.',
      },
      {
        name: 'list_pending_actions',
        description:
          'List all pending approvals or permissions that require user input.',
      },
      {
        name: 'query_local_context',
        description:
          'Run a read-only local context query for workspace/repo/file facts, then answer user directly.',
        parametersJsonSchema: {
          type: 'object',
          required: ['query'],
          properties: {
            query: {
              type: 'string',
              enum: [
                'workspace_summary',
                'git_status',
                'git_changed_files',
                'list_directory',
                'read_file_excerpt',
              ],
              description: 'Type of local context query.',
            },
            path: {
              type: 'string',
              description:
                'Optional relative path inside current workspace for list/read queries.',
            },
            limit: {
              type: 'number',
              description:
                'Optional max items for list queries (clamped to a safe maximum).',
            },
            maxLines: {
              type: 'number',
              description:
                'Optional max lines for file excerpt queries (clamped to a safe maximum).',
            },
          },
        },
      },
      {
        name: 'submit_user_request',
        description:
          'Submit a clear new user request to the coding agent when appropriate.',
        parametersJsonSchema: {
          type: 'object',
          required: ['text'],
          properties: {
            text: {
              type: 'string',
              description: 'Exact user request to submit.',
            },
          },
        },
      },
      {
        name: 'submit_user_hint',
        description:
          'Inject a live steering hint while the coding agent continues running.',
        parametersJsonSchema: {
          type: 'object',
          required: ['text'],
          properties: {
            text: {
              type: 'string',
              description: 'Hint or adjustment requested by user.',
            },
          },
        },
      },
      {
        name: 'resolve_pending_action',
        description:
          'Resolve a pending action with an explicit decision chosen by the user.',
        parametersJsonSchema: {
          type: 'object',
          required: ['decision'],
          properties: {
            actionId: {
              type: 'string',
              description:
                'Pending action id from list_pending_actions. Optional when only one action is pending.',
            },
            decision: {
              type: 'string',
              enum: [
                'allow',
                'deny',
                'allow_once',
                'allow_session',
                'allow_always',
                'allow_tool_session',
                'allow_server_session',
                'modify',
                'cancel',
                'answer',
                'implement_manual',
                'implement_auto_edit',
                'stay_in_plan',
                'keep',
                'disable',
              ],
              description: 'Explicit user decision for the selected action.',
            },
            answers: {
              type: 'object',
              description:
                'Answer map for ask_user prompts. Keys are question indices.',
              additionalProperties: {
                type: 'string',
              },
            },
            feedback: {
              type: 'string',
              description:
                'Optional feedback when staying in plan mode or rejecting implementation.',
            },
            approvalMode: {
              type: 'string',
              enum: ['default', 'auto_edit'],
              description:
                'Optional implementation approval mode for exit-plan prompts.',
            },
          },
        },
      },
      {
        name: 'stop_voice_assistant',
        description: 'Handle stop requests for assistant, active run, or both.',
        parametersJsonSchema: {
          type: 'object',
          properties: {
            target: {
              type: 'string',
              enum: ['assistant', 'run', 'both'],
              description:
                'What to stop. If unknown or omitted, clarify with the user.',
            },
          },
        },
      },
    ],
  },
];

function asObject(value: unknown): Record<string, unknown> {
  if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
    return Object.fromEntries(Object.entries(value));
  }
  return {};
}

function asStringArg(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}

function asAnswersArg(value: unknown): { [questionIndex: string]: string } {
  const obj = asObject(value);
  const answers: { [questionIndex: string]: string } = {};
  for (const [key, raw] of Object.entries(obj)) {
    if (typeof raw === 'string' && raw.trim().length > 0) {
      answers[key] = raw.trim();
    }
  }
  return answers;
}

function isLikelyFragment(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed) {
    return true;
  }

  if (trimmed.includes('?')) {
    return false;
  }

  const words = trimmed.toLowerCase().split(/\s+/).filter(Boolean);
  if (words.length === 0) {
    return true;
  }

  const first = words[0];
  if (QUESTION_LEADS.has(first) || COMMAND_LEADS.has(first)) {
    return false;
  }

  if (FRAGMENT_LEADS.has(first)) {
    return words.length <= 4;
  }

  // Very short utterances without clear question/command lead are often partial.
  return words.length <= 2;
}

function normalizeTranscriptText(text: string): string {
  return text
    .normalize('NFKC')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s']/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function containsNonLatinLetters(text: string): boolean {
  const normalized = text.normalize('NFKC');
  for (const char of normalized) {
    if (!/\p{L}/u.test(char)) {
      continue;
    }
    if (!/\p{Script=Latin}/u.test(char)) {
      return true;
    }
  }
  return false;
}

function containsAnyKeyword(text: string, keywords: readonly string[]) {
  return keywords.some((keyword) => text.includes(keyword));
}

function getEditDistance(a: string, b: string): number {
  if (a === b) {
    return 0;
  }
  if (a.length === 0) {
    return b.length;
  }
  if (b.length === 0) {
    return a.length;
  }

  const prev = new Array<number>(b.length + 1);
  const next = new Array<number>(b.length + 1);
  for (let j = 0; j <= b.length; j += 1) {
    prev[j] = j;
  }

  for (let i = 1; i <= a.length; i += 1) {
    next[0] = i;
    for (let j = 1; j <= b.length; j += 1) {
      const substitutionCost = a[i - 1] === b[j - 1] ? 0 : 1;
      next[j] = Math.min(
        prev[j] + 1,
        next[j - 1] + 1,
        prev[j - 1] + substitutionCost,
      );
    }
    for (let j = 0; j <= b.length; j += 1) {
      prev[j] = next[j];
    }
  }

  return prev[b.length];
}

function hasFuzzyTokenMatch(
  normalizedText: string,
  markers: readonly string[],
): boolean {
  const tokens = normalizedText.split(/\s+/).filter(Boolean);
  if (tokens.length === 0) {
    return false;
  }

  for (const token of tokens) {
    if (token.length < 2) {
      continue;
    }
    for (const marker of markers) {
      if (token === marker) {
        return true;
      }
      const distance = getEditDistance(token, marker);
      const maxDistance = marker.length <= 4 ? 1 : 2;
      if (distance <= maxDistance) {
        return true;
      }
    }
  }

  return false;
}

function computeAudioChunkLevel(chunk: Buffer): number {
  const sampleCount = Math.floor(chunk.length / 2);
  if (sampleCount <= 0) {
    return 0;
  }

  // Sample every few frames for low overhead while preserving responsiveness.
  const step = Math.max(1, Math.floor(sampleCount / 80));
  let sumSquares = 0;
  let measured = 0;
  for (let index = 0; index < sampleCount; index += step) {
    const pcm = chunk.readInt16LE(index * 2) / 32768;
    sumSquares += pcm * pcm;
    measured += 1;
  }
  if (measured === 0) {
    return 0;
  }

  const rms = Math.sqrt(sumSquares / measured);
  // Mild gain + clamp for a readable UI meter.
  return Math.max(0, Math.min(1, rms * 9));
}

function hashSeedToIndex(seed: string, length: number): number {
  if (length <= 0) {
    return 0;
  }
  let hash = 0;
  for (let i = 0; i < seed.length; i += 1) {
    hash = (hash * 31 + seed.charCodeAt(i)) >>> 0;
  }
  return hash % length;
}

function pickVoicePhrase(options: readonly string[], seedText: string): string {
  if (options.length === 0) {
    return '';
  }
  const normalizedSeed = normalizeTranscriptText(seedText) || seedText.trim();
  const index = hashSeedToIndex(normalizedSeed, options.length);
  return options[index] || options[0];
}

function tokenizeForEcho(text: string): string[] {
  return normalizeTranscriptText(text)
    .split(/\s+/)
    .filter((token) => token.length >= 2 && !/^\d+$/.test(token));
}

function hasStrongTokenOverlap(inputText: string, outputText: string): boolean {
  const inputTokens = Array.from(new Set(tokenizeForEcho(inputText)));
  const outputTokens = new Set(tokenizeForEcho(outputText));

  if (inputTokens.length < 2 || outputTokens.size < 2) {
    return false;
  }

  let sharedCount = 0;
  for (const token of inputTokens) {
    if (outputTokens.has(token)) {
      sharedCount += 1;
    }
  }

  if (sharedCount < 2) {
    return false;
  }

  const inputCoverage = sharedCount / inputTokens.length;
  const outputCoverage = sharedCount / outputTokens.size;
  return (
    inputCoverage >= 0.72 || (inputCoverage >= 0.55 && outputCoverage >= 0.35)
  );
}

interface TimedTranscriptSegment {
  at: number;
  text: string;
}

function pruneTranscriptSegments(
  segments: TimedTranscriptSegment[],
  now: number,
  maxAgeMs: number,
) {
  return segments.filter((segment) => now - segment.at <= maxAgeMs);
}

function buildCandidateUtterance(
  segments: TimedTranscriptSegment[],
  maxAgeMs: number,
  now: number,
) {
  const fresh = pruneTranscriptSegments(segments, now, maxAgeMs);
  if (fresh.length === 0) {
    return '';
  }

  let combined = '';
  for (const segment of fresh) {
    const raw = segment.text;
    if (!raw.trim()) {
      continue;
    }
    if (!combined) {
      combined = raw;
      continue;
    }

    const merged = mergeTranscriptChunk(combined, raw);
    combined = merged;
  }

  return combined.replace(/\s+/g, ' ').trim();
}

function mergeTranscriptChunk(base: string, chunk: string): string {
  if (!base) {
    return chunk;
  }

  const normalizedBase = base.toLowerCase();
  const normalizedChunk = chunk.toLowerCase();
  const maxOverlap = Math.min(
    normalizedBase.length,
    normalizedChunk.length,
    48,
  );

  for (let overlap = maxOverlap; overlap >= 1; overlap--) {
    if (
      normalizedBase.slice(normalizedBase.length - overlap) ===
      normalizedChunk.slice(0, overlap)
    ) {
      return base + chunk.slice(overlap);
    }
  }

  if (/\s$/.test(base) || /^\s/.test(chunk)) {
    return base + chunk;
  }

  if (/^[A-Z]/.test(chunk)) {
    return `${base} ${chunk}`;
  }

  return base + chunk;
}

function appendOutputTranscriptLine(
  previousTranscript: string,
  nextChunk: string,
  startNewLine: boolean,
): string {
  const normalizedNextChunk = nextChunk.replace(/\s+/g, ' ').trim();
  if (!normalizedNextChunk) {
    return previousTranscript;
  }

  const lines = previousTranscript
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length === 0) {
    return normalizedNextChunk;
  }

  if (startNewLine) {
    const lastLine = lines[lines.length - 1] || '';
    if (
      normalizeTranscriptText(lastLine) !==
      normalizeTranscriptText(normalizedNextChunk)
    ) {
      lines.push(normalizedNextChunk);
    }
    return lines.slice(-OUTPUT_TRANSCRIPT_MAX_LINES).join('\n');
  }

  const lastIndex = lines.length - 1;
  const merged = mergeTranscriptChunk(lines[lastIndex], normalizedNextChunk)
    .replace(/\s+/g, ' ')
    .trim();
  lines[lastIndex] = merged || normalizedNextChunk;
  return lines.slice(-OUTPUT_TRANSCRIPT_MAX_LINES).join('\n');
}

function stripConversationalPrefix(text: string): string {
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length === 0) {
    return '';
  }

  let removed = 0;
  while (words.length > 0 && removed < 3) {
    if (!CONVERSATIONAL_PREFIX_TERMS.has(words[0])) {
      break;
    }
    words.shift();
    removed += 1;
  }
  return words.join(' ');
}

function isLikelyAssistantEcho(
  inputText: string,
  outputSegments: TimedTranscriptSegment[],
  now: number,
) {
  // Never suppress explicit approval/deny intents as assistant echo.
  if (detectSimpleDecision(inputText)) {
    return false;
  }

  const input = normalizeTranscriptText(inputText);
  if (input.length < 4) {
    return false;
  }

  const freshOutputs = pruneTranscriptSegments(
    outputSegments,
    now,
    OUTPUT_ECHO_WINDOW_MS,
  );
  for (const output of freshOutputs) {
    const normalizedOutput = normalizeTranscriptText(output.text);
    if (!normalizedOutput) {
      continue;
    }
    if (
      normalizedOutput === input ||
      normalizedOutput.includes(input) ||
      input.includes(normalizedOutput)
    ) {
      return true;
    }
    if (hasStrongTokenOverlap(input, normalizedOutput)) {
      return true;
    }
  }

  return false;
}

function isLikelyMetaNarration(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed) {
    return false;
  }
  if (trimmed.startsWith('**')) {
    return true;
  }

  const normalized = normalizeTranscriptText(trimmed);
  return containsAnyKeyword(normalized, [
    'awaiting tool results',
    'holding pattern',
    'my next move',
    'i m currently',
    'i am currently',
    'i m focusing',
    'i am focusing',
    'clarifying ambiguity',
    'assessing local context',
    'submitting user request',
    'delegating',
    'i cannot directly',
    'submit a request to the coding agent',
    'call in the coding agent',
    'query local context',
    'tool explicitly rejected',
  ]);
}

function isGitDelegationIntent(query: string, fallbackText: string) {
  if (query === 'git_status' || query === 'git_changed_files') {
    return true;
  }
  const normalized = normalizeTranscriptText(fallbackText);
  if (!normalized) {
    return false;
  }
  return containsAnyKeyword(normalized, [
    'uncommitted',
    'changed file',
    'changed files',
    'git status',
    'staged',
    'unstaged',
    'untracked',
    'working tree',
  ]);
}

function detectSimpleDecision(text: string): 'approve' | 'reject' | null {
  const normalized = normalizeTranscriptText(text);
  if (!normalized) {
    return null;
  }

  if (AFFIRMATIVE_TERMS.has(normalized)) {
    return 'approve';
  }
  if (NEGATIVE_TERMS.has(normalized)) {
    return 'reject';
  }

  const hasAffirmative = hasMarkerSequenceMatch(
    normalized,
    AFFIRMATIVE_MARKERS,
  );
  const hasNegative = hasMarkerSequenceMatch(normalized, NEGATIVE_MARKERS);
  const fuzzyAffirmative = hasFuzzyTokenMatch(normalized, AFFIRMATIVE_MARKERS);
  const fuzzyNegative = hasFuzzyTokenMatch(normalized, NEGATIVE_FUZZY_MARKERS);
  const isAffirmative = hasAffirmative || fuzzyAffirmative;
  const isNegative = hasNegative || fuzzyNegative;
  if (isAffirmative && !isNegative) {
    return 'approve';
  }
  if (isNegative && !isAffirmative) {
    return 'reject';
  }
  return null;
}

function isLikelyGenericConfirmationReply(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed) {
    return false;
  }

  // Keep questions out of the fast-path.
  if (/[?؟]/u.test(trimmed)) {
    return false;
  }

  const normalized = normalizeTranscriptText(trimmed);
  if (!normalized) {
    return false;
  }

  const words = normalized.split(/\s+/).filter(Boolean);
  const decisionWords = words.filter(
    (word) => !NON_DECISION_NOISE_TOKENS.has(word),
  );
  if (decisionWords.length === 0 || decisionWords.length > 3) {
    return false;
  }

  const first = decisionWords[0];
  if (QUESTION_LEADS.has(first) || COMMAND_LEADS.has(first)) {
    return false;
  }
  if (NON_DECISION_LEADING_PREPOSITIONS.has(first)) {
    return false;
  }

  return true;
}

function shouldAutoHandoffToCodingAgent(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed) {
    return false;
  }

  if (detectSimpleDecision(trimmed)) {
    return false;
  }

  if (isLikelyFragment(trimmed)) {
    return false;
  }

  const normalized = stripConversationalPrefix(
    normalizeTranscriptText(trimmed),
  );
  if (!normalized) {
    return false;
  }
  const words = normalized.split(/\s+/).filter(Boolean);
  if (words.length < 3) {
    if (
      normalized.includes('help me') ||
      normalized.includes('find files') ||
      normalized.includes('find file')
    ) {
      return true;
    }
    return false;
  }

  const first = words[0];
  const last = words[words.length - 1];
  const ambiguousTailTokens = new Set([
    'me',
    'you',
    'us',
    'it',
    'this',
    'that',
    'please',
    'now',
  ]);
  if (ambiguousTailTokens.has(last)) {
    return false;
  }

  if (first === 'can' || first === 'could' || first === 'would') {
    if (words.length < 5) {
      return false;
    }
    return normalized.startsWith('can you ');
  }

  if (first === 'please') {
    return words.length >= 4;
  }

  if (QUESTION_LEADS.has(first) || COMMAND_LEADS.has(first)) {
    return true;
  }

  return false;
}

function pickDecisionForIntent(
  allowedDecisions: string[],
  intent: 'approve' | 'reject',
): string | null {
  const normalizedAllowed = allowedDecisions.map((decision) =>
    decision.toLowerCase(),
  );

  const preferred =
    intent === 'approve'
      ? [
          'allow',
          'allow_once',
          'allow_session',
          'allow_always',
          'allow_tool_session',
          'allow_server_session',
          'implement_auto_edit',
          'implement_manual',
          'keep',
        ]
      : ['deny', 'cancel', 'disable', 'stay_in_plan'];

  for (const candidate of preferred) {
    if (normalizedAllowed.includes(candidate)) {
      return candidate;
    }
  }

  return null;
}

function extractDecisionByOptionIndex(
  normalizedTranscript: string,
  allowedDecisions: string[],
): string | null {
  if (allowedDecisions.length === 0) {
    return null;
  }

  const directNumberMatch = normalizedTranscript.match(/^\d+$/);
  if (directNumberMatch) {
    const index = Number.parseInt(directNumberMatch[0], 10) - 1;
    if (index >= 0 && index < allowedDecisions.length) {
      return allowedDecisions[index];
    }
  }

  const numberedOptionMatch = normalizedTranscript.match(
    /\b(?:option|choice|select|number|no)\s*(\d+)\b/,
  );
  if (numberedOptionMatch?.[1]) {
    const index = Number.parseInt(numberedOptionMatch[1], 10) - 1;
    if (index >= 0 && index < allowedDecisions.length) {
      return allowedDecisions[index];
    }
  }

  for (const [ordinal, index] of Object.entries(DECISION_ORDINAL_TO_INDEX)) {
    if (
      hasMarkerSequenceMatch(normalizedTranscript, [
        `${ordinal} option`,
        `option ${ordinal}`,
        `${ordinal} choice`,
      ]) ||
      normalizedTranscript === ordinal
    ) {
      if (index >= 0 && index < allowedDecisions.length) {
        return allowedDecisions[index];
      }
    }
  }

  if (
    hasMarkerSequenceMatch(normalizedTranscript, ['last option']) ||
    normalizedTranscript === 'last'
  ) {
    return allowedDecisions[allowedDecisions.length - 1];
  }

  return null;
}

function pickDecisionFromTranscript(
  allowedDecisions: string[],
  transcript: string,
  fallbackApproveIfGenericReply = false,
): string | null {
  const normalizedTranscript = normalizeTranscriptText(transcript);
  if (!normalizedTranscript) {
    return null;
  }

  const normalizedAllowed = allowedDecisions.map((decision) =>
    decision.toLowerCase(),
  );
  if (normalizedAllowed.length === 0) {
    return null;
  }

  for (const decision of normalizedAllowed) {
    const decisionPhrase = decision.replace(/_/g, ' ');
    if (
      normalizedTranscript === decision ||
      normalizedTranscript === decisionPhrase ||
      hasMarkerSequenceMatch(normalizedTranscript, [decisionPhrase])
    ) {
      return decision;
    }
  }

  const optionIndexedDecision = extractDecisionByOptionIndex(
    normalizedTranscript,
    normalizedAllowed,
  );
  if (optionIndexedDecision) {
    return optionIndexedDecision;
  }

  for (const decision of DECISION_PRIORITY) {
    if (!normalizedAllowed.includes(decision)) {
      continue;
    }
    const phrases = DECISION_PHRASE_HINTS[decision];
    if (!phrases || phrases.length === 0) {
      continue;
    }
    if (hasMarkerSequenceMatch(normalizedTranscript, phrases)) {
      return decision;
    }
  }

  const intent = detectSimpleDecision(transcript);
  if (intent) {
    return pickDecisionForIntent(normalizedAllowed, intent);
  }

  if (fallbackApproveIfGenericReply) {
    return pickDecisionForIntent(normalizedAllowed, 'approve');
  }

  return null;
}

type LocalContextQuery =
  | 'workspace_summary'
  | 'git_status'
  | 'git_changed_files'
  | 'list_directory'
  | 'read_file_excerpt';
const LOCAL_CONTEXT_QUERIES: LocalContextQuery[] = [
  'workspace_summary',
  'git_status',
  'git_changed_files',
  'list_directory',
  'read_file_excerpt',
];

function isLocalContextQuery(value: string): value is LocalContextQuery {
  return LOCAL_CONTEXT_QUERIES.some((query) => query === value);
}

interface GitStatusSummary {
  isGitRepo: boolean;
  branch: string | null;
  ahead: number | null;
  behind: number | null;
  totalChangedFiles: number;
  stagedFiles: number;
  unstagedFiles: number;
  untrackedFiles: number;
  files: Array<{
    path: string;
    xy: string;
    staged: boolean;
    unstaged: boolean;
    untracked: boolean;
  }>;
}

function toClampedPositiveInt(
  value: unknown,
  fallback: number,
  maxValue: number,
) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return fallback;
  }
  return Math.max(1, Math.min(maxValue, Math.floor(value)));
}

function resolveWorkspacePath(workspaceRoot: string, rawPath: unknown): string {
  const requested =
    typeof rawPath === 'string' && rawPath.trim().length > 0
      ? rawPath.trim()
      : '.';
  const absolute = path.resolve(workspaceRoot, requested);
  const relative = path.relative(workspaceRoot, absolute);
  const isOutside = relative.startsWith('..') || path.isAbsolute(relative);
  if (isOutside) {
    throw new Error('Path must stay inside the current workspace.');
  }
  return absolute;
}

function parseGitBranchHeader(line: string) {
  const header = line.replace(/^##\s+/, '').trim();
  const [branchPart, trackingPart] = header.split('...');
  const branch = branchPart && branchPart.length > 0 ? branchPart : null;

  let ahead = 0;
  let behind = 0;
  if (trackingPart) {
    const bracketStart = trackingPart.indexOf('[');
    const bracketEnd = trackingPart.indexOf(']');
    if (bracketStart >= 0 && bracketEnd > bracketStart) {
      const meta = trackingPart.slice(bracketStart + 1, bracketEnd);
      const parts = meta.split(',').map((p) => p.trim());
      for (const part of parts) {
        if (part.startsWith('ahead ')) {
          ahead = Number(part.replace('ahead ', '')) || 0;
        } else if (part.startsWith('behind ')) {
          behind = Number(part.replace('behind ', '')) || 0;
        }
      }
    }
  }

  return {
    branch,
    ahead,
    behind,
  };
}

function parsePorcelainEntry(rawLine: string) {
  const line = rawLine.replace(/\r$/, '');
  if (line.length < 3) {
    return null;
  }

  const xy = line.slice(0, 2);
  let filePath = line.slice(3).trim();
  const renameSplit = filePath.split(' -> ');
  if (renameSplit.length > 1) {
    filePath = renameSplit[renameSplit.length - 1];
  }

  const untracked = xy === '??';
  const staged = !untracked && xy[0] !== ' ';
  const unstaged = !untracked && xy[1] !== ' ';

  return {
    path: filePath,
    xy,
    staged,
    unstaged,
    untracked,
  };
}

async function getGitStatusSummary(
  workspaceRoot: string,
): Promise<GitStatusSummary> {
  try {
    const { stdout } = await execFileAsync(
      'git',
      ['status', '--porcelain=v1', '-b'],
      {
        cwd: workspaceRoot,
        maxBuffer: 1024 * 1024,
      },
    );
    const lines = stdout.split('\n').map((line) => line.trimEnd());
    const headerLine = lines.find((line) => line.startsWith('## ')) || '';
    const fileLines = lines.filter(
      (line) => line.length > 0 && !line.startsWith('## '),
    );
    const files = fileLines
      .map((line) => parsePorcelainEntry(line))
      .filter((entry): entry is NonNullable<typeof entry> => Boolean(entry));

    const parsedHeader = headerLine
      ? parseGitBranchHeader(headerLine)
      : { branch: null, ahead: 0, behind: 0 };

    return {
      isGitRepo: true,
      branch: parsedHeader.branch,
      ahead: parsedHeader.ahead,
      behind: parsedHeader.behind,
      totalChangedFiles: files.length,
      stagedFiles: files.filter((file) => file.staged).length,
      unstagedFiles: files.filter((file) => file.unstaged).length,
      untrackedFiles: files.filter((file) => file.untracked).length,
      files,
    };
  } catch (error) {
    const message = getErrorMessage(error).toLowerCase();
    if (message.includes('not a git repository')) {
      return {
        isGitRepo: false,
        branch: null,
        ahead: null,
        behind: null,
        totalChangedFiles: 0,
        stagedFiles: 0,
        unstagedFiles: 0,
        untrackedFiles: 0,
        files: [],
      };
    }
    throw error;
  }
}

async function runLocalContextQuery(args: Record<string, unknown>) {
  const rawQuery = asStringArg(args['query']);
  if (!isLocalContextQuery(rawQuery)) {
    return {
      ok: false,
      message: `Unsupported query_local_context request: ${rawQuery || 'unknown'}.`,
    };
  }
  const query: LocalContextQuery = rawQuery;
  const workspaceRoot = process.cwd();

  switch (query) {
    case 'workspace_summary': {
      const git = await getGitStatusSummary(workspaceRoot);
      return {
        ok: true,
        workspaceRoot,
        git,
      };
    }
    case 'git_status': {
      const git = await getGitStatusSummary(workspaceRoot);
      return {
        ok: true,
        git,
      };
    }
    case 'git_changed_files': {
      const git = await getGitStatusSummary(workspaceRoot);
      const limit = toClampedPositiveInt(
        args['limit'],
        40,
        MAX_LOCAL_QUERY_LIST_LIMIT,
      );
      return {
        ok: true,
        isGitRepo: git.isGitRepo,
        totalChangedFiles: git.totalChangedFiles,
        files: git.files.slice(0, limit),
        truncated: git.files.length > limit,
      };
    }
    case 'list_directory': {
      const limit = toClampedPositiveInt(
        args['limit'],
        60,
        MAX_LOCAL_QUERY_LIST_LIMIT,
      );
      const targetPath = resolveWorkspacePath(workspaceRoot, args['path']);
      const entries = await fs.readdir(targetPath, { withFileTypes: true });
      const sorted = entries
        .map((entry) => ({
          name: entry.name,
          type: entry.isDirectory()
            ? 'directory'
            : entry.isFile()
              ? 'file'
              : entry.isSymbolicLink()
                ? 'symlink'
                : 'other',
        }))
        .sort((a, b) => {
          if (a.type !== b.type) {
            if (a.type === 'directory') return -1;
            if (b.type === 'directory') return 1;
          }
          return a.name.localeCompare(b.name);
        });

      return {
        ok: true,
        path: path.relative(workspaceRoot, targetPath) || '.',
        total: sorted.length,
        entries: sorted.slice(0, limit),
        truncated: sorted.length > limit,
      };
    }
    case 'read_file_excerpt': {
      const maxLines = toClampedPositiveInt(
        args['maxLines'],
        80,
        MAX_LOCAL_QUERY_FILE_LINES,
      );
      const targetPath = resolveWorkspacePath(workspaceRoot, args['path']);
      const stats = await fs.stat(targetPath);
      if (!stats.isFile()) {
        return {
          ok: false,
          message: 'Target path is not a file.',
        };
      }
      const content = await fs.readFile(targetPath, 'utf8');
      const lines = content.split('\n').slice(0, maxLines);
      return {
        ok: true,
        path: path.relative(workspaceRoot, targetPath),
        maxLines,
        excerpt: lines.join('\n'),
        totalLines: content.split('\n').length,
        truncated: content.split('\n').length > maxLines,
      };
    }
    default:
      return {
        ok: false,
        message: `Unsupported query_local_context request: ${query}.`,
      };
  }
}

export function useVoiceAssistantController({
  config,
  enabled,
  captureAudio,
  runtimeConfig,
  isAgentBusy,
  onDisableRequested,
  getRuntimeStatus,
  getPendingActions,
  submitUserRequest,
  submitUserHint,
  resolvePendingAction,
  cancelCurrentRun,
}: VoiceAssistantControllerParams) {
  const resolvedRuntimeConfig = useMemo(
    () => resolveVoiceAssistantRuntimeConfig(runtimeConfig),
    [runtimeConfig],
  );
  const [state, setState] =
    useState<VoiceAssistantControllerState>(INITIAL_STATE);
  const sessionRef = useRef<LiveVoiceSession | null>(null);
  const playbackRef = useRef<AudioPlayback | null>(null);
  const getRuntimeStatusRef = useRef(getRuntimeStatus);
  const getPendingActionsRef = useRef(getPendingActions);
  const isAgentBusyRef = useRef(isAgentBusy);
  const submitUserRequestRef = useRef(submitUserRequest);
  const submitUserHintRef = useRef(submitUserHint);
  const resolvePendingActionRef = useRef(resolvePendingAction);
  const cancelCurrentRunRef = useRef(cancelCurrentRun);
  const onDisableRequestedRef = useRef(onDisableRequested);
  const wasTalkingRef = useRef(false);
  const speechStartAtRef = useRef(0);
  const lastTalkingAtRef = useRef(0);
  const lastTurnEndSignalAtRef = useRef(0);
  const lastInputAtRef = useRef(0);
  const latestInputTranscriptRef = useRef('');
  const lastOutputAtRef = useRef(0);
  const lastToolCallAtRef = useRef(0);
  const lastTranscriptFallbackAtRef = useRef(0);
  const lastLocalFallbackTranscriptRef = useRef('');
  const lastDirectDecisionKeyRef = useRef('');
  const lastDirectDecisionAtRef = useRef(0);
  const lastAutoHandoffTranscriptRef = useRef('');
  const lastUserUtteranceRef = useRef('');
  const lastSubmittedRequestNormalizedRef = useRef('');
  const lastSubmittedRequestAtRef = useRef(0);
  const lastApprovalPromptAtRef = useRef(0);
  const recentInputSegmentsRef = useRef<TimedTranscriptSegment[]>([]);
  const recentOutputSegmentsRef = useRef<TimedTranscriptSegment[]>([]);
  const muteModelOutputUntilRef = useRef(0);
  const localFallbackTimerRef = useRef<NodeJS.Timeout | null>(null);
  const autoHandoffTimerRef = useRef<NodeJS.Timeout | null>(null);
  const silenceHandoffTimerRef = useRef<NodeJS.Timeout | null>(null);
  const modelInterpretSubmitFallbackTimerRef = useRef<NodeJS.Timeout | null>(
    null,
  );
  const localSpeechQueueRef = useRef<Promise<void>>(Promise.resolve());
  const localSpeechActiveRef = useRef(false);
  const localSpeechInputGuardUntilRef = useRef(0);
  const outputLevelRef = useRef(0);
  const outputLevelDecayTimerRef = useRef<NodeJS.Timeout | null>(null);
  const outputSpeakingDecayTimerRef = useRef<NodeJS.Timeout | null>(null);
  const captureAudioRef = useRef(captureAudio);
  const captureStopTimerRef = useRef<NodeJS.Timeout | null>(null);
  const [lastOutputAt, setLastOutputAt] = useState<number>(0);
  const [sessionRestartNonce, setSessionRestartNonce] = useState(0);
  const reconnectAttemptCountRef = useRef(0);
  const lastReconnectAttemptAtRef = useRef(0);

  useEffect(() => {
    getRuntimeStatusRef.current = getRuntimeStatus;
    getPendingActionsRef.current = getPendingActions;
    isAgentBusyRef.current = isAgentBusy;
    submitUserRequestRef.current = submitUserRequest;
    submitUserHintRef.current = submitUserHint;
    resolvePendingActionRef.current = resolvePendingAction;
    cancelCurrentRunRef.current = cancelCurrentRun;
    onDisableRequestedRef.current = onDisableRequested;
  }, [
    getRuntimeStatus,
    getPendingActions,
    isAgentBusy,
    submitUserRequest,
    submitUserHint,
    resolvePendingAction,
    cancelCurrentRun,
    onDisableRequested,
  ]);

  useEffect(() => {
    if (captureAudio) {
      if (captureStopTimerRef.current) {
        clearTimeout(captureStopTimerRef.current);
        captureStopTimerRef.current = null;
      }
      const wasCapturing = captureAudioRef.current;
      captureAudioRef.current = true;
      if (!wasCapturing) {
        voiceDebugLog('audio.capture.started');
      }
      return;
    }

    if (!captureAudioRef.current || captureStopTimerRef.current) {
      return;
    }

    voiceDebugLog('audio.capture.stop_queued', {
      drainMs: PUSH_TO_TALK_RELEASE_DRAIN_MS,
    });
    captureStopTimerRef.current = setTimeout(() => {
      captureStopTimerRef.current = null;
      if (!captureAudioRef.current) {
        return;
      }
      captureAudioRef.current = false;
      voiceDebugLog('audio.capture.stopped');
      if (sessionRef.current?.isConnected()) {
        sessionRef.current.sendAudioStreamEnd('push_to_talk_release');
      }
    }, PUSH_TO_TALK_RELEASE_DRAIN_MS);
    captureStopTimerRef.current.unref?.();
  }, [captureAudio]);

  useEffect(
    () => () => {
      if (captureStopTimerRef.current) {
        clearTimeout(captureStopTimerRef.current);
        captureStopTimerRef.current = null;
      }
      if (outputLevelDecayTimerRef.current) {
        clearTimeout(outputLevelDecayTimerRef.current);
        outputLevelDecayTimerRef.current = null;
      }
      if (outputSpeakingDecayTimerRef.current) {
        clearTimeout(outputSpeakingDecayTimerRef.current);
        outputSpeakingDecayTimerRef.current = null;
      }
      if (modelInterpretSubmitFallbackTimerRef.current) {
        clearTimeout(modelInterpretSubmitFallbackTimerRef.current);
        modelInterpretSubmitFallbackTimerRef.current = null;
      }
    },
    [],
  );

  useEffect(() => {
    if (!enabled) {
      if (captureStopTimerRef.current) {
        clearTimeout(captureStopTimerRef.current);
        captureStopTimerRef.current = null;
      }
      if (outputLevelDecayTimerRef.current) {
        clearTimeout(outputLevelDecayTimerRef.current);
        outputLevelDecayTimerRef.current = null;
      }
      if (outputSpeakingDecayTimerRef.current) {
        clearTimeout(outputSpeakingDecayTimerRef.current);
        outputSpeakingDecayTimerRef.current = null;
      }
      if (modelInterpretSubmitFallbackTimerRef.current) {
        clearTimeout(modelInterpretSubmitFallbackTimerRef.current);
        modelInterpretSubmitFallbackTimerRef.current = null;
      }
      captureAudioRef.current = false;
      outputLevelRef.current = 0;
      reconnectAttemptCountRef.current = 0;
      lastReconnectAttemptAtRef.current = 0;
      setState((prev) => ({
        ...prev,
        assistantSpeaking: false,
        outputLevel: 0,
      }));
    }
  }, [enabled]);

  const speak = useCallback((text: string) => {
    const prompt = sanitizeAnnouncementText(text);
    if (!prompt) {
      return false;
    }

    const now = Date.now();
    const promptNormalized = normalizeTranscriptText(prompt);
    if (
      promptNormalized.includes('approval required') ||
      (promptNormalized.includes('allow') && promptNormalized.includes('deny'))
    ) {
      lastApprovalPromptAtRef.current = now;
    }
    recentOutputSegmentsRef.current = pruneTranscriptSegments(
      recentOutputSegmentsRef.current,
      now,
      OUTPUT_ECHO_WINDOW_MS,
    );
    recentOutputSegmentsRef.current.push({
      at: now,
      text: prompt,
    });

    const speakViaModel = () => {
      if (!sessionRef.current?.isConnected()) {
        return false;
      }
      muteModelOutputUntilRef.current = 0;
      sessionRef.current.sendTextTurn(
        `Notification: ${prompt}\nReply in one short, warm, conversational sentence with light personality. No emojis. If paths appear, speak key path segments naturally instead of slash-by-slash.`,
      );
      return true;
    };

    if (!ENABLE_LOCAL_SYSTEM_ANNOUNCEMENTS) {
      return speakViaModel();
    }

    if (speakViaModel()) {
      return true;
    }

    localSpeechQueueRef.current = localSpeechQueueRef.current.then(async () => {
      localSpeechActiveRef.current = true;
      muteModelOutputUntilRef.current =
        Date.now() +
        LOCAL_ANNOUNCEMENT_TIMEOUT_MS +
        LOCAL_SPEECH_OUTPUT_MUTE_BUFFER_MS;
      try {
        playbackRef.current?.stop();
        const playbackStartResult = await playbackRef.current?.start();
        if (playbackStartResult && !playbackStartResult.available) {
          voiceDebugLog('announcement.local_speech.playback_unavailable', {
            message: playbackStartResult.message || null,
          });
        }
        voiceDebugLog('announcement.local_speech.start', {
          text: prompt,
        });
        await execFileAsync(
          'say',
          ['-r', LOCAL_ANNOUNCEMENT_SPEECH_RATE, prompt],
          {
            timeout: LOCAL_ANNOUNCEMENT_TIMEOUT_MS,
            maxBuffer: 1024 * 1024,
          },
        );
        voiceDebugLog('announcement.local_speech.done', {
          text: prompt,
        });
      } catch (error) {
        voiceDebugLog('announcement.local_speech.failed', {
          message: getErrorMessage(error),
        });
        speakViaModel();
      } finally {
        localSpeechActiveRef.current = false;
        outputLevelRef.current = 0;
        setState((prev) => ({
          ...prev,
          assistantSpeaking: false,
          outputLevel: 0,
        }));
        muteModelOutputUntilRef.current = Math.max(
          muteModelOutputUntilRef.current,
          Date.now() + LOCAL_SPEECH_OUTPUT_MUTE_BUFFER_MS,
        );
        localSpeechInputGuardUntilRef.current =
          Date.now() + LOCAL_SPEECH_INPUT_GUARD_MS;
      }
    });

    return true;
  }, []);

  useEffect(() => {
    if (!enabled || !config) {
      voiceDebugLog('controller.disabled');
      sessionRef.current?.close();
      sessionRef.current = null;
      playbackRef.current?.stop();
      playbackRef.current = null;
      if (autoHandoffTimerRef.current) {
        clearTimeout(autoHandoffTimerRef.current);
        autoHandoffTimerRef.current = null;
      }
      if (silenceHandoffTimerRef.current) {
        clearTimeout(silenceHandoffTimerRef.current);
        silenceHandoffTimerRef.current = null;
      }
      if (modelInterpretSubmitFallbackTimerRef.current) {
        clearTimeout(modelInterpretSubmitFallbackTimerRef.current);
        modelInterpretSubmitFallbackTimerRef.current = null;
      }
      if (outputLevelDecayTimerRef.current) {
        clearTimeout(outputLevelDecayTimerRef.current);
        outputLevelDecayTimerRef.current = null;
      }
      if (outputSpeakingDecayTimerRef.current) {
        clearTimeout(outputSpeakingDecayTimerRef.current);
        outputSpeakingDecayTimerRef.current = null;
      }
      localSpeechActiveRef.current = false;
      localSpeechInputGuardUntilRef.current = 0;
      outputLevelRef.current = 0;
      reconnectAttemptCountRef.current = 0;
      lastReconnectAttemptAtRef.current = 0;
      setState(INITIAL_STATE);
      return;
    }

    voiceDebugLog('controller.enabled');
    let active = true;
    const session = new LiveVoiceSession(config);
    const playback = new AudioPlayback();
    sessionRef.current = session;
    playbackRef.current = playback;
    wasTalkingRef.current = false;
    speechStartAtRef.current = 0;
    lastTalkingAtRef.current = 0;
    lastTurnEndSignalAtRef.current = 0;
    lastInputAtRef.current = 0;
    latestInputTranscriptRef.current = '';
    lastOutputAtRef.current = 0;
    lastToolCallAtRef.current = 0;
    lastTranscriptFallbackAtRef.current = 0;
    lastLocalFallbackTranscriptRef.current = '';
    lastDirectDecisionKeyRef.current = '';
    lastAutoHandoffTranscriptRef.current = '';
    lastUserUtteranceRef.current = '';
    lastSubmittedRequestNormalizedRef.current = '';
    lastSubmittedRequestAtRef.current = 0;
    recentInputSegmentsRef.current = [];
    recentOutputSegmentsRef.current = [];
    muteModelOutputUntilRef.current = 0;
    if (localFallbackTimerRef.current) {
      clearTimeout(localFallbackTimerRef.current);
      localFallbackTimerRef.current = null;
    }
    if (autoHandoffTimerRef.current) {
      clearTimeout(autoHandoffTimerRef.current);
      autoHandoffTimerRef.current = null;
    }
    if (silenceHandoffTimerRef.current) {
      clearTimeout(silenceHandoffTimerRef.current);
      silenceHandoffTimerRef.current = null;
    }
    if (modelInterpretSubmitFallbackTimerRef.current) {
      clearTimeout(modelInterpretSubmitFallbackTimerRef.current);
      modelInterpretSubmitFallbackTimerRef.current = null;
    }
    if (outputLevelDecayTimerRef.current) {
      clearTimeout(outputLevelDecayTimerRef.current);
      outputLevelDecayTimerRef.current = null;
    }
    if (outputSpeakingDecayTimerRef.current) {
      clearTimeout(outputSpeakingDecayTimerRef.current);
      outputSpeakingDecayTimerRef.current = null;
    }
    localSpeechActiveRef.current = false;
    localSpeechInputGuardUntilRef.current = 0;
    outputLevelRef.current = 0;

    setState((prev) => ({
      ...prev,
      connectionState: 'connecting',
      error: null,
      playbackWarning: null,
      inputTranscript: '',
      outputTranscript: '',
    }));

    const onOpen = () => {
      if (!active) {
        return;
      }
      voiceDebugLog('controller.connection.open');
      reconnectAttemptCountRef.current = 0;
      setState((prev) => ({
        ...prev,
        connectionState: 'connected',
      }));
    };

    const onClose = () => {
      if (!active) {
        return;
      }
      const now = Date.now();
      if (
        now - lastReconnectAttemptAtRef.current >
        SESSION_RECONNECT_RESET_WINDOW_MS
      ) {
        reconnectAttemptCountRef.current = 0;
      }
      lastReconnectAttemptAtRef.current = now;
      reconnectAttemptCountRef.current += 1;

      const attempt = reconnectAttemptCountRef.current;
      voiceDebugLog('controller.connection.closed', { attempt });
      if (attempt > SESSION_MAX_RECONNECT_ATTEMPTS) {
        setState((prev) => ({
          ...prev,
          connectionState: 'idle',
          error:
            'Voice session disconnected repeatedly. Hold Cmd+G to start a new session.',
        }));
        return;
      }

      setState((prev) => ({
        ...prev,
        connectionState: 'connecting',
        error: null,
      }));
      setSessionRestartNonce((prev) => prev + 1);
    };

    const normalizeUserRequestForDedupe = (text: string) =>
      stripConversationalPrefix(normalizeTranscriptText(text));

    const wasRecentlySubmitted = (text: string) => {
      const normalized = normalizeUserRequestForDedupe(text);
      if (!normalized) {
        return false;
      }
      const now = Date.now();
      return (
        normalized === lastSubmittedRequestNormalizedRef.current &&
        now - lastSubmittedRequestAtRef.current < REQUEST_DEDUP_WINDOW_MS
      );
    };

    const markSubmittedRequest = (text: string) => {
      const normalized = normalizeUserRequestForDedupe(text);
      if (!normalized) {
        return;
      }
      lastSubmittedRequestNormalizedRef.current = normalized;
      lastSubmittedRequestAtRef.current = Date.now();
      lastAutoHandoffTranscriptRef.current = text.trim();
    };

    const announceOutput = (message: string, muteWindowMs = 0) => {
      const trimmed = message.trim();
      if (!trimmed) {
        return;
      }
      const at = Date.now();
      setLastOutputAt(at);
      lastOutputAtRef.current = at;
      // With push-to-talk, we only need an output mute window while actively
      // capturing mic input. Otherwise this can hide the assistant's normal
      // spoken acknowledgements.
      if (muteWindowMs > 0 && captureAudioRef.current) {
        muteModelOutputUntilRef.current = Math.max(
          muteModelOutputUntilRef.current,
          at + muteWindowMs,
        );
      }
      setState((prev) => ({
        ...prev,
        outputTranscript: appendOutputTranscriptLine(
          prev.outputTranscript,
          trimmed,
          true,
        ),
      }));
      speak(trimmed);
    };

    const clearModelInterpretSubmitFallback = () => {
      if (modelInterpretSubmitFallbackTimerRef.current) {
        clearTimeout(modelInterpretSubmitFallbackTimerRef.current);
        modelInterpretSubmitFallbackTimerRef.current = null;
      }
    };

    const consumeInputTranscript = () => {
      latestInputTranscriptRef.current = '';
      setState((prev) =>
        prev.inputTranscript
          ? {
              ...prev,
              inputTranscript: '',
            }
          : prev,
      );
    };

    const tryResolveDirectDecision = (transcript: string) => {
      const pendingActions = getPendingActionsRef.current();
      if (pendingActions.length === 0) {
        return false;
      }

      const pending = pendingActions[0];
      if (isTooShortDecisionTranscript(transcript)) {
        return false;
      }
      const fallbackApproveIfGenericReply =
        Date.now() - lastApprovalPromptAtRef.current <=
          APPROVAL_REPLY_WINDOW_MS &&
        isLikelyGenericConfirmationReply(transcript);
      const mappedDecision = pickDecisionFromTranscript(
        pending.allowedDecisions,
        transcript,
        fallbackApproveIfGenericReply,
      );
      if (!mappedDecision) {
        return false;
      }

      const dedupeKey = `${pending.id}:${mappedDecision}:${normalizeTranscriptText(
        transcript,
      )}`;
      if (lastDirectDecisionKeyRef.current === dedupeKey) {
        return true;
      }

      lastDirectDecisionKeyRef.current = dedupeKey;
      lastDirectDecisionAtRef.current = Date.now();
      voiceDebugLog('controller.direct_decision.applied', {
        transcript,
        actionId: pending.id,
        decision: mappedDecision,
      });
      void resolvePendingActionRef
        .current({
          actionId: pending.id,
          decision: mappedDecision,
        })
        .then((message) => {
          if (!active) {
            return;
          }
          consumeInputTranscript();
          announceOutput(message);
        })
        .catch((error) => {
          lastDirectDecisionKeyRef.current = '';
          voiceDebugLog('controller.direct_decision.failed', {
            message: getErrorMessage(error),
          });
        });
      return true;
    };

    const shouldRouteTranscriptViaModelInterpreter = (transcript: string) => {
      const normalized = normalizeTranscriptText(transcript);
      if (!normalized) {
        return false;
      }
      if (detectSimpleDecision(normalized)) {
        return false;
      }
      const words = normalized.split(/\s+/).filter(Boolean);
      const hasPendingActions = getPendingActionsRef.current().length > 0;
      const recentApprovalPrompt =
        Date.now() - lastApprovalPromptAtRef.current <=
        APPROVAL_REPLY_WINDOW_MS;
      const likelyApprovalReply = isLikelyGenericConfirmationReply(transcript);

      // Keep multilingual approval replies flowing even when the transcript is
      // short (for example, single-word "allow" variants in non-Latin scripts).
      if (
        (hasPendingActions || recentApprovalPrompt) &&
        (containsNonLatinLetters(transcript) || likelyApprovalReply)
      ) {
        return words.length >= 1;
      }

      if (!containsNonLatinLetters(transcript)) {
        return false;
      }
      return words.length >= 2;
    };

    const routeTranscriptViaModelInterpreter = (
      transcript: string,
      source: 'auto_handoff' | 'local_fallback' | 'silence_handoff',
    ) => {
      const pendingActions = getPendingActionsRef
        .current()
        .slice(0, 3)
        .map((action) => ({
          id: action.id,
          type: action.type,
          title: action.title,
          allowedDecisions: action.allowedDecisions,
        }));
      const hasPendingActions = pendingActions.length > 0;
      const recentApprovalPrompt =
        Date.now() - lastApprovalPromptAtRef.current <=
        APPROVAL_REPLY_WINDOW_MS;
      const likelyApprovalReply = isLikelyGenericConfirmationReply(transcript);
      const requiresDecisionFlow =
        hasPendingActions || recentApprovalPrompt || likelyApprovalReply;
      const payload = JSON.stringify({
        userSpeech: transcript,
        source,
        pendingActions,
      });
      voiceDebugLog(`${source}.model_interpret`, {
        transcript,
        hasPendingActions,
        recentApprovalPrompt,
        likelyApprovalReply,
      });
      if (requiresDecisionFlow) {
        announceOutput(
          pickVoicePhrase(INTERPRET_PENDING_ACKS, transcript),
          POST_TOOL_MODEL_MUTE_MS,
        );
      }
      const instructionLines = [
        "Interpret the user's speech in any supported language or dialect and translate intent to clear English internally.",
        hasPendingActions || recentApprovalPrompt || likelyApprovalReply
          ? 'If this is an approval/denial for a pending action, call list_pending_actions if needed, then call resolve_pending_action.'
          : 'If intent is clear, call submit_user_request (or submit_user_hint for in-flight steering).',
        hasPendingActions || recentApprovalPrompt || likelyApprovalReply
          ? 'When user says generic allow/yes without scope, choose the least permissive allow option available (for example allow_once).'
          : 'Do not resolve pending actions unless the utterance is clearly a decision.',
        hasPendingActions || recentApprovalPrompt || likelyApprovalReply
          ? 'If the user specifies scope (session/always/tool/server), an explicit mode (manual/auto edit), or an option number, resolve to that exact allowed decision.'
          : 'Respect explicit user choice if they mention a concrete decision option.',
        'If this is clearly a new work request (not a decision), call submit_user_request or submit_user_hint.',
        'If intent is ambiguous, ask exactly one short clarification question before taking action.',
        `Payload: ${payload}`,
      ];
      session.sendTextTurn(instructionLines.join('\n'));
      clearModelInterpretSubmitFallback();
      const fallbackTranscript = transcript.trim();
      modelInterpretSubmitFallbackTimerRef.current = setTimeout(() => {
        if (!active) {
          return;
        }
        if (requiresDecisionFlow) {
          const pendingActions = getPendingActionsRef.current();
          if (pendingActions.length === 0) {
            return;
          }
          const pending = pendingActions[0];
          const approvalWindowOpen =
            Date.now() - lastApprovalPromptAtRef.current <=
            APPROVAL_REPLY_WINDOW_MS;
          const latestTranscript =
            latestInputTranscriptRef.current.trim() || fallbackTranscript;
          const modelOutputCandidate = buildCandidateUtterance(
            recentOutputSegmentsRef.current,
            OUTPUT_ECHO_WINDOW_MS,
            Date.now(),
          );
          let mappedDecision = pickDecisionFromTranscript(
            pending.allowedDecisions,
            latestTranscript,
            approvalWindowOpen &&
              isLikelyGenericConfirmationReply(latestTranscript),
          );
          let mappedFrom: 'input' | 'assistant_output' = 'input';
          if (!mappedDecision && modelOutputCandidate) {
            mappedDecision = pickDecisionFromTranscript(
              pending.allowedDecisions,
              modelOutputCandidate,
              approvalWindowOpen &&
                isLikelyGenericConfirmationReply(modelOutputCandidate),
            );
            if (mappedDecision) {
              mappedFrom = 'assistant_output';
            }
          }
          if (!mappedDecision) {
            voiceDebugLog(
              `${source}.model_interpret_decision_fallback.unresolved`,
              {
                transcript: latestTranscript,
                modelOutputCandidate,
                actionId: pending.id,
                allowedDecisions: pending.allowedDecisions,
              },
            );
            return;
          }

          const dedupeText =
            mappedFrom === 'assistant_output'
              ? modelOutputCandidate || latestTranscript
              : latestTranscript;
          const dedupeKey = `${pending.id}:${mappedDecision}:${normalizeTranscriptText(
            dedupeText,
          )}`;
          if (lastDirectDecisionKeyRef.current === dedupeKey) {
            return;
          }
          lastDirectDecisionKeyRef.current = dedupeKey;
          lastDirectDecisionAtRef.current = Date.now();
          voiceDebugLog(`${source}.model_interpret_decision_fallback.applied`, {
            transcript: latestTranscript,
            modelOutputCandidate,
            actionId: pending.id,
            decision: mappedDecision,
            mappedFrom,
          });
          void resolvePendingActionRef
            .current({
              actionId: pending.id,
              decision: mappedDecision,
            })
            .then((message) => {
              if (!active) {
                return;
              }
              consumeInputTranscript();
              announceOutput(message);
            })
            .catch((error) => {
              lastDirectDecisionKeyRef.current = '';
              voiceDebugLog(
                `${source}.model_interpret_decision_fallback.failed`,
                {
                  message: getErrorMessage(error),
                  actionId: pending.id,
                  decision: mappedDecision,
                },
              );
            });
          return;
        }

        // Safety net: if model interpretation does not call a submission tool,
        // hand off the request directly so voice input is never UI-only.
        const handoffTranscript =
          latestInputTranscriptRef.current.trim() || fallbackTranscript;
        if (
          !handoffTranscript ||
          isLikelyFragment(handoffTranscript) ||
          wasRecentlySubmitted(handoffTranscript)
        ) {
          return;
        }
        voiceDebugLog(`${source}.model_interpret_submit_fallback`, {
          transcript: handoffTranscript,
        });
        lastAutoHandoffTranscriptRef.current = handoffTranscript;
        lastToolCallAtRef.current = Date.now();
        void submitUserRequestRef
          .current(handoffTranscript)
          .then((result) => {
            if (!active) {
              return;
            }
            markSubmittedRequest(handoffTranscript);
            consumeInputTranscript();
            const message =
              (typeof result === 'string' && result.trim()) ||
              pickVoicePhrase(SUBMITTED_TO_AGENT_ACKS, handoffTranscript);
            announceOutput(message, POST_TOOL_MODEL_MUTE_MS);
          })
          .catch((error) => {
            voiceDebugLog(`${source}.model_interpret_submit_fallback.failed`, {
              message: getErrorMessage(error),
            });
          });
      }, MODEL_INTERPRET_SUBMIT_FALLBACK_MS);
      modelInterpretSubmitFallbackTimerRef.current.unref?.();
    };

    const onInputTranscript = (text: string) => {
      if (!active || !text.trim()) {
        return;
      }
      const now = Date.now();
      const trimmed = text.trim();
      if (
        !captureAudioRef.current &&
        (localSpeechActiveRef.current ||
          now < localSpeechInputGuardUntilRef.current)
      ) {
        // During local announcements, still accept explicit approve/deny intents.
        if (tryResolveDirectDecision(trimmed)) {
          voiceDebugLog('input.blocked.local_announcement.decision_resolved', {
            text: trimmed,
          });
          return;
        }
        voiceDebugLog('input.blocked.local_announcement', {
          text: trimmed,
          localSpeechActive: localSpeechActiveRef.current,
          guardMsRemaining: Math.max(
            0,
            localSpeechInputGuardUntilRef.current - now,
          ),
        });
        return;
      }
      lastInputAtRef.current = now;

      recentOutputSegmentsRef.current = pruneTranscriptSegments(
        recentOutputSegmentsRef.current,
        now,
        OUTPUT_ECHO_WINDOW_MS,
      );
      const pendingActionsAtInput = getPendingActionsRef.current();
      if (
        pendingActionsAtInput.length === 0 &&
        now - lastDirectDecisionAtRef.current < DIRECT_DECISION_TAIL_IGNORE_MS
      ) {
        voiceDebugLog('input.ignored.post_decision_tail', {
          text: trimmed,
          msSinceDecision: now - lastDirectDecisionAtRef.current,
        });
        return;
      }
      if (
        isLikelyAssistantEcho(trimmed, recentOutputSegmentsRef.current, now) &&
        !(detectSimpleDecision(trimmed) && pendingActionsAtInput.length > 0)
      ) {
        voiceDebugLog('input.echo_ignored', {
          text: trimmed,
        });
        return;
      }

      recentInputSegmentsRef.current = pruneTranscriptSegments(
        recentInputSegmentsRef.current,
        now,
        INPUT_UTTERANCE_WINDOW_MS,
      );
      recentInputSegmentsRef.current.push({
        at: now,
        text,
      });
      const candidateUtterance = buildCandidateUtterance(
        recentInputSegmentsRef.current,
        INPUT_UTTERANCE_WINDOW_MS,
        now,
      );
      latestInputTranscriptRef.current = candidateUtterance || trimmed;
      if (
        latestInputTranscriptRef.current.length >= 8 &&
        !isLikelyFragment(latestInputTranscriptRef.current)
      ) {
        lastUserUtteranceRef.current = latestInputTranscriptRef.current;
      }

      setState((prev) => ({
        ...prev,
        inputTranscript: latestInputTranscriptRef.current,
      }));

      if (tryResolveDirectDecision(latestInputTranscriptRef.current)) {
        return;
      }

      if (autoHandoffTimerRef.current) {
        clearTimeout(autoHandoffTimerRef.current);
      }
      const isAgentBusyNow = isAgentBusyRef.current();
      const autoHandoffDelayMs = isAgentBusyNow
        ? AUTO_HANDOFF_BUSY_FALLBACK_MS
        : AUTO_HANDOFF_FALLBACK_MS;
      autoHandoffTimerRef.current = setTimeout(() => {
        if (!active) {
          return;
        }

        const transcript = latestInputTranscriptRef.current.trim();
        if (shouldRouteTranscriptViaModelInterpreter(transcript)) {
          if (transcript === lastAutoHandoffTranscriptRef.current) {
            return;
          }
          if (wasRecentlySubmitted(transcript)) {
            return;
          }
          lastAutoHandoffTranscriptRef.current = transcript;
          lastToolCallAtRef.current = Date.now();
          routeTranscriptViaModelInterpreter(transcript, 'auto_handoff');
          return;
        }
        if (!shouldAutoHandoffToCodingAgent(transcript)) {
          return;
        }
        if (transcript === lastAutoHandoffTranscriptRef.current) {
          return;
        }

        const sinceLastToolCall = Date.now() - lastToolCallAtRef.current;
        const minToolCooldownMs = isAgentBusyRef.current()
          ? Math.floor(AUTO_HANDOFF_BUSY_FALLBACK_MS * 0.8)
          : AUTO_HANDOFF_FALLBACK_MS;
        if (sinceLastToolCall < minToolCooldownMs) {
          return;
        }
        if (wasRecentlySubmitted(transcript)) {
          return;
        }

        lastAutoHandoffTranscriptRef.current = transcript;
        lastToolCallAtRef.current = Date.now();
        voiceDebugLog('auto_handoff.submit_request', {
          transcript,
        });

        void submitUserRequestRef
          .current(transcript)
          .then((result) => {
            if (!active) {
              return;
            }
            markSubmittedRequest(transcript);
            consumeInputTranscript();
            const message =
              (typeof result === 'string' && result.trim()) ||
              pickVoicePhrase(SUBMITTED_TO_AGENT_ACKS, transcript);
            announceOutput(message, POST_TOOL_MODEL_MUTE_MS);
          })
          .catch((error) => {
            voiceDebugLog('auto_handoff.submit_request.failed', {
              message: getErrorMessage(error),
            });
          });
      }, autoHandoffDelayMs);

      if (!resolvedRuntimeConfig.localAssistantFallback) {
        return;
      }

      if (localFallbackTimerRef.current) {
        clearTimeout(localFallbackTimerRef.current);
      }
      localFallbackTimerRef.current = setTimeout(() => {
        if (!active) {
          return;
        }
        const transcript = latestInputTranscriptRef.current.trim();
        if (!transcript) {
          return;
        }

        if (shouldRouteTranscriptViaModelInterpreter(transcript)) {
          if (transcript === lastLocalFallbackTranscriptRef.current) {
            return;
          }
          if (wasRecentlySubmitted(transcript)) {
            return;
          }
          lastLocalFallbackTranscriptRef.current = transcript;
          lastToolCallAtRef.current = Date.now();
          routeTranscriptViaModelInterpreter(transcript, 'local_fallback');
          return;
        }

        // Avoid repeated local fallback for the exact same transcript text.
        if (transcript === lastLocalFallbackTranscriptRef.current) {
          return;
        }
        if (wasRecentlySubmitted(transcript)) {
          return;
        }

        const now = Date.now();
        const recentModelOutput =
          now - lastOutputAtRef.current <
          resolvedRuntimeConfig.localAssistantFallbackMs / 2;
        const recentToolCall =
          now - lastToolCallAtRef.current <
          resolvedRuntimeConfig.localAssistantFallbackMs / 2;
        if (recentModelOutput || recentToolCall) {
          return;
        }

        lastLocalFallbackTranscriptRef.current = transcript;
        voiceDebugLog('local_fallback.submit_request', {
          transcript,
        });
        void submitUserRequestRef
          .current(transcript)
          .then((result) => {
            if (!active) {
              return;
            }
            markSubmittedRequest(transcript);
            consumeInputTranscript();
            const message =
              (typeof result === 'string' && result.trim()) ||
              pickVoicePhrase(SUBMITTED_GENERIC_ACKS, transcript);
            announceOutput(message, POST_TOOL_MODEL_MUTE_MS);
          })
          .catch((error) => {
            voiceDebugLog('local_fallback.submit_request.failed', {
              message: getErrorMessage(error),
            });
          });
      }, resolvedRuntimeConfig.localAssistantFallbackMs);
    };

    const onOutputTranscript = (text: string) => {
      if (!active || !text.trim()) {
        return;
      }
      const now = Date.now();
      if (now < muteModelOutputUntilRef.current) {
        return;
      }

      const prunedOutputs = pruneTranscriptSegments(
        recentOutputSegmentsRef.current,
        now,
        OUTPUT_ECHO_WINDOW_MS,
      );
      const candidateOutputs = [
        ...prunedOutputs,
        {
          at: now,
          text,
        },
      ];
      const candidateOutput = buildCandidateUtterance(
        candidateOutputs,
        OUTPUT_ECHO_WINDOW_MS,
        now,
      );

      if (
        isLikelyMetaNarration(text) ||
        isLikelyMetaNarration(candidateOutput)
      ) {
        if (captureAudioRef.current) {
          muteModelOutputUntilRef.current = now + POST_TOOL_MODEL_MUTE_MS;
        }
        recentOutputSegmentsRef.current = prunedOutputs;
        voiceDebugLog('output.meta_suppressed', {
          text: candidateOutput || text,
        });
        return;
      }
      recentOutputSegmentsRef.current = candidateOutputs;
      const previousOutputAt = lastOutputAtRef.current;
      setLastOutputAt(now);
      lastOutputAtRef.current = now;
      setState((prev) => ({
        ...prev,
        outputTranscript: appendOutputTranscriptLine(
          prev.outputTranscript,
          text,
          now - previousOutputAt > OUTPUT_TRANSCRIPT_RESET_GAP_MS,
        ),
      }));
    };

    const onOutputAudioChunk = ({
      chunk,
    }: {
      chunk: Buffer;
      mimeType: string;
    }) => {
      if (!active) {
        return;
      }
      if (Date.now() < muteModelOutputUntilRef.current) {
        return;
      }
      playback.playChunk(chunk);

      // Keep assistant speaking state alive while audio chunks are arriving,
      // even when a given chunk has very low energy.
      if (outputSpeakingDecayTimerRef.current) {
        clearTimeout(outputSpeakingDecayTimerRef.current);
      }
      outputSpeakingDecayTimerRef.current = setTimeout(() => {
        setState((prev) => ({
          ...prev,
          assistantSpeaking: false,
          outputLevel: 0,
        }));
      }, OUTPUT_SPEAKING_HOLD_MS);
      outputSpeakingDecayTimerRef.current.unref?.();

      const chunkLevel = computeAudioChunkLevel(chunk);
      setState((prev) =>
        prev.assistantSpeaking
          ? prev
          : {
              ...prev,
              assistantSpeaking: true,
            },
      );
      if (chunkLevel < OUTPUT_AUDIO_ACTIVITY_THRESHOLD) {
        return;
      }
      const smoothedLevel = outputLevelRef.current * 0.2 + chunkLevel * 0.8;
      outputLevelRef.current = smoothedLevel;
      setState((prev) => ({
        ...prev,
        assistantSpeaking: true,
        outputLevel: smoothedLevel,
      }));

      if (outputLevelDecayTimerRef.current) {
        clearTimeout(outputLevelDecayTimerRef.current);
      }
      outputLevelDecayTimerRef.current = setTimeout(() => {
        outputLevelRef.current = 0;
        setState((prev) => ({
          ...prev,
          outputLevel: 0,
        }));
      }, OUTPUT_LEVEL_DECAY_MS);
      outputLevelDecayTimerRef.current.unref?.();
    };

    const onError = (error: Error) => {
      if (!active) {
        return;
      }
      voiceDebugLog('controller.connection.error', {
        message: error.message,
      });
      setState((prev) => ({
        ...prev,
        error: error.message,
        connectionState: 'idle',
      }));
    };

    const onToolCall = (functionCalls: FunctionCall[]) => {
      lastToolCallAtRef.current = Date.now();
      clearModelInterpretSubmitFallback();
      void (async () => {
        const responses: FunctionResponse[] = [];
        voiceDebugLog('controller.tool_call.received', {
          count: functionCalls.length,
          names: functionCalls.map((call) => call.name || 'unknown'),
        });
        for (const call of functionCalls) {
          if (call.name === 'submit_user_request') {
            const submitArgs = asObject(call.args);
            const argText = asStringArg(submitArgs['text']);
            const now = Date.now();
            const fallbackCandidate = buildCandidateUtterance(
              recentInputSegmentsRef.current,
              INPUT_UTTERANCE_WINDOW_MS,
              now,
            );
            const fallbackText =
              fallbackCandidate.trim() ||
              lastUserUtteranceRef.current.trim() ||
              latestInputTranscriptRef.current.trim();
            const intendedText = argText || fallbackText;
            if (intendedText && wasRecentlySubmitted(intendedText)) {
              responses.push({
                id: call.id,
                name: call.name || 'submit_user_request',
                response: {
                  ok: true,
                  deduped: true,
                  submittedText: intendedText,
                  message:
                    'I already submitted that request to the coding agent.',
                },
              });
              continue;
            }
          }

          const response = await handleAssistantToolCall({
            call,
            getRuntimeStatus: () => getRuntimeStatusRef.current(),
            getPendingActions: () => getPendingActionsRef.current(),
            submitUserRequest: (text) => submitUserRequestRef.current(text),
            submitUserHint: (text) => submitUserHintRef.current(text),
            resolvePendingAction: (request) =>
              resolvePendingActionRef.current(request),
            cancelCurrentRun: () => cancelCurrentRunRef.current(),
            onDisableRequested: () => onDisableRequestedRef.current(),
            getFallbackUserRequestText: () => {
              const now = Date.now();
              const candidate = buildCandidateUtterance(
                recentInputSegmentsRef.current,
                INPUT_UTTERANCE_WINDOW_MS,
                now,
              );
              return (
                candidate.trim() ||
                lastUserUtteranceRef.current.trim() ||
                latestInputTranscriptRef.current.trim()
              );
            },
          });
          responses.push(response);

          const responseBody = asObject(response.response);
          const responseMessage = asStringArg(responseBody['message']);
          if (
            responseMessage &&
            (response.name === 'submit_user_request' ||
              response.name === 'submit_user_hint' ||
              response.name === 'resolve_pending_action' ||
              (response.name === 'query_local_context' &&
                responseBody['delegated'] === true))
          ) {
            const submittedText = asStringArg(responseBody['submittedText']);
            if (
              (response.name === 'submit_user_request' ||
                response.name === 'query_local_context') &&
              submittedText
            ) {
              markSubmittedRequest(submittedText);
            }
            consumeInputTranscript();
            announceOutput(responseMessage, POST_TOOL_MODEL_MUTE_MS);
          }
        }
        if (responses.length > 0) {
          session.sendToolResponses(responses);
        }
      })().catch((error: unknown) => {
        const message =
          error instanceof Error
            ? error.message
            : String(error ?? 'Unknown error');
        voiceDebugLog('controller.tool_call.failed', { message });
        session.emit('error', new Error(message));
      });
    };

    session.on('open', onOpen);
    session.on('close', onClose);
    session.on('inputTranscript', onInputTranscript);
    session.on('outputTranscript', onOutputTranscript);
    session.on('outputAudioChunk', onOutputAudioChunk);
    session.on('toolCall', onToolCall);
    session.on('error', onError);

    const unsubscribePcm = audioEngine.subscribePcm((pcmBytes) => {
      if (!captureAudioRef.current) {
        return;
      }
      // While push-to-talk is actively held, forward audio even during local narration.
      session.sendAudioChunk(pcmBytes);
    });

    const maybeSendTranscriptFallback = (
      trigger: 'silence_commit' | 'max_speech_commit',
    ) => {
      if (!resolvedRuntimeConfig.transcriptTurnFallback) {
        return;
      }
      const transcript = latestInputTranscriptRef.current.trim();
      if (!transcript) {
        return;
      }
      const now = Date.now();
      const sinceLastFallback = now - lastTranscriptFallbackAtRef.current;
      if (sinceLastFallback < resolvedRuntimeConfig.transcriptTurnCooldownMs) {
        return;
      }

      lastTranscriptFallbackAtRef.current = now;
      voiceDebugLog('turn.transcript_fallback.send', {
        trigger,
        transcript,
      });
      session.sendTextTurn(`User said: ${transcript}`);
    };

    const unsubscribeAudioState = audioEngine.subscribe((audioState) => {
      if (!resolvedRuntimeConfig.forceTurnEndOnSilence) {
        return;
      }
      if (!captureAudioRef.current) {
        if (wasTalkingRef.current) {
          wasTalkingRef.current = false;
          speechStartAtRef.current = 0;
          lastTalkingAtRef.current = 0;
        }
        if (silenceHandoffTimerRef.current) {
          clearTimeout(silenceHandoffTimerRef.current);
          silenceHandoffTimerRef.current = null;
        }
        return;
      }

      const now = Date.now();
      if (audioState.isTalking) {
        if (!wasTalkingRef.current) {
          voiceDebugLog('turn.speech_start');
          speechStartAtRef.current = now;
          recentInputSegmentsRef.current = [];
          if (silenceHandoffTimerRef.current) {
            clearTimeout(silenceHandoffTimerRef.current);
            silenceHandoffTimerRef.current = null;
          }
        }
        wasTalkingRef.current = true;
        lastTalkingAtRef.current = now;

        const speechMs = now - speechStartAtRef.current;
        const sinceLastSignalMs = now - lastTurnEndSignalAtRef.current;
        if (
          speechMs >= resolvedRuntimeConfig.maxSpeechSegmentMs &&
          sinceLastSignalMs >= 1000
        ) {
          lastTurnEndSignalAtRef.current = now;
          speechStartAtRef.current = now;
          voiceDebugLog('turn.max_speech_commit', {
            speechMs,
            thresholdMs: resolvedRuntimeConfig.maxSpeechSegmentMs,
          });
          session.sendAudioStreamEnd('max_speech_commit');
          maybeSendTranscriptFallback('max_speech_commit');
        }
        return;
      }

      if (!wasTalkingRef.current) {
        return;
      }

      const silenceMs = now - lastTalkingAtRef.current;
      const sinceLastSignalMs = now - lastTurnEndSignalAtRef.current;
      if (
        silenceMs >= resolvedRuntimeConfig.turnEndSilenceMs &&
        sinceLastSignalMs >= 500
      ) {
        lastTurnEndSignalAtRef.current = now;
        wasTalkingRef.current = false;
        speechStartAtRef.current = 0;
        voiceDebugLog('turn.silence_commit', {
          silenceMs,
          thresholdMs: resolvedRuntimeConfig.turnEndSilenceMs,
        });
        session.sendAudioStreamEnd('silence_commit');
        maybeSendTranscriptFallback('silence_commit');
        if (silenceHandoffTimerRef.current) {
          clearTimeout(silenceHandoffTimerRef.current);
        }
        const isAgentBusyNow = isAgentBusyRef.current();
        const silenceHandoffDelayMs = isAgentBusyNow
          ? SILENCE_HANDOFF_BUSY_FALLBACK_MS
          : SILENCE_HANDOFF_FALLBACK_MS;
        silenceHandoffTimerRef.current = setTimeout(() => {
          if (!active) {
            return;
          }
          const transcript = latestInputTranscriptRef.current.trim();
          const currentNow = Date.now();
          if (shouldRouteTranscriptViaModelInterpreter(transcript)) {
            if (transcript === lastAutoHandoffTranscriptRef.current) {
              return;
            }
            if (wasRecentlySubmitted(transcript)) {
              return;
            }
            lastAutoHandoffTranscriptRef.current = transcript;
            lastToolCallAtRef.current = currentNow;
            routeTranscriptViaModelInterpreter(transcript, 'silence_handoff');
            return;
          }
          if (!shouldAutoHandoffToCodingAgent(transcript)) {
            return;
          }
          if (transcript === lastAutoHandoffTranscriptRef.current) {
            return;
          }
          if (wasRecentlySubmitted(transcript)) {
            return;
          }

          const sinceLastToolCall = currentNow - lastToolCallAtRef.current;
          const sinceLastOutput = currentNow - lastOutputAtRef.current;
          const sinceLastInput = currentNow - lastInputAtRef.current;
          const minHandoffCooldownMs = isAgentBusyRef.current()
            ? SILENCE_HANDOFF_BUSY_FALLBACK_MS
            : SILENCE_HANDOFF_FALLBACK_MS;
          if (
            sinceLastToolCall < minHandoffCooldownMs ||
            sinceLastOutput < minHandoffCooldownMs ||
            sinceLastInput < resolvedRuntimeConfig.turnEndSilenceMs
          ) {
            return;
          }

          lastAutoHandoffTranscriptRef.current = transcript;
          lastToolCallAtRef.current = currentNow;
          voiceDebugLog('silence_handoff.submit_request', {
            transcript,
            sinceLastToolCall,
            sinceLastOutput,
            sinceLastInput,
          });
          void submitUserRequestRef
            .current(transcript)
            .then((result) => {
              if (!active) {
                return;
              }
              markSubmittedRequest(transcript);
              consumeInputTranscript();
              const message =
                (typeof result === 'string' && result.trim()) ||
                pickVoicePhrase(SUBMITTED_TO_AGENT_ACKS, transcript);
              announceOutput(message, POST_TOOL_MODEL_MUTE_MS);
            })
            .catch((error) => {
              voiceDebugLog('silence_handoff.submit_request.failed', {
                message: getErrorMessage(error),
              });
            });
        }, silenceHandoffDelayMs);
      }
    });

    void (async () => {
      const playbackResult = await playback.start();
      if (!active) {
        return;
      }
      if (!playbackResult.available) {
        setState((prev) => ({
          ...prev,
          playbackWarning:
            playbackResult.message ??
            'Audio playback unavailable; transcripts only.',
        }));
      }

      try {
        const model = await session.start({
          model: resolvedRuntimeConfig.model,
          inputTranscriptionLanguageCode:
            resolvedRuntimeConfig.inputTranscriptionLanguageCode,
          advancedVad: resolvedRuntimeConfig.advancedVad,
          serverSilenceMs: resolvedRuntimeConfig.serverSilenceMs,
          setupWaitMs: resolvedRuntimeConfig.setupWaitMs,
          tools: VOICE_ASSISTANT_TOOLS,
          systemInstruction: VOICE_ASSISTANT_SYSTEM_INSTRUCTION,
        });
        if (!active) {
          return;
        }
        voiceDebugLog('controller.connection.ready', { model });
        setState((prev) => ({
          ...prev,
          connectionState: 'connected',
          model,
        }));
      } catch (error) {
        if (!active) {
          return;
        }
        voiceDebugLog('controller.connection.start_failed', {
          message: getErrorMessage(error),
        });
        setState((prev) => ({
          ...prev,
          connectionState: 'idle',
          error: getErrorMessage(error),
        }));
      }
    })();

    return () => {
      voiceDebugLog('controller.cleanup');
      active = false;
      if (localFallbackTimerRef.current) {
        clearTimeout(localFallbackTimerRef.current);
        localFallbackTimerRef.current = null;
      }
      if (autoHandoffTimerRef.current) {
        clearTimeout(autoHandoffTimerRef.current);
        autoHandoffTimerRef.current = null;
      }
      if (silenceHandoffTimerRef.current) {
        clearTimeout(silenceHandoffTimerRef.current);
        silenceHandoffTimerRef.current = null;
      }
      if (modelInterpretSubmitFallbackTimerRef.current) {
        clearTimeout(modelInterpretSubmitFallbackTimerRef.current);
        modelInterpretSubmitFallbackTimerRef.current = null;
      }
      if (outputLevelDecayTimerRef.current) {
        clearTimeout(outputLevelDecayTimerRef.current);
        outputLevelDecayTimerRef.current = null;
      }
      if (outputSpeakingDecayTimerRef.current) {
        clearTimeout(outputSpeakingDecayTimerRef.current);
        outputSpeakingDecayTimerRef.current = null;
      }
      localSpeechActiveRef.current = false;
      localSpeechInputGuardUntilRef.current = 0;
      outputLevelRef.current = 0;
      unsubscribePcm();
      unsubscribeAudioState();
      session.off('open', onOpen);
      session.off('close', onClose);
      session.off('inputTranscript', onInputTranscript);
      session.off('outputTranscript', onOutputTranscript);
      session.off('outputAudioChunk', onOutputAudioChunk);
      session.off('toolCall', onToolCall);
      session.off('error', onError);
      session.close();
      playback.stop();
      sessionRef.current = null;
      playbackRef.current = null;
    };
  }, [config, enabled, resolvedRuntimeConfig, sessionRestartNonce, speak]);

  return {
    ...state,
    lastOutputAt,
    speak,
  };
}

interface HandleAssistantToolCallParams {
  call: FunctionCall;
  getRuntimeStatus: () => string;
  getPendingActions: () => VoicePendingAction[];
  getFallbackUserRequestText: () => string;
  submitUserRequest: (text: string) => Promise<string | void>;
  submitUserHint: (text: string) => Promise<string | void> | string | void;
  resolvePendingAction: (
    request: ResolvePendingActionRequest,
  ) => Promise<string>;
  cancelCurrentRun: () => void;
  onDisableRequested: () => void;
}

async function handleAssistantToolCall(
  params: HandleAssistantToolCallParams,
): Promise<FunctionResponse> {
  const {
    call,
    getRuntimeStatus,
    getPendingActions,
    getFallbackUserRequestText,
    submitUserRequest,
    submitUserHint,
    resolvePendingAction,
    cancelCurrentRun,
    onDisableRequested,
  } = params;
  const name = call.name || '';
  const args = asObject(call.args);

  try {
    switch (name) {
      case 'get_runtime_status':
        return {
          id: call.id,
          name,
          response: { status: getRuntimeStatus() },
        };
      case 'list_pending_actions':
        return {
          id: call.id,
          name,
          response: { actions: getPendingActions() },
        };
      case 'query_local_context': {
        const query = asStringArg(args['query']);
        const fallbackRequestText = getFallbackUserRequestText().trim();
        if (isGitDelegationIntent(query, fallbackRequestText)) {
          const delegatedText =
            fallbackRequestText || 'List uncommitted files in this workspace.';
          const submitResult = await submitUserRequest(delegatedText);
          return {
            id: call.id,
            name,
            response: {
              ok: true,
              delegated: true,
              submittedText: delegatedText,
              message:
                submitResult ||
                pickVoicePhrase(SUBMITTED_TO_AGENT_ACKS, delegatedText),
            },
          };
        }
        if (
          !isLocalContextQuery(query) &&
          fallbackRequestText &&
          shouldAutoHandoffToCodingAgent(fallbackRequestText)
        ) {
          const submitResult = await submitUserRequest(fallbackRequestText);
          return {
            id: call.id,
            name,
            response: {
              ok: true,
              delegated: true,
              submittedText: fallbackRequestText,
              message:
                submitResult ||
                pickVoicePhrase(SUBMITTED_TO_AGENT_ACKS, fallbackRequestText),
            },
          };
        }
        const result = await runLocalContextQuery(args);
        return {
          id: call.id,
          name,
          response: result,
        };
      }
      case 'submit_user_request': {
        const argText = asStringArg(args['text']);
        const fallbackText = getFallbackUserRequestText().trim();
        const text = argText || fallbackText;
        if (!argText && fallbackText) {
          voiceDebugLog('tool.submit_user_request.fallback_text', {
            text: fallbackText,
          });
        }
        if (!text) {
          return {
            id: call.id,
            name,
            response: { ok: false, message: 'Missing request text.' },
          };
        }
        if (isLikelyFragment(text)) {
          return {
            id: call.id,
            name,
            response: {
              ok: false,
              partialUtterance: true,
              message:
                'Partial utterance detected. Wait for additional user speech before responding.',
            },
          };
        }
        const result = await submitUserRequest(text);
        return {
          id: call.id,
          name,
          response: {
            ok: true,
            submittedText: text,
            message: result || pickVoicePhrase(SUBMITTED_TO_AGENT_ACKS, text),
          },
        };
      }
      case 'submit_user_hint': {
        const text = asStringArg(args['text']);
        if (!text) {
          return {
            id: call.id,
            name,
            response: { ok: false, message: 'Missing hint text.' },
          };
        }
        const result = await submitUserHint(text);
        return {
          id: call.id,
          name,
          response: {
            ok: true,
            message: result || pickVoicePhrase(HINT_ADDED_ACKS, text),
          },
        };
      }
      case 'resolve_pending_action': {
        const rawDecision = asStringArg(args['decision']);
        if (!rawDecision) {
          return {
            id: call.id,
            name,
            response: { ok: false, message: 'Missing decision.' },
          };
        }
        const actionId = asStringArg(args['actionId']) || undefined;
        const pendingActions = getPendingActions();
        const targetAction = actionId
          ? pendingActions.find((action) => action.id === actionId)
          : pendingActions[0];
        let decision = rawDecision.toLowerCase();
        if (targetAction && !targetAction.allowedDecisions.includes(decision)) {
          const mapped = pickDecisionFromTranscript(
            targetAction.allowedDecisions,
            rawDecision,
          );
          if (mapped) {
            voiceDebugLog('tool.resolve_pending_action.decision_mapped', {
              original: rawDecision,
              mapped,
              actionId: targetAction.id,
            });
            decision = mapped;
          }
        }
        const request: ResolvePendingActionRequest = {
          actionId,
          decision,
          answers: asAnswersArg(args['answers']),
          feedback: asStringArg(args['feedback']) || undefined,
          approvalMode: asStringArg(args['approvalMode']) || undefined,
        };
        const message = await resolvePendingAction(request);
        return {
          id: call.id,
          name,
          response: { ok: true, message },
        };
      }
      case 'stop_voice_assistant': {
        const target = asStringArg(args['target']);
        if (!target) {
          return {
            id: call.id,
            name,
            response: {
              ok: false,
              needsClarification: true,
              question:
                'Do you want me to stop listening, stop the active run, or both?',
            },
          };
        }

        if (target === 'run' || target === 'both') {
          cancelCurrentRun();
        }
        if (target === 'assistant' || target === 'both') {
          onDisableRequested();
        }

        return {
          id: call.id,
          name,
          response: {
            ok: true,
            message:
              target === 'both'
                ? 'Stopped the active run and voice assistant.'
                : target === 'run'
                  ? 'Stopped the active run.'
                  : 'Stopped voice assistant listening.',
          },
        };
      }
      default:
        return {
          id: call.id,
          name: name || 'unknown_tool',
          response: {
            ok: false,
            message: `Unknown assistant tool: ${name}`,
          },
        };
    }
  } catch (error) {
    return {
      id: call.id,
      name: name || 'assistant_error',
      response: {
        ok: false,
        message: getErrorMessage(error),
      },
    };
  }
}
