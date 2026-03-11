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
import {
  estimateTokenCountSync,
  getErrorMessage,
  tokenLimit,
  type Config,
} from '@google/gemini-cli-core';
import {
  type FunctionCall,
  type FunctionResponse,
  type ToolListUnion,
} from '@google/genai';
import { audioEngine } from '../../services/audioEngine.js';
import { AudioPlayback } from '../../services/audioPlayback.js';
import { LiveVoiceSession } from '../../services/liveVoiceSession.js';
import {
  buildVoiceAssistantSystemInstruction,
  getVoicePersonaByName,
} from '../../services/voicePersonas.js';
import { voiceDebugLog } from '../../services/voiceDebugLogger.js';
import {
  appendOutputTranscriptChunk,
  mergeTranscriptChunk,
} from '../utils/voiceTranscript.js';
import type { VoicePendingAction } from '../utils/voiceAssistantState.js';

export type VoiceConnectionState = 'idle' | 'connecting' | 'connected';

export interface VoiceAssistantOutputItem {
  id: number;
  anchorHistoryId: number;
  text: string;
}

export interface VoiceAssistantControllerState {
  connectionState: VoiceConnectionState;
  inputTranscript: string;
  outputTranscript: string;
  outputHistory: VoiceAssistantOutputItem[];
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
  persona?: string;
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
  contextSyncGeneration?: number;
  contextSyncMessage?: string | null;
  isAgentBusy: () => boolean;
  onDisableRequested: () => void;
  getRuntimeStatus: () => string;
  getLatestHistoryId: () => number;
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
  outputHistory: [],
  assistantSpeaking: false,
  outputLevel: 0,
  model: null,
  error: null,
  playbackWarning: null,
};

const DEFAULT_FORCE_TURN_END_ON_SILENCE = true;
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
const OUTPUT_PLAYBACK_DRAIN_GUARD_MS = 180;
const OUTPUT_TRANSCRIPT_RESET_GAP_MS = 1400;
const OUTPUT_TRANSCRIPT_HARD_RESET_GAP_MS = 2600;
const OUTPUT_TRANSCRIPT_MAX_LINES = 10;
const OUTPUT_HISTORY_MAX_ITEMS = 8;
const DEFAULT_VOICE_CONTEXT_THRESHOLD = 0.5;
const VOICE_CONTEXT_MIN_TRIGGER_TOKENS = 1024;
const VOICE_CONTEXT_ROLLOVER_BUFFER_MIN_TOKENS = 160;
const VOICE_CONTEXT_ROLLOVER_BUFFER_MAX_TOKENS = 1024;
const VOICE_CONTEXT_ROLLOVER_RETRY_MS = 320;
const VOICE_CONTEXT_MAX_SUMMARY_CHARS = 1800;
const VOICE_CONTEXT_PREVIOUS_SUMMARY_MAX_CHARS = 720;
const VOICE_CONTEXT_RUNTIME_STATUS_MAX_CHARS = 260;
const VOICE_CONTEXT_ENTRY_MAX_CHARS = 220;
const VOICE_CONTEXT_RECENT_ENTRIES = 6;
const VOICE_CONTEXT_MAX_LEDGER_ENTRIES = 24;
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
const NON_LATIN_LANGUAGE_BASES = new Set([
  'am',
  'ar',
  'be',
  'bg',
  'bn',
  'bo',
  'dz',
  'el',
  'fa',
  'gu',
  'he',
  'hi',
  'hy',
  'ja',
  'ka',
  'kk',
  'km',
  'kn',
  'ko',
  'lo',
  'mk',
  'ml',
  'mn',
  'mr',
  'my',
  'ne',
  'or',
  'pa',
  'ps',
  'ru',
  'si',
  'sr',
  'ta',
  'te',
  'th',
  'ti',
  'uk',
  'ur',
  'yi',
  'zh',
]);
const NON_LATIN_SCRIPT_SUBTAGS = new Set([
  'Arab',
  'Armn',
  'Beng',
  'Cyrl',
  'Deva',
  'Ethi',
  'Geor',
  'Grek',
  'Gujr',
  'Guru',
  'Hang',
  'Hani',
  'Hans',
  'Hant',
  'Hebr',
  'Hira',
  'Kana',
  'Khmr',
  'Knda',
  'Laoo',
  'Mlym',
  'Mong',
  'Mymr',
  'Orya',
  'Sinh',
  'Taml',
  'Telu',
  'Thai',
  'Tibt',
]);

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
  allow_once: [
    'allow once',
    'just once',
    'only once',
    'one time',
    'this time',
    'for once',
  ],
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
  const persona =
    typeof config?.persona === 'string' && config.persona.trim().length > 0
      ? config.persona.trim()
      : undefined;
  const model =
    typeof config?.model === 'string' && config.model.trim().length > 0
      ? config.model.trim()
      : undefined;
  const inputTranscriptionLanguageCode =
    typeof config?.inputTranscriptionLanguageCode === 'string' &&
    config.inputTranscriptionLanguageCode.trim().length > 0
      ? config.inputTranscriptionLanguageCode.trim()
      : undefined;

  return {
    persona,
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

function compactVoiceContextText(text: string): string {
  return text
    .replace(/<[^>]+>/g, ' ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\s+/g, ' ')
    .trim();
}

function truncateVoiceContextText(text: string, maxChars: number): string {
  const compact = compactVoiceContextText(text);
  if (!compact || compact.length <= maxChars) {
    return compact;
  }

  let clipped = compact.slice(0, Math.max(1, maxChars - 1)).trim();
  const lastSpace = clipped.lastIndexOf(' ');
  if (lastSpace >= Math.floor(maxChars * 0.55)) {
    clipped = clipped.slice(0, lastSpace).trim();
  }
  clipped = clipped.replace(/[,:;.-]+$/g, '').trim();
  return clipped ? `${clipped}…` : compact.slice(0, maxChars);
}

function serializeVoiceContextPayload(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }

  try {
    return JSON.stringify(value);
  } catch {
    return String(value ?? '');
  }
}

function isVoiceContextClearEventMessage(message: string | null | undefined) {
  const normalized = compactVoiceContextText(message ?? '').toLowerCase();
  return normalized === 'conversation context has been cleared.';
}

type VoiceSessionContextRole = 'user' | 'assistant';

interface VoiceSessionContextEntry {
  id: number;
  role: VoiceSessionContextRole;
  text: string;
  at: number;
}

function formatPendingActionForVoiceContext(
  action: VoicePendingAction,
): string {
  const title = truncateVoiceContextText(
    action.title || action.detail || action.type,
    110,
  );
  const decisions = action.allowedDecisions.slice(0, 4).join(', ');
  return decisions ? `${title} [${decisions}]` : title;
}

function buildVoiceSessionSummary(params: {
  previousSummary: string;
  runtimeStatus: string;
  pendingActions: VoicePendingAction[];
  recentEntries: VoiceSessionContextEntry[];
  lastUserUtterance: string;
  contextSyncMessage: string | null;
}): string {
  const sections: string[] = [];
  const previousSummary = truncateVoiceContextText(
    params.previousSummary,
    VOICE_CONTEXT_PREVIOUS_SUMMARY_MAX_CHARS,
  );
  if (previousSummary) {
    sections.push(`Earlier memory: ${previousSummary}`);
  }

  const runtimeStatus = truncateVoiceContextText(
    params.runtimeStatus,
    VOICE_CONTEXT_RUNTIME_STATUS_MAX_CHARS,
  );
  if (runtimeStatus) {
    sections.push(`Coding agent status: ${runtimeStatus}`);
  }

  if (params.contextSyncMessage) {
    sections.push(
      `Latest context maintenance event: ${truncateVoiceContextText(params.contextSyncMessage, 180)}`,
    );
  }

  if (params.pendingActions.length > 0) {
    sections.push(
      `Pending actions: ${params.pendingActions
        .slice(0, 3)
        .map((action) => formatPendingActionForVoiceContext(action))
        .join(' | ')}`,
    );
  }

  const recentEntries = params.recentEntries.slice(
    -VOICE_CONTEXT_RECENT_ENTRIES,
  );
  if (recentEntries.length > 0) {
    const recentLines = recentEntries.map((entry) => {
      const roleLabel = entry.role === 'user' ? 'User' : 'Assistant';
      return `${roleLabel}: ${truncateVoiceContextText(entry.text, VOICE_CONTEXT_ENTRY_MAX_CHARS)}`;
    });
    sections.push(`Recent exchanges: ${recentLines.join(' | ')}`);
  }

  const lastUserUtterance = truncateVoiceContextText(
    params.lastUserUtterance,
    VOICE_CONTEXT_ENTRY_MAX_CHARS,
  );
  if (
    lastUserUtterance &&
    !recentEntries.some(
      (entry) =>
        entry.role === 'user' &&
        truncateVoiceContextText(entry.text, VOICE_CONTEXT_ENTRY_MAX_CHARS) ===
          lastUserUtterance,
    )
  ) {
    sections.push(`Latest user wording/language cue: ${lastUserUtterance}`);
  }

  sections.push(
    'Carry this forward silently. Keep replies aligned with the current coding-agent state and the user’s latest language.',
  );

  return truncateVoiceContextText(
    sections.filter(Boolean).join(' '),
    VOICE_CONTEXT_MAX_SUMMARY_CHARS,
  );
}

function appendVoiceSessionCarryoverInstruction(
  baseInstruction: string,
  summary: string,
): string {
  const compactSummary = compactVoiceContextText(summary);
  if (!compactSummary) {
    return baseInstruction;
  }

  return [
    baseInstruction,
    'Private carry-over memory for this restarted live voice session:',
    '<voice_session_memory>',
    compactSummary,
    '</voice_session_memory>',
    'Use this memory only to preserve continuity. Do not quote it verbatim unless the user asks.',
  ].join('\n');
}

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

const VOICE_ASSISTANT_TOOLS_TOKEN_ESTIMATE = estimateTokenCountSync([
  {
    text: JSON.stringify(VOICE_ASSISTANT_TOOLS),
  },
]);

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

function containsLatinLetters(text: string): boolean {
  return /\p{Script=Latin}/u.test(text.normalize('NFKC'));
}

function normalizeScriptTag(tag: string): string {
  if (tag.length !== 4) {
    return tag;
  }
  return `${tag[0]?.toUpperCase() || ''}${tag.slice(1).toLowerCase()}`;
}

function shouldPreferLatinTranscriptDisplay(languageCode: string | undefined) {
  if (!languageCode) {
    return false;
  }
  const normalized = languageCode.trim();
  if (!normalized) {
    return false;
  }

  const subtags = normalized.split(/[-_]/).filter(Boolean);
  const language = subtags[0]?.toLowerCase() || '';
  const scriptTag = subtags
    .slice(1)
    .find((subtag) => subtag.length === 4 && /^[a-z]+$/i.test(subtag));
  const normalizedScriptTag = scriptTag
    ? normalizeScriptTag(scriptTag)
    : undefined;

  if (normalizedScriptTag === 'Latn') {
    return true;
  }
  if (
    normalizedScriptTag &&
    NON_LATIN_SCRIPT_SUBTAGS.has(normalizedScriptTag)
  ) {
    return false;
  }
  if (!language) {
    return false;
  }
  return !NON_LATIN_LANGUAGE_BASES.has(language);
}

function shouldSuppressNonLatinTranscriptDisplay(
  transcript: string,
  inputLanguageCode: string | undefined,
) {
  if (!transcript.trim()) {
    return false;
  }
  if (!containsNonLatinLetters(transcript)) {
    return false;
  }
  if (containsLatinLetters(transcript)) {
    return false;
  }
  return shouldPreferLatinTranscriptDisplay(inputLanguageCode);
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

interface QueuedModelTurnRequest {
  kind: 'instruction' | 'notification';
  text: string;
}

interface QueuedModelTurn extends QueuedModelTurnRequest {
  id: number;
  enqueuedAt: number;
  preview: string;
}

type ActiveModelTurnKind = QueuedModelTurn['kind'] | 'assistant';

interface ActiveModelTurnState {
  id: number;
  kind: ActiveModelTurnKind;
  origin: 'queued' | 'ambient';
  startedAt: number;
  suppressedReason?: string;
  outputTranscriptChars: number;
  outputTranscriptChunks: number;
  outputAudioChunks: number;
  outputAudioBytes: number;
  toolCallCount: number;
  preview?: string;
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

function appendVoiceOutputHistory(
  history: VoiceAssistantOutputItem[],
  nextMessage: string,
  anchorHistoryId: number,
) {
  const trimmedMessage = nextMessage.trim();
  if (!trimmedMessage) {
    return history;
  }

  const lastMessage = history[history.length - 1]?.text ?? '';
  if (
    lastMessage &&
    normalizeTranscriptText(lastMessage) ===
      normalizeTranscriptText(trimmedMessage)
  ) {
    return history;
  }

  const lastId = history[history.length - 1]?.id ?? 0;

  return [
    ...history,
    {
      id: lastId + 1,
      anchorHistoryId: Math.max(0, anchorHistoryId),
      text: trimmedMessage,
    },
  ].slice(-OUTPUT_HISTORY_MAX_ITEMS);
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

const REDUNDANT_SUBMISSION_ACK_NORMALIZED = new Set(
  [
    ...SUBMITTED_TO_AGENT_ACKS,
    ...SUBMITTED_GENERIC_ACKS,
    "On it. I'll check that and report back.",
    'I already submitted that request to the coding agent.',
  ].map((message) => normalizeTranscriptText(message)),
);

function isRedundantSubmissionAck(message: string): boolean {
  const normalized = normalizeTranscriptText(message);
  if (!normalized) {
    return false;
  }
  return REDUNDANT_SUBMISSION_ACK_NORMALIZED.has(normalized);
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

function shouldDirectlySubmitVoiceRequest(text: string): boolean {
  if (shouldAutoHandoffToCodingAgent(text)) {
    return true;
  }

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
  if (!containsNonLatinLetters(trimmed)) {
    return false;
  }

  const normalized = normalizeTranscriptText(trimmed);
  if (!normalized) {
    return false;
  }

  const words = normalized.split(/\s+/).filter(Boolean);
  return words.length >= 4 || normalized.length >= 12;
}

function getAmbientAssistantSuppressionReason(
  transcript: string,
  hasPendingActions: boolean,
  recentApprovalPrompt: boolean,
): string | null {
  const trimmed = transcript.trim();
  if (!trimmed) {
    return null;
  }

  if (hasPendingActions || recentApprovalPrompt) {
    return 'pending_action_flow';
  }

  if (shouldDirectlySubmitVoiceRequest(trimmed)) {
    return 'tool_first_request';
  }

  return null;
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
  contextSyncGeneration = 0,
  contextSyncMessage = null,
  isAgentBusy,
  onDisableRequested,
  getRuntimeStatus,
  getLatestHistoryId,
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
  const selectedVoicePersona = useMemo(
    () => getVoicePersonaByName(resolvedRuntimeConfig.persona),
    [resolvedRuntimeConfig.persona],
  );
  const voiceAssistantSystemInstruction = useMemo(
    () => buildVoiceAssistantSystemInstruction(selectedVoicePersona),
    [selectedVoicePersona],
  );
  const [state, setState] =
    useState<VoiceAssistantControllerState>(INITIAL_STATE);
  const sessionRef = useRef<LiveVoiceSession | null>(null);
  const playbackRef = useRef<AudioPlayback | null>(null);
  const getRuntimeStatusRef = useRef(getRuntimeStatus);
  const getLatestHistoryIdRef = useRef(getLatestHistoryId);
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
  const latestDisplayInputTranscriptRef = useRef('');
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
  const ambientSuppressionReasonRef = useRef<string | null>(null);
  const recentInputSegmentsRef = useRef<TimedTranscriptSegment[]>([]);
  const recentOutputSegmentsRef = useRef<TimedTranscriptSegment[]>([]);
  const recentInstructionOutputSegmentsRef = useRef<TimedTranscriptSegment[]>(
    [],
  );
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
  const pendingModelTurnsRef = useRef<QueuedModelTurn[]>([]);
  const pendingModelTurnFlushTimerRef = useRef<NodeJS.Timeout | null>(null);
  const nextModelTurnIdRef = useRef(1);
  const activeModelTurnRef = useRef<ActiveModelTurnState | null>(null);
  const modelTurnActiveRef = useRef(false);
  const outputLevelRef = useRef(0);
  const outputLevelDecayTimerRef = useRef<NodeJS.Timeout | null>(null);
  const outputSpeakingDecayTimerRef = useRef<NodeJS.Timeout | null>(null);
  const resetOutputTranscriptOnNextChunkRef = useRef(false);
  const latestOutputTranscriptRef = useRef('');
  const captureAudioRef = useRef(captureAudio);
  const captureStopTimerRef = useRef<NodeJS.Timeout | null>(null);
  const [lastOutputAt, setLastOutputAt] = useState<number>(0);
  const [sessionRestartNonce, setSessionRestartNonce] = useState(0);
  const reconnectAttemptCountRef = useRef(0);
  const lastReconnectAttemptAtRef = useRef(0);
  const sessionRestartPendingRef = useRef(false);
  const currentVoiceModelRef = useRef<string | null>(
    resolvedRuntimeConfig.model ?? null,
  );
  const voiceContextThresholdFractionRef = useRef(
    DEFAULT_VOICE_CONTEXT_THRESHOLD,
  );
  const voiceContextEntriesRef = useRef<VoiceSessionContextEntry[]>([]);
  const voiceContextSummaryRef = useRef('');
  const voiceContextTokenEstimateRef = useRef(0);
  const nextVoiceContextEntryIdRef = useRef(1);
  const voiceContextRolloverPendingRef = useRef(false);
  const voiceContextRolloverReasonRef = useRef<string | null>(null);
  const voiceContextRolloverTimerRef = useRef<NodeJS.Timeout | null>(null);
  const lastVoiceContextUserKeyRef = useRef('');
  const lastVoiceContextUserAtRef = useRef(0);
  const lastContextSyncGenerationRef = useRef(contextSyncGeneration);
  const lastContextSyncMessageRef = useRef<string | null>(contextSyncMessage);

  useEffect(() => {
    getRuntimeStatusRef.current = getRuntimeStatus;
    getLatestHistoryIdRef.current = getLatestHistoryId;
    getPendingActionsRef.current = getPendingActions;
    isAgentBusyRef.current = isAgentBusy;
    submitUserRequestRef.current = submitUserRequest;
    submitUserHintRef.current = submitUserHint;
    resolvePendingActionRef.current = resolvePendingAction;
    cancelCurrentRunRef.current = cancelCurrentRun;
    onDisableRequestedRef.current = onDisableRequested;
  }, [
    getRuntimeStatus,
    getLatestHistoryId,
    getPendingActions,
    isAgentBusy,
    submitUserRequest,
    submitUserHint,
    resolvePendingAction,
    cancelCurrentRun,
    onDisableRequested,
  ]);

  useEffect(() => {
    let cancelled = false;
    voiceContextThresholdFractionRef.current = DEFAULT_VOICE_CONTEXT_THRESHOLD;

    if (!config || typeof config.getCompressionThreshold !== 'function') {
      return () => {
        cancelled = true;
      };
    }

    void config
      .getCompressionThreshold()
      .then((threshold) => {
        if (cancelled) {
          return;
        }
        voiceContextThresholdFractionRef.current =
          typeof threshold === 'number' &&
          Number.isFinite(threshold) &&
          threshold > 0
            ? threshold
            : DEFAULT_VOICE_CONTEXT_THRESHOLD;
        voiceDebugLog('voice_context.threshold_loaded', {
          fraction: voiceContextThresholdFractionRef.current,
        });
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        voiceContextThresholdFractionRef.current =
          DEFAULT_VOICE_CONTEXT_THRESHOLD;
        voiceDebugLog('voice_context.threshold_load_failed', {
          message: getErrorMessage(error),
        });
      });

    return () => {
      cancelled = true;
    };
  }, [config]);

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
      if (pendingModelTurnFlushTimerRef.current) {
        clearTimeout(pendingModelTurnFlushTimerRef.current);
        pendingModelTurnFlushTimerRef.current = null;
      }
      if (voiceContextRolloverTimerRef.current) {
        clearTimeout(voiceContextRolloverTimerRef.current);
        voiceContextRolloverTimerRef.current = null;
      }
    },
    [],
  );

  const clearVoiceContextRolloverTimer = useCallback(() => {
    if (voiceContextRolloverTimerRef.current) {
      clearTimeout(voiceContextRolloverTimerRef.current);
      voiceContextRolloverTimerRef.current = null;
    }
  }, []);

  const getVoiceContextBudget = useCallback(
    (modelOverride?: string | null) => {
      const model =
        modelOverride ||
        currentVoiceModelRef.current ||
        resolvedRuntimeConfig.model ||
        '';
      const modelTokenLimit = tokenLimit(model);
      const configuredThreshold =
        voiceContextThresholdFractionRef.current ||
        DEFAULT_VOICE_CONTEXT_THRESHOLD;
      const thresholdTokens = Math.max(
        VOICE_CONTEXT_MIN_TRIGGER_TOKENS,
        Math.floor(modelTokenLimit * configuredThreshold),
      );
      const rolloverBuffer = Math.min(
        VOICE_CONTEXT_ROLLOVER_BUFFER_MAX_TOKENS,
        Math.max(
          VOICE_CONTEXT_ROLLOVER_BUFFER_MIN_TOKENS,
          Math.floor(thresholdTokens * 0.15),
        ),
      );

      return {
        model,
        modelTokenLimit,
        thresholdTokens,
        rolloverTriggerTokens: Math.max(
          VOICE_CONTEXT_MIN_TRIGGER_TOKENS,
          thresholdTokens - rolloverBuffer,
        ),
      };
    },
    [resolvedRuntimeConfig.model],
  );

  const requestSessionRestart = useCallback(
    (reason: 'context_rollover' | 'context_sync') => {
      if (sessionRestartPendingRef.current) {
        return false;
      }
      sessionRestartPendingRef.current = true;
      voiceDebugLog('controller.session_restart.requested', {
        reason,
      });
      setState((prev) => ({
        ...prev,
        connectionState: 'connecting',
        error: null,
        inputTranscript: '',
        outputTranscript: '',
      }));
      setSessionRestartNonce((prev) => prev + 1);
      return true;
    },
    [],
  );

  const buildVoiceContextSummary = useCallback(
    () =>
      buildVoiceSessionSummary({
        previousSummary: voiceContextSummaryRef.current,
        runtimeStatus: getRuntimeStatusRef.current(),
        pendingActions: getPendingActionsRef.current(),
        recentEntries: voiceContextEntriesRef.current,
        lastUserUtterance: lastUserUtteranceRef.current,
        contextSyncMessage: lastContextSyncMessageRef.current,
      }),
    [],
  );

  const maybePerformVoiceContextRollover = useCallback(() => {
    if (
      !voiceContextRolloverPendingRef.current ||
      sessionRestartPendingRef.current
    ) {
      return false;
    }

    const session = sessionRef.current;
    if (!session?.isConnected()) {
      return false;
    }

    const pendingPlaybackMs = playbackRef.current?.getPendingPlaybackMs() ?? 0;
    const hasPendingInput = latestInputTranscriptRef.current.trim().length > 0;
    const captureBusy =
      captureAudioRef.current && (wasTalkingRef.current || hasPendingInput);
    const hasPendingAnnouncements = pendingModelTurnsRef.current.length > 0;
    const hasActiveModelTurn = modelTurnActiveRef.current;
    const hasDelayedFallback =
      autoHandoffTimerRef.current !== null ||
      localFallbackTimerRef.current !== null ||
      silenceHandoffTimerRef.current !== null ||
      modelInterpretSubmitFallbackTimerRef.current !== null;

    if (
      pendingPlaybackMs > 0 ||
      localSpeechActiveRef.current ||
      hasActiveModelTurn ||
      hasPendingAnnouncements ||
      captureBusy ||
      hasDelayedFallback
    ) {
      if (!voiceContextRolloverTimerRef.current) {
        voiceDebugLog('voice_context.rollover_deferred', {
          reason: voiceContextRolloverReasonRef.current,
          pendingPlaybackMs,
          localSpeechActive: localSpeechActiveRef.current,
          hasActiveModelTurn,
          hasPendingAnnouncements,
          captureBusy,
          hasDelayedFallback,
        });
        voiceContextRolloverTimerRef.current = setTimeout(() => {
          voiceContextRolloverTimerRef.current = null;
          maybePerformVoiceContextRollover();
        }, VOICE_CONTEXT_ROLLOVER_RETRY_MS);
        voiceContextRolloverTimerRef.current.unref?.();
      }
      return false;
    }

    clearVoiceContextRolloverTimer();
    const summary = buildVoiceContextSummary();
    voiceContextSummaryRef.current = summary;
    voiceContextRolloverPendingRef.current = false;
    const reason = voiceContextRolloverReasonRef.current;
    voiceContextRolloverReasonRef.current = null;
    voiceDebugLog('voice_context.rollover_executed', {
      reason,
      summaryChars: summary.length,
      totalTokens: voiceContextTokenEstimateRef.current,
    });
    return requestSessionRestart(
      reason === 'context_sync' ? 'context_sync' : 'context_rollover',
    );
  }, [
    buildVoiceContextSummary,
    clearVoiceContextRolloverTimer,
    requestSessionRestart,
  ]);

  const scheduleVoiceContextRollover = useCallback(
    (reason: 'context_rollover' | 'context_sync') => {
      if (!enabled) {
        return;
      }
      voiceContextRolloverPendingRef.current = true;
      voiceContextRolloverReasonRef.current = reason;
      voiceDebugLog('voice_context.rollover_needed', {
        reason,
        totalTokens: voiceContextTokenEstimateRef.current,
        ...getVoiceContextBudget(),
      });
      maybePerformVoiceContextRollover();
    },
    [enabled, getVoiceContextBudget, maybePerformVoiceContextRollover],
  );

  const resetVoiceContextLedger = useCallback(
    (systemInstruction: string, modelOverride?: string | null) => {
      const baseTokens =
        estimateTokenCountSync([{ text: systemInstruction }]) +
        VOICE_ASSISTANT_TOOLS_TOKEN_ESTIMATE;
      voiceContextEntriesRef.current = [];
      voiceContextTokenEstimateRef.current = baseTokens;
      nextVoiceContextEntryIdRef.current = 1;
      currentVoiceModelRef.current =
        modelOverride ||
        currentVoiceModelRef.current ||
        resolvedRuntimeConfig.model ||
        null;
      clearVoiceContextRolloverTimer();
      voiceContextRolloverPendingRef.current = false;
      voiceContextRolloverReasonRef.current = null;
      lastVoiceContextUserKeyRef.current = '';
      lastVoiceContextUserAtRef.current = 0;
      voiceDebugLog('voice_context.ledger_reset', {
        baseTokens,
        hasSummary: Boolean(voiceContextSummaryRef.current),
        summaryChars: voiceContextSummaryRef.current.length,
        model: currentVoiceModelRef.current,
      });
    },
    [clearVoiceContextRolloverTimer, resolvedRuntimeConfig.model],
  );

  const recordVoiceContextUsage = useCallback(
    (params: {
      countText: string;
      reason: string;
      role?: VoiceSessionContextRole;
      summaryText?: string;
      includeInSummary?: boolean;
      dedupeKey?: string;
    }) => {
      const countText = params.countText.trim();
      if (!countText) {
        return;
      }

      const includeInSummary = params.includeInSummary ?? false;
      const summaryText = truncateVoiceContextText(
        params.summaryText ?? countText,
        VOICE_CONTEXT_ENTRY_MAX_CHARS,
      );
      const now = Date.now();
      if (
        includeInSummary &&
        params.role === 'user' &&
        params.dedupeKey &&
        params.dedupeKey === lastVoiceContextUserKeyRef.current &&
        now - lastVoiceContextUserAtRef.current < REQUEST_DEDUP_WINDOW_MS
      ) {
        voiceDebugLog('voice_context.entry_deduped', {
          reason: params.reason,
        });
        return;
      }

      const tokenEstimate = estimateTokenCountSync([{ text: countText }]);
      voiceContextTokenEstimateRef.current += tokenEstimate;

      if (includeInSummary && params.role && summaryText) {
        if (params.role === 'user' && params.dedupeKey) {
          lastVoiceContextUserKeyRef.current = params.dedupeKey;
          lastVoiceContextUserAtRef.current = now;
        }
        voiceContextEntriesRef.current = [
          ...voiceContextEntriesRef.current,
          {
            id: nextVoiceContextEntryIdRef.current++,
            role: params.role,
            text: summaryText,
            at: now,
          },
        ].slice(-VOICE_CONTEXT_MAX_LEDGER_ENTRIES);
      }

      const budget = getVoiceContextBudget();
      voiceDebugLog('voice_context.entry_recorded', {
        reason: params.reason,
        tokenEstimate,
        totalTokens: voiceContextTokenEstimateRef.current,
        thresholdTokens: budget.thresholdTokens,
        rolloverTriggerTokens: budget.rolloverTriggerTokens,
        includeInSummary,
      });
      if (
        voiceContextTokenEstimateRef.current >= budget.rolloverTriggerTokens
      ) {
        scheduleVoiceContextRollover('context_rollover');
      }
    },
    [getVoiceContextBudget, scheduleVoiceContextRollover],
  );

  const recordCommittedUserVoiceContext = useCallback(
    (transcript: string, reason: string) => {
      const normalized = normalizeTranscriptText(transcript);
      if (!normalized) {
        return;
      }
      recordVoiceContextUsage({
        countText: transcript,
        summaryText: transcript,
        role: 'user',
        includeInSummary: true,
        dedupeKey: normalized,
        reason,
      });
    },
    [recordVoiceContextUsage],
  );

  useEffect(() => {
    lastContextSyncMessageRef.current = contextSyncMessage;
    if (lastContextSyncGenerationRef.current === contextSyncGeneration) {
      return;
    }
    lastContextSyncGenerationRef.current = contextSyncGeneration;
    if (isVoiceContextClearEventMessage(contextSyncMessage)) {
      voiceContextSummaryRef.current = '';
      voiceContextEntriesRef.current = [];
      lastUserUtteranceRef.current = '';
      lastVoiceContextUserKeyRef.current = '';
      lastVoiceContextUserAtRef.current = 0;
    }
    voiceDebugLog('voice_context.sync_event', {
      generation: contextSyncGeneration,
      message: contextSyncMessage,
    });
    if (!enabled) {
      return;
    }
    if (sessionRef.current?.isConnected()) {
      scheduleVoiceContextRollover('context_sync');
      return;
    }
    const bufferedSummary = buildVoiceContextSummary();
    voiceContextSummaryRef.current = bufferedSummary;
    voiceDebugLog('voice_context.sync_event.buffered', {
      generation: contextSyncGeneration,
      summaryChars: bufferedSummary.length,
    });
  }, [
    buildVoiceContextSummary,
    contextSyncGeneration,
    contextSyncMessage,
    enabled,
    scheduleVoiceContextRollover,
  ]);

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
      if (pendingModelTurnFlushTimerRef.current) {
        clearTimeout(pendingModelTurnFlushTimerRef.current);
        pendingModelTurnFlushTimerRef.current = null;
      }
      clearVoiceContextRolloverTimer();
      captureAudioRef.current = false;
      outputLevelRef.current = 0;
      pendingModelTurnsRef.current = [];
      activeModelTurnRef.current = null;
      modelTurnActiveRef.current = false;
      resetOutputTranscriptOnNextChunkRef.current = false;
      latestOutputTranscriptRef.current = '';
      reconnectAttemptCountRef.current = 0;
      lastReconnectAttemptAtRef.current = 0;
      sessionRestartPendingRef.current = false;
      currentVoiceModelRef.current = resolvedRuntimeConfig.model ?? null;
      voiceContextEntriesRef.current = [];
      voiceContextSummaryRef.current = '';
      voiceContextTokenEstimateRef.current = 0;
      voiceContextRolloverPendingRef.current = false;
      voiceContextRolloverReasonRef.current = null;
      lastVoiceContextUserKeyRef.current = '';
      lastVoiceContextUserAtRef.current = 0;
      lastContextSyncGenerationRef.current = contextSyncGeneration;
      lastContextSyncMessageRef.current = contextSyncMessage;
      setState((prev) => ({
        ...prev,
        inputTranscript: '',
        outputTranscript: '',
        outputHistory: [],
        assistantSpeaking: false,
        outputLevel: 0,
      }));
    }
  }, [
    clearVoiceContextRolloverTimer,
    contextSyncGeneration,
    contextSyncMessage,
    enabled,
    resolvedRuntimeConfig.model,
  ]);

  const clearActiveModelTurn = useCallback(
    (
      reason: 'turn_complete' | 'close' | 'error' | 'cleanup' | 'disabled',
      extra?: Record<string, unknown>,
    ) => {
      const activeTurn = activeModelTurnRef.current;
      if (activeTurn) {
        voiceDebugLog('model_turn.end', {
          id: activeTurn.id,
          kind: activeTurn.kind,
          origin: activeTurn.origin,
          durationMs: Date.now() - activeTurn.startedAt,
          outputTranscriptChars: activeTurn.outputTranscriptChars,
          outputTranscriptChunks: activeTurn.outputTranscriptChunks,
          outputAudioChunks: activeTurn.outputAudioChunks,
          outputAudioBytes: activeTurn.outputAudioBytes,
          toolCallCount: activeTurn.toolCallCount,
          reason,
          ...extra,
        });
      }
      activeModelTurnRef.current = null;
      modelTurnActiveRef.current = false;
    },
    [],
  );

  const ensureAmbientModelTurn = useCallback(
    (source: 'output_transcript' | 'output_audio' | 'tool_call') => {
      const existingTurn = activeModelTurnRef.current;
      if (existingTurn) {
        return existingTurn;
      }

      const suppressedReason = ambientSuppressionReasonRef.current || undefined;
      const nextTurn: ActiveModelTurnState = {
        id: nextModelTurnIdRef.current++,
        kind: 'assistant',
        origin: 'ambient',
        startedAt: Date.now(),
        suppressedReason,
        outputTranscriptChars: 0,
        outputTranscriptChunks: 0,
        outputAudioChunks: 0,
        outputAudioBytes: 0,
        toolCallCount: 0,
      };
      activeModelTurnRef.current = nextTurn;
      modelTurnActiveRef.current = true;
      voiceDebugLog('model_turn.start', {
        id: nextTurn.id,
        kind: nextTurn.kind,
        origin: nextTurn.origin,
        source,
        suppressedReason: suppressedReason ?? null,
      });
      return nextTurn;
    },
    [],
  );

  const flushPendingModelTurns = useCallback(() => {
    if (modelTurnActiveRef.current) {
      return false;
    }

    if (pendingModelTurnFlushTimerRef.current) {
      clearTimeout(pendingModelTurnFlushTimerRef.current);
      pendingModelTurnFlushTimerRef.current = null;
    }

    const nextTurn = pendingModelTurnsRef.current[0];
    if (!nextTurn || !sessionRef.current?.isConnected()) {
      maybePerformVoiceContextRollover();
      return false;
    }

    const pendingPlaybackMs = playbackRef.current?.getPendingPlaybackMs() ?? 0;
    if (pendingPlaybackMs > 0 || localSpeechActiveRef.current) {
      const waitMs =
        Math.max(
          pendingPlaybackMs,
          localSpeechActiveRef.current ? LOCAL_SPEECH_OUTPUT_MUTE_BUFFER_MS : 0,
        ) + OUTPUT_PLAYBACK_DRAIN_GUARD_MS;
      voiceDebugLog('model_turn.flush_deferred', {
        id: nextTurn.id,
        kind: nextTurn.kind,
        queueDepth: pendingModelTurnsRef.current.length,
        pendingPlaybackMs,
        localSpeechActive: localSpeechActiveRef.current,
        waitMs,
      });
      pendingModelTurnFlushTimerRef.current = setTimeout(() => {
        pendingModelTurnFlushTimerRef.current = null;
        flushPendingModelTurns();
      }, waitMs);
      pendingModelTurnFlushTimerRef.current.unref?.();
      return false;
    }

    pendingModelTurnsRef.current.shift();
    const activeTurn: ActiveModelTurnState = {
      id: nextTurn.id,
      kind: nextTurn.kind,
      origin: 'queued',
      startedAt: Date.now(),
      outputTranscriptChars: 0,
      outputTranscriptChunks: 0,
      outputAudioChunks: 0,
      outputAudioBytes: 0,
      toolCallCount: 0,
      preview: nextTurn.preview,
    };
    activeModelTurnRef.current = activeTurn;
    modelTurnActiveRef.current = true;
    muteModelOutputUntilRef.current = 0;
    voiceDebugLog('model_turn.start', {
      id: activeTurn.id,
      kind: activeTurn.kind,
      origin: activeTurn.origin,
      queuedForMs: Date.now() - nextTurn.enqueuedAt,
      queueDepth: pendingModelTurnsRef.current.length,
      preview: nextTurn.preview,
    });
    recordVoiceContextUsage({
      countText: nextTurn.text,
      reason: `model_turn.prompt.${nextTurn.kind}`,
    });
    sessionRef.current.sendTextTurn(nextTurn.text);
    return true;
  }, [maybePerformVoiceContextRollover, recordVoiceContextUsage]);

  const enqueueModelTurn = useCallback(
    (turn: QueuedModelTurnRequest) => {
      if (!sessionRef.current?.isConnected()) {
        return false;
      }

      const queuedTurn: QueuedModelTurn = {
        ...turn,
        id: nextModelTurnIdRef.current++,
        enqueuedAt: Date.now(),
        preview: turn.text.replace(/\s+/g, ' ').trim().slice(0, 160),
      };

      if (queuedTurn.kind === 'instruction') {
        const firstNotificationIndex = pendingModelTurnsRef.current.findIndex(
          (queuedTurn) => queuedTurn.kind === 'notification',
        );
        if (firstNotificationIndex === -1) {
          pendingModelTurnsRef.current.push(queuedTurn);
        } else {
          pendingModelTurnsRef.current.splice(
            firstNotificationIndex,
            0,
            queuedTurn,
          );
        }
      } else {
        pendingModelTurnsRef.current.push(queuedTurn);
      }

      voiceDebugLog('model_turn.queued', {
        id: queuedTurn.id,
        kind: queuedTurn.kind,
        queueDepth: pendingModelTurnsRef.current.length,
        preview: queuedTurn.preview,
      });

      flushPendingModelTurns();
      return true;
    },
    [flushPendingModelTurns],
  );

  const speak = useCallback(
    (text: string) => {
      const prompt = sanitizeAnnouncementText(text);
      if (!prompt) {
        return false;
      }

      const now = Date.now();
      const promptNormalized = normalizeTranscriptText(prompt);
      if (
        promptNormalized.includes('approval required') ||
        (promptNormalized.includes('allow') &&
          promptNormalized.includes('deny'))
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

      const speakViaModel = () =>
        enqueueModelTurn({
          kind: 'notification',
          text: [
            `Notification: ${prompt}`,
            'Speak this notification to the user in one or two short, warm, conversational sentences.',
            'This is a controller notification, not a new user request.',
            'Keep the original meaning, requested action, and any listed options intact.',
            'Do not call tools, revisit approvals, continue prior reasoning, or ask follow-up questions.',
            'Do not invent extra status updates, promises, or tool actions.',
            'If paths appear, speak key path segments naturally instead of slash-by-slash.',
          ].join('\n'),
        });

      if (!ENABLE_LOCAL_SYSTEM_ANNOUNCEMENTS) {
        return speakViaModel();
      }

      if (speakViaModel()) {
        return true;
      }

      localSpeechQueueRef.current = localSpeechQueueRef.current.then(
        async () => {
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
        },
      );

      return true;
    },
    [enqueueModelTurn],
  );

  const commitCompletedOutput = useCallback((text: string) => {
    latestOutputTranscriptRef.current = '';
    setState((prev) => {
      const nextOutputHistory = appendVoiceOutputHistory(
        prev.outputHistory,
        text,
        getLatestHistoryIdRef.current(),
      );
      if (!prev.outputTranscript && nextOutputHistory === prev.outputHistory) {
        return prev;
      }
      return {
        ...prev,
        outputTranscript: '',
        outputHistory: nextOutputHistory,
      };
    });
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
      if (pendingModelTurnFlushTimerRef.current) {
        clearTimeout(pendingModelTurnFlushTimerRef.current);
        pendingModelTurnFlushTimerRef.current = null;
      }
      ambientSuppressionReasonRef.current = null;
      localSpeechActiveRef.current = false;
      localSpeechInputGuardUntilRef.current = 0;
      pendingModelTurnsRef.current = [];
      recentInstructionOutputSegmentsRef.current = [];
      clearActiveModelTurn('disabled');
      outputLevelRef.current = 0;
      resetOutputTranscriptOnNextChunkRef.current = false;
      reconnectAttemptCountRef.current = 0;
      lastReconnectAttemptAtRef.current = 0;
      setState(INITIAL_STATE);
      return;
    }

    voiceDebugLog('controller.enabled');
    let active = true;
    const session = new LiveVoiceSession(config);
    const playback = new AudioPlayback();
    const sessionSystemInstruction = appendVoiceSessionCarryoverInstruction(
      voiceAssistantSystemInstruction,
      voiceContextSummaryRef.current,
    );
    resetVoiceContextLedger(
      sessionSystemInstruction,
      resolvedRuntimeConfig.model ?? currentVoiceModelRef.current,
    );
    sessionRef.current = session;
    playbackRef.current = playback;
    wasTalkingRef.current = false;
    speechStartAtRef.current = 0;
    lastTalkingAtRef.current = 0;
    lastTurnEndSignalAtRef.current = 0;
    lastInputAtRef.current = 0;
    latestInputTranscriptRef.current = '';
    latestDisplayInputTranscriptRef.current = '';
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
    ambientSuppressionReasonRef.current = null;
    recentInstructionOutputSegmentsRef.current = [];
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
    if (pendingModelTurnFlushTimerRef.current) {
      clearTimeout(pendingModelTurnFlushTimerRef.current);
      pendingModelTurnFlushTimerRef.current = null;
    }
    localSpeechActiveRef.current = false;
    localSpeechInputGuardUntilRef.current = 0;
    clearActiveModelTurn('cleanup');
    outputLevelRef.current = 0;
    resetOutputTranscriptOnNextChunkRef.current = false;
    latestOutputTranscriptRef.current = '';

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
      sessionRestartPendingRef.current = false;
      activeModelTurnRef.current = null;
      modelTurnActiveRef.current = false;
      setState((prev) => ({
        ...prev,
        connectionState: 'connected',
      }));
      flushPendingModelTurns();
      maybePerformVoiceContextRollover();
    };

    const onClose = () => {
      if (!active) {
        return;
      }
      if (pendingModelTurnFlushTimerRef.current) {
        clearTimeout(pendingModelTurnFlushTimerRef.current);
        pendingModelTurnFlushTimerRef.current = null;
      }
      clearActiveModelTurn('close');
      resetOutputTranscriptOnNextChunkRef.current = true;
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
        pendingModelTurnsRef.current = [];
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
      const modelConnected = sessionRef.current?.isConnected() ?? false;
      const spoken = speak(trimmed);
      if (!modelConnected || !spoken) {
        commitCompletedOutput(trimmed);
      }
    };

    const setAmbientSuppressionReason = (
      reason: string | null,
      transcript: string,
    ) => {
      if (ambientSuppressionReasonRef.current === reason) {
        return;
      }
      ambientSuppressionReasonRef.current = reason;
      voiceDebugLog(
        reason
          ? 'ambient_output.suppression_enabled'
          : 'ambient_output.suppression_cleared',
        {
          reason,
          transcript: transcript.trim() || undefined,
        },
      );
    };

    const clearModelInterpretSubmitFallback = () => {
      if (modelInterpretSubmitFallbackTimerRef.current) {
        clearTimeout(modelInterpretSubmitFallbackTimerRef.current);
        modelInterpretSubmitFallbackTimerRef.current = null;
      }
      recentInstructionOutputSegmentsRef.current = [];
    };

    const consumeInputTranscript = () => {
      latestInputTranscriptRef.current = '';
      latestDisplayInputTranscriptRef.current = '';
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
      recordCommittedUserVoiceContext(transcript, 'direct_decision');
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
      const hasPendingActions = getPendingActionsRef.current().length > 0;
      const recentApprovalPrompt =
        Date.now() - lastApprovalPromptAtRef.current <=
        APPROVAL_REPLY_WINDOW_MS;
      const likelyApprovalReply = isLikelyGenericConfirmationReply(transcript);
      return hasPendingActions || (recentApprovalPrompt && likelyApprovalReply);
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
      recentInstructionOutputSegmentsRef.current = [];
      voiceDebugLog(`${source}.model_interpret`, {
        transcript,
        hasPendingActions,
        recentApprovalPrompt,
        likelyApprovalReply,
      });
      recordCommittedUserVoiceContext(transcript, `${source}.model_interpret`);
      const instructionLines = [
        'Internal control turn. Do not address the user directly.',
        'Do not generate spoken natural-language output. Prefer tool calls as soon as intent is clear.',
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
        `Payload: ${payload}`,
      ];
      enqueueModelTurn({
        kind: 'instruction',
        text: instructionLines.join('\n'),
      });
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
            recentInstructionOutputSegmentsRef.current,
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
            if (!isRedundantSubmissionAck(message)) {
              announceOutput(message, POST_TOOL_MODEL_MUTE_MS);
            }
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
      const suppressDisplayTranscript = shouldSuppressNonLatinTranscriptDisplay(
        latestInputTranscriptRef.current,
        resolvedRuntimeConfig.inputTranscriptionLanguageCode,
      );
      if (!suppressDisplayTranscript) {
        latestDisplayInputTranscriptRef.current =
          latestInputTranscriptRef.current;
      }

      const displayTranscript = latestDisplayInputTranscriptRef.current;
      setState((prev) =>
        prev.inputTranscript === displayTranscript
          ? prev
          : {
              ...prev,
              inputTranscript: displayTranscript,
            },
      );

      const recentApprovalPrompt =
        now - lastApprovalPromptAtRef.current <= APPROVAL_REPLY_WINDOW_MS;
      setAmbientSuppressionReason(
        getAmbientAssistantSuppressionReason(
          latestInputTranscriptRef.current,
          pendingActionsAtInput.length > 0,
          recentApprovalPrompt,
        ),
        latestInputTranscriptRef.current,
      );

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
        if (!shouldDirectlySubmitVoiceRequest(transcript)) {
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
        recordCommittedUserVoiceContext(transcript, 'auto_handoff.submit');
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
            if (!isRedundantSubmissionAck(message)) {
              announceOutput(message, POST_TOOL_MODEL_MUTE_MS);
            }
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
        recordCommittedUserVoiceContext(transcript, 'local_fallback.submit');
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
            if (!isRedundantSubmissionAck(message)) {
              announceOutput(message, POST_TOOL_MODEL_MUTE_MS);
            }
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
      const activeTurn = ensureAmbientModelTurn('output_transcript');
      const now = Date.now();
      if (activeTurn.kind === 'instruction') {
        const instructionSegments = pruneTranscriptSegments(
          recentInstructionOutputSegmentsRef.current,
          now,
          OUTPUT_ECHO_WINDOW_MS,
        );
        instructionSegments.push({
          at: now,
          text,
        });
        recentInstructionOutputSegmentsRef.current = instructionSegments;
        activeTurn.outputTranscriptChunks += 1;
        activeTurn.outputTranscriptChars += text.length;
        voiceDebugLog('model_turn.transcript_suppressed', {
          id: activeTurn.id,
          kind: activeTurn.kind,
          length: text.length,
          reason: 'instruction',
          transcriptChunks: activeTurn.outputTranscriptChunks,
          transcriptLength: buildCandidateUtterance(
            instructionSegments,
            OUTPUT_ECHO_WINDOW_MS,
            now,
          ).length,
        });
        return;
      }
      if (activeTurn.suppressedReason) {
        activeTurn.outputTranscriptChunks += 1;
        activeTurn.outputTranscriptChars += text.length;
        voiceDebugLog('model_turn.transcript_suppressed', {
          id: activeTurn.id,
          kind: activeTurn.kind,
          length: text.length,
          reason: activeTurn.suppressedReason,
          transcriptChunks: activeTurn.outputTranscriptChunks,
        });
        return;
      }
      if (now < muteModelOutputUntilRef.current) {
        voiceDebugLog('model_turn.transcript_suppressed', {
          id: activeTurn.id,
          kind: activeTurn.kind,
          length: text.length,
          muteRemainingMs: muteModelOutputUntilRef.current - now,
        });
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
        recentOutputSegmentsRef.current = prunedOutputs;
        voiceDebugLog('output.meta_suppressed', {
          kind: activeTurn.kind,
          text: candidateOutput || text,
        });
        return;
      }
      recentOutputSegmentsRef.current = candidateOutputs;
      const previousOutputAt = lastOutputAtRef.current;
      const outputGapMs = now - previousOutputAt;
      const resetForCompletedTurn = resetOutputTranscriptOnNextChunkRef.current;
      if (resetForCompletedTurn) {
        resetOutputTranscriptOnNextChunkRef.current = false;
      }
      const nextOutputTranscript = appendOutputTranscriptChunk(
        resetForCompletedTurn ||
          outputGapMs > OUTPUT_TRANSCRIPT_HARD_RESET_GAP_MS
          ? ''
          : latestOutputTranscriptRef.current,
        text,
        outputGapMs > OUTPUT_TRANSCRIPT_RESET_GAP_MS,
        OUTPUT_TRANSCRIPT_MAX_LINES,
      );
      latestOutputTranscriptRef.current = nextOutputTranscript;
      activeTurn.outputTranscriptChunks += 1;
      activeTurn.outputTranscriptChars += text.length;
      voiceDebugLog('model_turn.transcript_chunk', {
        id: activeTurn.id,
        kind: activeTurn.kind,
        chunkLength: text.length,
        outputGapMs,
        resetForCompletedTurn,
        transcriptLength: nextOutputTranscript.length,
        transcriptChunks: activeTurn.outputTranscriptChunks,
      });
      setLastOutputAt(now);
      lastOutputAtRef.current = now;
      setState((prev) => {
        if (prev.outputTranscript === nextOutputTranscript) {
          return prev;
        }
        return {
          ...prev,
          outputTranscript: nextOutputTranscript,
        };
      });
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
      const activeTurn = ensureAmbientModelTurn('output_audio');
      if (activeTurn.kind === 'instruction') {
        activeTurn.outputAudioChunks += 1;
        activeTurn.outputAudioBytes += chunk.length;
        if (
          activeTurn.outputAudioChunks === 1 ||
          activeTurn.outputAudioChunks % 10 === 0
        ) {
          voiceDebugLog('model_turn.audio_suppressed', {
            id: activeTurn.id,
            kind: activeTurn.kind,
            chunkBytes: chunk.length,
            reason: 'instruction',
            outputAudioChunks: activeTurn.outputAudioChunks,
            outputAudioBytes: activeTurn.outputAudioBytes,
          });
        }
        return;
      }
      if (activeTurn.suppressedReason) {
        activeTurn.outputAudioChunks += 1;
        activeTurn.outputAudioBytes += chunk.length;
        if (
          activeTurn.outputAudioChunks === 1 ||
          activeTurn.outputAudioChunks % 10 === 0
        ) {
          voiceDebugLog('model_turn.audio_suppressed', {
            id: activeTurn.id,
            kind: activeTurn.kind,
            chunkBytes: chunk.length,
            reason: activeTurn.suppressedReason,
            outputAudioChunks: activeTurn.outputAudioChunks,
            outputAudioBytes: activeTurn.outputAudioBytes,
          });
        }
        return;
      }
      if (Date.now() < muteModelOutputUntilRef.current) {
        voiceDebugLog('model_turn.audio_suppressed', {
          id: activeTurn.id,
          kind: activeTurn.kind,
          chunkBytes: chunk.length,
          muteRemainingMs: muteModelOutputUntilRef.current - Date.now(),
        });
        return;
      }
      activeTurn.outputAudioChunks += 1;
      activeTurn.outputAudioBytes += chunk.length;
      if (
        activeTurn.outputAudioChunks === 1 ||
        activeTurn.outputAudioChunks % 10 === 0
      ) {
        voiceDebugLog('model_turn.audio_chunk', {
          id: activeTurn.id,
          kind: activeTurn.kind,
          chunkBytes: chunk.length,
          outputAudioChunks: activeTurn.outputAudioChunks,
          outputAudioBytes: activeTurn.outputAudioBytes,
          pendingPlaybackMs: playback.getPendingPlaybackMs(),
        });
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
      clearActiveModelTurn('error', {
        message: error.message,
      });
      setState((prev) => ({
        ...prev,
        error: error.message,
        connectionState: 'idle',
      }));
    };

    const onTurnComplete = (_reason: string | null) => {
      if (!active) {
        return;
      }
      const activeTurn = activeModelTurnRef.current;
      const now = Date.now();
      const completedTranscript = activeTurn?.suppressedReason
        ? ''
        : latestOutputTranscriptRef.current.trim();
      const instructionTranscript =
        activeTurn?.kind === 'instruction'
          ? buildCandidateUtterance(
              recentInstructionOutputSegmentsRef.current,
              OUTPUT_ECHO_WINDOW_MS,
              now,
            ).trim()
          : '';
      if (activeTurn?.kind === 'assistant' && completedTranscript) {
        recordVoiceContextUsage({
          countText: completedTranscript,
          summaryText: completedTranscript,
          role: 'assistant',
          includeInSummary: true,
          reason: 'assistant.turn_complete',
        });
      } else if (completedTranscript) {
        recordVoiceContextUsage({
          countText: completedTranscript,
          reason: `${activeTurn?.kind ?? 'unknown'}.turn_complete`,
        });
      } else if (instructionTranscript) {
        recordVoiceContextUsage({
          countText: instructionTranscript,
          reason: 'instruction.turn_complete',
        });
      }
      if (completedTranscript) {
        commitCompletedOutput(completedTranscript);
      }
      clearActiveModelTurn('turn_complete', {
        pendingPlaybackMs: playbackRef.current?.getPendingPlaybackMs() ?? 0,
        committedTranscriptLength: completedTranscript.length,
      });
      resetOutputTranscriptOnNextChunkRef.current = true;
      flushPendingModelTurns();
      maybePerformVoiceContextRollover();
    };

    const onToolCall = (functionCalls: FunctionCall[]) => {
      const activeTurn = ensureAmbientModelTurn('output_transcript');
      activeTurn.toolCallCount += functionCalls.length;
      if (functionCalls.length > 0) {
        recordVoiceContextUsage({
          countText: serializeVoiceContextPayload(
            functionCalls.map((call) => ({
              name: call.name || 'unknown_tool',
              args: call.args ?? null,
            })),
          ),
          reason: 'tool_call.received',
        });
      }
      voiceDebugLog('model_turn.tool_call', {
        id: activeTurn.id,
        kind: activeTurn.kind,
        count: functionCalls.length,
        names: functionCalls.map((call) => call.name || 'unknown'),
      });
      if (activeTurn.kind === 'notification') {
        voiceDebugLog('notification.tool_call_ignored', {
          id: activeTurn.id,
          count: functionCalls.length,
          names: functionCalls.map((call) => call.name || 'unknown'),
        });
        if (functionCalls.length > 0) {
          const ignoredResponses = functionCalls.map((call) => ({
            id: call.id,
            name: call.name || 'unknown_tool',
            response: {
              ok: false,
              ignored: true,
              message: 'Ignored unexpected tool call from a notification turn.',
            },
          }));
          recordVoiceContextUsage({
            countText: serializeVoiceContextPayload(ignoredResponses),
            reason: 'tool_response.ignored',
          });
          session.sendToolResponses(ignoredResponses);
        }
        return;
      }
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
            if (argText) {
              latestDisplayInputTranscriptRef.current = argText;
              setState((prev) =>
                prev.inputTranscript === argText
                  ? prev
                  : {
                      ...prev,
                      inputTranscript: argText,
                    },
              );
            }
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
            if (intendedText) {
              recordCommittedUserVoiceContext(
                intendedText,
                'tool_call.submit_user_request',
              );
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
            const shouldSuppressAnnouncement =
              (response.name === 'submit_user_request' ||
                (response.name === 'query_local_context' &&
                  responseBody['delegated'] === true)) &&
              isRedundantSubmissionAck(responseMessage);
            if (!shouldSuppressAnnouncement) {
              announceOutput(responseMessage, POST_TOOL_MODEL_MUTE_MS);
            }
          }
        }
        if (responses.length > 0) {
          recordVoiceContextUsage({
            countText: serializeVoiceContextPayload(responses),
            reason: 'tool_response.sent',
          });
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
    session.on('turnComplete', onTurnComplete);
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
      enqueueModelTurn({
        kind: 'instruction',
        text: `User said: ${transcript}`,
      });
    };

    const useClientSilenceTurnDetection =
      resolvedRuntimeConfig.forceTurnEndOnSilence &&
      !resolvedRuntimeConfig.advancedVad;

    const unsubscribeAudioState = audioEngine.subscribe((audioState) => {
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

      if (!useClientSilenceTurnDetection) {
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
          if (!shouldDirectlySubmitVoiceRequest(transcript)) {
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
          recordCommittedUserVoiceContext(transcript, 'silence_handoff.submit');
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
              if (!isRedundantSubmissionAck(message)) {
                announceOutput(message, POST_TOOL_MODEL_MUTE_MS);
              }
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
          voiceName: selectedVoicePersona?.name,
          inputTranscriptionLanguageCode:
            resolvedRuntimeConfig.inputTranscriptionLanguageCode,
          advancedVad: resolvedRuntimeConfig.advancedVad,
          serverSilenceMs: resolvedRuntimeConfig.serverSilenceMs,
          setupWaitMs: resolvedRuntimeConfig.setupWaitMs,
          tools: VOICE_ASSISTANT_TOOLS,
          systemInstruction: sessionSystemInstruction,
        });
        if (!active) {
          return;
        }
        currentVoiceModelRef.current = model;
        sessionRestartPendingRef.current = false;
        voiceDebugLog('controller.connection.ready', {
          model,
          persona: selectedVoicePersona?.name || null,
        });
        setState((prev) => ({
          ...prev,
          connectionState: 'connected',
          model,
        }));
      } catch (error) {
        if (!active) {
          return;
        }
        sessionRestartPendingRef.current = false;
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
      if (enabled && !sessionRestartPendingRef.current) {
        const carryoverSummary = buildVoiceContextSummary();
        if (carryoverSummary) {
          voiceContextSummaryRef.current = carryoverSummary;
          voiceDebugLog('voice_context.carryover_refreshed', {
            summaryChars: carryoverSummary.length,
          });
        }
      }
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
      if (pendingModelTurnFlushTimerRef.current) {
        clearTimeout(pendingModelTurnFlushTimerRef.current);
        pendingModelTurnFlushTimerRef.current = null;
      }
      ambientSuppressionReasonRef.current = null;
      localSpeechActiveRef.current = false;
      localSpeechInputGuardUntilRef.current = 0;
      outputLevelRef.current = 0;
      recentInstructionOutputSegmentsRef.current = [];
      clearActiveModelTurn('cleanup');
      unsubscribePcm();
      unsubscribeAudioState();
      session.off('open', onOpen);
      session.off('close', onClose);
      session.off('turnComplete', onTurnComplete);
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
  }, [
    clearActiveModelTurn,
    commitCompletedOutput,
    config,
    enabled,
    enqueueModelTurn,
    ensureAmbientModelTurn,
    flushPendingModelTurns,
    maybePerformVoiceContextRollover,
    buildVoiceContextSummary,
    recordCommittedUserVoiceContext,
    recordVoiceContextUsage,
    resolvedRuntimeConfig,
    resetVoiceContextLedger,
    selectedVoicePersona,
    sessionRestartNonce,
    speak,
    voiceAssistantSystemInstruction,
  ]);

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
          shouldDirectlySubmitVoiceRequest(fallbackRequestText)
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
