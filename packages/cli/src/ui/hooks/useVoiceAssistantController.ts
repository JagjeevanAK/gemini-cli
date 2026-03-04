/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { execFile } from 'node:child_process';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { promisify } from 'node:util';
import { useCallback, useEffect, useRef, useState } from 'react';
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

interface VoiceAssistantControllerParams {
  config: Config | undefined;
  enabled: boolean;
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
  model: null,
  error: null,
  playbackWarning: null,
};

const FORCE_TURN_END_ON_SILENCE =
  process.env['GEMINI_CLI_VOICE_FORCE_TURN_END'] !== '0';
const TURN_END_SILENCE_MS = Math.max(
  300,
  Number(process.env['GEMINI_CLI_VOICE_TURN_END_SILENCE_MS'] || '1600'),
);
const MAX_SPEECH_SEGMENT_MS = Math.max(
  1200,
  Number(process.env['GEMINI_CLI_VOICE_MAX_SPEECH_MS'] || '9000'),
);
// Temporarily disabled while isolating live-voice response behavior.
// const TRANSCRIPT_TURN_FALLBACK =
//   process.env['GEMINI_CLI_VOICE_TRANSCRIPT_FALLBACK'] !== '0';
const TRANSCRIPT_TURN_FALLBACK = false;
const TRANSCRIPT_TURN_COOLDOWN_MS = Math.max(
  1200,
  Number(process.env['GEMINI_CLI_VOICE_TRANSCRIPT_COOLDOWN_MS'] || '4000'),
);
// Temporarily disabled while isolating live-voice response behavior.
// const LOCAL_ASSISTANT_FALLBACK_ENABLED =
//   process.env['GEMINI_CLI_VOICE_LOCAL_FALLBACK'] !== '0';
const LOCAL_ASSISTANT_FALLBACK_ENABLED = false;
const LOCAL_ASSISTANT_FALLBACK_MS = Math.max(
  1500,
  Number(process.env['GEMINI_CLI_VOICE_LOCAL_FALLBACK_MS'] || '3200'),
);
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

const VOICE_ASSISTANT_SYSTEM_INSTRUCTION = `
You are Gemini CLI Voice Assistant, a concise personal assistant controlling a coding agent.
Hard rules:
- Respond in plain text using one short sentence unless you need one clarification question.
- Never narrate internal reasoning, tool usage, or phrases like "initiating analysis".
- Do not use markdown headings or status banners.
- If the transcript sounds partial or ambiguous, ask for a full one-sentence request before submitting it.
- For factual workspace/repo/file questions, call query_local_context first and answer directly.
- Only delegate with submit_user_request when the user clearly asks the coding agent to do work.
- Use submit_user_request only for clear, complete requests intended for the coding agent.
- For status, approvals, permissions, or control, use tools instead of guessing.
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
    return true;
  }

  // Short utterances without clear question/command lead are often partial.
  return words.length <= 4;
}

function normalizeTranscriptText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s']/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
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
  onDisableRequested,
  getRuntimeStatus,
  getPendingActions,
  submitUserRequest,
  submitUserHint,
  resolvePendingAction,
  cancelCurrentRun,
}: VoiceAssistantControllerParams) {
  const [state, setState] =
    useState<VoiceAssistantControllerState>(INITIAL_STATE);
  const sessionRef = useRef<LiveVoiceSession | null>(null);
  const playbackRef = useRef<AudioPlayback | null>(null);
  const wasTalkingRef = useRef(false);
  const speechStartAtRef = useRef(0);
  const lastTalkingAtRef = useRef(0);
  const lastTurnEndSignalAtRef = useRef(0);
  const latestInputTranscriptRef = useRef('');
  const lastOutputAtRef = useRef(0);
  const lastToolCallAtRef = useRef(0);
  const lastTranscriptFallbackAtRef = useRef(0);
  const lastLocalFallbackTranscriptRef = useRef('');
  const lastDirectDecisionKeyRef = useRef('');
  const localFallbackTimerRef = useRef<NodeJS.Timeout | null>(null);
  const [lastOutputAt, setLastOutputAt] = useState<number>(0);

  const speak = useCallback((text: string) => {
    const prompt = text.trim();
    if (!prompt || !sessionRef.current?.isConnected()) {
      return false;
    }
    sessionRef.current.sendTextTurn(
      `Notification: ${prompt}\nReply in one short sentence.`,
    );
    return true;
  }, []);

  useEffect(() => {
    if (!enabled || !config) {
      voiceDebugLog('controller.disabled');
      sessionRef.current?.close();
      sessionRef.current = null;
      playbackRef.current?.stop();
      playbackRef.current = null;
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
    latestInputTranscriptRef.current = '';
    lastOutputAtRef.current = 0;
    lastToolCallAtRef.current = 0;
    lastTranscriptFallbackAtRef.current = 0;
    lastLocalFallbackTranscriptRef.current = '';
    lastDirectDecisionKeyRef.current = '';
    if (localFallbackTimerRef.current) {
      clearTimeout(localFallbackTimerRef.current);
      localFallbackTimerRef.current = null;
    }

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
      setState((prev) => ({
        ...prev,
        connectionState: 'connected',
      }));
    };

    const onInputTranscript = (text: string) => {
      if (!active || !text.trim()) {
        return;
      }
      latestInputTranscriptRef.current = text.trim();
      setState((prev) => ({
        ...prev,
        inputTranscript: text,
      }));

      const decisionIntent = detectSimpleDecision(
        latestInputTranscriptRef.current,
      );
      if (decisionIntent) {
        const pendingActions = getPendingActions();
        if (pendingActions.length > 0) {
          const pending = pendingActions[0];
          const mappedDecision = pickDecisionForIntent(
            pending.allowedDecisions,
            decisionIntent,
          );
          if (mappedDecision) {
            const dedupeKey = `${pending.id}:${mappedDecision}:${normalizeTranscriptText(
              latestInputTranscriptRef.current,
            )}`;
            if (lastDirectDecisionKeyRef.current !== dedupeKey) {
              lastDirectDecisionKeyRef.current = dedupeKey;
              void resolvePendingAction({
                actionId: pending.id,
                decision: mappedDecision,
              })
                .then((message) => {
                  if (!active) {
                    return;
                  }
                  const at = Date.now();
                  setLastOutputAt(at);
                  lastOutputAtRef.current = at;
                  setState((prev) => ({
                    ...prev,
                    outputTranscript: message,
                  }));
                })
                .catch((error) => {
                  voiceDebugLog('controller.direct_decision.failed', {
                    message: getErrorMessage(error),
                  });
                });
            }
            return;
          }
        }
      }

      if (!LOCAL_ASSISTANT_FALLBACK_ENABLED) {
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

        // Avoid repeated local fallback for the exact same transcript text.
        if (transcript === lastLocalFallbackTranscriptRef.current) {
          return;
        }

        const now = Date.now();
        const recentModelOutput =
          now - lastOutputAtRef.current < LOCAL_ASSISTANT_FALLBACK_MS / 2;
        const recentToolCall =
          now - lastToolCallAtRef.current < LOCAL_ASSISTANT_FALLBACK_MS / 2;
        if (recentModelOutput || recentToolCall) {
          return;
        }

        lastLocalFallbackTranscriptRef.current = transcript;
        voiceDebugLog('local_fallback.submit_request', {
          transcript,
        });
        void submitUserRequest(transcript)
          .then((result) => {
            if (!active) {
              return;
            }
            const message =
              (typeof result === 'string' && result.trim()) ||
              'Submitted your request.';
            const at = Date.now();
            setLastOutputAt(at);
            lastOutputAtRef.current = at;
            setState((prev) => ({
              ...prev,
              outputTranscript: message,
            }));
          })
          .catch((error) => {
            voiceDebugLog('local_fallback.submit_request.failed', {
              message: getErrorMessage(error),
            });
          });
      }, LOCAL_ASSISTANT_FALLBACK_MS);
    };

    const onOutputTranscript = (text: string) => {
      if (!active || !text.trim()) {
        return;
      }
      setLastOutputAt(Date.now());
      lastOutputAtRef.current = Date.now();
      setState((prev) => ({
        ...prev,
        outputTranscript: text,
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
      playback.playChunk(chunk);
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
      void (async () => {
        const responses: FunctionResponse[] = [];
        voiceDebugLog('controller.tool_call.received', {
          count: functionCalls.length,
          names: functionCalls.map((call) => call.name || 'unknown'),
        });
        for (const call of functionCalls) {
          responses.push(
            await handleAssistantToolCall({
              call,
              getRuntimeStatus,
              getPendingActions,
              submitUserRequest,
              submitUserHint,
              resolvePendingAction,
              cancelCurrentRun,
              onDisableRequested,
            }),
          );
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
    session.on('inputTranscript', onInputTranscript);
    session.on('outputTranscript', onOutputTranscript);
    session.on('outputAudioChunk', onOutputAudioChunk);
    session.on('toolCall', onToolCall);
    session.on('error', onError);

    const unsubscribePcm = audioEngine.subscribePcm((pcmBytes) => {
      session.sendAudioChunk(pcmBytes);
    });

    const maybeSendTranscriptFallback = (
      trigger: 'silence_commit' | 'max_speech_commit',
    ) => {
      if (!TRANSCRIPT_TURN_FALLBACK) {
        return;
      }
      const transcript = latestInputTranscriptRef.current.trim();
      if (!transcript) {
        return;
      }
      const now = Date.now();
      const sinceLastFallback = now - lastTranscriptFallbackAtRef.current;
      if (sinceLastFallback < TRANSCRIPT_TURN_COOLDOWN_MS) {
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
      if (!FORCE_TURN_END_ON_SILENCE) {
        return;
      }

      const now = Date.now();
      if (audioState.isTalking) {
        if (!wasTalkingRef.current) {
          voiceDebugLog('turn.speech_start');
          speechStartAtRef.current = now;
        }
        wasTalkingRef.current = true;
        lastTalkingAtRef.current = now;

        const speechMs = now - speechStartAtRef.current;
        const sinceLastSignalMs = now - lastTurnEndSignalAtRef.current;
        if (speechMs >= MAX_SPEECH_SEGMENT_MS && sinceLastSignalMs >= 1000) {
          lastTurnEndSignalAtRef.current = now;
          speechStartAtRef.current = now;
          voiceDebugLog('turn.max_speech_commit', {
            speechMs,
            thresholdMs: MAX_SPEECH_SEGMENT_MS,
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
      if (silenceMs >= TURN_END_SILENCE_MS && sinceLastSignalMs >= 500) {
        lastTurnEndSignalAtRef.current = now;
        wasTalkingRef.current = false;
        speechStartAtRef.current = 0;
        voiceDebugLog('turn.silence_commit', {
          silenceMs,
          thresholdMs: TURN_END_SILENCE_MS,
        });
        session.sendAudioStreamEnd('silence_commit');
        maybeSendTranscriptFallback('silence_commit');
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
      unsubscribePcm();
      unsubscribeAudioState();
      session.off('open', onOpen);
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
    cancelCurrentRun,
    config,
    enabled,
    getPendingActions,
    getRuntimeStatus,
    onDisableRequested,
    resolvePendingAction,
    submitUserHint,
    submitUserRequest,
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
        const result = await runLocalContextQuery(args);
        return {
          id: call.id,
          name,
          response: result,
        };
      }
      case 'submit_user_request': {
        const text = asStringArg(args['text']);
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
              needsClarification: true,
              question:
                'I caught a partial phrase. Please say your full request in one sentence.',
            },
          };
        }
        const result = await submitUserRequest(text);
        return {
          id: call.id,
          name,
          response: {
            ok: true,
            message: result || 'Submitted the request to the coding agent.',
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
            message: result || 'Added your update as a live hint.',
          },
        };
      }
      case 'resolve_pending_action': {
        const decision = asStringArg(args['decision']);
        if (!decision) {
          return {
            id: call.id,
            name,
            response: { ok: false, message: 'Missing decision.' },
          };
        }
        const request: ResolvePendingActionRequest = {
          actionId: asStringArg(args['actionId']) || undefined,
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
