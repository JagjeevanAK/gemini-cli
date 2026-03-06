/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from 'vitest';
import { CoreToolCallStatus } from '@google/gemini-cli-core';
import {
  buildRuntimeStatusSummary,
  getVoicePendingActions,
} from './voiceAssistantState.js';
import { StreamingState } from '../types.js';

describe('voiceAssistantState', () => {
  it('returns a tool pending action with strict allowed decisions', () => {
    const actions = getVoicePendingActions({
      pendingHistoryItems: [
        {
          type: 'tool_group',
          tools: [
            {
              callId: 'tool-1',
              status: CoreToolCallStatus.AwaitingApproval,
              name: 'run_shell_command',
              description: 'Run command',
              confirmationDetails: {
                type: 'exec',
                title: 'Execute command',
                command: 'npm test',
                rootCommand: 'npm',
                rootCommands: ['npm'],
              },
            },
          ],
        } as never,
      ],
      commandConfirmationRequest: null,
      authConsentRequest: null,
      permissionConfirmationRequest: null,
      hasConfirmUpdateExtensionRequests: false,
      hasLoopDetectionConfirmationRequest: false,
    });

    expect(actions).toHaveLength(1);
    expect(actions[0].id).toBe('tool:tool-1');
    expect(actions[0].allowedDecisions).toContain('allow_once');
    expect(actions[0].allowedDecisions).toContain('cancel');
  });

  it('includes non-tool pending actions', () => {
    const actions = getVoicePendingActions({
      pendingHistoryItems: [],
      commandConfirmationRequest: {
        prompt: 'Allow command?',
        onConfirm: () => {},
      },
      authConsentRequest: {
        prompt: 'Allow auth?',
        onConfirm: () => {},
      },
      permissionConfirmationRequest: {
        files: ['/tmp/example.txt'],
        onComplete: () => {},
      },
      hasConfirmUpdateExtensionRequests: true,
      hasLoopDetectionConfirmationRequest: true,
    });

    expect(actions.map((a) => a.type)).toEqual([
      'command',
      'auth',
      'permission',
      'extension_update',
      'loop_detection',
    ]);
  });

  it('builds a running status summary with active tool and goal', () => {
    const summary = buildRuntimeStatusSummary({
      streamingState: StreamingState.Responding,
      pendingHistoryItems: [
        {
          type: 'tool_group',
          tools: [
            {
              callId: 'exec-1',
              name: 'read_file',
              status: CoreToolCallStatus.Executing,
              description: 'Read file',
              confirmationDetails: undefined,
            },
          ],
        } as never,
      ],
      commandConfirmationRequest: null,
      authConsentRequest: null,
      permissionConfirmationRequest: null,
      hasConfirmUpdateExtensionRequests: false,
      hasLoopDetectionConfirmationRequest: false,
      thoughtSubject: null,
      history: [
        {
          id: 1,
          type: 'user',
          text: 'Refactor the parser and update tests',
        } as never,
      ],
    });

    expect(summary).toContain('Current goal:');
    expect(summary).toContain('read_file');
  });

  it('builds an approval-waiting summary when pending action exists', () => {
    const summary = buildRuntimeStatusSummary({
      streamingState: StreamingState.Responding,
      pendingHistoryItems: [
        {
          type: 'tool_group',
          tools: [
            {
              callId: 'ask-1',
              name: 'run_shell_command',
              status: CoreToolCallStatus.AwaitingApproval,
              description: 'Run command',
              confirmationDetails: {
                type: 'exec',
                title: 'Execute command',
                command: 'npm test',
                rootCommand: 'npm',
                rootCommands: ['npm'],
              },
            },
          ],
        } as never,
      ],
      commandConfirmationRequest: null,
      authConsentRequest: null,
      permissionConfirmationRequest: null,
      hasConfirmUpdateExtensionRequests: false,
      hasLoopDetectionConfirmationRequest: false,
      thoughtSubject: 'running tests',
      history: [],
    });

    expect(summary).toContain('Waiting for approval');
  });
});
