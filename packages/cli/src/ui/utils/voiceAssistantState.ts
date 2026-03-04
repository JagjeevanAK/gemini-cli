/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  CoreToolCallStatus,
  type SerializableConfirmationDetails,
} from '@google/gemini-cli-core';
import { type ReactNode } from 'react';
import {
  type ConfirmationRequest,
  type HistoryItem,
  type HistoryItemToolGroup,
  type HistoryItemWithoutId,
  type PermissionConfirmationRequest,
  StreamingState,
} from '../types.js';
import { getConfirmingToolState } from './confirmingTool.js';

export type VoicePendingActionType =
  | 'tool'
  | 'command'
  | 'auth'
  | 'permission'
  | 'extension_update'
  | 'loop_detection';

export interface VoicePendingAction {
  id: string;
  type: VoicePendingActionType;
  title: string;
  detail: string;
  toolCallId?: string;
  allowedDecisions: string[];
}

export interface VoicePendingActionsInput {
  pendingHistoryItems: HistoryItemWithoutId[];
  commandConfirmationRequest: ConfirmationRequest | null;
  authConsentRequest: ConfirmationRequest | null;
  permissionConfirmationRequest: PermissionConfirmationRequest | null;
  hasConfirmUpdateExtensionRequests: boolean;
  hasLoopDetectionConfirmationRequest: boolean;
}

export interface VoiceRuntimeStatusInput {
  streamingState: StreamingState;
  pendingHistoryItems: HistoryItemWithoutId[];
  commandConfirmationRequest: ConfirmationRequest | null;
  authConsentRequest: ConfirmationRequest | null;
  permissionConfirmationRequest: PermissionConfirmationRequest | null;
  hasConfirmUpdateExtensionRequests: boolean;
  hasLoopDetectionConfirmationRequest: boolean;
  thoughtSubject: string | null;
  history: HistoryItem[];
}

function nodeToText(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') {
    return String(node);
  }
  if (Array.isArray(node)) {
    return node.map((part) => nodeToText(part)).join(' ');
  }
  return 'Details available in terminal.';
}

function compressWhitespace(text: string): string {
  return text.replace(/\s+/g, ' ').trim();
}

function truncate(text: string, max = 140): string {
  if (text.length <= max) {
    return text;
  }
  return `${text.slice(0, max - 3)}...`;
}

function extractToolDetail(
  details: SerializableConfirmationDetails | undefined,
  fallback: string,
): string {
  if (!details) {
    return fallback;
  }

  switch (details.type) {
    case 'edit':
      return `Edit request for ${details.fileName}.`;
    case 'exec':
      if (details.commands && details.commands.length > 1) {
        return `${details.commands.length} commands requested.`;
      }
      return `Command: ${details.rootCommand}`;
    case 'info':
      return details.prompt || fallback;
    case 'mcp':
      return `MCP ${details.serverName}/${details.toolName}.`;
    case 'ask_user':
      return (
        details.questions.at(0)?.question ||
        details.questions.at(0)?.header ||
        fallback
      );
    case 'exit_plan_mode':
      return 'Plan is ready for implementation mode.';
    default:
      return fallback;
  }
}

function getToolAllowedDecisions(
  details: SerializableConfirmationDetails | undefined,
): string[] {
  if (!details) {
    return ['allow_once', 'cancel'];
  }

  switch (details.type) {
    case 'edit':
      return [
        'allow_once',
        'allow_session',
        'allow_always',
        'modify',
        'cancel',
      ];
    case 'exec':
    case 'info':
      return ['allow_once', 'allow_session', 'allow_always', 'cancel'];
    case 'mcp':
      return [
        'allow_once',
        'allow_tool_session',
        'allow_server_session',
        'allow_always',
        'cancel',
      ];
    case 'ask_user':
      return ['answer', 'cancel'];
    case 'exit_plan_mode':
      return ['implement_manual', 'implement_auto_edit', 'stay_in_plan'];
    default:
      return ['allow_once', 'cancel'];
  }
}

function getExecutingToolNames(pendingHistoryItems: HistoryItemWithoutId[]) {
  const toolGroups = pendingHistoryItems.filter(
    (item): item is HistoryItemToolGroup => item.type === 'tool_group',
  );
  return toolGroups
    .flatMap((group) => group.tools)
    .filter((tool) => tool.status === CoreToolCallStatus.Executing)
    .map((tool) => tool.name);
}

function getLastUserIntent(history: HistoryItem[]): string | null {
  const lastUserMessage = [...history]
    .reverse()
    .find((item) => item.type === 'user' || item.type === 'user_shell');
  if (
    !lastUserMessage ||
    !('text' in lastUserMessage) ||
    !lastUserMessage.text
  ) {
    return null;
  }
  return truncate(compressWhitespace(lastUserMessage.text), 120);
}

export function getVoicePendingActions(
  input: VoicePendingActionsInput,
): VoicePendingAction[] {
  const {
    pendingHistoryItems,
    commandConfirmationRequest,
    authConsentRequest,
    permissionConfirmationRequest,
    hasConfirmUpdateExtensionRequests,
    hasLoopDetectionConfirmationRequest,
  } = input;
  const actions: VoicePendingAction[] = [];

  const confirmingTool = getConfirmingToolState(pendingHistoryItems)?.tool;
  if (confirmingTool) {
    const title =
      confirmingTool.confirmationDetails?.title ||
      confirmingTool.description ||
      confirmingTool.name;
    actions.push({
      id: `tool:${confirmingTool.callId}`,
      type: 'tool',
      title: title || 'Tool confirmation required',
      detail: truncate(
        compressWhitespace(
          extractToolDetail(
            confirmingTool.confirmationDetails,
            confirmingTool.description || 'Tool action pending.',
          ),
        ),
      ),
      toolCallId: confirmingTool.callId,
      allowedDecisions: getToolAllowedDecisions(
        confirmingTool.confirmationDetails,
      ),
    });
  }

  if (commandConfirmationRequest) {
    actions.push({
      id: 'command_confirmation',
      type: 'command',
      title: 'Command confirmation required',
      detail: truncate(
        compressWhitespace(nodeToText(commandConfirmationRequest.prompt)),
      ),
      allowedDecisions: ['allow', 'deny'],
    });
  }

  if (authConsentRequest) {
    actions.push({
      id: 'auth_confirmation',
      type: 'auth',
      title: 'Authentication confirmation required',
      detail: truncate(
        compressWhitespace(nodeToText(authConsentRequest.prompt)),
      ),
      allowedDecisions: ['allow', 'deny'],
    });
  }

  if (permissionConfirmationRequest) {
    const fileCount = permissionConfirmationRequest.files.length;
    actions.push({
      id: 'permission_confirmation',
      type: 'permission',
      title: 'Filesystem permission required',
      detail:
        fileCount === 1
          ? `Read access requested for ${permissionConfirmationRequest.files[0]}.`
          : `Read access requested for ${fileCount} files outside workspace.`,
      allowedDecisions: ['allow', 'deny'],
    });
  }

  if (hasConfirmUpdateExtensionRequests) {
    actions.push({
      id: 'extension_update_confirmation',
      type: 'extension_update',
      title: 'Extension update confirmation required',
      detail: 'An extension update is waiting for your approval.',
      allowedDecisions: ['allow', 'deny'],
    });
  }

  if (hasLoopDetectionConfirmationRequest) {
    actions.push({
      id: 'loop_detection_confirmation',
      type: 'loop_detection',
      title: 'Loop detection confirmation required',
      detail:
        'A potential loop was detected. Choose whether to keep or disable loop detection.',
      allowedDecisions: ['keep', 'disable'],
    });
  }

  return actions;
}

export function buildRuntimeStatusSummary(
  input: VoiceRuntimeStatusInput,
): string {
  const { streamingState, pendingHistoryItems, thoughtSubject, history } =
    input;

  const goal = getLastUserIntent(history);
  const pendingAction = getVoicePendingActions({
    pendingHistoryItems,
    commandConfirmationRequest: input.commandConfirmationRequest,
    authConsentRequest: input.authConsentRequest,
    permissionConfirmationRequest: input.permissionConfirmationRequest,
    hasConfirmUpdateExtensionRequests: input.hasConfirmUpdateExtensionRequests,
    hasLoopDetectionConfirmationRequest:
      input.hasLoopDetectionConfirmationRequest,
  })[0];

  if (pendingAction) {
    return goal
      ? `Current goal: ${goal}. Waiting for approval: ${pendingAction.title}.`
      : `Waiting for approval: ${pendingAction.title}.`;
  }

  if (streamingState === StreamingState.Responding) {
    const executingTools = getExecutingToolNames(pendingHistoryItems);
    if (executingTools.length > 0) {
      const activeToolList = truncate(executingTools.join(', '), 80);
      return goal
        ? `Current goal: ${goal}. Agent is running: ${activeToolList}.`
        : `Agent is running: ${activeToolList}.`;
    }

    if (thoughtSubject?.trim()) {
      const subject = truncate(compressWhitespace(thoughtSubject), 100);
      return goal
        ? `Current goal: ${goal}. Agent is reasoning about: ${subject}.`
        : `Agent is reasoning about: ${subject}.`;
    }

    return goal
      ? `Current goal: ${goal}. Agent is actively working.`
      : 'Agent is actively working.';
  }

  if (streamingState === StreamingState.WaitingForConfirmation) {
    return goal
      ? `Current goal: ${goal}. Agent is waiting for confirmation.`
      : 'Agent is waiting for confirmation.';
  }

  return goal ? `Agent is idle. Last goal: ${goal}.` : 'Agent is idle.';
}
