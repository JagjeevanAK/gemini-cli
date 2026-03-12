/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  type ConfirmationRequest,
  type HistoryItemWithoutId,
  type PermissionConfirmationRequest,
} from '../types.js';
import { type ReactNode } from 'react';
import { type RunEventNotificationEvent } from '../../utils/terminalNotifications.js';
import { getConfirmingToolState } from './confirmingTool.js';

export interface PendingAttentionNotification {
  key: string;
  event: RunEventNotificationEvent;
}

function shortenForSpeech(text: string, maxChars: number) {
  const cleaned = text.replace(/\s+/g, ' ').trim();
  if (!cleaned) {
    return '';
  }
  if (cleaned.length <= maxChars) {
    return cleaned;
  }
  return `${cleaned.slice(0, maxChars - 3)}...`;
}

function getToolDecisionPrompt(
  details: NonNullable<
    ReturnType<typeof getConfirmingToolState>
  >['tool']['confirmationDetails'],
) {
  if (!details) {
    return 'Say "allow once", "allow for this session", "always allow", or "cancel".';
  }

  switch (details.type) {
    case 'edit':
      return 'Say "allow once", "allow for this session", "always allow", "modify", or "cancel".';
    case 'exec':
    case 'info':
      return 'Say "allow once", "allow for this session", "always allow", or "cancel".';
    case 'mcp':
      return 'Say "allow once", "allow for this tool", "allow for this server", "always allow", or "cancel".';
    case 'exit_plan_mode':
      return 'Say "manual", "auto edit", or "stay in plan".';
    default:
      return 'Respond with one of the available options shown in the terminal.';
  }
}

function buildToolPermissionDetail(
  details: NonNullable<
    ReturnType<typeof getConfirmingToolState>
  >['tool']['confirmationDetails'],
  fallbackTitle: string | undefined,
) {
  const optionsPrompt = getToolDecisionPrompt(details);
  if (!details) {
    return `I need your permission to continue this task. ${optionsPrompt}`;
  }

  if (details.type === 'exec') {
    const command = shortenForSpeech(
      details.command || details.rootCommand,
      90,
    );
    if (command) {
      return `I need permission to run "${command}" to continue this work. ${optionsPrompt}`;
    }
    return `I need permission to run a command to continue this work. ${optionsPrompt}`;
  }

  if (details.type === 'edit') {
    const fileName = shortenForSpeech(details.fileName, 60);
    return fileName
      ? `I need permission to edit ${fileName} to complete this work. ${optionsPrompt}`
      : `I need permission to edit a file to complete this work. ${optionsPrompt}`;
  }

  if (details.type === 'mcp') {
    const toolName = shortenForSpeech(details.toolDisplayName, 60);
    const serverName = shortenForSpeech(details.serverName, 40);
    if (toolName && serverName) {
      return `I need permission to run ${toolName} on ${serverName} for this task. ${optionsPrompt}`;
    }
  }

  const title = shortenForSpeech(details.title || fallbackTitle || '', 80);
  if (title) {
    return `I need your permission for ${title} to continue this task. ${optionsPrompt}`;
  }

  return `I need your permission to continue this task. ${optionsPrompt}`;
}

function keyFromReactNode(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') {
    return String(node);
  }
  if (Array.isArray(node)) {
    return node.map((item) => keyFromReactNode(item)).join('|');
  }
  return 'react-node';
}

export function getPendingAttentionNotification(
  pendingHistoryItems: HistoryItemWithoutId[],
  commandConfirmationRequest: ConfirmationRequest | null,
  authConsentRequest: ConfirmationRequest | null,
  permissionConfirmationRequest: PermissionConfirmationRequest | null,
  hasConfirmUpdateExtensionRequests: boolean,
  hasLoopDetectionConfirmationRequest: boolean,
): PendingAttentionNotification | null {
  const confirmingToolState = getConfirmingToolState(pendingHistoryItems);
  if (confirmingToolState) {
    const details = confirmingToolState.tool.confirmationDetails;
    if (details?.type === 'ask_user') {
      const firstQuestion = details.questions.at(0)?.header;
      return {
        key: `ask_user:${confirmingToolState.tool.callId}`,
        event: {
          type: 'attention',
          heading: 'Answer requested by agent',
          detail: firstQuestion || 'The agent needs your response to continue.',
        },
      };
    }

    const toolTitle = details?.title || confirmingToolState.tool.description;
    return {
      key: `tool_confirmation:${confirmingToolState.tool.callId}`,
      event: {
        type: 'attention',
        heading: 'Approval required',
        detail: buildToolPermissionDetail(details, toolTitle),
      },
    };
  }

  if (commandConfirmationRequest) {
    const promptKey = keyFromReactNode(commandConfirmationRequest.prompt);
    return {
      key: `command_confirmation:${promptKey}`,
      event: {
        type: 'attention',
        heading: 'Confirmation required',
        detail:
          'A command is waiting for your confirmation. Say "allow" or "deny".',
      },
    };
  }

  if (authConsentRequest) {
    const promptKey = keyFromReactNode(authConsentRequest.prompt);
    return {
      key: `auth_consent:${promptKey}`,
      event: {
        type: 'attention',
        heading: 'Authentication confirmation required',
        detail:
          'Authentication is waiting for your confirmation. Say "allow" or "deny".',
      },
    };
  }

  if (permissionConfirmationRequest) {
    const filesKey = permissionConfirmationRequest.files.join('|');
    return {
      key: `filesystem_permission_confirmation:${filesKey}`,
      event: {
        type: 'attention',
        heading: 'Filesystem permission required',
        detail:
          'Read-only path access is waiting for your confirmation. Say "allow" or "deny".',
      },
    };
  }

  if (hasConfirmUpdateExtensionRequests) {
    return {
      key: 'extension_update_confirmation',
      event: {
        type: 'attention',
        heading: 'Extension update confirmation required',
        detail:
          'An extension update is waiting for your confirmation. Say "allow" or "deny".',
      },
    };
  }

  if (hasLoopDetectionConfirmationRequest) {
    return {
      key: 'loop_detection_confirmation',
      event: {
        type: 'attention',
        heading: 'Loop detection confirmation required',
        detail:
          'A loop detection prompt is waiting for your response. Say "keep" or "disable".',
      },
    };
  }

  return null;
}
