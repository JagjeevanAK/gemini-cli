/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  type Config,
  type SerializableConfirmationDetails,
  type ToolConfirmationPayload,
  ToolConfirmationOutcome,
  MessageBusType,
  type IdeClient,
  debugLogger,
} from '@google/gemini-cli-core';
import type { IndividualToolCallDisplay } from '../types.js';

type LegacyConfirmationDetails = SerializableConfirmationDetails & {
  onConfirm: (
    outcome: ToolConfirmationOutcome,
    payload?: ToolConfirmationPayload,
  ) => Promise<void>;
};

function hasLegacyCallback(
  details: SerializableConfirmationDetails | undefined,
): details is LegacyConfirmationDetails {
  return (
    !!details &&
    'onConfirm' in details &&
    typeof details.onConfirm === 'function'
  );
}

export interface ResolveToolConfirmationParams {
  config: Config;
  toolCalls: IndividualToolCallDisplay[];
  callId: string;
  outcome: ToolConfirmationOutcome;
  payload?: ToolConfirmationPayload;
  ideClient?: IdeClient | null;
  isDiffingEnabled?: boolean;
}

export type ResolveToolConfirmationResult =
  | 'resolved'
  | 'not_found'
  | 'no_handler';

export async function resolveToolConfirmation(
  params: ResolveToolConfirmationParams,
): Promise<ResolveToolConfirmationResult> {
  const {
    config,
    toolCalls,
    callId,
    outcome,
    payload,
    ideClient = null,
    isDiffingEnabled = false,
  } = params;

  const tool = toolCalls.find((t) => t.callId === callId);
  if (!tool) {
    debugLogger.warn(`ToolActions: Tool ${callId} not found`);
    return 'not_found';
  }

  const details = tool.confirmationDetails;

  if (
    details?.type === 'edit' &&
    isDiffingEnabled &&
    'filePath' in details &&
    ideClient
  ) {
    const cliOutcome =
      outcome === ToolConfirmationOutcome.Cancel ? 'rejected' : 'accepted';
    await ideClient.resolveDiffFromCli(details.filePath, cliOutcome);
  }

  if (tool.correlationId) {
    await config.getMessageBus().publish({
      type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
      correlationId: tool.correlationId,
      confirmed: outcome !== ToolConfirmationOutcome.Cancel,
      requiresUserConfirmation: false,
      outcome,
      payload,
    });
    return 'resolved';
  }

  if (hasLegacyCallback(details)) {
    await details.onConfirm(outcome, payload);
    return 'resolved';
  }

  debugLogger.warn(`ToolActions: No correlationId or callback for ${callId}`);
  return 'no_handler';
}
