/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it, vi } from 'vitest';
import {
  CoreToolCallStatus,
  MessageBusType,
  ToolConfirmationOutcome,
  type Config,
  type SerializableConfirmationDetails,
} from '@google/gemini-cli-core';
import { resolveToolConfirmation } from './toolConfirmationResolver.js';
import type { IndividualToolCallDisplay } from '../types.js';

describe('resolveToolConfirmation', () => {
  it('publishes to message bus when correlationId exists', async () => {
    const publish = vi.fn().mockResolvedValue(undefined);
    const config = {
      getMessageBus: () => ({ publish }),
    } as unknown as Config;

    const toolCalls: IndividualToolCallDisplay[] = [
      {
        callId: 'tool-1',
        correlationId: 'corr-1',
        name: 'tool',
        description: 'desc',
        status: CoreToolCallStatus.AwaitingApproval,
        resultDisplay: undefined,
        confirmationDetails: { type: 'info', title: 't', prompt: 'p' },
      },
    ];

    const result = await resolveToolConfirmation({
      config,
      toolCalls,
      callId: 'tool-1',
      outcome: ToolConfirmationOutcome.ProceedOnce,
    });

    expect(result).toBe('resolved');
    expect(publish).toHaveBeenCalledWith({
      type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
      correlationId: 'corr-1',
      confirmed: true,
      requiresUserConfirmation: false,
      outcome: ToolConfirmationOutcome.ProceedOnce,
      payload: undefined,
    });
  });

  it('falls back to legacy callback when correlationId is missing', async () => {
    const onConfirm = vi.fn().mockResolvedValue(undefined);
    const config = {
      getMessageBus: () => ({ publish: vi.fn() }),
    } as unknown as Config;

    const toolCalls: IndividualToolCallDisplay[] = [
      {
        callId: 'tool-legacy',
        name: 'legacy',
        description: 'desc',
        status: CoreToolCallStatus.AwaitingApproval,
        resultDisplay: undefined,
        confirmationDetails: {
          type: 'exec',
          title: 'Run',
          command: 'ls',
          rootCommand: 'ls',
          rootCommands: ['ls'],
          onConfirm,
        } as unknown as SerializableConfirmationDetails,
      },
    ];

    const result = await resolveToolConfirmation({
      config,
      toolCalls,
      callId: 'tool-legacy',
      outcome: ToolConfirmationOutcome.Cancel,
    });

    expect(result).toBe('resolved');
    expect(onConfirm).toHaveBeenCalledWith(
      ToolConfirmationOutcome.Cancel,
      undefined,
    );
  });

  it('returns not_found for unknown call ids', async () => {
    const config = {
      getMessageBus: () => ({ publish: vi.fn() }),
    } as unknown as Config;

    const result = await resolveToolConfirmation({
      config,
      toolCalls: [],
      callId: 'missing',
      outcome: ToolConfirmationOutcome.Cancel,
    });

    expect(result).toBe('not_found');
  });
});
