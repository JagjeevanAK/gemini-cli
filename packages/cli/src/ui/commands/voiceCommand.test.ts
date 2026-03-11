/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { beforeEach, describe, expect, it } from 'vitest';
import { SettingScope } from '../../config/settings.js';
import { createMockCommandContext } from '../../test-utils/mockCommandContext.js';
import type { CommandContext } from './types.js';
import { voiceCommand } from './voiceCommand.js';

describe('voiceCommand', () => {
  let mockContext: CommandContext;

  beforeEach(() => {
    mockContext = createMockCommandContext();
  });

  it('shows root guidance for the persona subcommand', async () => {
    if (!voiceCommand.action) {
      throw new Error('The voice command must have an action.');
    }

    const result = await voiceCommand.action(mockContext, '');

    expect(result).toEqual({
      type: 'message',
      messageType: 'info',
      content:
        'Use /voice persona to view or change the assistant voice persona.',
    });
  });

  it('lists the current persona and supported catalog when no persona arg is provided', async () => {
    mockContext = createMockCommandContext({
      services: {
        settings: {
          merged: {
            ui: {
              voiceAssistant: {
                persona: 'Aoede',
              },
            },
          },
        },
      },
    });

    const personaCommand = voiceCommand.subCommands?.find(
      (command) => command.name === 'persona',
    );

    const result = await personaCommand!.action!(mockContext, '');
    expect(result).toEqual(
      expect.objectContaining({
        type: 'message',
        messageType: 'info',
      }),
    );
    if (!result || result.type !== 'message') {
      throw new Error('Expected a message action.');
    }
    expect(result.content).toContain('Current persona: Aoede');
    expect(result.content).toContain('Derived grammar gender: feminine');
    expect(result.content).toContain(
      'Light, quick, and energetic for lively back-and-forth. Zephyr (Bright, feminine grammar)',
    );
    expect(result.content).toContain(
      'Playful and upbeat for quick energetic exchanges. Puck (Upbeat, masculine grammar)',
    );
  });

  it('persists a selected persona using the canonical catalog name', async () => {
    const personaCommand = voiceCommand.subCommands?.find(
      (command) => command.name === 'persona',
    );

    const result = await personaCommand!.action!(mockContext, 'zephyr');

    expect(mockContext.services.settings.setValue).toHaveBeenCalledWith(
      SettingScope.User,
      'ui.voiceAssistant.persona',
      'Zephyr',
    );
    expect(result).toEqual(
      expect.objectContaining({
        type: 'message',
        messageType: 'info',
        content: expect.stringContaining(
          'Voice persona set to Zephyr (Bright, feminine grammar)',
        ),
      }),
    );
  });

  it('clears the persisted persona when default is requested', async () => {
    const personaCommand = voiceCommand.subCommands?.find(
      (command) => command.name === 'persona',
    );

    const result = await personaCommand!.action!(mockContext, 'default');

    expect(mockContext.services.settings.setValue).toHaveBeenCalledWith(
      SettingScope.User,
      'ui.voiceAssistant.persona',
      undefined,
    );
    expect(result).toEqual(
      expect.objectContaining({
        type: 'message',
        messageType: 'info',
        content: expect.stringContaining('reset to the Live API default'),
      }),
    );
  });

  it('rejects unsupported personas', async () => {
    const personaCommand = voiceCommand.subCommands?.find(
      (command) => command.name === 'persona',
    );

    const result = await personaCommand!.action!(mockContext, 'unknown-voice');

    expect(mockContext.services.settings.setValue).not.toHaveBeenCalled();
    expect(result).toEqual(
      expect.objectContaining({
        type: 'message',
        messageType: 'error',
        content: expect.stringContaining(
          'Unknown voice persona "unknown-voice"',
        ),
      }),
    );
  });

  it('offers persona completion for default and known voices', () => {
    const personaCommand = voiceCommand.subCommands?.find(
      (command) => command.name === 'persona',
    );

    const completions = personaCommand!.completion!(mockContext, 'pu');

    expect(completions).toContain('Puck');
    expect(completions).not.toContain('Aoede');
    expect(personaCommand!.completion!(mockContext, '')).toContain('default');
  });

  it('has the expected command shape', () => {
    expect(voiceCommand.name).toBe('voice');
    expect(voiceCommand.description).toBe(
      'Manage voice assistant persona settings',
    );
    expect(voiceCommand.kind).toBe('built-in');
    expect(voiceCommand.autoExecute).toBe(false);
  });
});
