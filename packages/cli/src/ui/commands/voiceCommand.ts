/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { MessageActionReturn } from '@google/gemini-cli-core';
import { SettingScope } from '../../config/settings.js';
import {
  formatVoicePersona,
  formatVoicePersonaWithDescription,
  getVoicePersonaByName,
  getVoicePersonas,
} from '../../services/voicePersonas.js';
import { CommandKind, type SlashCommand } from './types.js';

function buildPersonaStatusMessage(configuredPersonaName: string | undefined) {
  const configuredPersona = getVoicePersonaByName(configuredPersonaName);
  const currentPersonaLine = configuredPersona
    ? formatVoicePersona(configuredPersona)
    : configuredPersonaName
      ? `${configuredPersonaName} (unsupported custom value)`
      : 'server default';
  const currentGender = configuredPersona?.grammaticalGender ?? 'neutral';
  const personaLines = getVoicePersonas()
    .map((persona) => `- ${formatVoicePersonaWithDescription(persona)}`)
    .join('\n');

  return [
    `Current persona: ${currentPersonaLine}`,
    `Derived grammar gender: ${currentGender}`,
    '',
    'Usage:',
    '/voice persona <persona-name>',
    '/voice persona default',
    '',
    'Available personas:',
    personaLines,
  ].join('\n');
}

const personaCommand: SlashCommand = {
  name: 'persona',
  description:
    'View or set the assistant voice persona. Usage: /voice persona [persona-name|default]',
  kind: CommandKind.BUILT_IN,
  autoExecute: false,
  completion: (_context, partialArg) => {
    const options = ['default', ...getVoicePersonas().map((p) => p.name)];
    const normalizedPartial = partialArg.trim().toLowerCase();
    if (!normalizedPartial) {
      return options;
    }
    return options.filter((option) =>
      option.toLowerCase().startsWith(normalizedPartial),
    );
  },
  action: async (context, args): Promise<MessageActionReturn> => {
    const trimmedArgs = args.trim();
    const configuredPersonaName =
      context.services.settings.merged.ui.voiceAssistant.persona;

    if (!trimmedArgs) {
      return {
        type: 'message',
        messageType: 'info',
        content: buildPersonaStatusMessage(configuredPersonaName),
      };
    }

    if (trimmedArgs.toLowerCase() === 'default') {
      context.services.settings.setValue(
        SettingScope.User,
        'ui.voiceAssistant.persona',
        undefined,
      );
      return {
        type: 'message',
        messageType: 'info',
        content:
          'Voice persona reset to the Live API default. If voice mode is active, it will reconnect automatically.',
      };
    }

    const selectedPersona = getVoicePersonaByName(trimmedArgs);
    if (!selectedPersona) {
      return {
        type: 'message',
        messageType: 'error',
        content: `Unknown voice persona "${trimmedArgs}". Run /voice persona to view the supported Google Live personas.`,
      };
    }

    context.services.settings.setValue(
      SettingScope.User,
      'ui.voiceAssistant.persona',
      selectedPersona.name,
    );

    return {
      type: 'message',
      messageType: 'info',
      content: `Voice persona set to ${formatVoicePersona(selectedPersona)}. If voice mode is active, it will reconnect automatically.`,
    };
  },
};

export const voiceCommand: SlashCommand = {
  name: 'voice',
  description: 'Manage voice assistant persona settings',
  kind: CommandKind.BUILT_IN,
  autoExecute: false,
  subCommands: [personaCommand],
  action: async (): Promise<MessageActionReturn> => ({
    type: 'message',
    messageType: 'info',
    content:
      'Use /voice persona to view or change the assistant voice persona.',
  }),
};
