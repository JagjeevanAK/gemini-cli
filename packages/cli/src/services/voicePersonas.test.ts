/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from 'vitest';
import {
  buildVoiceAssistantSystemInstruction,
  getVoicePersonaByName,
  getVoicePersonas,
} from './voicePersonas.js';

describe('voicePersonas', () => {
  it('contains the full Google Live persona catalog with unique names', () => {
    const personas = getVoicePersonas();
    expect(personas).toHaveLength(30);
    expect(new Set(personas.map((persona) => persona.name)).size).toBe(
      personas.length,
    );
    expect(
      personas.every((persona) => persona.description.trim().length > 0),
    ).toBe(true);
  });

  it('resolves personas case-insensitively with grammar metadata', () => {
    expect(getVoicePersonaByName('zephyr')).toEqual(
      expect.objectContaining({
        name: 'Zephyr',
        style: 'Bright',
        description: 'Light, quick, and energetic for lively back-and-forth.',
        grammaticalGender: 'feminine',
      }),
    );
    expect(getVoicePersonaByName('PUCK')).toEqual(
      expect.objectContaining({
        name: 'Puck',
        grammaticalGender: 'masculine',
      }),
    );
    expect(getVoicePersonaByName('unknown-voice')).toBeUndefined();
  });

  it('builds persona-aware system instructions for gendered languages', () => {
    const instruction = buildVoiceAssistantSystemInstruction(
      getVoicePersonaByName('Aoede'),
    );

    expect(instruction).toContain('Selected voice persona: Aoede (Breezy).');
    expect(instruction).toContain('feminine self-reference and agreement');
    expect(instruction).toContain(
      'Indic, Arabic, Romance, and other gendered languages',
    );
  });

  it('falls back to neutral phrasing guidance when no persona is selected', () => {
    const instruction = buildVoiceAssistantSystemInstruction(undefined);

    expect(instruction).toContain('prefer neutral phrasing');
    expect(instruction).not.toContain('Selected voice persona:');
  });
});
