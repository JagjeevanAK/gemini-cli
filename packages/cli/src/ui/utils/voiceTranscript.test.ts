/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from 'vitest';
import {
  appendOutputTranscriptChunk,
  mergeTranscriptChunk,
} from './voiceTranscript.js';

describe('voiceTranscript', () => {
  it('preserves multiline output chunks', () => {
    expect(
      appendOutputTranscriptChunk('', 'First line\nSecond line', false, 10),
    ).toBe('First line\nSecond line');
  });

  it('merges repeated multiline transcript updates without dropping earlier lines', () => {
    expect(
      appendOutputTranscriptChunk(
        'Step one\nStep two',
        'Step one\nStep two\nStep three',
        false,
        10,
      ),
    ).toBe('Step one\nStep two\nStep three');
  });

  it('appends distinct completed turns as separate transcript blocks', () => {
    expect(
      appendOutputTranscriptChunk(
        'First response',
        'Second response\nWith detail',
        true,
        10,
      ),
    ).toBe('First response\nSecond response\nWith detail');
  });

  it('preserves spaces between streamed transcript chunks', () => {
    expect(mergeTranscriptChunk('Just need', ' your approval')).toBe(
      'Just need your approval',
    );
    expect(appendOutputTranscriptChunk('Got it,', ' the git', false, 10)).toBe(
      'Got it, the git',
    );
  });
});
