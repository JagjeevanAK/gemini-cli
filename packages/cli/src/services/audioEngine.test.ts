/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from 'vitest';
import { __testables } from './audioEngine.js';

describe('audioEngine VAD helpers', () => {
  it('uses an inference interval that can keep up with the current frame size', () => {
    expect(__testables.getTargetInferenceIntervalMs(512)).toBe(32);
    expect(__testables.getTargetInferenceIntervalMs(256)).toBe(16);
  });

  it('requires sustained speech before switching to talking', () => {
    const initialState = {
      isTalking: false,
      speechCandidateSince: null,
      silenceCandidateSince: null,
    };

    const candidateState = __testables.reduceTalkingState(
      initialState,
      0.8,
      1000,
    );
    expect(candidateState).toEqual({
      isTalking: false,
      speechCandidateSince: 1000,
      silenceCandidateSince: null,
    });

    const talkingState = __testables.reduceTalkingState(
      candidateState,
      0.82,
      1055,
    );
    expect(talkingState.isTalking).toBe(true);
    expect(talkingState.silenceCandidateSince).toBeNull();
  });

  it('keeps talking through short probability dips and only releases after sustained silence', () => {
    const talkingState = {
      isTalking: true,
      speechCandidateSince: 1000,
      silenceCandidateSince: null,
    };

    const hysteresisState = __testables.reduceTalkingState(
      talkingState,
      0.5,
      2000,
    );
    expect(hysteresisState.isTalking).toBe(true);
    expect(hysteresisState.silenceCandidateSince).toBeNull();

    const shortDipState = __testables.reduceTalkingState(
      hysteresisState,
      0.2,
      2100,
    );
    expect(shortDipState.isTalking).toBe(true);
    expect(shortDipState.silenceCandidateSince).toBe(2100);

    const releasedState = __testables.reduceTalkingState(
      shortDipState,
      0.2,
      2325,
    );
    expect(releasedState).toEqual({
      isTalking: false,
      speechCandidateSince: null,
      silenceCandidateSince: 2100,
    });
  });
});
