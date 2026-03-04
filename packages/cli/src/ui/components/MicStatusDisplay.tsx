/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type React from 'react';
import { Box, Text } from 'ink';
import { useEffect, useMemo, useRef, useState } from 'react';
import { audioEngine, type AudioState } from '../../services/audioEngine.js';
import { theme } from '../semantic-colors.js';
import { useKeypress } from '../hooks/useKeypress.js';
import { KeypressPriority } from '../contexts/KeypressContext.js';
import { useOptionalVoiceAssistant } from '../contexts/VoiceAssistantContext.js';

const BAR_LENGTH = 12;
const ACTIVE_BLOCK = '█';
const INACTIVE_BLOCK = '░';

const INITIAL_STATE: AudioState = {
  rms: 0,
  smoothedRms: 0,
  level: 0,
  probability: 0,
  isTalking: false,
  timestamp: 0,
  error: null,
  permissionRequired: false,
};

export const MicStatusDisplay: React.FC = () => {
  const voiceAssistant = useOptionalVoiceAssistant();
  const isTestEnv =
    process.env['NODE_ENV'] === 'test' || process.env['VITEST'] === 'true';
  const [state, setState] = useState<AudioState>(INITIAL_STATE);
  const unsubscribeRef = useRef<(() => void) | null>(null);
  const enabled = voiceAssistant?.enabled ?? false;
  const connectionState = voiceAssistant?.connectionState ?? 'idle';
  const inputTranscript = voiceAssistant?.inputTranscript ?? '';
  const outputTranscript = voiceAssistant?.outputTranscript ?? '';
  const model = voiceAssistant?.model ?? null;
  const playbackWarning = voiceAssistant?.playbackWarning ?? null;
  const voiceError = voiceAssistant?.error ?? null;

  useKeypress(
    (key) => {
      if (key.cmd && key.name === 'g' && !key.ctrl && !key.alt) {
        voiceAssistant?.toggle();
        return true;
      }
      return false;
    },
    { isActive: !isTestEnv, priority: KeypressPriority.Normal },
  );

  useEffect(() => {
    if (isTestEnv) {
      return;
    }
    if (!enabled) {
      if (unsubscribeRef.current) {
        unsubscribeRef.current();
        unsubscribeRef.current = null;
      }
      setState(INITIAL_STATE);
      return;
    }

    const unsubscribe = audioEngine.subscribe(setState);
    unsubscribeRef.current = unsubscribe;
    return () => {
      unsubscribe();
      unsubscribeRef.current = null;
    };
  }, [enabled, isTestEnv]);

  const level = enabled ? state.level : 0;

  const bar = useMemo(() => {
    const activeBlocks = Math.max(
      0,
      Math.min(BAR_LENGTH, Math.floor(level * BAR_LENGTH)),
    );

    return (
      ACTIVE_BLOCK.repeat(activeBlocks) +
      INACTIVE_BLOCK.repeat(BAR_LENGTH - activeBlocks)
    );
  }, [level]);

  if (isTestEnv) {
    return null;
  }

  return (
    <Box flexDirection="column">
      {enabled && (
        <Box flexDirection="row">
          <Text>{bar}</Text>
          {state.isTalking ? (
            <Text color={theme.status.success}> Talking...</Text>
          ) : connectionState === 'connected' ? (
            <Text color={theme.text.secondary}> Listening...</Text>
          ) : (
            <Text color={theme.text.secondary}> Silent</Text>
          )}
          {connectionState === 'connecting' ? (
            <Text color={theme.text.secondary}> Connecting...</Text>
          ) : connectionState === 'connected' ? (
            <Text color={theme.status.success}> Live</Text>
          ) : null}
        </Box>
      )}
      {!enabled && <Text color={theme.text.secondary}>Mic off (Cmd+G)</Text>}
      {enabled &&
      connectionState === 'connected' &&
      !state.isTalking &&
      !inputTranscript &&
      !outputTranscript ? (
        <Text color={theme.text.secondary}>
          Connected. Speak to control the coding agent.
        </Text>
      ) : null}
      {enabled && inputTranscript ? (
        <Text color={theme.text.secondary}>You: {inputTranscript}</Text>
      ) : null}
      {enabled && outputTranscript ? (
        <Text color={theme.text.secondary}>Assistant: {outputTranscript}</Text>
      ) : null}
      {enabled && model ? (
        <Text color={theme.text.secondary}>Model: {model}</Text>
      ) : null}
      {enabled && playbackWarning ? (
        <Text color={theme.status.warning}>{playbackWarning}</Text>
      ) : null}
      {state.permissionRequired ? (
        <Text color={theme.status.warning}>
          Enable mic in System Settings &gt; Privacy &amp; Security &gt;
          Microphone, then restart Gemini CLI.
        </Text>
      ) : state.error ? (
        <Text color={theme.status.warning}>{state.error}</Text>
      ) : voiceError ? (
        <Text color={theme.status.warning}>{voiceError}</Text>
      ) : null}
    </Box>
  );
};
