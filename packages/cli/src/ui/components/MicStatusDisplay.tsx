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
  const isTestEnv = process.env['NODE_ENV'] === 'test';
  const [state, setState] = useState<AudioState>(INITIAL_STATE);
  const [enabled, setEnabled] = useState(false);
  const unsubscribeRef = useRef<(() => void) | null>(null);

  useKeypress(
    (key) => {
      if (key.cmd && key.name === 'g' && !key.ctrl && !key.alt) {
        setEnabled((prev) => !prev);
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
          {state.isTalking && (
            <Text color={theme.status.success}> Talking...</Text>
          )}
        </Box>
      )}
      {!enabled && <Text color={theme.text.secondary}>Mic off (Cmd+G)</Text>}
      {state.permissionRequired ? (
        <Text color={theme.status.warning}>
          Enable mic in System Settings &gt; Privacy &amp; Security &gt;
          Microphone, then restart Gemini CLI.
        </Text>
      ) : state.error ? (
        <Text color={theme.status.warning}>{state.error}</Text>
      ) : null}
    </Box>
  );
};
