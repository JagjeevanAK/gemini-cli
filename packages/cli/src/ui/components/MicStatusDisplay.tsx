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
import { ThemedGradient } from './ThemedGradient.js';
import { GeminiSpinner } from './GeminiRespondingSpinner.js';
import { useKeypress } from '../hooks/useKeypress.js';
import { KeypressPriority } from '../contexts/KeypressContext.js';
import { useOptionalVoiceAssistant } from '../contexts/VoiceAssistantContext.js';

const BAR_LENGTH = 12;
const ACTIVE_BLOCK = '█';
const INACTIVE_BLOCK = '░';
const PUSH_TO_TALK_FALLBACK_RELEASE_MS = 900;
const AWAITING_ASSISTANT_RESPONSE_TIMEOUT_MS = 8000;
const ASSISTANT_PULSE_HOLD_MS = 1100;

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
  const pushToTalkActiveRef = useRef(false);
  const pushToTalkReleaseTimerRef = useRef<NodeJS.Timeout | null>(null);
  const awaitingResponseTimerRef = useRef<NodeJS.Timeout | null>(null);
  const assistantPulseHoldTimerRef = useRef<NodeJS.Timeout | null>(null);
  const wasActiveCaptureRef = useRef(false);
  const [assistantPulseHeld, setAssistantPulseHeld] = useState(false);
  const enabled = voiceAssistant?.enabled ?? false;
  const listening = voiceAssistant?.listening ?? false;
  const connectionState = voiceAssistant?.connectionState ?? 'idle';
  const inputTranscript = voiceAssistant?.inputTranscript ?? '';
  const assistantSpeaking = voiceAssistant?.assistantSpeaking ?? false;
  const assistantOutputLevel = voiceAssistant?.outputLevel ?? 0;
  const model = voiceAssistant?.model ?? null;
  const playbackWarning = voiceAssistant?.playbackWarning ?? null;
  const voiceError = voiceAssistant?.error ?? null;
  const isActiveCapture = enabled && listening;
  const [awaitingAssistantResponse, setAwaitingAssistantResponse] =
    useState(false);
  const showPassiveMicOff =
    !isActiveCapture &&
    !assistantSpeaking &&
    !assistantPulseHeld &&
    !awaitingAssistantResponse &&
    connectionState !== 'connecting' &&
    connectionState !== 'idle';

  useKeypress(
    (key) => {
      if (key.cmd && key.name === 'g' && !key.ctrl && !key.alt) {
        const clearFallbackReleaseTimer = () => {
          if (pushToTalkReleaseTimerRef.current) {
            clearTimeout(pushToTalkReleaseTimerRef.current);
            pushToTalkReleaseTimerRef.current = null;
          }
        };

        const stopPushToTalkSession = () => {
          clearFallbackReleaseTimer();
          voiceAssistant?.setListening(false);
          pushToTalkActiveRef.current = false;
        };

        if (key.phase === 'release') {
          stopPushToTalkSession();
          return true;
        }

        // Any non-release Cmd+G event means user is still holding the chord.
        // Always clear a previously scheduled fallback release timeout to
        // prevent accidental mid-hold capture drops.
        clearFallbackReleaseTimer();

        if (!pushToTalkActiveRef.current) {
          pushToTalkActiveRef.current = true;
          if (!enabled) {
            voiceAssistant?.enable();
          }
          voiceAssistant?.setListening(true);
        } else if (!listening) {
          voiceAssistant?.setListening(true);
        }

        // Fallback for terminals that do not emit key-release events.
        if (!key.phase) {
          pushToTalkReleaseTimerRef.current = setTimeout(() => {
            stopPushToTalkSession();
          }, PUSH_TO_TALK_FALLBACK_RELEASE_MS);
        }
        return true;
      }
      return false;
    },
    // Keep push-to-talk available even when confirmation dialogs consume keys.
    { isActive: !isTestEnv, priority: KeypressPriority.Critical },
  );

  useEffect(() => {
    if (!enabled) {
      pushToTalkActiveRef.current = false;
      if (pushToTalkReleaseTimerRef.current) {
        clearTimeout(pushToTalkReleaseTimerRef.current);
        pushToTalkReleaseTimerRef.current = null;
      }
    }
  }, [enabled]);

  useEffect(
    () => () => {
      if (pushToTalkReleaseTimerRef.current) {
        clearTimeout(pushToTalkReleaseTimerRef.current);
        pushToTalkReleaseTimerRef.current = null;
      }
      if (awaitingResponseTimerRef.current) {
        clearTimeout(awaitingResponseTimerRef.current);
        awaitingResponseTimerRef.current = null;
      }
      if (assistantPulseHoldTimerRef.current) {
        clearTimeout(assistantPulseHoldTimerRef.current);
        assistantPulseHoldTimerRef.current = null;
      }
    },
    [],
  );

  useEffect(() => {
    if (!enabled) {
      if (assistantPulseHoldTimerRef.current) {
        clearTimeout(assistantPulseHoldTimerRef.current);
        assistantPulseHoldTimerRef.current = null;
      }
      setAssistantPulseHeld(false);
      return;
    }

    if (assistantSpeaking || assistantOutputLevel > 0.01) {
      setAssistantPulseHeld(true);
      if (assistantPulseHoldTimerRef.current) {
        clearTimeout(assistantPulseHoldTimerRef.current);
      }
      assistantPulseHoldTimerRef.current = setTimeout(() => {
        setAssistantPulseHeld(false);
        assistantPulseHoldTimerRef.current = null;
      }, ASSISTANT_PULSE_HOLD_MS);
      assistantPulseHoldTimerRef.current.unref?.();
    }
  }, [assistantOutputLevel, assistantSpeaking, enabled]);

  useEffect(() => {
    const hadActiveCapture = wasActiveCaptureRef.current;
    wasActiveCaptureRef.current = isActiveCapture;

    if (!enabled) {
      if (awaitingResponseTimerRef.current) {
        clearTimeout(awaitingResponseTimerRef.current);
        awaitingResponseTimerRef.current = null;
      }
      setAwaitingAssistantResponse(false);
      return;
    }

    if (isActiveCapture) {
      if (awaitingResponseTimerRef.current) {
        clearTimeout(awaitingResponseTimerRef.current);
        awaitingResponseTimerRef.current = null;
      }
      setAwaitingAssistantResponse(false);
      return;
    }

    if (assistantSpeaking) {
      if (awaitingResponseTimerRef.current) {
        clearTimeout(awaitingResponseTimerRef.current);
        awaitingResponseTimerRef.current = null;
      }
      setAwaitingAssistantResponse(false);
      return;
    }

    const justReleasedAfterCapture = hadActiveCapture && !isActiveCapture;
    if (!justReleasedAfterCapture) {
      return;
    }

    setAwaitingAssistantResponse(true);
    if (awaitingResponseTimerRef.current) {
      clearTimeout(awaitingResponseTimerRef.current);
    }
    awaitingResponseTimerRef.current = setTimeout(() => {
      setAwaitingAssistantResponse(false);
      awaitingResponseTimerRef.current = null;
    }, AWAITING_ASSISTANT_RESPONSE_TIMEOUT_MS);
    awaitingResponseTimerRef.current.unref?.();
  }, [assistantSpeaking, enabled, isActiveCapture]);

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

  const assistantLevel =
    assistantSpeaking || assistantPulseHeld ? assistantOutputLevel : 0;
  const showUserTalkingActivity = isActiveCapture && state.isTalking;
  const showAssistantActivity =
    enabled &&
    !showUserTalkingActivity &&
    (assistantSpeaking || assistantPulseHeld);
  const showVoiceActivityBar = showAssistantActivity || showUserTalkingActivity;
  const level = showAssistantActivity
    ? assistantLevel
    : showUserTalkingActivity
      ? state.level
      : 0;
  const activeBlocks = useMemo(() => {
    const blocks = Math.max(
      0,
      Math.min(BAR_LENGTH, Math.floor(level * BAR_LENGTH)),
    );
    if (showAssistantActivity) {
      return Math.max(1, blocks);
    }
    return blocks;
  }, [level, showAssistantActivity]);
  const inactiveBlocks = BAR_LENGTH - activeBlocks;
  const userBar = useMemo(
    () =>
      ACTIVE_BLOCK.repeat(activeBlocks) + INACTIVE_BLOCK.repeat(inactiveBlocks),
    [activeBlocks, inactiveBlocks],
  );

  if (isTestEnv) {
    return null;
  }

  return (
    <Box flexDirection="column">
      {enabled && !showPassiveMicOff && (
        <Box flexDirection="row">
          {showVoiceActivityBar &&
            (showAssistantActivity ? (
              <Box flexDirection="row">
                <Text color={theme.text.secondary}>
                  {INACTIVE_BLOCK.repeat(inactiveBlocks)}
                </Text>
                {activeBlocks > 0 ? (
                  <ThemedGradient>
                    {ACTIVE_BLOCK.repeat(activeBlocks)}
                  </ThemedGradient>
                ) : null}
              </Box>
            ) : (
              <Text>{userBar}</Text>
            ))}
          {showAssistantActivity ? (
            <ThemedGradient> Assistant speaking...</ThemedGradient>
          ) : isActiveCapture && state.isTalking ? (
            <Text color={theme.status.success}> Talking...</Text>
          ) : awaitingAssistantResponse ? (
            <Box flexDirection="row">
              <GeminiSpinner
                spinnerType="dots"
                altText="Waiting for response"
              />
              <Text color={theme.text.secondary}> Waiting for response...</Text>
            </Box>
          ) : connectionState === 'connected' && isActiveCapture ? (
            <Text color={theme.text.secondary}> Listening...</Text>
          ) : connectionState === 'connected' ? (
            <Text color={theme.text.secondary}> Standby</Text>
          ) : (
            <Text color={theme.text.secondary}> Silent</Text>
          )}
          {connectionState === 'connecting' ? (
            <Text color={theme.text.secondary}> Connecting...</Text>
          ) : connectionState === 'connected' && isActiveCapture ? (
            <Text color={theme.status.success}> Live</Text>
          ) : null}
        </Box>
      )}
      {(!enabled || showPassiveMicOff) && (
        <Text color={theme.text.secondary}>Mic off (Hold Cmd+G to talk)</Text>
      )}
      {enabled && !listening && !showPassiveMicOff ? (
        <Text color={theme.text.secondary}>Hold Cmd+G to speak.</Text>
      ) : null}
      {enabled &&
      listening &&
      connectionState === 'connected' &&
      !state.isTalking &&
      !inputTranscript &&
      !voiceAssistant?.outputTranscript ? (
        <Text color={theme.text.secondary}>
          Connected. Speak to control the coding agent.
        </Text>
      ) : null}
      {enabled && model && !showPassiveMicOff ? (
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
