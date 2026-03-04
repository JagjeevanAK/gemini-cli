/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef, useState } from 'react';
import { getErrorMessage, type Config } from '@google/gemini-cli-core';
import { audioEngine } from '../../services/audioEngine.js';
import { AudioPlayback } from '../../services/audioPlayback.js';
import { LiveVoiceSession } from '../../services/liveVoiceSession.js';

export type VoiceConnectionState = 'idle' | 'connecting' | 'connected';

export interface VoiceModeState {
  connectionState: VoiceConnectionState;
  inputTranscript: string;
  outputTranscript: string;
  model: string | null;
  error: string | null;
  playbackWarning: string | null;
}

const INITIAL_STATE: VoiceModeState = {
  connectionState: 'idle',
  inputTranscript: '',
  outputTranscript: '',
  model: null,
  error: null,
  playbackWarning: null,
};

export function useVoiceMode(
  config: Config | undefined,
  enabled: boolean,
): VoiceModeState {
  const [state, setState] = useState<VoiceModeState>(INITIAL_STATE);
  const sessionRef = useRef<LiveVoiceSession | null>(null);
  const playbackRef = useRef<AudioPlayback | null>(null);

  useEffect(() => {
    if (!enabled || !config) {
      sessionRef.current?.close();
      sessionRef.current = null;
      playbackRef.current?.stop();
      playbackRef.current = null;
      setState(INITIAL_STATE);
      return;
    }

    let active = true;
    const session = new LiveVoiceSession(config);
    const playback = new AudioPlayback();
    sessionRef.current = session;
    playbackRef.current = playback;

    setState((prev) => ({
      ...prev,
      connectionState: 'connecting',
      error: null,
      playbackWarning: null,
      inputTranscript: '',
      outputTranscript: '',
    }));

    const onOpen = () => {
      if (!active) return;
      setState((prev) => ({
        ...prev,
        connectionState: 'connected',
      }));
    };

    const onInputTranscript = (text: string) => {
      if (!active || !text.trim()) return;
      setState((prev) => ({
        ...prev,
        inputTranscript: text,
      }));
    };

    const onOutputTranscript = (text: string) => {
      if (!active || !text.trim()) return;
      setState((prev) => ({
        ...prev,
        outputTranscript: text,
      }));
    };

    const onOutputAudioChunk = ({
      chunk,
    }: {
      chunk: Buffer;
      mimeType: string;
    }) => {
      if (!active) return;
      playback.playChunk(chunk);
    };

    const onError = (error: Error) => {
      if (!active) return;
      setState((prev) => ({
        ...prev,
        error: error.message,
        connectionState: 'idle',
      }));
    };

    session.on('open', onOpen);
    session.on('inputTranscript', onInputTranscript);
    session.on('outputTranscript', onOutputTranscript);
    session.on('outputAudioChunk', onOutputAudioChunk);
    session.on('error', onError);

    const unsubscribePcm = audioEngine.subscribePcm((pcmBytes) => {
      session.sendAudioChunk(pcmBytes);
    });

    void (async () => {
      const playbackResult = await playback.start();
      if (!active) return;
      if (!playbackResult.available) {
        setState((prev) => ({
          ...prev,
          playbackWarning:
            playbackResult.message ??
            'Audio playback unavailable; transcripts only.',
        }));
      }

      try {
        const model = await session.start();
        if (!active) return;
        setState((prev) => ({
          ...prev,
          connectionState: 'connected',
          model,
        }));
      } catch (error) {
        if (!active) return;
        setState((prev) => ({
          ...prev,
          connectionState: 'idle',
          error: getErrorMessage(error),
        }));
      }
    })();

    return () => {
      active = false;
      unsubscribePcm();
      session.off('open', onOpen);
      session.off('inputTranscript', onInputTranscript);
      session.off('outputTranscript', onOutputTranscript);
      session.off('outputAudioChunk', onOutputAudioChunk);
      session.off('error', onError);
      session.close();
      playback.stop();
      sessionRef.current = null;
      playbackRef.current = null;
    };
  }, [config, enabled]);

  return state;
}
