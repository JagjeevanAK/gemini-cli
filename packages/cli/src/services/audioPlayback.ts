/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import commandExists from 'command-exists';
import { voiceDebugLog } from './voiceDebugLogger.js';

const PLAYER_COMMAND = 'play';
const PCM_BYTES_PER_SECOND = 24000 * 2;
const PLAYER_ARGS = [
  '-q',
  '-t',
  'raw',
  '-b',
  '16',
  '-e',
  'signed-integer',
  '-c',
  '1',
  '-r',
  '24000',
  '-',
];

export interface AudioPlaybackStartResult {
  available: boolean;
  message?: string;
}

export class AudioPlayback {
  private player?: ChildProcessWithoutNullStreams;
  private available = false;
  private startPromise: Promise<AudioPlaybackStartResult> | null = null;
  private nextRestartAllowedAt = 0;
  private playbackEndsAt = 0;

  async start(): Promise<AudioPlaybackStartResult> {
    if (this.player && this.player.stdin.writable) {
      return { available: true };
    }

    if (this.startPromise) {
      return this.startPromise;
    }

    this.startPromise = this.startInternal().finally(() => {
      this.startPromise = null;
    });
    return this.startPromise;
  }

  private startInternal(): Promise<AudioPlaybackStartResult> {
    if (!commandExists.sync(PLAYER_COMMAND)) {
      this.available = false;
      voiceDebugLog('audio.playback.missing_binary', {
        command: PLAYER_COMMAND,
      });
      return Promise.resolve({
        available: false,
        message:
          "Missing 'play' binary (sox). Install sox for native audio playback.",
      });
    }

    try {
      const player = spawn(PLAYER_COMMAND, PLAYER_ARGS, {
        stdio: ['pipe', 'pipe', 'pipe'],
      });
      player.stdin.on('error', (error) => {
        voiceDebugLog('audio.playback.stdin_error', {
          message: getErrorMessage(error),
        });
        // Ignore write-after-close and transport errors during shutdown.
      });
      player.stderr.on('data', (chunk: Buffer | string) => {
        const text = (
          typeof chunk === 'string' ? chunk : chunk.toString('utf8')
        ).trim();
        if (!text) {
          return;
        }
        voiceDebugLog('audio.playback.stderr', {
          text: text.slice(0, 300),
        });
      });
      player.on('exit', (code, signal) => {
        voiceDebugLog('audio.playback.exit', {
          code,
          signal: signal || null,
        });
        this.player = undefined;
        this.available = false;
        this.playbackEndsAt = 0;
      });
      player.on('error', (error) => {
        voiceDebugLog('audio.playback.error', {
          message: getErrorMessage(error),
        });
        this.player = undefined;
        this.available = false;
        this.playbackEndsAt = 0;
      });
      this.player = player;
      this.available = true;
      this.nextRestartAllowedAt = 0;
      voiceDebugLog('audio.playback.started', {
        command: PLAYER_COMMAND,
      });
      return Promise.resolve({ available: true });
    } catch (error) {
      this.available = false;
      this.nextRestartAllowedAt = Date.now() + 1000;
      voiceDebugLog('audio.playback.start_failed', {
        message: getErrorMessage(error),
      });
      return Promise.resolve({
        available: false,
        message:
          error instanceof Error
            ? error.message
            : 'Unable to start audio playback process.',
      });
    }
  }

  playChunk(chunk: Buffer) {
    if (!this.available || !this.player || !this.player.stdin.writable) {
      const now = Date.now();
      if (now >= this.nextRestartAllowedAt && !this.startPromise) {
        this.nextRestartAllowedAt = now + 1000;
        void this.start();
      }
      return;
    }
    const durationMs = Math.max(
      1,
      Math.round((chunk.length / PCM_BYTES_PER_SECOND) * 1000),
    );
    this.playbackEndsAt =
      Math.max(this.playbackEndsAt, Date.now()) + durationMs;
    this.player.stdin.write(chunk);
  }

  getPendingPlaybackMs() {
    return Math.max(0, this.playbackEndsAt - Date.now());
  }

  stop() {
    if (!this.player) {
      this.playbackEndsAt = 0;
      return;
    }

    try {
      this.player.stdin.end();
    } catch {
      // Best-effort cleanup.
    }

    try {
      this.player.kill();
    } catch {
      // Best-effort cleanup.
    } finally {
      this.player = undefined;
      this.available = false;
      this.nextRestartAllowedAt = 0;
      this.playbackEndsAt = 0;
    }
  }
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return String(error);
}
