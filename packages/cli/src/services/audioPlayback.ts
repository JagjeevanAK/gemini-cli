/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import commandExists from 'command-exists';

const PLAYER_COMMAND = 'play';
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

  async start(): Promise<AudioPlaybackStartResult> {
    if (this.player) {
      return { available: true };
    }

    if (!commandExists.sync(PLAYER_COMMAND)) {
      this.available = false;
      return {
        available: false,
        message:
          "Missing 'play' binary (sox). Install sox for native audio playback.",
      };
    }

    try {
      const player = spawn(PLAYER_COMMAND, PLAYER_ARGS, {
        stdio: ['pipe', 'pipe', 'pipe'],
      });
      player.stdin.on('error', () => {
        // Ignore write-after-close and transport errors during shutdown.
      });
      player.on('exit', () => {
        this.player = undefined;
      });
      player.on('error', () => {
        this.player = undefined;
      });
      this.player = player;
      this.available = true;
      return { available: true };
    } catch (error) {
      this.available = false;
      return {
        available: false,
        message:
          error instanceof Error
            ? error.message
            : 'Unable to start audio playback process.',
      };
    }
  }

  playChunk(chunk: Buffer) {
    if (!this.available || !this.player || !this.player.stdin.writable) {
      return;
    }
    this.player.stdin.write(chunk);
  }

  stop() {
    if (!this.player) {
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
    }
  }
}
