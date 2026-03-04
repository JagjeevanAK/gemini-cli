/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const IS_DEV_RUNTIME = process.env['DEV'] === 'true';
const IS_NPM_START = process.env['npm_lifecycle_event'] === 'start';
const IS_SANDBOX_RUNTIME = process.env['SANDBOX'] === 'true';
// Temporarily disabled for local troubleshooting while keeping code in place.
// Re-enable by restoring the original env-driven expression below.
const DEBUG_ENABLED = false;
// const DEBUG_ENABLED =
//   process.env['GEMINI_CLI_VOICE_DEBUG'] !== '0' &&
//   (
//     process.env['GEMINI_CLI_VOICE_DEBUG'] === '1' ||
//     IS_DEV_RUNTIME ||
//     IS_NPM_START ||
//     IS_SANDBOX_RUNTIME
//   );

const explicitLogPath = process.env['GEMINI_CLI_VOICE_LOG_FILE']?.trim();
const candidateLogPaths = [
  ...(explicitLogPath ? [explicitLogPath] : []),
  path.resolve(process.cwd(), '.gemini-cli-voice.log'),
  path.join(os.homedir(), '.gemini', 'voice-debug.log'),
  path.join(os.tmpdir(), 'gemini-cli-voice.log'),
];

let stream: fs.WriteStream | null = null;
let activeLogPath: string | null = null;
let startupLogged = false;
let noPathWarningPrinted = false;
const attemptedPaths = new Set<string>();
const brokenPaths = new Set<string>();

const getStream = () => {
  if (!DEBUG_ENABLED) {
    return null;
  }
  if (stream) {
    return stream;
  }

  for (const logPath of candidateLogPaths) {
    if (brokenPaths.has(logPath)) {
      continue;
    }
    if (attemptedPaths.has(logPath) && !explicitLogPath) {
      continue;
    }
    attemptedPaths.add(logPath);
    try {
      fs.mkdirSync(path.dirname(logPath), { recursive: true });
      const nextStream = fs.createWriteStream(logPath, {
        flags: 'a',
        encoding: 'utf8',
      });
      nextStream.on('error', () => {
        if (activeLogPath) {
          brokenPaths.add(activeLogPath);
        }
        stream = null;
        activeLogPath = null;
      });
      stream = nextStream;
      activeLogPath = logPath;
      break;
    } catch {
      brokenPaths.add(logPath);
      continue;
    }
  }

  if (!stream) {
    if (!noPathWarningPrinted) {
      noPathWarningPrinted = true;
      process.stderr.write(
        `[voice-debug] unable to open log file; attempted: ${candidateLogPaths.join(', ')}\n`,
      );
    }
    return null;
  }

  if (!startupLogged) {
    startupLogged = true;
    const mode =
      process.env['GEMINI_CLI_VOICE_DEBUG'] === '1'
        ? 'env'
        : IS_DEV_RUNTIME
          ? 'dev-default'
          : IS_SANDBOX_RUNTIME
            ? 'sandbox-default'
            : 'npm-start-default';
    stream.write(
      `${toSafeString({
        ts: new Date().toISOString(),
        event: 'voice_debug.start',
        details: {
          path: activeLogPath,
          mode,
          dev: IS_DEV_RUNTIME,
          npmStart: IS_NPM_START,
          sandbox: IS_SANDBOX_RUNTIME,
        },
      })}\n`,
    );
  }

  return stream;
};

const toSafeString = (value: unknown) => {
  try {
    return JSON.stringify(value);
  } catch {
    return '"[unserializable]"';
  }
};

export const isVoiceDebugEnabled = () => DEBUG_ENABLED;

export const getVoiceDebugLogPath = () =>
  activeLogPath || explicitLogPath || candidateLogPaths[0];

export const voiceDebugLog = (
  event: string,
  details?: Record<string, unknown>,
) => {
  if (!DEBUG_ENABLED) {
    return;
  }

  const writer = getStream();
  if (!writer) {
    return;
  }

  const payload: Record<string, unknown> = {
    ts: new Date().toISOString(),
    event,
  };
  if (details && Object.keys(details).length > 0) {
    payload['details'] = details;
  }

  writer.write(`${toSafeString(payload)}\n`);
};
