/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
declare module 'mic' {
  import type { Readable } from 'node:stream';

  interface MicOptions {
    rate?: string;
    channels?: string;
    bitwidth?: string;
    encoding?: string;
    endian?: string;
    device?: string;
    fileType?: string;
    debug?: boolean;
    exitOnSilence?: number;
  }

  interface MicInstance {
    getAudioStream(): Readable;
    start(): void;
    stop(): void;
    pause(): void;
    resume(): void;
  }

  // eslint-disable-next-line import/no-default-export
  export default function mic(options?: MicOptions): MicInstance;
}
