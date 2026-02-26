/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { EventEmitter } from 'node:events';
import fs, { promises as fsPromises } from 'node:fs';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import mic from 'mic';
import * as ort from 'onnxruntime-node';
import commandExists from 'command-exists';
import { fetch } from 'undici';

const SAMPLE_RATE = 16000;
const CHANNELS = 1;
const BIT_WIDTH = 16;
const ENCODING = 'signed-integer';
const ENDIAN = 'little';
const RMS_ALPHA = 0.2;
const RMS_SCALE = 20.0;
const UI_UPDATE_INTERVAL_MS = 30;
const INFERENCE_INTERVAL_MS = 50;
const SPEECH_THRESHOLD = 0.6;
const TALKING_HOLD_MS = 50;
const NOISE_FLOOR = 0.015;
const DEFAULT_FRAME_SIZE = 512;
const MAX_BUFFER_FRAMES = 50;
const MODEL_FILENAME = 'silero_vad.onnx';
const MODEL_URL =
  'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx';
const MIC_COMMAND = 'rec';
const SOURCE_RATE_WINDOW_MS = 1200;
const SOURCE_RATE_THRESHOLD = 32000;

let modelDownloadPromise: Promise<string> | null = null;

export interface AudioState {
  rms: number;
  smoothedRms: number;
  level: number;
  probability: number;
  isTalking: boolean;
  timestamp: number;
  error: string | null;
  permissionRequired: boolean;
}

type AudioEvents = {
  data: [AudioState];
  error: [Error];
};

type MicInstance = ReturnType<typeof mic>;

type TensorMetadata = Extract<
  ort.InferenceSession.ValueMetadata,
  { isTensor: true }
>;

type SessionOutputs = ort.InferenceSession.ReturnType;

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const normalizeError = (error: unknown) =>
  error instanceof Error
    ? error
    : new Error(typeof error === 'string' ? error : String(error));

const computeRms = (samples: Float32Array) => {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const sample = samples[i];
    sum += sample * sample;
  }
  return Math.sqrt(sum / samples.length);
};

const toFloat32 = (buffer: Buffer) => {
  const sampleCount = Math.floor(buffer.length / 2);
  const float32 = new Float32Array(sampleCount);
  for (let i = 0; i < sampleCount; i += 1) {
    float32[i] = buffer.readInt16LE(i * 2) / 32768;
  }
  return float32;
};

const downsampleByFactor = (samples: Float32Array, factor: number) => {
  if (factor <= 1) return samples;
  const outputLength = Math.floor(samples.length / factor);
  if (outputLength <= 0) return samples;

  const output = new Float32Array(outputLength);
  let inputIndex = 0;

  for (let i = 0; i < outputLength; i += 1) {
    let sum = 0;
    for (let j = 0; j < factor; j += 1) {
      sum += samples[inputIndex];
      inputIndex += 1;
    }
    output[i] = sum / factor;
  }

  return output;
};

const fileExists = async (filePath: string) => {
  try {
    await fsPromises.access(filePath, fs.constants.R_OK);
    return true;
  } catch {
    return false;
  }
};

const findPackageRoot = async (startDir: string) => {
  let current = startDir;
  while (true) {
    const pkgPath = path.join(current, 'package.json');
    try {
      const contents = await fsPromises.readFile(pkgPath, 'utf8');
      const parsed: unknown = JSON.parse(contents);
      const pkgName =
        isRecord(parsed) && typeof parsed['name'] === 'string'
          ? parsed['name']
          : null;
      if (pkgName === '@google/gemini-cli') {
        return current;
      }
    } catch {
      // Keep searching up the tree.
    }

    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }

  return null;
};

const getModelPaths = async () => {
  const cwdPath = path.resolve(process.cwd(), 'model', MODEL_FILENAME);
  const moduleDir = path.dirname(fileURLToPath(import.meta.url));
  const packageRoot = await findPackageRoot(moduleDir);
  const packagePath = packageRoot
    ? path.join(packageRoot, 'model', MODEL_FILENAME)
    : null;

  return { cwdPath, packagePath };
};

const downloadModel = async (targetPath: string) => {
  await fsPromises.mkdir(path.dirname(targetPath), { recursive: true });
  const tempPath = `${targetPath}.download`;

  const response = await fetch(MODEL_URL);
  if (!response.ok || !response.body) {
    throw new Error(
      `Failed to download silero_vad.onnx (${response.status} ${response.statusText})`,
    );
  }

  const readable = Readable.fromWeb(response.body);
  try {
    await pipeline(readable, fs.createWriteStream(tempPath));
    await fsPromises.rename(tempPath, targetPath);
    return targetPath;
  } catch (error) {
    await fsPromises.rm(tempPath, { force: true });
    throw error;
  }
};

const ensureModelPath = async () => {
  const { cwdPath, packagePath } = await getModelPaths();

  if (await fileExists(cwdPath)) return cwdPath;
  if (packagePath && (await fileExists(packagePath))) return packagePath;

  if (!modelDownloadPromise) {
    modelDownloadPromise = (async () => {
      if (await fileExists(cwdPath)) return cwdPath;
      return downloadModel(cwdPath);
    })();
  }

  try {
    return await modelDownloadPromise;
  } catch (error) {
    modelDownloadPromise = null;
    throw error;
  }
};

const isTensorMetadata = (
  metadata: ort.InferenceSession.ValueMetadata | undefined,
): metadata is TensorMetadata => Boolean(metadata?.isTensor);

const createZeroTensor = (metadata: TensorMetadata) => {
  const dims = (metadata.shape ?? []).map((dim: number | string) =>
    typeof dim === 'number' && dim > 0 ? dim : 1,
  );
  const size = dims.reduce((acc: number, dim: number) => acc * dim, 1) || 1;

  if (metadata.type === 'float32') {
    return new ort.Tensor('float32', new Float32Array(size), dims);
  }

  if (metadata.type === 'int64') {
    return new ort.Tensor('int64', new BigInt64Array(size), dims);
  }

  if (metadata.type === 'int32') {
    return new ort.Tensor('int32', new Int32Array(size), dims);
  }

  return undefined;
};

class AudioEngine extends EventEmitter<AudioEvents> {
  private micInstance?: MicInstance;
  private micStream?: NodeJS.ReadableStream;
  private session?: ort.InferenceSession;
  private inferenceTimer?: NodeJS.Timeout;
  private running = false;
  private starting = false;
  private inferenceInFlight = false;
  private smoothedRms = 0;
  private lastRms = 0;
  private lastLevel = 0;
  private lastProbability = 0;
  private isTalking = false;
  private lastEmit = 0;
  private lastErrorMessage: string | null = null;
  private permissionRequired = false;
  private speechCandidateSince: number | null = null;
  private subscribers = 0;
  private sourceSampleRate = SAMPLE_RATE;
  private sourceRateWindowStart = 0;
  private sourceRateWindowSamples = 0;
  private frameSize = DEFAULT_FRAME_SIZE;
  private audioChunks: Float32Array[] = [];
  private audioChunkSamples = 0;
  private audioInputName?: string;
  private srInputName?: string;
  private stateOutputName?: string;
  private srTensor?: ort.Tensor;
  private inputMetadataByName?: Map<string, ort.InferenceSession.ValueMetadata>;
  private stateInputNames: string[] = [];
  private stateTensorsByName = new Map<string, ort.Tensor>();

  constructor() {
    super();
    this.on('error', (_error) => {});
  }

  subscribe(handler: (state: AudioState) => void) {
    this.on('data', handler);
    this.subscribers += 1;

    if (this.subscribers === 1) {
      void this.ensureStarted();
    }

    handler(this.getCurrentState());

    return () => {
      this.off('data', handler);
      this.subscribers = Math.max(0, this.subscribers - 1);
      if (this.subscribers === 0) {
        this.stop();
      }
    };
  }

  private getCurrentState(timestamp = Date.now()): AudioState {
    return {
      rms: this.lastRms,
      smoothedRms: this.smoothedRms,
      level: this.lastLevel,
      probability: this.lastProbability,
      isTalking: this.isTalking,
      timestamp,
      error: this.lastErrorMessage,
      permissionRequired: this.permissionRequired,
    };
  }

  private emitIfNeeded(force = false) {
    const now = Date.now();
    if (!force && !this.running) return;
    if (!force && now - this.lastEmit < UI_UPDATE_INTERVAL_MS) return;

    this.lastEmit = now;
    this.emit('data', this.getCurrentState(now));
  }

  private setErrorState(error: Error) {
    const message = error.message || String(error);
    const permissionRequired = /permission|not permitted|denied/i.test(message);

    this.permissionRequired = permissionRequired;
    this.lastErrorMessage = permissionRequired
      ? 'Microphone permission required. Enable it in System Settings > Privacy & Security > Microphone.'
      : message;

    this.emitIfNeeded(true);
  }

  private clearErrorState() {
    if (!this.lastErrorMessage && !this.permissionRequired) return;
    this.lastErrorMessage = null;
    this.permissionRequired = false;
  }

  private async ensureStarted() {
    if (this.running || this.starting) return;
    this.starting = true;

    try {
      if (!this.session) {
        const modelPath = await ensureModelPath();
        this.session = await ort.InferenceSession.create(modelPath);
      }
      if (this.session && !this.audioInputName) {
        this.configureSession();
      }

      if (this.subscribers === 0) {
        return;
      }

      if (!this.startMic()) {
        return;
      }
      this.startInferenceLoop();
      this.running = true;
    } catch (error) {
      const normalizedError = normalizeError(error);
      this.setErrorState(normalizedError);
      this.emit('error', normalizedError);
      this.stop();
    } finally {
      this.starting = false;
    }
  }

  private stop() {
    this.running = false;

    if (this.inferenceTimer) {
      clearInterval(this.inferenceTimer);
      this.inferenceTimer = undefined;
    }

    if (this.micStream) {
      this.micStream.off('data', this.handleAudioChunk);
      this.micStream.removeAllListeners('error');
      this.micStream.removeAllListeners('end');
    }

    if (this.micInstance) {
      try {
        this.micInstance.stop();
      } catch {
        // Best-effort cleanup.
      }
    }

    this.micInstance = undefined;
    this.micStream = undefined;
    this.inferenceInFlight = false;
    this.speechCandidateSince = null;
    this.isTalking = false;
    this.lastProbability = 0;
    this.sourceSampleRate = SAMPLE_RATE;
    this.sourceRateWindowStart = 0;
    this.sourceRateWindowSamples = 0;
    this.audioChunks = [];
    this.audioChunkSamples = 0;
    this.frameSize = DEFAULT_FRAME_SIZE;
    this.audioInputName = undefined;
    this.srInputName = undefined;
    this.stateOutputName = undefined;
    this.inputMetadataByName = undefined;
    this.stateInputNames = [];
    this.stateTensorsByName.clear();
    this.clearErrorState();
  }

  private startMic() {
    if (!commandExists.sync(MIC_COMMAND)) {
      const error = new Error(
        `Missing audio capture binary '${MIC_COMMAND}'. Install sox to enable microphone input.`,
      );
      this.setErrorState(error);
      this.emit('error', error);
      return false;
    }

    const micInstance = mic({
      rate: String(SAMPLE_RATE),
      channels: String(CHANNELS),
      bitwidth: String(BIT_WIDTH),
      encoding: ENCODING,
      endian: ENDIAN,
      fileType: 'raw',
      debug: false,
      exitOnSilence: 0,
    });

    const stream = micInstance.getAudioStream();

    stream.on('data', this.handleAudioChunk);
    stream.on('error', (error) => {
      const resolvedError =
        error instanceof Error ? error : new Error(String(error));
      this.setErrorState(resolvedError);
      this.emit('error', resolvedError);
    });

    this.micInstance = micInstance;
    this.micStream = stream;
    micInstance.start();
    return true;
  }

  private startInferenceLoop() {
    if (this.inferenceTimer) {
      clearInterval(this.inferenceTimer);
    }

    this.inferenceTimer = setInterval(() => {
      void this.runInferenceTick();
    }, INFERENCE_INTERVAL_MS);

    this.inferenceTimer.unref?.();
  }

  private handleAudioChunk = (buffer: Buffer) => {
    const rawSamples = toFloat32(buffer);
    const now = Date.now();

    if (this.sourceRateWindowStart === 0) {
      this.sourceRateWindowStart = now;
    }
    this.sourceRateWindowSamples += rawSamples.length;

    if (now - this.sourceRateWindowStart >= SOURCE_RATE_WINDOW_MS) {
      const elapsedSeconds = (now - this.sourceRateWindowStart) / 1000;
      if (elapsedSeconds > 0) {
        const estimatedRate = this.sourceRateWindowSamples / elapsedSeconds;
        this.sourceSampleRate =
          estimatedRate > SOURCE_RATE_THRESHOLD ? 48000 : SAMPLE_RATE;
      }
      this.sourceRateWindowStart = now;
      this.sourceRateWindowSamples = 0;
    }

    const downsampleFactor = Math.round(this.sourceSampleRate / SAMPLE_RATE);
    const samples =
      downsampleFactor > 1
        ? downsampleByFactor(rawSamples, downsampleFactor)
        : rawSamples;
    const rms = computeRms(samples);

    this.lastRms = rms;
    this.smoothedRms = RMS_ALPHA * rms + (1 - RMS_ALPHA) * this.smoothedRms;
    const gatedRms =
      this.smoothedRms > NOISE_FLOOR ? this.smoothedRms - NOISE_FLOOR : 0;
    this.lastLevel = clamp(gatedRms * RMS_SCALE, 0, 1);
    this.clearErrorState();

    this.enqueueSamples(samples);

    this.emitIfNeeded();
  };

  private configureSession() {
    if (!this.session) return;

    const { inputNames, outputNames, inputMetadata } = this.session;
    const metadataByName = new Map<
      string,
      ort.InferenceSession.ValueMetadata
    >();
    inputMetadata.forEach((metadata) => {
      metadataByName.set(metadata.name, metadata);
    });
    this.inputMetadataByName = metadataByName;

    const floatInputs = inputNames.filter((name) => {
      const metadata = metadataByName.get(name);
      return isTensorMetadata(metadata) && metadata.type === 'float32';
    });
    const intInputs = inputNames.filter((name) => {
      const metadata = metadataByName.get(name);
      if (!isTensorMetadata(metadata)) return false;
      const type = metadata.type;
      return type === 'int64' || type === 'int32';
    });

    this.audioInputName = inputNames.includes('input')
      ? 'input'
      : floatInputs[0];
    this.srInputName = inputNames.includes('sr') ? 'sr' : intInputs[0];

    if (!this.audioInputName) {
      throw new Error('Silero VAD model missing float32 audio input.');
    }

    this.stateInputNames = inputNames.filter(
      (name) => name !== this.audioInputName && name !== this.srInputName,
    );

    this.stateTensorsByName.clear();
    for (const name of this.stateInputNames) {
      const metadata = metadataByName.get(name);
      if (metadata && isTensorMetadata(metadata)) {
        const zeroTensor = createZeroTensor(metadata);
        if (zeroTensor) {
          this.stateTensorsByName.set(name, zeroTensor);
        }
      }
    }

    this.stateOutputName = outputNames.includes('state') ? 'state' : undefined;

    const audioMetadata = metadataByName.get(this.audioInputName);
    if (audioMetadata && isTensorMetadata(audioMetadata)) {
      const numericDims = audioMetadata.shape.filter(
        (dim): dim is number => typeof dim === 'number' && dim > 0,
      );
      const resolved =
        numericDims.length > 0
          ? numericDims[numericDims.length - 1]
          : undefined;
      this.frameSize = resolved ?? DEFAULT_FRAME_SIZE;
    } else {
      this.frameSize = DEFAULT_FRAME_SIZE;
    }
  }

  private getSampleRateTensor() {
    if (this.srTensor) return this.srTensor;

    const metadata =
      this.srInputName && this.inputMetadataByName
        ? this.inputMetadataByName.get(this.srInputName)
        : undefined;
    const type =
      metadata && isTensorMetadata(metadata) ? metadata.type : 'int64';

    if (type === 'int32') {
      this.srTensor = new ort.Tensor('int32', Int32Array.from([SAMPLE_RATE]), [
        1,
      ]);
    } else {
      this.srTensor = new ort.Tensor(
        'int64',
        BigInt64Array.from([BigInt(SAMPLE_RATE)]),
        [1],
      );
    }

    return this.srTensor;
  }

  private buildFeeds(audio: Float32Array): ort.InferenceSession.FeedsType {
    if (!this.session || !this.audioInputName) return {};

    const feeds: Record<string, ort.OnnxValue> = {};
    feeds[this.audioInputName] = new ort.Tensor('float32', audio, [
      1,
      audio.length,
    ]);

    if (this.srInputName) {
      feeds[this.srInputName] = this.getSampleRateTensor();
    }

    if (this.stateInputNames.length > 0) {
      for (const name of this.stateInputNames) {
        let tensor = this.stateTensorsByName.get(name);
        if (!tensor && this.inputMetadataByName) {
          const metadata = this.inputMetadataByName.get(name);
          if (metadata && isTensorMetadata(metadata)) {
            tensor = createZeroTensor(metadata);
            if (tensor) {
              this.stateTensorsByName.set(name, tensor);
            }
          }
        }
        if (tensor) {
          feeds[name] = tensor;
        }
      }
    }

    return feeds as ort.InferenceSession.FeedsType;
  }

  private getStateOutputNames() {
    const names = new Set<string>();
    if (this.stateOutputName) {
      names.add(this.stateOutputName);
    }
    for (const name of this.stateInputNames) {
      names.add(name);
      names.add(`${name}n`);
    }
    return names;
  }

  private extractProbability(outputs: SessionOutputs) {
    const preferredNames = ['output', 'prob', 'probability'];

    for (const name of preferredNames) {
      const tensor = outputs[name];
      if (tensor instanceof ort.Tensor) {
        const data = tensor.data;
        if (data && data.length > 0) {
          const first = data[0];
          if (typeof first === 'number' || typeof first === 'bigint') {
            return Number(first);
          }
        }
      }
    }

    const excluded = this.getStateOutputNames();
    const fallbackName = Object.keys(outputs).find(
      (name) => !excluded.has(name),
    );
    if (!fallbackName) return 0;

    const fallbackTensor = outputs[fallbackName];
    if (!(fallbackTensor instanceof ort.Tensor)) return 0;

    const fallbackData = fallbackTensor.data;
    if (!fallbackData || fallbackData.length === 0) return 0;

    const first = fallbackData[0];
    if (typeof first === 'number' || typeof first === 'bigint') {
      return Number(first);
    }
    return 0;
  }

  private updateTalkingState(probability: number) {
    const now = Date.now();

    if (probability > SPEECH_THRESHOLD) {
      if (this.speechCandidateSince === null) {
        this.speechCandidateSince = now;
      }

      if (now - this.speechCandidateSince >= TALKING_HOLD_MS) {
        this.isTalking = true;
      }
    } else {
      this.speechCandidateSince = null;
      this.isTalking = false;
    }
  }

  private updateState(outputs: SessionOutputs) {
    if (this.stateInputNames.length === 0) return;

    for (const name of this.stateInputNames) {
      const direct = outputs[name];
      const prefixed = outputs[`new_${name}`];
      const fallback = outputs[`${name}n`];
      const nextState = prefixed ?? direct ?? fallback;
      if (nextState instanceof ort.Tensor) {
        this.stateTensorsByName.set(name, nextState);
      }
    }
  }

  private async runInferenceTick() {
    if (!this.session || this.inferenceInFlight) return;
    const frame = this.dequeueFrame();
    if (!frame) return;

    this.inferenceInFlight = true;

    try {
      const outputs = await this.session.run(this.buildFeeds(frame));

      if (!this.running) return;

      this.clearErrorState();
      const probability = this.extractProbability(outputs);
      this.lastProbability = probability;
      this.updateTalkingState(probability);
      this.updateState(outputs);
      this.emitIfNeeded(true);
    } catch (error) {
      const normalizedError = normalizeError(error);
      this.setErrorState(normalizedError);
      this.emit('error', normalizedError);
    } finally {
      this.inferenceInFlight = false;
    }
  }

  private enqueueSamples(samples: Float32Array) {
    if (samples.length === 0) return;
    this.audioChunks.push(samples);
    this.audioChunkSamples += samples.length;

    const maxSamples = this.frameSize * MAX_BUFFER_FRAMES;
    while (this.audioChunkSamples > maxSamples && this.audioChunks.length > 0) {
      const chunk = this.audioChunks.shift();
      if (!chunk) break;
      this.audioChunkSamples -= chunk.length;
    }
  }

  private dequeueFrame() {
    if (this.frameSize <= 0) return null;
    if (this.audioChunkSamples < this.frameSize) return null;

    const frame = new Float32Array(this.frameSize);
    let offset = 0;

    while (offset < this.frameSize && this.audioChunks.length > 0) {
      const chunk = this.audioChunks[0];
      const remaining = this.frameSize - offset;

      if (chunk.length <= remaining) {
        frame.set(chunk, offset);
        offset += chunk.length;
        this.audioChunks.shift();
        this.audioChunkSamples -= chunk.length;
      } else {
        frame.set(chunk.subarray(0, remaining), offset);
        this.audioChunks[0] = chunk.subarray(remaining);
        this.audioChunkSamples -= remaining;
        offset += remaining;
      }
    }

    return frame;
  }
}

export const audioEngine = new AudioEngine();
