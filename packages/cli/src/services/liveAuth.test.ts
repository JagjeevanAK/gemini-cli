/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AuthType, type Config } from '@google/gemini-cli-core';
import * as core from '@google/gemini-cli-core';
import { GoogleGenAI } from '@google/genai';
import { createLiveGoogleGenAI } from './liveAuth.js';

vi.mock('@google/genai');

describe('createLiveGoogleGenAI', () => {
  const mockGenAIInstance = {} as unknown as GoogleGenAI;

  beforeEach(() => {
    vi.clearAllMocks();
    vi.unstubAllEnvs();
    vi.mocked(GoogleGenAI).mockImplementation(() => mockGenAIInstance);
    vi.spyOn(core, 'getOauthClient').mockResolvedValue({} as never);
    vi.spyOn(core, 'getCodeAssistServer').mockReturnValue(undefined);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
  });

  it('reuses project from existing Google auth state and defaults location to global', async () => {
    const authClient = {} as never;
    vi.mocked(core.getOauthClient).mockResolvedValue(authClient);
    vi.mocked(core.getCodeAssistServer).mockReturnValue({
      projectId: 'code-assist-project',
    } as never);

    const config = {
      getContentGeneratorConfig: () => ({
        authType: AuthType.LOGIN_WITH_GOOGLE,
      }),
    } as unknown as Config;

    await createLiveGoogleGenAI(config);

    expect(GoogleGenAI).toHaveBeenCalledWith({
      vertexai: true,
      project: 'code-assist-project',
      location: 'global',
      googleAuthOptions: {
        authClient,
      },
    });
  });

  it('uses env project/location when present', async () => {
    vi.stubEnv('GOOGLE_CLOUD_PROJECT', 'env-project');
    vi.stubEnv('GOOGLE_CLOUD_LOCATION', 'us-central1');
    vi.mocked(core.getCodeAssistServer).mockReturnValue({
      projectId: 'code-assist-project',
    } as never);

    const config = {
      getContentGeneratorConfig: () => ({
        authType: AuthType.LOGIN_WITH_GOOGLE,
      }),
    } as unknown as Config;

    await createLiveGoogleGenAI(config);

    expect(GoogleGenAI).toHaveBeenCalledWith(
      expect.objectContaining({
        vertexai: true,
        project: 'env-project',
        location: 'us-central1',
      }),
    );
  });

  it('throws when project cannot be resolved for Google auth', async () => {
    const config = {
      getContentGeneratorConfig: () => ({
        authType: AuthType.LOGIN_WITH_GOOGLE,
      }),
    } as unknown as Config;

    await expect(createLiveGoogleGenAI(config)).rejects.toThrow(
      'Voice mode with Google auth requires a Google Cloud project.',
    );
  });

  it('defaults Vertex AI location to global when only project is set', async () => {
    vi.stubEnv('GOOGLE_CLOUD_PROJECT', 'vertex-project');
    const config = {
      getContentGeneratorConfig: () => ({
        authType: AuthType.USE_VERTEX_AI,
      }),
    } as unknown as Config;

    await createLiveGoogleGenAI(config);

    expect(GoogleGenAI).toHaveBeenCalledWith({
      vertexai: true,
      project: 'vertex-project',
      location: 'global',
    });
  });
});
