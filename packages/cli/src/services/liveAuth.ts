/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  AuthType,
  getCodeAssistServer,
  getOauthClient,
  type Config,
} from '@google/gemini-cli-core';
import { GoogleGenAI, type GoogleGenAIOptions } from '@google/genai';

const DEFAULT_API_VERSION = process.env['GOOGLE_GENAI_API_VERSION'];
const DEFAULT_VERTEX_LOCATION = 'global';

function resolveProject(config: Config) {
  const envProject =
    process.env['GOOGLE_CLOUD_PROJECT'] ||
    process.env['GOOGLE_CLOUD_PROJECT_ID'] ||
    undefined;
  if (envProject) {
    return envProject;
  }

  return getCodeAssistServer(config)?.projectId;
}

function resolveLocation() {
  return process.env['GOOGLE_CLOUD_LOCATION'] || DEFAULT_VERTEX_LOCATION;
}

function resolveApiVersion() {
  return DEFAULT_API_VERSION || undefined;
}

export async function createLiveGoogleGenAI(
  config: Config,
): Promise<GoogleGenAI> {
  const contentConfig = config.getContentGeneratorConfig();
  const authType = contentConfig.authType;
  const project = resolveProject(config);
  const location = resolveLocation();
  const apiVersion = resolveApiVersion();

  const baseOptions: GoogleGenAIOptions = {
    ...(apiVersion ? { apiVersion } : {}),
  };

  if (authType === AuthType.USE_GEMINI) {
    const apiKey =
      contentConfig.apiKey ||
      process.env['GEMINI_API_KEY'] ||
      process.env['GOOGLE_API_KEY'];
    if (!apiKey) {
      throw new Error(
        'Voice mode requires GEMINI_API_KEY when using Gemini API auth.',
      );
    }

    return new GoogleGenAI({
      ...baseOptions,
      apiKey,
    });
  }

  if (authType === AuthType.USE_VERTEX_AI) {
    if (contentConfig.apiKey) {
      return new GoogleGenAI({
        ...baseOptions,
        vertexai: true,
        apiKey: contentConfig.apiKey,
      });
    }

    if (!project) {
      throw new Error(
        'Voice mode requires a Google Cloud project for Vertex AI (set GOOGLE_CLOUD_PROJECT).',
      );
    }

    return new GoogleGenAI({
      ...baseOptions,
      vertexai: true,
      project,
      location,
    });
  }

  if (
    authType === AuthType.LOGIN_WITH_GOOGLE ||
    authType === AuthType.COMPUTE_ADC
  ) {
    if (!project) {
      throw new Error(
        'Voice mode with Google auth requires a Google Cloud project.',
      );
    }

    if (authType === AuthType.LOGIN_WITH_GOOGLE) {
      const authClient = await getOauthClient(authType, config);
      return new GoogleGenAI({
        ...baseOptions,
        vertexai: true,
        project,
        location,
        googleAuthOptions: {
          authClient,
        },
      });
    }

    return new GoogleGenAI({
      ...baseOptions,
      vertexai: true,
      project,
      location,
    });
  }

  throw new Error(
    `Voice mode is not supported for auth type: ${String(authType)}`,
  );
}
