/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { createContext, useContext } from 'react';
import type { VoiceAssistantControllerState } from '../hooks/useVoiceAssistantController.js';

export interface VoiceAssistantContextValue
  extends VoiceAssistantControllerState {
  enabled: boolean;
  toggle: () => void;
  disable: () => void;
  speak: (text: string) => boolean;
  lastOutputAt: number;
}

const VoiceAssistantContext = createContext<VoiceAssistantContextValue | null>(
  null,
);

export const useOptionalVoiceAssistant = () =>
  useContext(VoiceAssistantContext);

export const useVoiceAssistant = () => {
  const context = useOptionalVoiceAssistant();
  if (!context) {
    throw new Error(
      'useVoiceAssistant must be used within VoiceAssistantContext provider',
    );
  }
  return context;
};

export const VoiceAssistantContextProvider = VoiceAssistantContext.Provider;
