/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type React from 'react';
import {
  createContext,
  useContext,
  useCallback,
  useState,
  useEffect,
} from 'react';
import {
  IdeClient,
  ToolConfirmationOutcome,
  type Config,
  type ToolConfirmationPayload,
  debugLogger,
} from '@google/gemini-cli-core';
import type { IndividualToolCallDisplay } from '../types.js';
import { resolveToolConfirmation } from '../utils/toolConfirmationResolver.js';

interface ToolActionsContextValue {
  confirm: (
    callId: string,
    outcome: ToolConfirmationOutcome,
    payload?: ToolConfirmationPayload,
  ) => Promise<void>;
  cancel: (callId: string) => Promise<void>;
  isDiffingEnabled: boolean;
}

const ToolActionsContext = createContext<ToolActionsContextValue | null>(null);

export const useToolActions = () => {
  const context = useContext(ToolActionsContext);
  if (!context) {
    throw new Error('useToolActions must be used within a ToolActionsProvider');
  }
  return context;
};

interface ToolActionsProviderProps {
  children: React.ReactNode;
  config: Config;
  toolCalls: IndividualToolCallDisplay[];
}

export const ToolActionsProvider: React.FC<ToolActionsProviderProps> = (
  props: ToolActionsProviderProps,
) => {
  const { children, config, toolCalls } = props;

  // Hoist IdeClient logic here to keep UI pure
  const [ideClient, setIdeClient] = useState<IdeClient | null>(null);
  const [isDiffingEnabled, setIsDiffingEnabled] = useState(false);

  useEffect(() => {
    let isMounted = true;
    if (config.getIdeMode()) {
      IdeClient.getInstance()
        .then((client) => {
          if (!isMounted) return;
          setIdeClient(client);
          setIsDiffingEnabled(client.isDiffingEnabled());

          const handleStatusChange = () => {
            if (isMounted) {
              setIsDiffingEnabled(client.isDiffingEnabled());
            }
          };

          client.addStatusChangeListener(handleStatusChange);
          // Return a cleanup function for the listener
          return () => {
            client.removeStatusChangeListener(handleStatusChange);
          };
        })
        .catch((error) => {
          debugLogger.error('Failed to get IdeClient instance:', error);
        });
    }
    return () => {
      isMounted = false;
    };
  }, [config]);

  const confirm = useCallback(
    async (
      callId: string,
      outcome: ToolConfirmationOutcome,
      payload?: ToolConfirmationPayload,
    ) => {
      await resolveToolConfirmation({
        config,
        toolCalls,
        callId,
        outcome,
        payload,
        ideClient,
        isDiffingEnabled,
      });
    },
    [config, ideClient, toolCalls, isDiffingEnabled],
  );

  const cancel = useCallback(
    async (callId: string) => {
      await confirm(callId, ToolConfirmationOutcome.Cancel);
    },
    [confirm],
  );

  return (
    <ToolActionsContext.Provider value={{ confirm, cancel, isDiffingEnabled }}>
      {children}
    </ToolActionsContext.Provider>
  );
};
