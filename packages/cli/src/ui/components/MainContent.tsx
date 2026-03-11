/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Box, Static } from 'ink';
import { HistoryItemDisplay } from './HistoryItemDisplay.js';
import { useUIState } from '../contexts/UIStateContext.js';
import { useAppContext } from '../contexts/AppContext.js';
import { AppHeader } from './AppHeader.js';
import { useAlternateBuffer } from '../hooks/useAlternateBuffer.js';
import {
  SCROLL_TO_ITEM_END,
  type VirtualizedListRef,
} from './shared/VirtualizedList.js';
import { ScrollableList } from './shared/ScrollableList.js';
import { useMemo, memo, useCallback, useEffect, useRef } from 'react';
import { MAX_GEMINI_MESSAGE_LINES } from '../constants.js';
import { useConfirmingTool } from '../hooks/useConfirmingTool.js';
import { ToolConfirmationQueue } from './ToolConfirmationQueue.js';
import { useOptionalVoiceAssistant } from '../contexts/VoiceAssistantContext.js';
import type { HistoryItem } from '../types.js';

const MemoizedHistoryItemDisplay = memo(HistoryItemDisplay);
const MemoizedAppHeader = memo(AppHeader);

// Limit Gemini messages to a very high number of lines to mitigate performance
// issues in the worst case if we somehow get an enormous response from Gemini.
// This threshold is arbitrary but should be high enough to never impact normal
// usage.
export const MainContent = () => {
  const { version } = useAppContext();
  const uiState = useUIState();
  const isAlternateBuffer = useAlternateBuffer();
  const voiceAssistant = useOptionalVoiceAssistant();

  const confirmingTool = useConfirmingTool();
  const showConfirmationQueue = confirmingTool !== null;
  const confirmingToolCallId = confirmingTool?.tool.callId;
  const liveVoiceAssistantOutput =
    voiceAssistant?.enabled && voiceAssistant.outputTranscript.trim().length > 0
      ? voiceAssistant.outputTranscript.trim()
      : null;

  const scrollableListRef = useRef<VirtualizedListRef<unknown>>(null);

  useEffect(() => {
    if (showConfirmationQueue) {
      scrollableListRef.current?.scrollToEnd();
    }
  }, [showConfirmationQueue, confirmingToolCallId]);

  const {
    pendingHistoryItems,
    mainAreaWidth,
    staticAreaMaxItemHeight,
    cleanUiDetailsVisible,
  } = uiState;
  const showHeaderDetails = cleanUiDetailsVisible;

  const combinedHistoryItems = useMemo<HistoryItem[]>(() => {
    const voiceOutputs = voiceAssistant?.outputHistory ?? [];
    if (voiceOutputs.length === 0) {
      return uiState.history;
    }

    const mergedHistory: HistoryItem[] = [];
    const insertedVoiceAnchors = new Set<number>();
    const voiceOutputsByAnchor = new Map<
      number,
      {
        id: number;
        text: string;
      }
    >();

    for (const output of voiceOutputs) {
      const trimmedText = output.text.trim();
      if (!trimmedText) {
        continue;
      }

      const existing = voiceOutputsByAnchor.get(output.anchorHistoryId);
      if (!existing) {
        voiceOutputsByAnchor.set(output.anchorHistoryId, {
          id: output.id,
          text: trimmedText,
        });
        continue;
      }

      existing.text = `${existing.text}\n${trimmedText}`.trim();
    }

    const appendVoiceOutputs = (anchorHistoryId: number) => {
      const anchoredOutput = voiceOutputsByAnchor.get(anchorHistoryId);
      if (!anchoredOutput) {
        return;
      }

      insertedVoiceAnchors.add(anchorHistoryId);
      mergedHistory.push({
        id: -anchoredOutput.id,
        type: 'gemini',
        text: anchoredOutput.text,
      });
    };

    appendVoiceOutputs(0);
    for (const item of uiState.history) {
      mergedHistory.push(item);
      appendVoiceOutputs(item.id);
    }

    for (const [anchorHistoryId, output] of voiceOutputsByAnchor.entries()) {
      if (insertedVoiceAnchors.has(anchorHistoryId)) {
        continue;
      }
      mergedHistory.push({
        id: -output.id,
        type: 'gemini',
        text: output.text,
      });
    }

    return mergedHistory;
  }, [uiState.history, voiceAssistant?.outputHistory]);

  const lastUserPromptIndex = useMemo(() => {
    for (let i = combinedHistoryItems.length - 1; i >= 0; i--) {
      const type = combinedHistoryItems[i].type;
      if (type === 'user' || type === 'user_shell') {
        return i;
      }
    }
    return -1;
  }, [combinedHistoryItems]);

  const historyItems = useMemo(
    () =>
      combinedHistoryItems.map((h, index) => {
        const isExpandable = index > lastUserPromptIndex;
        return (
          <MemoizedHistoryItemDisplay
            terminalWidth={mainAreaWidth}
            availableTerminalHeight={
              uiState.constrainHeight || !isExpandable
                ? staticAreaMaxItemHeight
                : undefined
            }
            availableTerminalHeightGemini={MAX_GEMINI_MESSAGE_LINES}
            key={h.id}
            item={h}
            isPending={false}
            commands={uiState.slashCommands}
            isExpandable={isExpandable}
          />
        );
      }),
    [
      combinedHistoryItems,
      mainAreaWidth,
      staticAreaMaxItemHeight,
      uiState.slashCommands,
      uiState.constrainHeight,
      lastUserPromptIndex,
    ],
  );

  const staticHistoryItems = useMemo(
    () => historyItems.slice(0, lastUserPromptIndex + 1),
    [historyItems, lastUserPromptIndex],
  );

  const lastResponseHistoryItems = useMemo(
    () => historyItems.slice(lastUserPromptIndex + 1),
    [historyItems, lastUserPromptIndex],
  );

  const pendingItems = useMemo(
    () => (
      <Box flexDirection="column">
        {pendingHistoryItems.map((item, i) => (
          <HistoryItemDisplay
            key={i}
            availableTerminalHeight={
              uiState.constrainHeight ? staticAreaMaxItemHeight : undefined
            }
            terminalWidth={mainAreaWidth}
            item={{ ...item, id: 0 }}
            isPending={true}
            isExpandable={true}
          />
        ))}
        {showConfirmationQueue && confirmingTool && (
          <ToolConfirmationQueue confirmingTool={confirmingTool} />
        )}
        {liveVoiceAssistantOutput && (
          <HistoryItemDisplay
            key="voice-assistant-live-output"
            availableTerminalHeight={
              uiState.constrainHeight ? staticAreaMaxItemHeight : undefined
            }
            terminalWidth={mainAreaWidth}
            item={{ id: 0, type: 'gemini', text: liveVoiceAssistantOutput }}
            isPending={false}
            isExpandable={true}
          />
        )}
      </Box>
    ),
    [
      pendingHistoryItems,
      uiState.constrainHeight,
      staticAreaMaxItemHeight,
      mainAreaWidth,
      showConfirmationQueue,
      confirmingTool,
      liveVoiceAssistantOutput,
    ],
  );

  const virtualizedData = useMemo(
    () => [
      { type: 'header' as const },
      ...combinedHistoryItems.map((item, index) => ({
        type: 'history' as const,
        item,
        isExpandable: index > lastUserPromptIndex,
      })),
      { type: 'pending' as const },
    ],
    [combinedHistoryItems, lastUserPromptIndex],
  );

  const renderItem = useCallback(
    ({ item }: { item: (typeof virtualizedData)[number] }) => {
      if (item.type === 'header') {
        return (
          <MemoizedAppHeader
            key="app-header"
            version={version}
            showDetails={showHeaderDetails}
          />
        );
      } else if (item.type === 'history') {
        return (
          <MemoizedHistoryItemDisplay
            terminalWidth={mainAreaWidth}
            availableTerminalHeight={
              uiState.constrainHeight || !item.isExpandable
                ? staticAreaMaxItemHeight
                : undefined
            }
            availableTerminalHeightGemini={MAX_GEMINI_MESSAGE_LINES}
            key={item.item.id}
            item={item.item}
            isPending={false}
            commands={uiState.slashCommands}
            isExpandable={item.isExpandable}
          />
        );
      } else {
        return pendingItems;
      }
    },
    [
      showHeaderDetails,
      version,
      mainAreaWidth,
      uiState.slashCommands,
      pendingItems,
      uiState.constrainHeight,
      staticAreaMaxItemHeight,
    ],
  );

  if (isAlternateBuffer) {
    return (
      <ScrollableList
        ref={scrollableListRef}
        hasFocus={!uiState.isEditorDialogOpen && !uiState.embeddedShellFocused}
        width={uiState.terminalWidth}
        data={virtualizedData}
        renderItem={renderItem}
        estimatedItemHeight={() => 100}
        keyExtractor={(item, _index) => {
          if (item.type === 'header') return 'header';
          if (item.type === 'history') return item.item.id.toString();
          return 'pending';
        }}
        initialScrollIndex={SCROLL_TO_ITEM_END}
        initialScrollOffsetInIndex={SCROLL_TO_ITEM_END}
      />
    );
  }

  return (
    <>
      <Static
        key={uiState.historyRemountKey}
        items={[
          <AppHeader key="app-header" version={version} />,
          ...staticHistoryItems,
          ...lastResponseHistoryItems,
        ]}
      >
        {(item) => item}
      </Static>
      {pendingItems}
    </>
  );
};
