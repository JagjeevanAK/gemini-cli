/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

function normalizeTranscriptText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function normalizeTranscriptBlock(text: string): string {
  const normalizedLines = text
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .split('\n')
    .map((line) => line.replace(/[^\S\n]+/g, ' ').trim());

  while (normalizedLines.length > 0 && normalizedLines[0] === '') {
    normalizedLines.shift();
  }
  while (
    normalizedLines.length > 0 &&
    normalizedLines[normalizedLines.length - 1] === ''
  ) {
    normalizedLines.pop();
  }

  const compactedLines: string[] = [];
  let previousLineWasBlank = false;
  for (const line of normalizedLines) {
    if (!line) {
      if (!previousLineWasBlank && compactedLines.length > 0) {
        compactedLines.push('');
      }
      previousLineWasBlank = true;
      continue;
    }
    compactedLines.push(line);
    previousLineWasBlank = false;
  }

  return compactedLines.join('\n');
}

function normalizeInlineTranscriptChunk(text: string): string {
  const normalized = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  if (normalized.includes('\n')) {
    return normalizeTranscriptBlock(normalized);
  }
  return normalized.replace(/[^\S\n]+/g, ' ');
}

function limitTranscriptLines(text: string, maxLines: number): string {
  if (maxLines <= 0) {
    return '';
  }
  return text.split('\n').slice(-maxLines).join('\n');
}

function hasTrailingTranscriptBlock(
  previousTranscript: string,
  nextTranscript: string,
): boolean {
  const previousLines = previousTranscript.split('\n');
  const nextLines = nextTranscript.split('\n');
  if (nextLines.length > previousLines.length) {
    return false;
  }

  const trailingLines = previousLines.slice(-nextLines.length);
  return trailingLines.every((line, index) => {
    const nextLine = nextLines[index] ?? '';
    if (!line && !nextLine) {
      return true;
    }
    return normalizeTranscriptText(line) === normalizeTranscriptText(nextLine);
  });
}

export function mergeTranscriptChunk(base: string, chunk: string): string {
  if (!base) {
    return chunk;
  }
  if (!chunk) {
    return base;
  }

  const normalizedBase = base.toLowerCase();
  const normalizedChunk = chunk.toLowerCase();
  const maxOverlap = Math.min(
    normalizedBase.length,
    normalizedChunk.length,
    48,
  );

  for (let overlap = maxOverlap; overlap >= 1; overlap--) {
    if (
      normalizedBase.slice(normalizedBase.length - overlap) ===
      normalizedChunk.slice(0, overlap)
    ) {
      return base + chunk.slice(overlap);
    }
  }

  if (/\s$/.test(base) || /^\s/.test(chunk)) {
    return base + chunk;
  }

  if (/^[,.;:!?)}\]]/.test(chunk) || /[[({'"`/-]$/.test(base)) {
    return base + chunk;
  }

  const baseEndsWord = /[0-9A-Za-z\u00c0-\uffff]$/.test(base);
  const chunkStartsWord = /^[0-9A-Za-z\u00c0-\uffff]/.test(chunk);
  if (baseEndsWord && chunkStartsWord) {
    return `${base} ${chunk}`;
  }

  if (/^[A-Z]/.test(chunk)) {
    return `${base} ${chunk}`;
  }

  return base + chunk;
}

export function appendOutputTranscriptChunk(
  previousTranscript: string,
  nextChunk: string,
  startNewLine: boolean,
  maxLines: number,
): string {
  const normalizedPrevious = normalizeTranscriptBlock(previousTranscript);
  const normalizedNextChunk = normalizeTranscriptBlock(nextChunk);
  const normalizedInlineChunk = normalizeInlineTranscriptChunk(nextChunk);
  if (!normalizedNextChunk && !normalizedInlineChunk.trim()) {
    return normalizedPrevious;
  }

  if (!normalizedPrevious) {
    return limitTranscriptLines(
      normalizedNextChunk || normalizeTranscriptBlock(normalizedInlineChunk),
      maxLines,
    );
  }

  if (startNewLine) {
    if (hasTrailingTranscriptBlock(normalizedPrevious, normalizedNextChunk)) {
      return limitTranscriptLines(normalizedPrevious, maxLines);
    }

    return limitTranscriptLines(
      `${normalizedPrevious}\n${normalizedNextChunk}`,
      maxLines,
    );
  }

  const mergedTranscript = normalizeTranscriptBlock(
    mergeTranscriptChunk(normalizedPrevious, normalizedInlineChunk),
  );
  return limitTranscriptLines(
    mergedTranscript || normalizedNextChunk,
    maxLines,
  );
}
