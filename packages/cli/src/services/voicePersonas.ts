/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

export type GrammaticalGender = 'feminine' | 'masculine' | 'neutral';

export interface VoicePersonaDefinition {
  name: string;
  style: string;
  description: string;
  grammaticalGender: GrammaticalGender;
}

const VOICE_PERSONAS: readonly VoicePersonaDefinition[] = [
  {
    name: 'Zephyr',
    style: 'Bright',
    description: 'Light, quick, and energetic for lively back-and-forth.',
    grammaticalGender: 'feminine',
  },
  {
    name: 'Kore',
    style: 'Firm',
    description: 'Grounded and direct for decisive, no-nonsense replies.',
    grammaticalGender: 'feminine',
  },
  {
    name: 'Orus',
    style: 'Firm',
    description: 'Steady and authoritative with calm confidence.',
    grammaticalGender: 'masculine',
  },
  {
    name: 'Autonoe',
    style: 'Bright',
    description: 'Crisp and upbeat with polished momentum.',
    grammaticalGender: 'feminine',
  },
  {
    name: 'Umbriel',
    style: 'Easy-going',
    description: 'Relaxed and conversational for casual guidance.',
    grammaticalGender: 'masculine',
  },
  {
    name: 'Erinome',
    style: 'Clear',
    description: 'Clear and composed for straightforward explanations.',
    grammaticalGender: 'feminine',
  },
  {
    name: 'Laomedeia',
    style: 'Upbeat',
    description: 'Cheerful and animated with forward energy.',
    grammaticalGender: 'feminine',
  },
  {
    name: 'Schedar',
    style: 'Even',
    description: 'Balanced and measured for calm delivery.',
    grammaticalGender: 'masculine',
  },
  {
    name: 'Achird',
    style: 'Friendly',
    description: 'Warm and approachable for collaborative chats.',
    grammaticalGender: 'masculine',
  },
  {
    name: 'Sadachbia',
    style: 'Lively',
    description: 'Expressive and lively with extra spark.',
    grammaticalGender: 'masculine',
  },
  {
    name: 'Puck',
    style: 'Upbeat',
    description: 'Playful and upbeat for quick energetic exchanges.',
    grammaticalGender: 'masculine',
  },
  {
    name: 'Fenrir',
    style: 'Excitable',
    description: 'High-energy and animated when you want urgency.',
    grammaticalGender: 'masculine',
  },
  {
    name: 'Aoede',
    style: 'Breezy',
    description: 'Airy and breezy with an easy natural cadence.',
    grammaticalGender: 'feminine',
  },
  {
    name: 'Enceladus',
    style: 'Breathy',
    description: 'Soft and breathy with a subtle dramatic edge.',
    grammaticalGender: 'masculine',
  },
  {
    name: 'Algieba',
    style: 'Smooth',
    description: 'Polished and smooth for refined conversation.',
    grammaticalGender: 'masculine',
  },
  {
    name: 'Algenib',
    style: 'Gravelly',
    description: 'Textured and gravelly with extra weight.',
    grammaticalGender: 'masculine',
  },
  {
    name: 'Achernar',
    style: 'Soft',
    description: 'Gentle and soft for calm reassuring speech.',
    grammaticalGender: 'feminine',
  },
  {
    name: 'Gacrux',
    style: 'Mature',
    description: 'Mature and steady with seasoned warmth.',
    grammaticalGender: 'feminine',
  },
  {
    name: 'Zubenelgenubi',
    style: 'Casual',
    description: 'Casual and laid-back for informal chats.',
    grammaticalGender: 'masculine',
  },
  {
    name: 'Sadaltager',
    style: 'Knowledgeable',
    description: 'Smart and composed for thoughtful guidance.',
    grammaticalGender: 'masculine',
  },
  {
    name: 'Charon',
    style: 'Informative',
    description: 'Clear and informative for explanatory replies.',
    grammaticalGender: 'masculine',
  },
  {
    name: 'Leda',
    style: 'Youthful',
    description: 'Youthful and bright with fresh energy.',
    grammaticalGender: 'feminine',
  },
  {
    name: 'Callirrhoe',
    style: 'Easy-going',
    description: 'Relaxed and easy-going with friendly warmth.',
    grammaticalGender: 'feminine',
  },
  {
    name: 'Iapetus',
    style: 'Clear',
    description: 'Clean and articulate for precise answers.',
    grammaticalGender: 'masculine',
  },
  {
    name: 'Despina',
    style: 'Smooth',
    description: 'Silky and smooth with a graceful tone.',
    grammaticalGender: 'feminine',
  },
  {
    name: 'Rasalgethi',
    style: 'Informative',
    description: 'Assured and informative for detailed guidance.',
    grammaticalGender: 'masculine',
  },
  {
    name: 'Alnilam',
    style: 'Firm',
    description: 'Firm and controlled for no-nonsense delivery.',
    grammaticalGender: 'masculine',
  },
  {
    name: 'Pulcherrima',
    style: 'Forward',
    description: 'Bold and forward for proactive conversations.',
    grammaticalGender: 'feminine',
  },
  {
    name: 'Vindemiatrix',
    style: 'Gentle',
    description: 'Gentle and calm with a soft touch.',
    grammaticalGender: 'feminine',
  },
  {
    name: 'Sulafat',
    style: 'Warm',
    description: 'Warm and welcoming for empathetic delivery.',
    grammaticalGender: 'feminine',
  },
] as const;

const PERSONA_INDEX = new Map(
  VOICE_PERSONAS.map((persona) => [persona.name.toLowerCase(), persona]),
);

export function getVoicePersonas(): readonly VoicePersonaDefinition[] {
  return VOICE_PERSONAS;
}

export function getVoicePersonaNames(): string[] {
  return VOICE_PERSONAS.map((persona) => persona.name);
}

export function getVoicePersonaByName(
  name: string | null | undefined,
): VoicePersonaDefinition | undefined {
  if (!name) {
    return undefined;
  }
  const normalized = name.trim().toLowerCase();
  if (!normalized) {
    return undefined;
  }
  return PERSONA_INDEX.get(normalized);
}

export function formatVoicePersona(persona: VoicePersonaDefinition): string {
  return `${persona.name} (${persona.style}, ${persona.grammaticalGender} grammar)`;
}

export function formatVoicePersonaWithDescription(
  persona: VoicePersonaDefinition,
): string {
  return `${persona.description} ${formatVoicePersona(persona)}`;
}

export function buildVoiceAssistantSystemInstruction(
  persona: VoicePersonaDefinition | null | undefined,
): string {
  const lines = [
    'You are Gemini CLI Voice Assistant, a concise personal assistant controlling a coding agent.',
    'Hard rules:',
    '- Sound like a friendly, confident coding partner and keep replies concise.',
    '- Use natural conversational phrasing with contractions and varied wording; avoid robotic status narration.',
    '- Keep the vibe human and lightly playful while staying professional and clear.',
    '- Avoid repeating the same acknowledgement opener across turns.',
    '- Respond in plain text using one to two short sentences unless you need one clarification question.',
    '- Never narrate internal reasoning, tool usage, or phrases like "initiating analysis".',
    '- Do not use markdown headings or status banners.',
    '- Do not dump long raw file-path lists; summarize counts and mention at most two examples.',
    '- If paths must be spoken, mention key path segments naturally and avoid reading slash characters one by one.',
    '- If the transcript sounds partial, wait for additional speech instead of immediately asking for restatement.',
    '- Never invent workspace facts, file names, git counts, paths, or command outputs.',
    '- Default to submit_user_request for user work requests so the coding agent does the real execution.',
    '- Use query_local_context only for lightweight read-only checks when delegation is unnecessary.',
    '- For status, approvals, permissions, or control, use tools instead of guessing.',
    '- If a request needs a tool, call the tool first and wait for the result before speaking.',
    '- Do not speak filler like "sure," "on it," or "let me check" before submit_user_request, query_local_context, or resolve_pending_action.',
    '- When a controller turn starts with "Notification:", only restate that notification for the user and then stop.',
    '- Notification turns are not user requests: never call tools, revisit approvals, or continue a previous task during a notification turn.',
    '- For git status, changed files, or uncommitted file questions, delegate via submit_user_request.',
    '- Support speech in any language or dialect supported by the model, including European and Asian languages.',
    "- Reply in the language or dialect of the user's most recent utterance unless they explicitly ask for a different language.",
    '- If the user switches languages mid-conversation, switch immediately on the next reply and continue in that language until they switch again.',
    '- Infer the spoken language from likely words and intent, not script alone. Automatic transcription may render English or other Latin-script speech phonetically in Devanagari or another script.',
    '- If a non-Latin transcript is clearly phonetic English or another transliterated foreign-language utterance, answer in that spoken language instead of the script language.',
    '- For mixed-language turns, answer in the dominant language of the latest utterance unless the user explicitly asks for translation or another language.',
    '- Translate user intent internally to clear English before delegating to coding tools.',
    '- If the meaning is ambiguous, ask one short clarification question before taking action.',
    '- Never auto-approve actions without an explicit user decision.',
    '- If the user says "stop" ambiguously, ask whether they mean stop listening, stop the active run, or both.',
    '- For mid-run changes while the agent is working, submit a steering hint so work continues.',
  ];

  if (!persona) {
    lines.push(
      '- If the language uses grammatical gender or agreement, prefer neutral phrasing when natural and keep self-reference consistent.',
    );
    return lines.join('\n');
  }

  lines.push(`- Selected voice persona: ${persona.name} (${persona.style}).`);
  lines.push(
    `- When speaking in languages with grammatical gender or agreement, use ${persona.grammaticalGender} self-reference and agreement that match this persona.`,
  );
  lines.push(
    '- Apply that gender consistently in Indic, Arabic, Romance, and other gendered languages without explaining the rule aloud.',
  );
  return lines.join('\n');
}
