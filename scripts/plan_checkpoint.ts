#!/usr/bin/env npx tsx
/**
 * Plan Checkpoint Script
 *
 * Triggered by PreToolUse hook on AskUserQuestion and ExitPlanMode tools.
 * Sends partial transcript to Letta at these natural pause points so the
 * Subconscious agent can provide guidance before Claude proceeds.
 *
 * Environment Variables:
 *   LETTA_API_KEY - API key for Letta authentication
 *   LETTA_CHECKPOINT_MODE - Mode: 'blocking' (default), 'async', or 'off'
 *
 * Hook Input (via stdin):
 *   - session_id: Current session ID
 *   - transcript_path: Path to conversation JSONL file
 *   - tool_name: The tool being called (AskUserQuestion or ExitPlanMode)
 *   - tool_input: The tool's input parameters
 *   - cwd: Current working directory
 *
 * Exit Codes:
 *   0 - Success
 *   1 - Error (non-blocking)
 *
 * Log file: $TMPDIR/letta-claude-sync-$UID/plan_checkpoint.log
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { getAgentId } from './agent_config.js';
import {
  LETTA_API_BASE,
  loadSyncState,
  getOrCreateConversation,
  saveSyncState,
  spawnSilentWorker,
  getSyncStateFile,
  LogFn,
  getTempStateDir,
} from './conversation_utils.js';
import {
  readTranscript,
  formatMessagesForLetta,
  formatAsXmlTranscript,
} from './transcript_utils.js';

// ESM-compatible __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const TEMP_STATE_DIR = getTempStateDir();
const LOG_FILE = path.join(TEMP_STATE_DIR, 'plan_checkpoint.log');

type CheckpointMode = 'blocking' | 'async' | 'off';

interface HookInput {
  session_id: string;
  transcript_path: string;
  tool_name: string;
  tool_input: any;
  cwd: string;
}

interface HookOutput {
  hookSpecificOutput?: {
    hookEventName: string;
    additionalContext?: string;
  };
}

/**
 * Ensure temp log directory exists
 */
function ensureLogDir(): void {
  if (!fs.existsSync(TEMP_STATE_DIR)) {
    fs.mkdirSync(TEMP_STATE_DIR, { recursive: true });
  }
}

/**
 * Log message to file
 */
function log(message: string): void {
  ensureLogDir();
  const timestamp = new Date().toISOString();
  const logLine = `[${timestamp}] ${message}\n`;
  fs.appendFileSync(LOG_FILE, logLine);
}

/**
 * Get checkpoint mode from environment
 */
function getCheckpointMode(): CheckpointMode {
  const mode = process.env.LETTA_CHECKPOINT_MODE?.toLowerCase();
  if (mode === 'async' || mode === 'off') return mode;
  return 'blocking';
}

/**
 * Read hook input from stdin
 */
async function readHookInput(): Promise<HookInput> {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('readable', () => {
      let chunk;
      while ((chunk = process.stdin.read()) !== null) {
        data += chunk;
      }
    });
    process.stdin.on('end', () => {
      try {
        resolve(JSON.parse(data));
      } catch (e) {
        reject(new Error(`Failed to parse hook input: ${e}`));
      }
    });
    process.stdin.on('error', reject);
  });
}

/**
 * Format tool context for the checkpoint message
 */
function formatToolContext(toolName: string, toolInput: any): string {
  if (toolName === 'AskUserQuestion') {
    const questions = toolInput?.questions;
    if (Array.isArray(questions) && questions.length > 0) {
      const questionTexts = questions.map((q: any) => {
        let text = q.question || '';
        if (q.options && Array.isArray(q.options)) {
          const optionLabels = q.options.map((o: any) => o.label).join(', ');
          text += ` [Options: ${optionLabels}]`;
        }
        return text;
      }).join('\n');
      return `<current_tool name="AskUserQuestion">
Claude Code is about to ask the user:
${questionTexts}
</current_tool>`;
    }
  } else if (toolName === 'ExitPlanMode') {
    return `<current_tool name="ExitPlanMode">
Claude Code is finishing plan mode and requesting user approval to proceed with implementation.
</current_tool>`;
  }
  return '';
}

/**
 * Send message to Letta and wait for response (blocking mode)
 */
async function sendAndWaitForResponse(
  apiKey: string,
  conversationId: string,
  message: string,
  log: LogFn
): Promise<string | null> {
  const url = `${LETTA_API_BASE}/conversations/${conversationId}/messages`;

  log(`Sending blocking message to conversation ${conversationId}`);

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      messages: [{ role: 'user', content: message }],
    }),
  });

  if (response.status === 409) {
    log(`Conversation busy (409) - skipping checkpoint`);
    return null;
  }

  if (!response.ok) {
    const errorText = await response.text();
    log(`Error response: ${errorText}`);
    throw new Error(`Letta API error (${response.status}): ${errorText}`);
  }

  // Read the full streaming response and extract assistant message
  const reader = response.body?.getReader();
  if (!reader) {
    log(`No response body`);
    return null;
  }

  let fullResponse = '';
  const decoder = new TextDecoder();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      fullResponse += decoder.decode(value, { stream: true });
    }
  } finally {
    reader.releaseLock();
  }

  log(`Received response (${fullResponse.length} chars)`);

  // Parse SSE events to extract assistant message
  // Format: data: {"message_type": "assistant_message", "content": "..."}
  const lines = fullResponse.split('\n');
  let assistantContent = '';

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      try {
        const data = JSON.parse(line.substring(6));
        if (data.message_type === 'assistant_message' && data.content) {
          assistantContent += data.content;
        }
      } catch (e) {
        // Skip non-JSON lines
      }
    }
  }

  if (assistantContent) {
    log(`Extracted assistant message (${assistantContent.length} chars)`);
    return assistantContent;
  }

  log(`No assistant message found in response`);
  return null;
}

/**
 * Main function
 */
async function main(): Promise<void> {
  log('='.repeat(60));
  log('plan_checkpoint.ts started');

  const mode = getCheckpointMode();
  log(`Checkpoint mode: ${mode}`);

  if (mode === 'off') {
    log('Checkpoint mode is off, exiting');
    process.exit(0);
  }

  const apiKey = process.env.LETTA_API_KEY;

  if (!apiKey) {
    log('ERROR: LETTA_API_KEY not set');
    process.exit(0); // Exit silently - don't block Claude
  }

  try {
    // Get agent ID
    const agentId = await getAgentId(apiKey, log);
    log(`Using agent: ${agentId}`);

    // Read hook input
    log('Reading hook input from stdin...');
    const hookInput = await readHookInput();
    log(`Hook input received:`);
    log(`  session_id: ${hookInput.session_id}`);
    log(`  transcript_path: ${hookInput.transcript_path}`);
    log(`  tool_name: ${hookInput.tool_name}`);
    log(`  cwd: ${hookInput.cwd}`);

    // Read transcript
    log(`Reading transcript from: ${hookInput.transcript_path}`);
    const messages = await readTranscript(hookInput.transcript_path, log);
    log(`Found ${messages.length} messages in transcript`);

    if (messages.length === 0) {
      log('No messages found, exiting');
      process.exit(0);
    }

    // Load sync state (don't update lastProcessedIndex - let Stop hook do that)
    const state = loadSyncState(hookInput.cwd, hookInput.session_id, log);

    // Format new messages since last sync
    const newMessages = formatMessagesForLetta(messages, state.lastProcessedIndex, log);

    // Get or create conversation
    const conversationId = await getOrCreateConversation(
      apiKey,
      agentId,
      hookInput.session_id,
      hookInput.cwd,
      state,
      log
    );
    log(`Using conversation: ${conversationId}`);

    // Save state with conversation ID
    saveSyncState(hookInput.cwd, state, log);

    // Build checkpoint message
    const toolContext = formatToolContext(hookInput.tool_name, hookInput.tool_input);
    const transcriptXml = newMessages.length > 0 ? formatAsXmlTranscript(newMessages) : '';

    const checkpointMessage = `<claude_code_checkpoint>
<session_id>${hookInput.session_id}</session_id>
<checkpoint_type>${hookInput.tool_name}</checkpoint_type>

${toolContext}

${transcriptXml ? `<recent_transcript>\n${transcriptXml}\n</recent_transcript>` : ''}

<instructions>
Claude Code is at a checkpoint (${hookInput.tool_name}). This is a good moment to provide guidance if you have any.

Your response will be injected as additionalContext before Claude proceeds. Keep it brief and actionable.
If you have no guidance, you can respond with just "No guidance needed" or similar.
</instructions>
</claude_code_checkpoint>`;

    log(`Built checkpoint message (${checkpointMessage.length} chars)`);

    if (mode === 'blocking') {
      // Wait for Letta response and inject as additionalContext
      const assistantResponse = await sendAndWaitForResponse(
        apiKey,
        conversationId,
        checkpointMessage,
        log
      );

      if (assistantResponse) {
        const output: HookOutput = {
          hookSpecificOutput: {
            hookEventName: 'PreToolUse',
            additionalContext: `<letta_message checkpoint="${hookInput.tool_name}">\n${assistantResponse}\n</letta_message>`,
          },
        };
        console.log(JSON.stringify(output));
        log('Wrote additionalContext to stdout');
      } else {
        log('No response to inject');
      }
    } else {
      // Async mode: spawn worker and don't wait
      const payloadFile = path.join(TEMP_STATE_DIR, `checkpoint-${hookInput.session_id}-${Date.now()}.json`);
      const payload = {
        apiKey,
        conversationId,
        sessionId: hookInput.session_id,
        message: checkpointMessage,
        stateFile: getSyncStateFile(hookInput.cwd, hookInput.session_id),
        // Don't update lastProcessedIndex for checkpoints
        newLastProcessedIndex: null,
      };
      fs.writeFileSync(payloadFile, JSON.stringify(payload), 'utf-8');
      log(`Wrote payload to ${payloadFile}`);

      const workerScript = path.join(__dirname, 'send_worker.ts');
      const child = spawnSilentWorker(workerScript, payloadFile, hookInput.cwd);
      log(`Spawned background worker (PID: ${child.pid})`);
    }

    log('Checkpoint completed');

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    log(`ERROR: ${errorMessage}`);
    if (error instanceof Error && error.stack) {
      log(`Stack trace: ${error.stack}`);
    }
    // Don't exit with error code - don't block Claude
    process.exit(0);
  }
}

// Run main function
main();
