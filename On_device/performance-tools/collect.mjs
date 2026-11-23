import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import readline from 'node:readline';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

export function parseEvent(line) {
  const start = line.indexOf('{');
  if (start < 0) return null;
  try {
    const e = JSON.parse(line.slice(start));
    if (e.schema !== 1 || !e.session || !Number.isInteger(e.trial_id) || e.trial_id <= 0 ||
        typeof e.event !== 'string' || !/^\d+$/.test(e.t_ns)) return null;
    return e;
  } catch { return null; }
}
function difference(a, b) {
  if (a == null || b == null) return '';
  const d = BigInt(a) - BigInt(b);
  return Number(d) / 1e6;
}
export function summarize(events) {
  const trials = new Map();
  for (const e of [...events].sort((a, b) => BigInt(a.t_ns) < BigInt(b.t_ns) ? -1 : BigInt(a.t_ns) > BigInt(b.t_ns) ? 1 : 0)) {
    const key = `${e.session}/${e.trial_id}`;
    if (!trials.has(key)) trials.set(key, { events: [], session: e.session, trial_id: e.trial_id, expected: e.expected });
    trials.get(key).events.push(e);
  }
  const rows = [];
  for (const trial of trials.values()) {
    const list = trial.events;
    const start = list.find(e => e.event === 'trial_start');
    const hand = list.find(e => e.event === 'hand_seen');
    const end = list.find(e => e.event === 'manual_gesture_end');
    // Only a specified target can be scored as correct. Empty target leaves latency unscored.
    const recognized = trial.expected ? list.find(e => e.event === 'recognition_result' && e.accepted_threshold === true &&
      e.label?.toLowerCase() === trial.expected.toLowerCase()) : null;
    const requests = [...new Set(list.filter(e => e.event === 'sentence_submitted').map(e => e.request_id))];
    if (!requests.length) requests.push('');
    for (const request of requests) {
      const find = name => list.find(e => e.event === name && e.request_id === request);
      const submitted = find('sentence_submitted');
      const returned = find('sentence_returned');
      const generation = find('generation_status');
      const tts = find('tts_start_callback');
      const ttsError = find('tts_error') || find('tts_stopped') || find('tts_skipped');
      rows.push({
        session: trial.session, trial_id: trial.trial_id, expected: trial.expected, request_id: request,
        recognition_status: !trial.expected ? 'unscored_no_target' : recognized ? 'target_observed' : 'target_not_observed',
        recognized_label: recognized?.label || '',
        trial_start_ns: start?.t_ns || '', first_hand_frame_ns: hand?.frame_observed_ns || '',
        recognition_ns: recognized?.t_ns || '', manual_end_ns: end?.t_ns || '',
        manual_start_to_target_ms: difference(recognized?.t_ns, start?.t_ns),
        first_hand_frame_to_target_ms: difference(recognized?.t_ns, hand?.frame_observed_ns),
        manual_end_to_target_ms: difference(recognized?.t_ns, end?.t_ns),
        trigger: submitted?.trigger || '', glosses: submitted?.glosses || '',
        generation_status: generation?.status || (submitted ? 'incomplete_or_logs_missing' : 'not_requested'),
        submit_to_text_ms: difference(returned?.t_ns, submitted?.submitted_ns),
        submit_to_tts_callback_ms: difference(tts?.t_ns, submitted?.submitted_ns),
        manual_end_to_text_ms: difference(returned?.t_ns, end?.t_ns),
        tts_status: ttsError?.event || (tts ? 'started_callback' : 'not_observed'),
        text: returned?.text || '',
      });
    }
  }
  return rows;
}
export function toCsv(rows) {
  if (!rows.length) return '';
  const columns = Object.keys(rows[0]);
  const quote = x => '"' + String(x ?? '').replaceAll('"', '""') + '"';
  return '\uFEFF' + [columns.map(quote).join(','), ...rows.map(r => columns.map(c => quote(r[c])).join(','))].join('\r\n') + '\r\n';
}
export function summarizeWords(events) {
  const sorted = [...events].sort((a, b) => BigInt(a.t_ns) < BigInt(b.t_ns) ? -1 : BigInt(a.t_ns) > BigInt(b.t_ns) ? 1 : 0);
  return sorted.filter(e => e.event === 'word_started').map(start => {
    const same = e => e.session === start.session && e.trial_id === start.trial_id && e.episode_id === start.episode_id;
    const recognized = sorted.find(e => same(e) && e.event === 'word_recognized');
    const removed = sorted.find(e => same(e) && e.event === 'word_hand_removed');
    return {
      session: start.session, trial_id: start.trial_id, sentence_id: start.sentence_id,
      word_index: start.word_index, word_position: start.word_index === 1 ? 'first' : 'subsequent',
      episode_id: start.episode_id,
      first_hand_frame_ns: start.start_ns, first_hand_detected_ns: start.hand_seen_ns,
      recognition_ns: recognized?.t_ns || '', label: recognized?.label || '',
      confidence: recognized?.confidence ?? '',
      hand_detected_to_50pct_ms: difference(recognized?.t_ns, start.hand_seen_ns),
      first_hand_frame_to_50pct_ms: difference(recognized?.t_ns, start.start_ns),
      hand_removed_ns: removed?.removed_ns || '',
      status: recognized ? 'threshold_reached' : removed ? 'hand_removed_before_threshold' : 'incomplete_or_logs_missing',
    };
  });
}
export function summarizeSentences(events) {
  return events.filter(e => e.event === 'sentence_submitted').map(start => {
    const same = e => e.session === start.session && e.trial_id === start.trial_id && e.request_id === start.request_id;
    const find = event => events.find(e => same(e) && e.event === event);
    const generated = find('sentence_returned');
    const status = find('generation_status');
    const begun = find('generation_started');
    const tts = find('tts_start_callback');
    return {
      session: start.session, trial_id: start.trial_id, sentence_id: start.sentence_id,
      request_id: start.request_id, word_count: start.word_count, trigger: start.trigger,
      glosses: start.glosses, submitted_ns: start.submitted_ns,
      text_returned_ns: generated?.t_ns || '',
      submit_to_text_ms: difference(generated?.t_ns, start.submitted_ns),
      generation_call_ms: difference(generated?.t_ns, begun?.t_ns),
      submit_to_tts_callback_ms: difference(tts?.t_ns, start.submitted_ns),
      status: status?.status || 'incomplete_or_logs_missing', text: generated?.text || '',
    };
  });
}
async function main() {
  const args = process.argv.slice(2);
  const value = (name, fallback) => {
    const i = args.indexOf(name);
    return i < 0 ? fallback : args[i + 1];
  };
  if (args.includes('--help')) {
    console.log('node collect.mjs [--seconds 120] [--adb PATH] [--device SERIAL] [--package com.sign2speak.app.perf] [--out DIR] [--input saved-logcat.txt]');
    return;
  }
  const seconds = Number(value('--seconds', '120'));
  if (!Number.isFinite(seconds) || seconds <= 0) throw new Error('--seconds must be positive');
  const input = value('--input');
  const adb = value('--adb', 'adb');
  const serial = value('--device');
  const appPackage = value('--package', 'com.sign2speak.app.perf');
  const prefix = serial ? ['-s', serial] : [];
  let child;
  if (!input) {
    const devices = spawnSync(adb, ['devices'], { encoding: 'utf8', timeout: 10000, windowsHide: true });
    if (devices.error) throw devices.error;
    const connected = (devices.stdout || '').split(/\r?\n/).filter(l => /\tdevice$/.test(l)).map(l => l.split('\t')[0]);
    if ((serial && !connected.includes(serial)) || (!serial && connected.length !== 1)) {
      throw new Error('Exactly one authorized device is required, or pass --device SERIAL.\n' + devices.stdout);
    }
    const state = spawnSync(adb, [...prefix, 'shell', 'pidof', appPackage], { encoding: 'utf8', timeout: 10000, windowsHide: true });
    if (!state.stdout?.trim()) console.warn('Sign2Speak is not running. Open it on your phone during capture.');
    // -T 1 starts near the present; never clears the device log buffer.
    child = spawn(adb, [...prefix, 'logcat', '-T', '1', '-v', 'threadtime', 'S2S_PERF:I', 'Gloss2TextEngine:I', 'SignRecognizer:I', 'CameraManager:I', 'TtsEngine:I', '*:S'], { windowsHide: true });
  }
  const out = path.resolve(value('--out', path.join(os.homedir(), 'Sign2SpeakTiming', 'results', new Date().toISOString().replaceAll(':', '-'))));
  if (input && path.resolve(input) === path.join(out, 'raw-logcat.txt')) {
    if (child) child.kill();
    throw new Error('Input log and output raw-logcat.txt must be different paths. Choose a new --out folder.');
  }
  fs.mkdirSync(out, { recursive: true });
  const raw = fs.createWriteStream(path.join(out, 'raw-logcat.txt'));
  const eventFile = fs.createWriteStream(path.join(out, 'events.ndjson'));
  const events = [];
  let lineCount = 0;
  let finished = false;
  let timer;
  const stream = input ? fs.createReadStream(path.resolve(input)) : child.stdout;
  const lines = readline.createInterface({ input: stream, crlfDelay: Infinity });
  const finish = () => {
    if (finished) return;
    finished = true;
    clearTimeout(timer);
    if (child && !child.killed) child.kill();
    lines.close();
    raw.end(); eventFile.end();
    const rows = summarize(events);
    const words = summarizeWords(events);
    const sentences = summarizeSentences(events);
    const notesPath = path.join(out, 'run-notes.json');
    const notes = fs.existsSync(notesPath) ? JSON.parse(fs.readFileSync(notesPath, 'utf8')) : null;
    const annotated = data => data.map(row => ({ ...row,
      include_in_formal_results: notes?.include_in_formal_results ?? 'unclassified',
      exclusion_reason: notes?.include_in_formal_results === false ? notes.reason : '',
    }));
    fs.writeFileSync(path.join(out, 'metrics.csv'), toCsv(rows));
    fs.writeFileSync(path.join(out, 'words.csv'), toCsv(annotated(words)));
    fs.writeFileSync(path.join(out, 'sentences.csv'), toCsv(annotated(sentences)));
    fs.writeFileSync(path.join(out, 'capture.json'), JSON.stringify({ events: events.length, rawLines: lineCount,
      trials: new Set(events.map(e => `${e.session}/${e.trial_id}`)).size,
      note: 'Device monotonic timestamps; hand_seen is detection, manual marker is not physical gesture onset. sentence_returned is model return, not screen presentation.',
      include_in_formal_results: notes?.include_in_formal_results ?? 'unclassified',
    }, null, 2));
    console.log(`Saved ${events.length} events, ${words.length} words and ${sentences.length} sentences to ${out}`);
    if (!events.length) console.warn('No S2S_PERF events: install the instrumented build and press Start trial. Stock logs cannot reconstruct gesture onset/recognition timestamps.');
  };
  lines.on('line', line => {
    if (finished) return;
    lineCount++;
    raw.write(line + '\n');
    const e = parseEvent(line);
    if (e) {
      events.push(e); eventFile.write(JSON.stringify(e) + '\n');
      console.log(`[trial ${e.trial_id}, sentence ${e.sentence_id ?? '-'}, word ${e.word_index ?? '-'}] ${e.event} ${e.label || e.status || e.text || ''}`);
    } else if (line.trim()) console.log('[stock] ' + line);
  });
  lines.on('close', finish);
  process.once('SIGINT', finish);
  process.once('SIGTERM', finish);
  if (child) {
    child.on('error', error => { console.error(error.message); process.exitCode = 1; finish(); });
    child.stderr.on('data', data => process.stderr.write(data));
    child.on('close', code => { if (code && !finished) process.exitCode = 1; finish(); });
    timer = setTimeout(finish, seconds * 1000);
    console.log(`Capturing for ${seconds}s. Hand entry/50% threshold are recorded automatically. Output: ${out}`);
  }
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(e => { console.error(e.message); process.exitCode = 1; });
}
