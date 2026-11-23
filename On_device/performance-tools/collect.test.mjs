import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseEvent, summarize, toCsv, summarizeWords, summarizeSentences } from './collect.mjs';
const e = (event, t, extra = {}) => ({ schema: 1, session: 'test', trial_id: 1, expected: 'water', event, t_ns: String(t), ...extra });
test('parses tagged JSON and rejects unrelated logs', () => {
  assert.equal(parseEvent('I/S2S_PERF: ' + JSON.stringify(e('trial_start', 1))).event, 'trial_start');
  assert.equal(parseEvent('unrelated {"other":1}'), null);
  assert.equal(parseEvent('{broken'), null);
});
test('matches the target, preserves failed generation, correlates requests and uses device time', () => {
  const rows = summarize([
    e('trial_start', 1000000),
    e('hand_seen', 12000000, { frame_observed_ns: '10000000' }),
    e('recognition_result', 20000000, { label: 'food', accepted_threshold: true }),
    e('recognition_result', 30000000, { label: 'water', accepted_threshold: false }),
    e('recognition_result', 50000000, { label: 'water', accepted_threshold: true }),
    e('sentence_submitted', 60000000, { request_id: 'r1', submitted_ns: '55000000', trigger: 'manual' }),
    e('generation_status', 70000000, { request_id: 'r1', status: 'model_unavailable' }),
    e('sentence_returned', 80000000, { request_id: 'r1', text: 'Water' }),
    e('tts_start_callback', 90000000, { trial_id: 2, request_id: 'r1' }),
  ].reverse());
  assert.equal(rows[0].manual_start_to_target_ms, 49);
  assert.equal(rows[0].first_hand_frame_to_target_ms, 40);
  assert.equal(rows[0].submit_to_text_ms, 25);
  assert.equal(rows[0].generation_status, 'model_unavailable');
  assert.equal(rows[0].submit_to_tts_callback_ms, '');
});
test('does not score an unspecified target and reports missing returns', () => {
  const rows = summarize([e('trial_start', 1, { expected: '' }),
    e('recognition_result', 2, { expected: '', label: 'water', accepted_threshold: true }),
    e('sentence_submitted', 3, { expected: '', request_id: 'a', submitted_ns: '3' })]);
  assert.equal(rows[0].recognition_status, 'unscored_no_target');
  assert.equal(rows[0].generation_status, 'incomplete_or_logs_missing');
  assert.equal(rows[0].submit_to_text_ms, '');
});
test('BigInt subtraction avoids precision loss and manual end permits early recognition', () => {
  const base = 9007199254740993000n;
  const rows = summarize([e('trial_start', base), e('manual_gesture_end', base + 2000000n),
    e('recognition_result', base + 1000000n, { label: 'water', accepted_threshold: true })]);
  assert.equal(rows[0].manual_start_to_target_ms, 1);
  assert.equal(rows[0].manual_end_to_target_ms, -1);
  assert.match(toCsv([{ text: 'a,"b"\n' }]), /"a,""b""\n"/);
});
test('separates first and subsequent words, includes failed episodes, associates sentence generation', () => {
  const events = [
    e('word_started', 10000000, { sentence_id: 1, word_index: 1, episode_id: 1, start_ns: '1000000', hand_seen_ns: '5000000' }),
    e('word_recognized', 50000000, { sentence_id: 1, word_index: 1, episode_id: 1, label: 'water', confidence: 0.6 }),
    e('word_started', 60000000, { sentence_id: 1, word_index: 2, episode_id: 2, start_ns: '55000000', hand_seen_ns: '58000000' }),
    e('word_hand_removed', 80000000, { sentence_id: 1, word_index: 2, episode_id: 2, removed_ns: '75000000' }),
    e('sentence_submitted', 90000000, { sentence_id: 1, request_id: 'r', submitted_ns: '90000000', word_count: 2 }),
    e('generation_started', 100000000, { request_id: 'r' }),
    e('generation_status', 120000000, { request_id: 'r', status: 'generation_error' }),
    e('sentence_returned', 130000000, { request_id: 'r', text: 'Water' }),
    e('word_started', 140000000, { sentence_id: 2, word_index: 1, episode_id: 3, start_ns: '135000000', hand_seen_ns: '138000000' }),
  ];
  const words = summarizeWords(events.reverse());
  assert.equal(words[0].word_position, 'first');
  assert.equal(words[0].hand_detected_to_50pct_ms, 45);
  assert.equal(words[1].word_position, 'subsequent');
  assert.equal(words[1].status, 'hand_removed_before_threshold');
  assert.equal(words[2].word_position, 'first');
  const sentences = summarizeSentences(events);
  assert.equal(sentences[0].submit_to_text_ms, 40);
  assert.equal(sentences[0].generation_call_ms, 30);
  assert.equal(sentences[0].status, 'generation_error');
});
