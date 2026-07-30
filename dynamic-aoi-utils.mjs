// Post-recording dynamic AOI helpers. These do not participate in gaze
// estimation, calibration, coordinate correction, or heatmap rendering.

export const DYNAMIC_AOI_DEFAULT_INTERVAL_MS = 500;
export const DYNAMIC_AOI_SLOW_INTERVAL_MS = 1000;
export const DYNAMIC_AOI_FRAME_TOLERANCE_MS = 600;

const number = (value) => Number.isFinite(Number(value)) ? Number(value) : 0;
const gazeTime = (sample) => Number.isFinite(Number(sample?.sync_ms)) ? Number(sample.sync_ms) : number(sample?.elapsed_ms);
const validGaze = (sample) => sample?.gaze_valid_for_content !== 0 && sample?.gaze_x !== "" && sample?.gaze_y !== "";

export function boxIou(a, b) {
  const left = Math.max(number(a?.x), number(b?.x));
  const top = Math.max(number(a?.y), number(b?.y));
  const right = Math.min(number(a?.x) + number(a?.width), number(b?.x) + number(b?.width));
  const bottom = Math.min(number(a?.y) + number(a?.height), number(b?.y) + number(b?.height));
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = number(a?.width) * number(a?.height) + number(b?.width) * number(b?.height) - intersection;
  return union > 0 ? intersection / union : 0;
}

export function assignDynamicTracks(frames = [], { minIou = 0.25, maxMissingMs = 1600 } = {}) {
  const tracks = [];
  let nextId = 1;
  for (const frame of frames.slice().sort((a, b) => number(a.sync_ms) - number(b.sync_ms))) {
    const usedTrackIds = new Set();
    for (const detection of frame.detections || []) {
      const candidate = tracks
        .filter((track) => track.label === detection.label && !usedTrackIds.has(track.id) && number(frame.sync_ms) - track.last_ms <= maxMissingMs)
        .map((track) => ({ track, iou: boxIou(track.last_box, detection) }))
        .filter((item) => item.iou >= minIou)
        .sort((a, b) => b.iou - a.iou)[0]?.track;
      const track = candidate || { id: `object-${nextId++}`, label: detection.label, name: detection.display_name || detection.label, first_ms: number(frame.sync_ms), last_ms: number(frame.sync_ms), detections: 0 };
      if (!candidate) tracks.push(track);
      usedTrackIds.add(track.id);
      detection.track_id = track.id;
      track.last_ms = number(frame.sync_ms);
      track.last_box = { x: detection.x, y: detection.y, width: detection.width, height: detection.height };
      track.detections += 1;
    }
  }
  return tracks.map(({ last_box, ...track }) => track);
}

function nearestFrame(frames, time, toleranceMs) {
  let best = null;
  for (const frame of frames) {
    const distance = Math.abs(number(frame.sync_ms) - time);
    if (distance <= toleranceMs && (!best || distance < best.distance)) best = { frame, distance };
  }
  return best?.frame || null;
}

export function calculateDynamicAoiMetrics(track, frames = [], samples = [], { intervalMs = 200, frameToleranceMs = DYNAMIC_AOI_FRAME_TOLERANCE_MS, minDwellMs = 200, missingGapMs = 400 } = {}) {
  const hits = [];
  let validSamples = 0;
  for (const sample of samples.slice().sort((a, b) => gazeTime(a) - gazeTime(b))) {
    if (!validGaze(sample)) continue;
    validSamples += 1;
    const time = gazeTime(sample);
    const frame = nearestFrame(frames, time, frameToleranceMs);
    const box = frame?.detections?.find((item) => item.track_id === track.id);
    const x = number(sample.gaze_x), y = number(sample.gaze_y);
    if (box && x >= box.x && x <= box.x + box.width && y >= box.y && y <= box.y + box.height) hits.push(time);
  }
  const visits = [];
  let active = null;
  for (const time of hits) {
    if (active && time - active.last_ms <= missingGapMs) active.last_ms = time;
    else {
      if (active) visits.push(active);
      active = { first_ms: time, last_ms: time };
    }
  }
  if (active) visits.push(active);
  const qualified = visits.map((visit) => ({ ...visit, duration_ms: visit.last_ms - visit.first_ms + intervalMs }))
    .filter((visit) => visit.duration_ms >= minDwellMs);
  const dwellMs = qualified.reduce((sum, visit) => sum + visit.duration_ms, 0);
  return {
    track_id: track.id,
    first_arrival_ms: qualified[0]?.first_ms ?? null,
    first_dwell_ms: qualified[0]?.duration_ms ?? null,
    dwell_ms: dwellMs,
    average_dwell_ms: qualified.length ? dwellMs / qualified.length : null,
    entries: qualified.length,
    revisits: Math.max(0, qualified.length - 1),
    seen: qualified.length > 0,
    valid_time_ratio: validSamples ? dwellMs / (validSamples * intervalMs) : 0,
  };
}

export function dynamicAoiAtTime(frames = [], time = 0, toleranceMs = DYNAMIC_AOI_FRAME_TOLERANCE_MS) {
  return nearestFrame(frames, number(time), toleranceMs)?.detections || [];
}
