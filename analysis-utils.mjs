// Lightweight presentation analysis helpers. These deliberately do not alter
// gaze estimation, calibration, or coordinate correction.
export const AOI_MIN_DWELL_MS = 200;
export const AOI_MISSING_GAP_MS = 400;

const value = (input) => Number.isFinite(Number(input)) ? Number(input) : 0;
const sampleTime = (sample) => value(sample.image_elapsed_ms ?? sample.elapsed_ms);
const validGaze = (sample) => sample?.gaze_valid_for_content !== 0 && sample?.gaze_x !== "" && sample?.gaze_y !== "";
const inside = (sample, aoi) => {
  const x = value(sample.gaze_x), y = value(sample.gaze_y);
  return x >= aoi.x && x <= aoi.x + aoi.width && y >= aoi.y && y <= aoi.y + aoi.height;
};

export function segmentSamples(samples = [], segmentSeconds = 10, durationMs = 0) {
  const span = Math.max(1, Number(segmentSeconds) || 10) * 1000;
  const last = Math.max(durationMs || 0, ...samples.map(sampleTime), 0);
  const count = Math.max(1, Math.ceil(Math.max(last, 1) / span));
  return Array.from({ length: count }, (_, index) => {
    const start_ms = index * span;
    const end_ms = Math.min((index + 1) * span, Math.max(last, span));
    const segmentSamples = samples.filter((sample) => {
      const time = Number.isFinite(Number(sample.sync_ms)) ? Number(sample.sync_ms) : sampleTime(sample);
      return time >= start_ms && time < end_ms;
    });
    return { start_ms, end_ms, valid_gaze_samples: segmentSamples.filter(validGaze).length, samples: segmentSamples };
  });
}

function qualifiedAoiVisits(aoi, samples = [], { intervalMs = 200, minDwellMs = AOI_MIN_DWELL_MS, missingGapMs = AOI_MISSING_GAP_MS } = {}) {
  const ordered = samples.slice().sort((a, b) => sampleTime(a) - sampleTime(b));
  const visits = [];
  let active = null;
  for (const sample of ordered) {
    const time = sampleTime(sample);
    if (validGaze(sample) && inside(sample, aoi)) {
      if (active && time - active.last_ms <= missingGapMs) active.last_ms = time;
      else {
        if (active) visits.push(active);
        active = { first_ms: time, last_ms: time };
      }
      continue;
    }
    // A short invalid gap can be rejoined when the next valid gaze returns to
    // this AOI. A valid gaze outside the AOI ends the visit immediately.
    if (active && validGaze(sample)) { visits.push(active); active = null; }
    else if (active && time - active.last_ms > missingGapMs) { visits.push(active); active = null; }
  }
  if (active) visits.push(active);
  return visits.map((visit) => ({ ...visit, duration_ms: visit.last_ms - visit.first_ms + intervalMs }))
    .filter((visit) => visit.duration_ms >= minDwellMs);
}

export function calculateAoiMetrics(aoi, samples = [], options = {}) {
  const intervalMs = options.intervalMs ?? 200;
  const totalValidMs = samples.filter(validGaze).length * intervalMs;
  const qualified = qualifiedAoiVisits(aoi, samples, options);
  const dwell_ms = qualified.reduce((sum, visit) => sum + visit.duration_ms, 0);
  return {
    aoi_id: aoi.id,
    first_arrival_ms: qualified[0]?.first_ms ?? null,
    dwell_ms,
    entries: qualified.length,
    first_dwell_ms: qualified[0]?.duration_ms ?? null,
    average_dwell_ms: qualified.length ? dwell_ms / qualified.length : null,
    revisits: Math.max(0, qualified.length - 1),
    seen: qualified.length > 0,
    valid_time_ratio: totalValidMs ? dwell_ms / totalValidMs : 0,
  };
}

export function calculateAoiJourney(aois = [], samples = [], options = {}) {
  const visits = aois.flatMap((aoi) => qualifiedAoiVisits(aoi, samples, options)
    .map((visit) => ({ ...visit, aoi_id: aoi.id, aoi_name: aoi.name }))
  ).sort((a, b) => a.first_ms - b.first_ms || a.last_ms - b.last_ms);
  const sequence = [];
  for (const visit of visits) {
    if (sequence.at(-1)?.aoi_id !== visit.aoi_id) sequence.push(visit);
  }
  const transitions = [];
  for (let index = 1; index < sequence.length; index++) {
    const from = sequence[index - 1], to = sequence[index];
    const existing = transitions.find((item) => item.from_aoi_id === from.aoi_id && item.to_aoi_id === to.aoi_id);
    if (existing) existing.count += 1;
    else transitions.push({ from_aoi_id: from.aoi_id, from_name: from.aoi_name, to_aoi_id: to.aoi_id, to_name: to.aoi_name, count: 1 });
  }
  return { sequence: sequence.map(({ aoi_id, aoi_name, first_ms }) => ({ aoi_id, aoi_name, first_ms })), transitions };
}
