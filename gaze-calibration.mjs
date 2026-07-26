const MODES = {
  affine2d: 3,
  quadratic2d: 6,
};

export function median(values) {
  if (!values.length) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function evaluatePoseQuality(metrics, pose) {
  if (!pose || !metrics?.faceDetected) {
    return { level: metrics?.faceDetected ? "good" : "unavailable", weight: metrics?.faceDetected ? 1 : 0, severity: 0, direction: "" };
  }
  const deviations = {
    horizontal: Math.abs(metrics.faceCenterX - pose.faceCenterX) / 0.1,
    vertical: Math.abs(metrics.faceCenterY - pose.faceCenterY) / 0.1,
    yaw: Math.abs(metrics.yaw - pose.yaw) / 0.14,
    pitch: Math.abs(metrics.pitch - pose.pitch) / 0.1,
    distance: Math.abs(metrics.eyeDistance / Math.max(pose.eyeDistance, 1e-6) - 1) / 0.18,
  };
  const [cause, severity] = Object.entries(deviations).sort((a, b) => b[1] - a[1])[0];
  const level = severity <= 1 ? "good" : severity <= 2.2 ? "usable" : "excluded";
  const weight = level === "good" ? 1 : level === "usable" ? Math.max(0.45, 1 - (severity - 1) * 0.35) : 0;
  let direction = "";
  if (cause === "horizontal") direction = metrics.faceCenterX > pose.faceCenterX ? "顔を少し左へ" : "顔を少し右へ";
  else if (cause === "vertical") direction = metrics.faceCenterY > pose.faceCenterY ? "顔を少し上へ" : "顔を少し下へ";
  else if (cause === "distance") direction = metrics.eyeDistance > pose.eyeDistance ? "カメラから少し離れて" : "カメラへ少し近づいて";
  else direction = "顔を少し正面へ";
  return { level, weight, severity, direction, deviations };
}

export function filterCalibrationSamples(samples, minimumDistance = 0.04) {
  if (!samples.length) return { samples: [], excludedCount: 0, rawSpread: NaN, spread: NaN };
  const centerX = median(samples.map((sample) => sample.screenX));
  const centerY = median(samples.map((sample) => sample.screenY));
  const ranked = samples.map((sample) => ({
    sample,
    distance: Math.hypot(sample.screenX - centerX, sample.screenY - centerY),
  })).sort((a, b) => a.distance - b.distance);
  const rawSpread = median(ranked.map((item) => item.distance));
  const cutoff = Math.max(minimumDistance, rawSpread * 2.5);
  let kept = ranked.filter((item) => item.distance <= cutoff);
  // Preserve a usable pair even when one model frame jumps far away.
  if (kept.length < 2) kept = ranked.slice(0, Math.min(2, ranked.length));
  const inliers = kept.map((item) => item.sample);
  const filteredX = median(inliers.map((sample) => sample.screenX));
  const filteredY = median(inliers.map((sample) => sample.screenY));
  const spread = median(inliers.map((sample) => Math.hypot(sample.screenX - filteredX, sample.screenY - filteredY)));
  return {
    samples: inliers,
    excludedCount: samples.length - inliers.length,
    rawSpread,
    spread,
  };
}

function mappingFeatures(screenX, screenY, mapping) {
  const x = (screenX - mapping.center.x) / mapping.scale.x;
  const y = (screenY - mapping.center.y) / mapping.scale.y;
  const affine = [1, x, y];
  return mapping.mode === "quadratic2d" ? [...affine, x * x, x * y, y * y] : affine;
}

export function applyDirectGazeMapping(screenX, screenY, mapping) {
  if (!mapping || !MODES[mapping.mode] || !Number.isFinite(screenX) || !Number.isFinite(screenY)) return null;
  const features = mappingFeatures(screenX, screenY, mapping);
  const x = features.reduce((sum, value, index) => sum + value * mapping.x_coefficients[index], 0);
  const y = features.reduce((sum, value, index) => sum + value * mapping.y_coefficients[index], 0);
  return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null;
}

export function fitDirectGazeMapping(points, mode = "affine2d") {
  const size = MODES[mode];
  if (!size || points.length < size) return null;
  const centerX = median(points.map((point) => point.screenX));
  const centerY = median(points.map((point) => point.screenY));
  const rangeX = Math.max(...points.map((point) => point.screenX)) - Math.min(...points.map((point) => point.screenX));
  const rangeY = Math.max(...points.map((point) => point.screenY)) - Math.min(...points.map((point) => point.screenY));
  const mapping = {
    mode,
    center: { x: centerX, y: centerY },
    scale: { x: Math.max(rangeX / 2, 0.03), y: Math.max(rangeY / 2, 0.03) },
  };
  const normal = Array.from({ length: size }, (_, row) => Array.from({ length: size }, (_, column) => {
    if (row !== column) return 0;
    if (row === 0) return 0.002;
    return row < 3 ? 0.01 : 0.04;
  }));
  const targetX = Array(size).fill(0);
  const targetY = Array(size).fill(0);
  for (const point of points) {
    const row = mappingFeatures(point.screenX, point.screenY, mapping);
    for (let i = 0; i < size; i++) {
      targetX[i] += row[i] * point.targetX;
      targetY[i] += row[i] * point.targetY;
      for (let j = 0; j < size; j++) normal[i][j] += row[i] * row[j];
    }
  }
  const xCoefficients = solveLinearSystem(normal, targetX);
  const yCoefficients = solveLinearSystem(normal, targetY);
  const coefficients = [...(xCoefficients || []), ...(yCoefficients || [])];
  if (!xCoefficients || !yCoefficients || !coefficients.every(Number.isFinite) || coefficients.some((value) => Math.abs(value) > 10)) return null;
  return { ...mapping, x_coefficients: xCoefficients, y_coefficients: yCoefficients };
}

function leaveOneOutScore(points, mode) {
  const minimumTrainingPoints = MODES[mode];
  if (points.length - 1 < minimumTrainingPoints) return Infinity;
  const errors = [];
  for (let index = 0; index < points.length; index++) {
    const training = points.filter((_, pointIndex) => pointIndex !== index);
    const mapping = fitDirectGazeMapping(training, mode);
    const predicted = mapping ? applyDirectGazeMapping(points[index].screenX, points[index].screenY, mapping) : null;
    if (!predicted) return Infinity;
    errors.push(Math.hypot(predicted.x - points[index].targetX, predicted.y - points[index].targetY));
  }
  return errors.reduce((sum, value) => sum + value, 0) / errors.length;
}

export function selectDirectGazeMapping(points) {
  const affineScore = leaveOneOutScore(points, "affine2d");
  const quadraticScore = leaveOneOutScore(points, "quadratic2d");
  // Use the more flexible model only when its held-out error is clearly lower.
  const selectedMode = quadraticScore < affineScore * 0.9 ? "quadratic2d" : "affine2d";
  const mapping = fitDirectGazeMapping(points, selectedMode);
  if (!mapping) return null;
  return {
    ...mapping,
    selection: {
      selected_mode: selectedMode,
      affine_leave_one_out_error: Number.isFinite(affineScore) ? affineScore : null,
      quadratic_leave_one_out_error: Number.isFinite(quadraticScore) ? quadraticScore : null,
    },
  };
}

function solveLinearSystem(matrix, vector) {
  const rows = matrix.map((row, index) => [...row, vector[index]]);
  const size = matrix.length;
  for (let column = 0; column < size; column++) {
    let pivot = column;
    for (let row = column + 1; row < size; row++) {
      if (Math.abs(rows[row][column]) > Math.abs(rows[pivot][column])) pivot = row;
    }
    if (Math.abs(rows[pivot][column]) < 1e-8) return null;
    [rows[column], rows[pivot]] = [rows[pivot], rows[column]];
    const divisor = rows[column][column];
    for (let index = column; index <= size; index++) rows[column][index] /= divisor;
    for (let row = 0; row < size; row++) {
      if (row === column) continue;
      const factor = rows[row][column];
      for (let index = column; index <= size; index++) rows[row][index] -= factor * rows[column][index];
    }
  }
  return rows.map((row) => row[size]);
}
