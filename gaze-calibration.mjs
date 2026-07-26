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
