// Wheel drop physics, mirroring hazard.py so the page can re-roll any tyre for any pothole
// width without another scan. The API sends the pothole's relief patch (mm below the road,
// 0 outside the outline); the scale comes from the pothole's width.

/** Surface height (negative = below road) along a travel line through the deepest cell. */
export function bandProfile(relief, axis, bandCells) {
  const rows = relief.length;
  const cols = relief[0].length;
  let best = -1, by = 0, bx = 0;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const v = Math.max(0, relief[y][x]);
      if (v > best) { best = v; by = y; bx = x; }
    }
  }
  const half = Math.max(0, Math.floor(bandCells / 2));
  const out = [];
  if (axis === 0) {
    for (let y = 0; y < rows; y++) {
      let m = Infinity;
      for (let x = Math.max(0, bx - half); x <= Math.min(cols - 1, bx + half); x++) m = Math.min(m, Math.max(0, relief[y][x]));
      out.push(-m);
    }
  } else {
    for (let x = 0; x < cols; x++) {
      let m = Infinity;
      for (let y = Math.max(0, by - half); y <= Math.min(rows - 1, by + half); y++) m = Math.min(m, Math.max(0, relief[y][x]));
      out.push(-m);
    }
  }
  return out;
}

/** Wheel centre height at every point of `height` (flat road assumed beyond it). */
export function wheelCentre(height, stepMm, radiusMm) {
  const half = Math.ceil(radiusMm / stepMm);
  const lift = [];
  for (let o = -half; o <= half; o++) lift.push(Math.sqrt(Math.max(0, radiusMm ** 2 - (o * stepMm) ** 2)));
  const n = height.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    let c = -Infinity;
    for (let o = -half; o <= half; o++) {
      const j = i + o;
      const h = j < 0 || j >= n ? 0 : height[j];
      const v = h + lift[o + half];
      if (v > c) c = v;
    }
    out[i] = c;
  }
  return out;
}

/** Drop of every tyre, rolling along both image axes and keeping the worse (as hazard.wheel_drops). */
export function allDrops(patch, tyres, mmPerCell) {
  return tyres.map((t) => {
    const band = Math.max(1, Math.round(t.width_mm / mmPerCell));
    let best = { drop: 0, axis: 0 };
    for (const axis of [0, 1]) {
      const prof = bandProfile(patch.relief_mm, axis, band);
      const c = wheelCentre(prof, mmPerCell, t.radius_mm);
      const drop = Math.max(0, t.radius_mm - Math.min(...c));
      if (drop >= best.drop) best = { drop, axis };
    }
    return { ...t, drop: best.drop, axis: best.axis };
  });
}

/** Profile padded with flat road so the wheel can roll in and out; returns x (mm), surface, centre. */
export function rollPath(patch, tyre, axis, mmPerCell) {
  const band = Math.max(1, Math.round(tyre.width_mm / mmPerCell));
  const prof = bandProfile(patch.relief_mm, axis, band);
  const surface = bandProfile(patch.relief_mm, axis, 1);
  const pad = Math.ceil((tyre.radius_mm * 1.15) / mmPerCell);
  const h = [...new Array(pad).fill(0), ...prof, ...new Array(pad).fill(0)];
  const s = [...new Array(pad).fill(0), ...surface, ...new Array(pad).fill(0)];
  const centre = wheelCentre(h, mmPerCell, tyre.radius_mm);
  const x = h.map((_, i) => (i - pad) * mmPerCell);
  return { x, surface: s, centre, profileMm: prof.length * mmPerCell };
}

/** Response category for a risk score (hazard.response_for). */
export function responseFor(score, route) {
  if (score >= 25) return { category: 'D1', response: '2 hours' };
  if (score >= 20) return { category: 'D2', response: route === 'county' ? '2 working days' : '5 working days' };
  if (score >= 10) return { category: score === 15 || score === 16 ? 'D3A' : 'D3', response: '90 working days' };
  return { category: 'Observation', response: 'next routine inspection' };
}
