const SVG_NS = 'http://www.w3.org/2000/svg';
const COLORS = {fill: '#dcebe3', edge: '#146f68', center: '#8ba69c', text: '#52655d', muted: '#7a827e'};

function element(document, tag, attributes = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

function number(value) { return Number(value.toFixed(4)); }

/**
 * The faithful solver has N - 1 finite tube sections, each L/N long. Its last
 * area is the lip radiation aperture, not another finite-length tube section.
 * Display positions are in centimeters along the straight tube.
 */
export function createTractGeometry(areas, {lengthCm = 17.5, faithful = true} = {}) {
  if (!(lengthCm > 0) || !Number.isFinite(lengthCm)) throw new RangeError('Tract length must be positive and finite.');
  const values = Array.from(areas, area => {
    if (!Number.isFinite(area) || area < 0) throw new RangeError('Tract areas must be finite and nonnegative.');
    return area;
  });
  if (values.length < 2) throw new RangeError('At least two tract areas are required.');
  const count = faithful ? values.length - 1 : values.length;
  const sectionLengthCm = lengthCm / values.length;
  const internalLengthCm = count * sectionLengthCm;
  const radii = values.map(area => Math.sqrt(area / Math.PI));
  const sections = values.slice(0, count).map((area, index) => ({
    index, area, radiusCm: radii[index],
    startCm: index * sectionLengthCm, endCm: (index + 1) * sectionLengthCm,
  }));
  // Restore the original connected-radius display. These are visual samples;
  // interpolation does not modify the areas passed to the acoustic solver.
  const profile = radii.map((radiusCm, index) => ({
    positionCm: index / (radii.length - 1) * internalLengthCm, radiusCm,
  }));
  return {
    sections, radii, profile, sectionLengthCm, internalLengthCm,
    nominalLengthCm: lengthCm, lipRadiusCm: radii.at(-1), lipArea: values.at(-1),
    hasSeparateLip: faithful,
  };
}

/** Render equivalent radii mirrored around a straight centerline. */
export function renderTractView(svg, areas, {lengthCm = 17.5, radiusLimitCm, faithful = true, targetAreas, selectedIndex} = {}) {
  const geometry = createTractGeometry(areas, {lengthCm, faithful});
  const target = targetAreas ? createTractGeometry(targetAreas, {lengthCm, faithful}) : null;
  const document = svg.ownerDocument;
  const node = (tag, attributes, text) => element(document, tag, attributes, text);
  for (const child of [...svg.children]) if (!['title', 'desc'].includes(child.localName)) child.remove();
  const title = [...svg.children].find(child => child.localName === 'title') ?? svg.insertBefore(node('title'), svg.firstChild);
  const description = [...svg.children].find(child => child.localName === 'desc') ?? svg.insertBefore(node('desc'), title.nextSibling);
  title.textContent = 'Vocal tract radii along a straight tube';
  description.textContent = `Each radius is the square root of area divided by pi. The outlines connect the radius samples continuously using linear interpolation for display. Radius is mirrored above and below the centerline; the horizontal and vertical scales are labeled separately. The displayed profile spans ${geometry.internalLengthCm.toFixed(3)} centimeters.${faithful ? ' The final radius is the lip radiation aperture, marked at the endpoint.' : ''}`;
  if (selectedIndex !== undefined) description.textContent += ' Drag either outline, or use the section and area sliders, to edit the tube.';
  if (target) description.textContent += ' The dashed orange outlines show the ground-truth Fant vowel profile on the same radius scale as the current tube.';
  svg.setAttribute('viewBox', '0 0 760 235');
  svg.setAttribute('data-view', 'straight');
  svg.setAttribute('data-internal-length-cm', geometry.internalLengthCm);
  const peak = Math.max(...geometry.radii, ...(target?.radii ?? []));
  // A trajectory-wide bound keeps animation fixed. Never crop an unexpected
  // off-grid peak, even if a caller supplies a smaller bound than this frame.
  const limit = Math.max(.1, peak, Number.isFinite(radiusLimitCm) && radiusLimitCm > 0 ? radiusLimitCm : 2.5);
  const label = (x, y, text, {size = 15, anchor = 'start', detail = false, ...attributes} = {}) => {
    svg.append(node('text', {
      x: number(x), y: number(y), 'text-anchor': anchor,
      class: detail ? 'tract-view-detail' : 'tract-view-label',
      'font-family': 'var(--sans, sans-serif)', 'font-size': size,
      fill: detail ? COLORS.muted : COLORS.text, ...attributes,
    }, text));
  };
  const sectionTitle = section => `Sample ${section.index + 1}: area ${section.area.toFixed(3)} cm²; radius ${section.radiusCm.toFixed(3)} cm. The outline interpolates between samples.`;

  const x = position => 58 + position / geometry.internalLengthCm * 675;
  const y = radius => 108 - radius / limit * 78;
  for (const value of [-limit, -limit / 2, 0, limit / 2, limit]) {
    if (value !== 0) svg.append(node('line', {x1: 58, x2: 733, y1: y(value), y2: y(value), stroke: '#e7eae5', 'stroke-width': 1}));
    label(47, y(value) + 4, value === 0 ? '0' : value.toFixed(2), {anchor: 'end', size: 12, detail: true});
  }
  const path = (samples, sign, initial = 'M') => samples.map((sample, index) => `${index ? 'L' : initial}${number(x(sample.positionCm))},${number(y(sign * sample.radiusCm))}`).join(' ');
  const boundary = sign => path(geometry.profile, sign);
  const upper = boundary(1), lower = boundary(-1);
  const reverseLower = path([...geometry.profile].reverse(), -1, 'L');
  svg.append(node('path', {d: `${upper} ${reverseLower} Z`, fill: COLORS.fill, 'data-tube-fill': ''}));
  for (let index = 0; index < geometry.profile.length - 1; index++) {
    const section = geometry.sections[index], interval = geometry.profile.slice(index, index + 2);
    const region = node('path', {d: `${path(interval, 1)} ${path([...interval].reverse(), -1, 'L')} Z`, fill: 'transparent', 'data-section': section.index, 'data-area-cm2': section.area});
    region.append(node('title', {}, sectionTitle(section)));
    svg.append(region);
  }
  svg.append(node('line', {x1: 58, x2: 733, y1: 108, y2: 108, stroke: COLORS.center, 'stroke-width': 1, 'stroke-dasharray': '5 4', 'data-centerline': '', 'pointer-events': 'none'}));
  for (const [sign, path] of [[1, upper], [-1, lower]]) svg.append(node('path', {d: path, fill: 'none', stroke: COLORS.edge, 'stroke-width': 2, 'stroke-linejoin': 'round', 'data-boundary': sign > 0 ? 'upper' : 'lower', 'pointer-events': 'none'}));
  const lip = node('line', {x1: 733, x2: 733, y1: y(geometry.lipRadiusCm), y2: y(-geometry.lipRadiusCm), stroke: COLORS.edge, 'stroke-width': 2, 'data-lip-boundary': '', 'data-area-cm2': geometry.lipArea});
  lip.append(node('title', {}, `Lip${faithful ? ' radiation' : ''} aperture: area ${geometry.lipArea.toFixed(3)} cm²; radius ${geometry.lipRadiusCm.toFixed(3)} cm.`));
  svg.append(lip);
  if (target) {
    for (const sign of [1, -1]) svg.append(node('path', {
      d: path(target.profile, sign), fill: 'none', stroke: '#b96537', 'stroke-width': 3,
      'stroke-dasharray': '7 5', 'stroke-linejoin': 'round', 'data-target-boundary': sign > 0 ? 'upper' : 'lower', 'pointer-events': 'none',
    }));
    svg.append(node('line', {
      x1: 733, x2: 733, y1: y(target.lipRadiusCm), y2: y(-target.lipRadiusCm),
      stroke: '#b96537', 'stroke-width': 3, 'stroke-dasharray': '7 5', 'data-target-lip-boundary': '', 'pointer-events': 'none',
    }));
  }
  const selected = geometry.profile[selectedIndex];
  if (selected) {
    const cx = x(selected.positionCm);
    svg.append(node('line', {x1: cx, x2: cx, y1: y(selected.radiusCm), y2: y(-selected.radiusCm), stroke: COLORS.edge, opacity: .3, 'pointer-events': 'none'}));
    for (const sign of [1, -1]) svg.append(node('circle', {
      cx, cy: y(sign * selected.radiusCm), r: 4.5, fill: COLORS.edge, stroke: '#fff', 'stroke-width': 2,
      'data-edit-handle': sign > 0 ? 'upper' : 'lower', 'pointer-events': 'none',
    }));
  }
  label(58, 17, 'Radius · cm', {size: 12, detail: true});
  label(58, 211, 'Glottis · 0 cm', {size: 12, detail: true});
  label(733, 211, 'Lips', {anchor: 'end', size: 12, detail: true});
  for (const distance of [5, 10, 15]) if (distance < geometry.internalLengthCm) label(x(distance), 211, `${distance}`, {anchor: 'middle', size: 12, detail: true});
  label(395, 230, 'Distance from glottis · cm', {anchor: 'middle', size: 12, detail: true});
  // Invert the same coordinates used for drawing, including the squared
  // radius-to-area conversion. Either half of the tube edits the same area.
  return {...geometry, plot: {
    sectionAt: px => Math.min(areas.length - 1, Math.max(0, Math.round((px - 58) / 675 * (areas.length - 1)))),
    areaAt: py => Math.PI * ((108 - py) / 78 * limit) ** 2,
    contains: (px, py) => px >= 58 && px <= 733 && py >= 30 && py <= 186,
  }};
}
