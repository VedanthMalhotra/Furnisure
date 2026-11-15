// An L-shaped hex key, shown turning each screw as it goes in (scan player and kit player).
// Built one unit long: the tip that sits in the screw head is at the origin, the short arm runs
// along +Z (straight out of the head) and the long arm along +X. The caller scales it to the screw.
import * as THREE from 'three';

const R = 0.11, LONG = 2.4;           // half-width of the hex bar, and the long arm, in short-arm lengths

let geometry;
function shape() {
  if (geometry) return geometry;
  const short = new THREE.CylinderGeometry(R, R, 1 + R, 6).rotateX(Math.PI / 2).translate(0, 0, (1 + R) / 2);
  const long = new THREE.CylinderGeometry(R, R, LONG, 6).rotateZ(Math.PI / 2).translate(LONG / 2 - R, 0, 1);
  // one geometry from the two bars, so a key is a single mesh
  const position = new Float32Array([...short.attributes.position.array, ...long.attributes.position.array]);
  const normal = new Float32Array([...short.attributes.normal.array, ...long.attributes.normal.array]);
  const offset = short.attributes.position.count;
  const index = [...short.index.array, ...[...long.index.array].map(i => i + offset)];
  geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normal, 3));
  geometry.setIndex(index);
  return geometry;
}

// Each key has its own material so it can fade in and out by itself (key.userData.material.opacity)
export function makeAllenKey() {
  const material = new THREE.MeshStandardMaterial({ color: 0xd3d8de, metalness: 0.35, roughness: 0.3,
    emissive: 0xff8a1f, emissiveIntensity: 0.18, transparent: true, opacity: 0 });
  const key = new THREE.Mesh(shape(), material);
  key.userData.material = material;
  key.visible = false;
  key.renderOrder = 1;
  return key;
}

// How big to draw the key for a screw of this length and head width (same units as the scene)
export const keySize = (length, headWidth) => Math.max(length * 0.7, headWidth * 3);
