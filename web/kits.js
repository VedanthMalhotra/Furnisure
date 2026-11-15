import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {makeAllenKey, keySize} from './allenkey.js';

const $ = id => document.getElementById(id);
const viewAngle = 40;
const camera = new THREE.PerspectiveCamera(viewAngle,1,.001,100);
const views = ['left','right'].map(id => {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xe8ede2);
  scene.add(new THREE.HemisphereLight(0xffffff,0x717971,2));
  const light = new THREE.DirectionalLight(0xffffff,2);
  light.position.set(2,4,3);
  scene.add(light);
  const renderer = new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
  $(id).append(renderer.domElement);
  return {scene,renderer,host:$(id)};
});
const controls = new OrbitControls(camera,views[0].renderer.domElement);
const GLOW = new THREE.Color(0xff8a1f);   // the parts being fitted in this step
let glide;                                // camera move towards the parts of the current step
controls.addEventListener('start',() => { glide = undefined; });   // the user's own orbit wins
const loader = new GLTFLoader();
const assemblyCenter = new THREE.Vector3();
let kit, model, parts, prediction, actions = [], step = 0, started = 0;
let busy = false, playing = false, recording = false, recordFrame, travel, sourceUrl;
const interval = 1700;
let extras = [];                         // ids of hardware drawn in because the model leaves it out
let hardwareSpecs = {};                  // id -> {kind, ...} for that drawn hardware
const hardwareMeshes = new Map();        // IKEA part number -> real hardware mesh from the server
const HARDWARE_LOOK = {
  screw: {color:0x9aa3ab,metalness:.8,roughness:.35},
  dowel: {color:0xc89b62,metalness:0,roughness:.8},
  cam_nut: {color:0xb9c0c6,metalness:.8,roughness:.3}
};

// Makes a model readable. Some exporters (trimesh) store near-black vertex colours, or none on some
// parts, so boards, holes and steps disappear into each other. Untextured dark parts get a neutral
// grey, small hardware a light metal finish, and every part thin outlines so edges, holes and a
// raised panel stand out. Textured models (the IKEA kits) keep their own look.
const EDGE = new THREE.LineBasicMaterial({color:0x1f2226,transparent:true,opacity:.55});
function finishLook(root) {
  const size = new THREE.Box3().setFromObject(root).getSize(new THREE.Vector3()).length();
  root.traverse(node => {
    if (!node.isMesh) return;
    const geometry = node.geometry;
    for (const material of [node.material].flat()) {
      if (material.map) continue;
      const colours = geometry.attributes.color;
      let light = 1;
      if (material.vertexColors && colours) {
        light = 0;
        const step = Math.max(1,Math.floor(colours.count/400));
        let n = 0;
        for (let i = 0; i < colours.count; i += step, n++) light += (colours.getX(i)+colours.getY(i)+colours.getZ(i))/3;
        light /= n;
      }
      geometry.computeBoundingBox();
      const small = geometry.boundingBox.getSize(new THREE.Vector3()).length() < 0.05*size;
      if (small) {
        material.vertexColors = false;
        material.color.set(0xc9ced4);
        material.metalness = .6; material.roughness = .35;
      } else if (light < .25 || (material.vertexColors && !colours) || !material.vertexColors && material.color.getHSL({}).l > .9) {
        material.vertexColors = false;
        material.color.set(0x7d838b);
        material.metalness = .1; material.roughness = .7;
      }
      material.needsUpdate = true;
    }
    const triangles = (geometry.index ? geometry.index.count : geometry.attributes.position.count)/3;
    if (triangles < 20000) node.add(new THREE.LineSegments(new THREE.EdgesGeometry(geometry,25),EDGE));
  });
}

// Turns a part about its own long axis through its own centre. Some exporters (trimesh) store every
// part around the model's origin, and turning those about their origin would swing them in a circle.
function spin(part, angle) {
  const {object,localAxis,localCenter} = part;
  const turn = new THREE.Quaternion().setFromAxisAngle(localAxis,angle);
  const shift = localCenter.clone().sub(localCenter.clone().applyQuaternion(turn))
    .multiply(object.scale).applyQuaternion(object.quaternion);
  object.quaternion.multiply(turn);
  object.position.add(shift);
}

function status(message) { $('status').textContent = message; }

// An allen key on every screw that is being turned. `driven` lists them for this frame:
// { part, from (the way the screw came in, or null), angle (how far it has turned) }.
const keys = [], Z = new THREE.Vector3(0,0,1);
function driveKeys(driven, progress, settle) {
  driven.forEach(({part,from,angle}, i) => {
    if (!keys[i]) views[0].scene.add(keys[i] = makeAllenKey());
    const key = keys[i], {object} = part;
    object.updateWorldMatrix(true,true);
    const box = new THREE.Box3().setFromObject(object), centre = box.getCenter(new THREE.Vector3());
    const axis = part.localAxis.clone().applyQuaternion(object.getWorldQuaternion(new THREE.Quaternion())).normalize();
    // the head is at the end the screw came in from; without a path, the end facing away from the product
    const outward = from && from.lengthSq() > 1e-12 ? from : centre.clone().sub(assemblyCenter);
    const side = axis.dot(outward) < 0 ? -1 : 1, dir = axis.clone().multiplyScalar(side);
    const size = box.getSize(new THREE.Vector3());
    const length = Math.abs(dir.x)*size.x + Math.abs(dir.y)*size.y + Math.abs(dir.z)*size.z;
    const width = Math.max(size.x*(1-Math.abs(dir.x)),size.y*(1-Math.abs(dir.y)),size.z*(1-Math.abs(dir.z)));
    const scale = keySize(length,width);
    key.scale.setScalar(scale);
    key.position.copy(centre).addScaledVector(dir,length/2 + scale*(.6*settle-.08));   // lifts off once the screw is in
    key.quaternion.setFromUnitVectors(Z,dir).multiply(new THREE.Quaternion().setFromAxisAngle(Z,angle*side));
    key.userData.material.opacity = Math.min(1,progress*6)*(1-settle);
    key.visible = key.userData.material.opacity > .01;
  });
  for (let i = driven.length; i < keys.length; i++) keys[i].visible = false;
}

// Hardware drawn from a profile when no real IKEA model fits: the same shapes as templates.js,
// in millimetres, tip at the origin, +Z toward the head.
function profileGeometry(kind) {
  const L = kind === 'dowel' ? 30 : kind === 'cam_nut' ? 12 : 40, r = kind === 'dowel' ? 4 : kind === 'cam_nut' ? 7.5 : 2;
  let points;
  if (kind === 'dowel') { const c = Math.min(r*.3,1.5); points = [[0,0],[r-c,0],[r,c],[r,L-c],[r-c,L],[0,L]]; }
  else if (kind === 'cam_nut') points = [[0,0],[r,0],[r,L],[0,L]];
  else { const R = r*1.9, hh = Math.max(2,r*1.2); points = [[0,0],[r*.45,0],[r,Math.min(r*1.4,L*.3)],[r,L],[R,L],[R,L+hh],[0,L+hh]]; }
  const geometry = new THREE.LatheGeometry(points.map(([x,y]) => new THREE.Vector2(x,y)),32);
  geometry.rotateX(Math.PI/2);
  return {geometry,length:L,width:kind === 'dowel' ? 2*r : kind === 'cam_nut' ? 2*r : 3.8*r};
}

// A real IKEA screw, dowel or cam lock taken from a kit whose model includes it.
function libraryGeometry(number) {
  if (!hardwareMeshes.has(number)) {
    hardwareMeshes.set(number,fetch('/api/kits/hardware/' + encodeURIComponent(number)).then(response => {
      if (!response.ok) throw new Error('No hardware model ' + number);
      return response.json();
    }).then(data => {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position',new THREE.Float32BufferAttribute(data.positions,3));
      geometry.setIndex(data.indices);
      geometry.computeVertexNormals();
      return {geometry,length:data.length_mm,width:data.width_mm};
    }));
  }
  return hardwareMeshes.get(number);
}

function clearHardware() {
  for (const id of extras) {
    const part = parts?.get(id);
    if (!part) continue;
    part.object.removeFromParent();
    part.object.children[0]?.material?.dispose();
    parts.delete(id);
  }
  extras = [];
}

// Adds the screws, dowels and cam locks the server placed at joints the model leaves bare. They get
// a faint blue glow, since their positions are estimates.
async function addHardware(specs) {
  clearHardware();
  hardwareSpecs = specs ?? {};
  for (const [id,spec] of Object.entries(specs ?? {})) {
    const {geometry,length,width} = spec.number
      ? await libraryGeometry(spec.number).catch(() => profileGeometry(spec.kind))
      : profileGeometry(spec.kind);
    const look = HARDWARE_LOOK[spec.kind] ?? HARDWARE_LOOK.screw;
    const material = new THREE.MeshStandardMaterial({...look,emissive:0x2a6f97,emissiveIntensity:.12});
    material.userData.baseEmissive = material.emissive.clone();
    const mesh = new THREE.Mesh(geometry,material);
    // keep the real part's proportions, but never fatter than the parts it joins allow
    const along = spec.length/length;
    const across = Math.min(along,(spec.width ?? Infinity)/width);
    mesh.scale.set(across,across,along);
    const holder = new THREE.Group();
    holder.add(mesh);
    holder.position.fromArray(spec.position);
    holder.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),new THREE.Vector3(...spec.direction));
    model.add(holder);
    holder.updateMatrixWorld(true);
    parts.set(id,{
      object:holder, position:holder.position.clone(), rotation:holder.quaternion.clone(),
      localAxis:new THREE.Vector3(0,0,1), localCenter:new THREE.Vector3(), slender:true, length:spec.length, centered:false,
      worldPosition:holder.getWorldPosition(new THREE.Vector3()), worldRotation:holder.getWorldQuaternion(new THREE.Quaternion()),
      materials:[material]
    });
    extras.push(id);
  }
}

// A point `t` (0..1) of the way along a list of offsets, by distance travelled. The server sends a
// two-leg path when a part has to lift out of its joint before it can move away without hitting anything.
function alongPath(points, t) {
  const vectors = points.map(point => new THREE.Vector3(...point));
  const lengths = vectors.slice(1).map((vector,i) => vector.distanceTo(vectors[i]));
  let remaining = t*(lengths.reduce((a,b) => a+b,0) || 1);
  for (let i = 0; i < lengths.length; i++) {
    if (remaining <= lengths[i] || i === lengths.length-1) {
      return vectors[i].clone().lerp(vectors[i+1],lengths[i] ? Math.min(1,remaining/lengths[i]) : 1);
    }
    remaining -= lengths[i];
  }
  return vectors.at(-1);
}

function updateControls() {
  const locked = busy || recording;
  for (const element of [$('pdf'),$('scan').querySelector('button')]) element.disabled = locked;
  $('prediction').disabled = locked || !model;
  $('use-pa').disabled = locked || !prediction || !actions.length;
  $('prev').disabled = locked || !actions.length || step === 0;
  $('next').disabled = locked || !actions.length || step === actions.length-1;
  $('play').disabled = $('export').disabled = locked || !actions.length;
  $('check-step').hidden = !scanBridge() || !actions[step]?.nodes?.length;
  $('check-step').disabled = locked;
  $('play').textContent = playing ? 'Pause' : 'Play all';
}

// Centre the view between the whole product and the parts this step fits, so the move stays in frame
function focusOn(action) {
  const box = new THREE.Box3();
  const host = views[0].host;
  const portrait = host.clientWidth < host.clientHeight;
  const visibleNodes = portrait && action ? actions.slice(0, step + 1).flatMap(item => item.nodes) : action?.nodes ?? [];
  for (const id of visibleNodes) {
    const part = parts.get(id);
    if (part && portrait) box.union(part.worldBounds);
    else if (part) box.expandByPoint(part.worldPosition);
  }
  const to = box.isEmpty() ? assemblyCenter.clone()
    : box.getCenter(new THREE.Vector3()).lerp(assemblyCenter, portrait ? 0 : .35);
  if (to.distanceTo(controls.target) > 1e-6) glide = {t:0,from:controls.target.clone(),to};
}

function showStep(index) {
  step = index;
  started = performance.now();
  const action = actions[index];
  $('stepLabel').textContent = action ? 'Step ' + action.step.replace(/^step\s+/i, '') : '';
  $('stepLabel').hidden = !action;
  $('step').textContent = action?.caption ?? 'Finished model';
  if (parts) focusOn(action);
  $('source').hidden = true;     // the manual page stays out of the way of the 3D model
  if (prediction) {
    const predicted = prediction.steps.find(item => item.step === action?.step);
    const expected = new Set(actions.filter(item => item.step === action?.step).flatMap(item => item.nodes));
    const matched = predicted?.poses.filter(pose => expected.has(pose.node)) ?? [];
    const error = matched.reduce((totals, pose) => {
      const part = parts.get(pose.node);
      totals.mm += part.worldPosition.distanceTo(new THREE.Vector3(...pose.position))*1000;
      totals.degrees += THREE.MathUtils.radToDeg(part.worldRotation.angleTo(new THREE.Quaternion(...pose.quaternion).normalize()));
      return totals;
    }, {mm:0,degrees:0});
    $('comparison').textContent = predicted
      ? predicted.poses.length + ' predicted placements · ' + matched.length + ' of ' + expected.size + ' FitFrame parts also predicted'
        + (matched.length ? ' · Mean difference from IKEA poses: ' + (error.mm/matched.length).toFixed(1)
          + ' mm, ' + (error.degrees/matched.length).toFixed(1) + '°' : '')
      : 'No prediction for this step.';
  }
  updateControls();
}

async function load(item) {
  if (kit?.article_number === item.article_number) return;
  // a new query each load, so a model replaced on the server is never served from the browser's cache
  const asset = await loader.loadAsync('/api/kits/' + item.article_number + '/model?t=' + Date.now());
  const mapped = new Map();
  asset.scene.updateMatrixWorld(true);
  asset.scene.traverse(object => {
    const id = asset.parser.associations.get(object)?.nodes;
    if (id !== undefined && asset.parser.json.nodes[id].mesh !== undefined) {
      const mesh = asset.parser.json.meshes[asset.parser.json.nodes[id].mesh];
      const bounds = new THREE.Box3();
      for (const primitive of mesh.primitives) {
        const accessor = asset.parser.json.accessors[primitive.attributes.POSITION];
        if (accessor.min && accessor.max) {
          bounds.expandByPoint(new THREE.Vector3(...accessor.min));
          bounds.expandByPoint(new THREE.Vector3(...accessor.max));
        }
      }
      const size = bounds.getSize(new THREE.Vector3());
      const longest = size.x >= size.y && size.x >= size.z ? 0 : size.y >= size.z ? 1 : 2;
      const axes = [0,1,2].map(axis => size.getComponent(axis)).sort((a,b) => b-a);
      const axis = new THREE.Vector3().setComponent(longest,
        bounds.min.getComponent(longest) + bounds.max.getComponent(longest) <= 0 ? 1 : -1);
      mapped.set(id,{
        object, position:object.position.clone(), rotation:object.quaternion.clone(),
        localAxis:axis, localCenter:bounds.getCenter(new THREE.Vector3()), slender:axes[0] > 2*axes[1], length:axes[0],
        centered:Math.abs(bounds.min.getComponent(longest)+bounds.max.getComponent(longest)) < axes[0]*.1,
        worldBounds:bounds.clone().applyMatrix4(object.matrixWorld),
        worldPosition:object.getWorldPosition(new THREE.Vector3()),
        worldRotation:object.getWorldQuaternion(new THREE.Quaternion())
      });
    }
  });
  // A catalog loaded before a model was replaced can hold an old part count; the model itself is the truth.
  if (mapped.size !== item.mesh_nodes) console.warn('Model has ' + mapped.size + ' parts; the catalog listed ' + item.mesh_nodes + '.');
  if (model) {
    model.removeFromParent();
    const resources = new Set();
    model.traverse(node => {
      if (node.geometry) resources.add(node.geometry);
      for (const material of [node.material].flat().filter(Boolean)) {
        resources.add(material);
        for (const value of Object.values(material)) if (value?.isTexture) resources.add(value);
      }
    });
    for (const resource of resources) resource.dispose();
  }
  model = asset.scene;
  model.traverse(node => {
    if (!node.isMesh) return;
    node.material = [node.material].flat().map(material => {
      const own = material.clone();
      own.userData.baseEmissive = (own.emissive ?? new THREE.Color()).clone();
      return own;
    });
    if (node.material.length === 1) node.material = node.material[0];
  });
  finishLook(model);      // after each mesh has its own material copy, so one part's finish stays its own
  for (const part of mapped.values()) {
    part.materials = [];
    part.object.traverse(node => { if (node.isMesh) part.materials.push(...[node.material].flat().filter(m => m.emissive)); });
  }
  parts = mapped;
  extras = [];
  kit = item;
  actions = [];
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  sourceUrl = undefined;
  prediction = undefined;
  views[0].scene.add(model);
  $('right').hidden = true;
  $('prediction').value = '';
  $('use-pa').checked = false;
  $('comparison').textContent = 'No Manual-PA run available for this kit.';
  $('kitName').textContent = kit.name;
  $('manual').href = '/api/kits/' + kit.article_number + '/manual';
  const bounds = new THREE.Box3().setFromObject(model);
  const size = bounds.getSize(new THREE.Vector3()).length();
  bounds.getCenter(assemblyCenter);
  controls.target.copy(assemblyCenter);
  camera.position.copy(controls.target).addScaledVector(new THREE.Vector3(.9,.65,1.2),size);
  camera.near = Math.max(.00001,size/1000);
  camera.far = size*100;
  travel = size*.16;
  controls.update();
  showStep(0);
}

function draw(view) {
  const {clientWidth:width,clientHeight:height} = view.host;
  if (!width || !height) return;
  const canvas = view.renderer.domElement;
  if (canvas.width !== width || canvas.height !== height) view.renderer.setSize(width,height,false);
  camera.aspect = width/height;
  camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(viewAngle / 2))
    * Math.max(1, 1 / camera.aspect)));
  camera.updateProjectionMatrix();
  view.renderer.render(view.scene,camera);
}

function posesThrough(index) {
  const poses = new Map();
  if (index < 0) return poses;
  for (const item of prediction.steps.slice(0,index+1)) {
    for (const pose of item.poses) poses.set(pose.node,pose);
  }
  return poses;
}

function setWorldPose(object, pose) {
  object.position.copy(object.parent.worldToLocal(new THREE.Vector3(...pose.position)));
  object.quaternion.copy(object.parent.getWorldQuaternion(new THREE.Quaternion()).invert())
    .multiply(new THREE.Quaternion(...pose.quaternion).normalize());
}

function tick(now) {
  requestAnimationFrame(tick);
  if (!model) return;
  // a step plays for as long as its manual says (duration_s), otherwise the default
  const span = actions[step]?.duration_s ? actions[step].duration_s*1000 : interval;
  if (playing && now-started >= span) {
    if (step < actions.length-1) showStep(step+1);
    else { playing = false; updateControls(); }
  }
  const progress = Math.min(1,Math.max(0,(now-started)/(span*.75)));
  const action = actions[step];
  const previous = new Set(actions.slice(0,step).flatMap(item => item.nodes));
  const active = new Set(action?.nodes);
  const savedOffsets = new Map();
  for (const earlier of actions.slice(0,step)) {
    for (const id of earlier.nodes) {
      const end = earlier.motions?.[id]?.end ?? earlier.end_offset;
      if (end) savedOffsets.set(id,end);
    }
  }
  // Per-part motions come from the server (worked out from the finished model); a saved plan may
  // instead carry one hand-entered offset for the whole action.
  const keyed = Boolean(action?.motions || action?.start_offset);
  const eased = progress < .5 ? 4*progress**3 : 1-(-2*progress+2)**3/2;
  const predictedStep = prediction?.steps.findIndex(item => item.step === action?.step) ?? -1;
  const compared = prediction ? posesThrough(predictedStep) : new Map();
  const usePrediction = $('use-pa').checked && prediction && action;
  const earlierSteps = new Set(actions.slice(0,step+1).map(item => item.step));
  const playback = new Map();
  if (usePrediction) {
    for (const item of prediction.steps) {
      if (earlierSteps.has(item.step)) for (const pose of item.poses) playback.set(pose.node,pose);
    }
  }
  const driven = [];                     // screws being turned in this frame, for the allen keys
  for (const [id,part] of parts) {
    const {object,position,rotation,localAxis,slender,length,centered} = part;
    object.visible = !action || previous.has(id) || active.has(id);
    object.position.copy(position);
    object.quaternion.copy(rotation);
    const pose = playback.get(id);
    if (pose) setWorldPose(object,pose);
    const isFastener = slender && ['insert','tighten'].includes(action?.motion);
    if (keyed) {
      const move = action.motions?.[id]
        ?? (action.start_offset && {start:action.start_offset,end:action.end_offset,turns:action.turns});
      const offset = active.has(id) && move
        ? (move.path ? alongPath(move.path,eased) : new THREE.Vector3(...move.start).lerp(new THREE.Vector3(...move.end),eased))
        : new THREE.Vector3(...(savedOffsets.get(id) ?? [0,0,0]));
      const world = object.getWorldPosition(new THREE.Vector3()).add(offset);
      object.position.copy(object.parent.worldToLocal(world));
      if (active.has(id) && move?.turns) {
        spin(part,(progress-1)*Math.PI*2*move.turns);
        if (slender) driven.push({part,angle:(progress-1)*Math.PI*2*move.turns,
          from:new THREE.Vector3(...(move.path?.[0] ?? move.start)).sub(new THREE.Vector3(...(move.path?.at(-1) ?? move.end)))});
      }
    } else if (active.has(id) && action.entry_target !== undefined) {
      const target = parts.get(action.entry_target).worldPosition;
      const direction = part.worldPosition.clone().sub(target);
      if (action.motion === 'tighten') {
        const shaft = localAxis.clone().applyQuaternion(object.getWorldQuaternion(new THREE.Quaternion()));
        direction.copy(shaft.multiplyScalar(direction.dot(shaft) < 0 ? -1 : 1));
        spin(part,(progress-1)*Math.PI*3);
        driven.push({part,from:direction.clone(),angle:(progress-1)*Math.PI*3});
      }
      const distance = THREE.MathUtils.lerp(action.distance,action.seat_offset ?? 0,progress);
      const position = object.getWorldPosition(new THREE.Vector3()).addScaledVector(direction.normalize(),distance);
      object.position.copy(object.parent.worldToLocal(position));
    } else if (active.has(id) && !previous.has(id)) {
      const world = object.getWorldPosition(new THREE.Vector3());
      const direction = new THREE.Vector3(0,1,0);
      if (isFastener) {
        direction.copy(localAxis).applyQuaternion(object.getWorldQuaternion(new THREE.Quaternion()));
        if (centered && direction.dot(world.clone().sub(assemblyCenter)) < 0) direction.negate();
      } else if (action.motion === 'slide') {
        const delta = world.clone().sub(assemblyCenter);
        const side = Math.abs(delta.x) > Math.abs(delta.z) ? 0 : 2;
        if (Math.abs(delta.getComponent(side)) > .00001) {
          direction.set(0,0,0).setComponent(side,Math.sign(delta.getComponent(side)));
        }
      }
      const distance = isFastener ? Math.min(travel,length*1.35) : travel;
      world.addScaledVector(direction,distance*(1-progress)**2);
      object.position.copy(object.parent.worldToLocal(world));
    }
    if (active.has(id) && action.motion === 'tighten' && isFastener && action.entry_target === undefined && !keyed) {
      spin(part,(progress-1)*Math.PI*2);
      driven.push({part,from:null,angle:(progress-1)*Math.PI*2});
    }
  }
  // Glow on the parts being fitted, fading out once they are seated
  const settle = Math.min(1,Math.max(0,(now-started-span*.75)/350));
  driveKeys(driven,progress,settle);
  for (const [id,part] of parts) {
    const k = action && active.has(id) ? 0.32*(1-settle) : 0;
    for (const material of part.materials) material.emissive.copy(material.userData.baseEmissive).lerp(GLOW,k);
  }
  if (glide) {
    glide.t = Math.min(1,glide.t+(now-(glide.last ?? now))/700);
    glide.last = now;
    const k = glide.t < .5 ? 4*glide.t**3 : 1-(-2*glide.t+2)**3/2;
    const target = glide.from.clone().lerp(glide.to,k);
    camera.position.add(target.clone().sub(controls.target));
    controls.target.copy(target);
    if (glide.t >= 1) glide = undefined;
  }
  draw(views[0]);
  recordFrame?.();
  if (!prediction || !action) return;
  for (const [id,{object}] of parts) {
    const pose = compared.get(id);
    object.visible = Boolean(pose);
    if (pose) setWorldPose(object,pose);
  }
  views[1].scene.add(model);
  draw(views[1]);
  views[0].scene.add(model);
}

requestAnimationFrame(tick);

async function perform(work) {
  busy = true;
  playing = false;
  updateControls();
  try { await work(); } catch (error) { status(error.message); }
  finally { busy = false; updateControls(); }
}

async function readManual(pdf, fresh = false) {
  status('Reading the PDF and preparing the walkthrough…');
  const body = new FormData();
  body.append('file', pdf, pdf.name || 'manual.pdf');
  const response = await fetch(`/api/kits/scan?fresh=${fresh}`, {method:'POST', body});
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail);
  await load(result.kit);
  await addHardware(result.plan.hardware);
  setPlan(result.plan, pdf);
  status('');
}

function setPlan(plan, pdf) {
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  sourceUrl = URL.createObjectURL(pdf);
  actions = plan.actions;
  showStep(0);
}

// While Gemini reads a manual, the model stays hidden behind a progress card. When the steps are ready
// the card says so, and the walkthrough starts from step 1 once the person taps OK.
async function prepare(work) {
  const stages = [[0,'Gemini is reading the pages.'],[8,'Finding the parts on each page…'],
    [20,'Working out the order of the steps…'],[40,'Building the 3D animation…'],[75,'Almost there, a big manual takes longer…']];
  document.body.classList.add('preparing');
  $('prep').hidden = false;
  $('prep-wait').hidden = false;
  $('prep-ready').hidden = true;
  const started = Date.now();
  const tick = () => {
    const seconds = (Date.now()-started)/1000;
    $('prep-stage').textContent = stages.filter(([at]) => seconds >= at).at(-1)[1];
    $('prep-bar').style.width = Math.min(95, 100*(1-Math.exp(-seconds/30))) + '%';   // never claims to be done early
    $('prep-time').textContent = seconds < 5 ? 'This usually takes up to a minute.' : Math.round(seconds) + ' s';
  };
  tick();
  const timer = setInterval(tick, 1000);
  try {
    await work();
  } catch (error) {
    document.body.classList.remove('preparing');
    $('prep').hidden = true;
    throw error;
  } finally { clearInterval(timer); }
  const printed = new Set(actions.map(item => item.step)).size;
  $('prep-bar').style.width = '100%';
  $('prep-summary').textContent = kit?.name ? `${kit.name}: ${printed} step${printed === 1 ? '' : 's'}, animated in 3D.`
    : `${printed} step${printed === 1 ? '' : 's'}, animated in 3D.`;
  $('prep-wait').hidden = true;
  $('prep-ready').hidden = false;
  $('prep-ok').focus();
  await new Promise(resolve => { $('prep-ok').onclick = resolve; });
  $('prep').hidden = true;
  document.body.classList.remove('preparing');
  showStep(0);   // replays step 1 now that it can be seen
}

$('scan').onsubmit = event => {
  event.preventDefault();
  const file = $('pdf').files[0];
  if (file) perform(() => prepare(() => readManual(file)));
};
$('prediction').onchange = async () => {
  try {
    const result = JSON.parse(await $('prediction').files[0].text());
    const vector = (a,n) => Array.isArray(a) && a.length === n && a.every(Number.isFinite);
    if (result.article_number !== kit.article_number || !Array.isArray(result.steps) || !result.steps.length) {
      throw new Error('Choose a prediction export for this kit.');
    }
    const stepIds = new Set();
    for (const entry of result.steps) {
      if (typeof entry.step !== 'string' || stepIds.has(entry.step) || !Array.isArray(entry.poses)) {
        throw new Error('Invalid or duplicate prediction step.');
      }
      stepIds.add(entry.step);
      const ids = new Set();
      for (const pose of entry.poses) {
        if (!parts.has(pose.node) || ids.has(pose.node) || !vector(pose.position,3) ||
            !vector(pose.quaternion,4) || Math.abs(Math.hypot(...pose.quaternion)-1)>.02) {
          throw new Error('Invalid prediction node or pose.');
        }
        ids.add(pose.node);
      }
    }
    prediction = result;
    $('use-pa').checked = false;
    $('right').hidden = false;
    showStep(step);
    if (!actions.length) $('comparison').textContent = 'Prediction loaded. Read the manual to synchronize steps.';
  } catch (error) { $('comparison').textContent = error.message; }
};
$('use-pa').onchange = () => {
  showStep(step);
  status($('use-pa').checked
    ? 'Playback uses imported positions where available. Remaining parts use IKEA positions.'
    : 'Playback uses the original IKEA positions.');
};

// ===================== Scan my step (iPhone app with LiDAR) =====================
// The app opens an AR camera, lets the person line a ghost of this step up with their furniture,
// scans it, and has the server compare it with the model (server/stepcheck.py).
const scanBridge = () => window.webkit?.messageHandlers?.furnisure;

function stepGroup(index) {
  const number = actions[index]?.step;
  const first = actions.findIndex(item => item.step === number);
  let last = first;
  while (last+1 < actions.length && actions[last+1].step === number) last++;
  return {number,first,last};
}

$('check-step').onclick = () => {
  const {number,first,last} = stepGroup(step);
  const ids = list => [...new Set(list.flatMap(item => item.nodes))];
  const installed = ids(actions.slice(0,first)).filter(id => typeof id === 'number');
  const active = ids(actions.slice(first,last+1)).filter(id => typeof id === 'number' && !installed.includes(id));
  const placed = new Set([...installed,...active]);
  const future = [...parts.keys()].filter(id => typeof id === 'number' && !placed.has(id));
  const hardware = {};
  for (const id of ids(actions.slice(first,last+1))) {
    const kind = hardwareSpecs[id]?.kind;
    if (kind) hardware[kind] = (hardware[kind] ?? 0) + 1;
  }
  scanBridge().postMessage({
    type:'scanStep', kit:kit.article_number, name:kit.name, step:String(number),
    stepIndex:first, lastIndex:last, isLast:last === actions.length-1,
    captions:[...new Set(actions.slice(first,last+1).map(item => item.caption))],
    installed, active, future,
    hardware:Object.entries(hardware).map(([name,count]) => ({name,count}))
  });
};

// Called by the app when the person closes the scan: move on, or stay to fix and rescan
window.furnisureScanDone = ({next} = {}) => {
  if (!next) return;
  const {last} = stepGroup(step);
  playing = false;
  showStep(Math.min(last+1,actions.length-1));
};
$('prev').onclick = () => { playing = false; showStep(step-1); };
$('next').onclick = () => { playing = false; showStep(step+1); };
$('play').onclick = () => { playing = !playing; if (playing) showStep(step === actions.length-1 ? 0 : step); updateControls(); };

$('export').onclick = () => {
  const mimeType = window.MediaRecorder && ['video/mp4','video/webm;codecs=vp9','video/webm']
    .find(type => MediaRecorder.isTypeSupported(type));
  if (!mimeType || !HTMLCanvasElement.prototype.captureStream) return status('Video recording is unavailable in this browser.');
  const canvas = document.createElement('canvas');
  canvas.width = 1280;
  canvas.height = 720;
  const context = canvas.getContext('2d');
  const stream = canvas.captureStream(30);
  const chunks = [];
  function cleanup() {
    recordFrame = undefined;
    recording = playing = false;
    controls.enabled = true;
    stream.getTracks().forEach(track => track.stop());
    updateControls();
  }
  try {
    const recorder = new MediaRecorder(stream,{mimeType});
    recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
    recorder.onerror = () => { cleanup(); status('Recording failed.'); };
    recorder.onstop = () => {
      if (!recording) return;
      const extension = mimeType.includes('mp4') ? 'mp4' : 'webm';
      const link = document.createElement('a');
      link.href = URL.createObjectURL(new Blob(chunks,{type:mimeType}));
      link.download = kit.article_number + '-assembly.' + extension;
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href),1000);
      cleanup();
      status('Exported ' + extension.toUpperCase() + '.');
    };
    recording = playing = true;
    controls.enabled = false;
    showStep(0);
    recordFrame = () => {
      const frame = views[0].renderer.domElement;
      const scale = Math.min(1280/frame.width,610/frame.height);
      const width = frame.width*scale, height = frame.height*scale;
      context.fillStyle = '#142622';
      context.fillRect(0,0,1280,720);
      context.drawImage(frame,(1280-width)/2,(610-height)/2,width,height);
      context.fillStyle = '#fff';
      context.font = '24px system-ui';
      context.fillText($('stepLabel').textContent + ' · ' + $('step').textContent,32,650,1216);
      context.font = '17px system-ui';
      context.fillText(($('use-pa').checked ? 'Imported poses where available · ' : '') + 'IKEA 3D Assembly Dataset · Inter IKEA Systems B.V. · CC BY-NC-SA 4.0',32,687);
      if (!playing && recorder.state === 'recording') recorder.stop();
    };
    recorder.start();
    status('Recording all steps. Keep this tab visible.');
  } catch (error) { cleanup(); status(error.message); }
};

perform(async () => {
  const response = await fetch('/api/kits');
  if (!response.ok) throw new Error('Could not load the kit catalog.');
  const catalog = await response.json();
  const params = new URLSearchParams(location.search);
  const requested = params.get('kit');
  const selected = catalog.find(item => item.article_number === requested);
  if (!selected) {
    await load(catalog[0]);
    status('Kit loaded. Upload the matching PDF to generate steps.');
  } else if (params.has('read')) {
    // a manual from the home page: keep the model hidden until its steps are ready
    await prepare(() => openKit(selected, params, true));
  } else await openKit(selected, params, false);
});

async function openKit(selected, params, read) {
  await load(selected);
  const auto = params.has('auto');
  const fresh = read && params.has('fresh');
  const kitUrl = '/api/kits/' + encodeURIComponent(selected.article_number);
  const reference = fresh ? null : await fetch(kitUrl + '/reference-plan' + (auto ? '?auto=true' : ''));
  if (reference && !reference.ok && reference.status !== 404) {
    throw new Error('Could not load the annotated steps for this model.');
  }
  if (!reference?.ok && !read) {
    status('Model loaded. Upload the matching PDF to generate steps with Gemini.');
    return;
  }
  const manual = await fetch(kitUrl + '/manual');
  if (!manual.ok) throw new Error('Could not load the matching manual.');
  const pdf = await manual.blob();
  if (!reference?.ok) {
    // Home's match uses the PDF hash, so these are the exact bytes the user selected.
    await readManual(pdf, fresh);
    return;
  }
  const result = await reference.json();
  setPlan(result.plan, pdf);
  status('');
}
