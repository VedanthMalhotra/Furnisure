const $ = id => document.getElementById(id);

// ---- Drag and drop: hand the dropped files to the same input main.js already listens to
const drop = $('drop');
['dragenter', 'dragover'].forEach(type => drop.addEventListener(type, event => {
  event.preventDefault();
  drop.classList.add('over');
}));
['dragleave', 'drop'].forEach(type => drop.addEventListener(type, () => drop.classList.remove('over')));
drop.addEventListener('drop', event => {
  event.preventDefault();
  const files = [...event.dataTransfer.files].filter(f => f.type === 'application/pdf' || f.type.startsWith('image/'));
  if (!files.length) return;
  const list = new DataTransfer();
  files.forEach(f => list.items.add(f));
  $('files').files = list.files;
  $('files').dispatchEvent(new Event('change'));
});

const preview = $('assemblyPreview');
const toggle = $('previewToggle');
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
let previewVisible = false;
let userWantsPlayback;

function updatePreview() {
  const shouldPlay = userWantsPlayback ?? !reduceMotion.matches;
  if (shouldPlay && previewVisible && !document.hidden) {
    preview.play().catch(() => { /* The play button stays available if autoplay is blocked. */ });
  } else {
    preview.pause();
  }
}

for (const [event, label] of [['play', 'Pause preview'], ['pause', 'Play preview']]) {
  preview.addEventListener(event, () => { toggle.textContent = label; });
}
preview.addEventListener('error', () => { toggle.hidden = true; });
toggle.addEventListener('click', () => {
  userWantsPlayback = preview.paused;
  if (userWantsPlayback) preview.play().catch(() => {});
  else preview.pause();
});

const observer = new IntersectionObserver(([entry]) => {
  previewVisible = entry.isIntersecting;
  updatePreview();
}, { threshold: 0.25 });
observer.observe(preview);
document.addEventListener('visibilitychange', updatePreview);
reduceMotion.addEventListener('change', updatePreview);
