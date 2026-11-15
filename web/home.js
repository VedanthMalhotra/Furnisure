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
