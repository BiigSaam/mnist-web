const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);

const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const resultEl = document.getElementById('result');

let drawing = false;

canvas.addEventListener('mousedown', () => drawing = true);
canvas.addEventListener('mouseup', () => drawing = false);
canvas.addEventListener('mouseout', () => drawing = false);
canvas.addEventListener('mousemove', draw);

function draw(e) {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  ctx.fillStyle = 'white'; // Dessin blanc sur fond noir
  ctx.beginPath();
  ctx.arc(x, y, 10, 0, Math.PI * 2);
  ctx.fill();
}

// Nettoyer le canvas
clearBtn.addEventListener('click', () => {
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  resultEl.textContent = '';
});

// Prédire
predictBtn.addEventListener('click', async () => {
  const imageData = ctx.getImageData(0, 0, 280, 280);
  const processed = downscale(imageData, 28, 28);

  const inputTensor = new ort.Tensor('float32', Float32Array.from(processed), [1, 1, 28, 28]);

  try {
    const session = await ort.InferenceSession.create('mnist_cnn.onnx');
    const feeds = { input: inputTensor };
    const output = await session.run(feeds);
    const outputData = output[Object.keys(output)[0]].data;

    const predicted = outputData.indexOf(Math.max(...outputData));
    resultEl.textContent = `Classe prédite : ${predicted}`;
  } catch (err) {
    console.error(err);
    resultEl.textContent = 'Erreur : ' + err.message;
  }
});

// Fonction améliorée : centrage automatique + inversion
function downscale(imageData, width, height) {
  const tmpCanvas = document.createElement('canvas');
  const tmpCtx = tmpCanvas.getContext('2d');

  tmpCanvas.width = imageData.width;
  tmpCanvas.height = imageData.height;
  tmpCtx.putImageData(imageData, 0, 0);

  const pixels = tmpCtx.getImageData(0, 0, tmpCanvas.width, tmpCanvas.height).data;
  const binary = [];

  for (let i = 0; i < pixels.length; i += 4) {
    const avg = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
    binary.push(avg < 128 ? 1 : 0);
  }

  let [minX, minY, maxX, maxY] = [imageData.width, imageData.height, 0, 0];

  for (let y = 0; y < imageData.height; y++) {
    for (let x = 0; x < imageData.width; x++) {
      if (binary[y * imageData.width + x]) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  const boxWidth = maxX - minX + 1;
  const boxHeight = maxY - minY + 1;

  const digitCanvas = document.createElement('canvas');
  const digitCtx = digitCanvas.getContext('2d');
  digitCanvas.width = width;
  digitCanvas.height = height;

  const scale = Math.min(width / boxWidth, height / boxHeight);
  const xOffset = (width - boxWidth * scale) / 2;
  const yOffset = (height - boxHeight * scale) / 2;

  digitCtx.fillStyle = 'black';
  digitCtx.fillRect(0, 0, width, height);
  digitCtx.drawImage(tmpCanvas,
    minX, minY, boxWidth, boxHeight,
    xOffset, yOffset, boxWidth * scale, boxHeight * scale
  );

  const finalData = digitCtx.getImageData(0, 0, width, height).data;
  const grayscale = [];

  for (let i = 0; i < finalData.length; i += 4) {
    const avg = (finalData[i] + finalData[i + 1] + finalData[i + 2]) / 3;
    grayscale.push(avg / 255); // Normalisation entre 0 et 1
  }
  
  console.log('Grayscale data:', grayscale);
  return grayscale;
}
