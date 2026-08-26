let session = null;
let currentImage = null;
let currentOutputTensor = null;

const MODEL_WIDTH = 640;
const MODEL_HEIGHT = 640;

// DOMの読み込み完了後にすべての処理・初期化を実行
document.addEventListener('DOMContentLoaded', async () => {
  const imageInput = document.getElementById('imageInput');
  const thresholdInput = document.getElementById('thresholdInput');
  const thresholdValue = document.getElementById('thresholdValue');
  const canvas = document.getElementById('outputCanvas');
  const ctx = canvas.getContext('2d');

  // モデルの初期化
  async function initModel() {
    try {
      session = await ort.InferenceSession.create('./yolo12n.onnx', {
        executionProviders: ['wasm']
      });
      console.log('Model loaded successfully');
    } catch (e) {
      console.error('Failed to load model:', e);
    }
  }

  // スライダー変更時の処理（数値ラベルの更新と描画）
  thresholdInput.addEventListener('input', (e) => {
    const val = e.target.value;
    thresholdValue.textContent = `${val}%`;
    
    if (currentImage && currentOutputTensor) {
      redraw();
    }
  });

  // 画像選択時の処理
  imageInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file || !session) return;

    const img = new Image();
    img.src = URL.createObjectURL(file);
    await img.decode();
    
    currentImage = img;
    canvas.width = img.width;
    canvas.height = img.height;

    // 前処理と推論
    const inputTensor = await preprocess(img, MODEL_WIDTH, MODEL_HEIGHT);
    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;
    
    const results = await session.run(feeds);
    currentOutputTensor = results[session.outputNames[0]];

    redraw();
  });

  // 再描画処理
  function redraw() {
    const threshold = parseFloat(thresholdInput.value) / 100;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(currentImage, 0, 0);

    processAndDrawBoxes(currentOutputTensor, currentImage.width, currentImage.height, ctx, threshold);
  }

  // モデル読み込みの開始
  await initModel();
});

// 前処理
async function preprocess(img, targetWidth, targetHeight) {
  const offscreenCanvas = document.createElement('canvas');
  offscreenCanvas.width = targetWidth;
  offscreenCanvas.height = targetHeight;
  const offCtx = offscreenCanvas.getContext('2d');

  offCtx.drawImage(img, 0, 0, targetWidth, targetHeight);
  const imageData = offCtx.getImageData(0, 0, targetWidth, targetHeight);
  const { data } = imageData;

  const float32Data = new Float32Array(3 * targetWidth * targetHeight);

  for (let i = 0; i < targetWidth * targetHeight; i++) {
    float32Data[i] = data[i * 4] / 255.0;
    float32Data[targetWidth * targetHeight + i] = data[i * 4 + 1] / 255.0;
    float32Data[2 * targetWidth * targetHeight + i] = data[i * 4 + 2] / 255.0;
  }

  return new ort.Tensor('float32', float32Data, [1, 3, targetHeight, targetWidth]);
}

// 描画処理
function processAndDrawBoxes(outputTensor, origWidth, origHeight, ctx, threshold) {
  const dims = outputTensor.dims;
  const data = outputTensor.data;

  let numBoxes, numChannels;
  let isTransposed = false;

  if (dims.length === 3) {
    if (dims[1] < dims[2]) {
      numChannels = dims[1];
      numBoxes = dims[2];
      isTransposed = false;
    } else {
      numBoxes = dims[1];
      numChannels = dims[2];
      isTransposed = true;
    }
  } else {
    return;
  }

  const scaleX = origWidth / MODEL_WIDTH;
  const scaleY = origHeight / MODEL_HEIGHT;

  for (let i = 0; i < numBoxes; i++) {
    let cx, cy, w, h, personScore;

    if (!isTransposed) {
      cx = data[0 * numBoxes + i];
      cy = data[1 * numBoxes + i];
      w  = data[2 * numBoxes + i];
      h  = data[3 * numBoxes + i];
      personScore = data[4 * numBoxes + i];
    } else {
      const stride = numChannels;
      cx = data[i * stride + 0];
      cy = data[i * stride + 1];
      w  = data[i * stride + 2];
      h  = data[i * stride + 3];
      personScore = data[i * stride + 4];
    }

    if (personScore >= threshold) {
      const x = (cx - w / 2) * scaleX;
      const y = (cy - h / 2) * scaleY;
      const boxWidth = w * scaleX;
      const boxHeight = h * scaleY;

      ctx.beginPath();
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, boxWidth, boxHeight);

      ctx.fillStyle = '#00FF00';
      ctx.font = '16px sans-serif';
      ctx.fillText(
        `person: ${(personScore * 100).toFixed(1)}%`,
        x,
        y > 20 ? y - 5 : y + 15
      );
    }
  }
}
