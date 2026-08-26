let session = null;
let currentImage = null;
let currentOutputTensor = null;

const MODEL_WIDTH = 640;
const MODEL_HEIGHT = 640;

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

// 画像が選択された時の処理
imageInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file || !session) return;

  const img = new Image();
  img.src = URL.createObjectURL(file);
  await img.decode();
  
  // 保持
  currentImage = img;

  canvas.width = img.width;
  canvas.height = img.height;

  // 1. 前処理
  const inputTensor = await preprocess(img, MODEL_WIDTH, MODEL_HEIGHT);

  // 2. 推論実行
  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;
  const results = await session.run(feeds);

  // 出力データの保持
  currentOutputTensor = results[session.outputNames[0]];

  // 3. 描画の実行
  redraw();
});

// スライダー変更時の処理
thresholdInput.addEventListener('input', (e) => {
  const val = e.target.value;
  thresholdValue.textContent = `${val}%`;
  
  // 画像と推論結果があれば再描画
  if (currentImage && currentOutputTensor) {
    redraw();
  }
});

// 描画処理関数
function redraw() {
  const threshold = parseFloat(thresholdInput.value) / 100;

  // 元画像を再描画してキャンバスをクリア
  ctx.drawImage(currentImage, 0, 0);

  // 閾値を適用して枠を描画
  processAndDrawBoxes(currentOutputTensor, currentImage.width, currentImage.height, ctx, threshold);
}

// 画像の前処理
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

// ボックス描画処理
function processAndDrawBoxes(outputTensor, origWidth, origHeight, ctx, threshold) {
  const [batch, channels, numBoxes] = outputTensor.dims;
  const data = outputTensor.data;

  const scaleX = origWidth / MODEL_WIDTH;
  const scaleY = origHeight / MODEL_HEIGHT;

  for (let i = 0; i < numBoxes; i++) {
    const cx = data[0 * numBoxes + i];
    const cy = data[1 * numBoxes + i];
    const w  = data[2 * numBoxes + i];
    const h  = data[3 * numBoxes + i];

    const personScore = data[4 * numBoxes + i];

    // 指定された閾値以上のものだけを描画
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

// 初期化実行
initModel();