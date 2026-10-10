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
      // WASM環境でFP16を扱うためにSIMDを強制有効化（セッション作成前に実行が必要）
      ort.env.wasm.simd = true;

      session = await ort.InferenceSession.create('./yolo11n-fp16.onnx', {
        executionProviders: ['wasm'],
        // 実行プロバイダ（WASM）のオプションでSIMDを有効化
        providerOptions: {
          wasm: {
            simd: true
          }
        }
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

// DataViewを使用した正確なFloat16/Float32相互変換ユーティリティ
const fp16Buffer = new ArrayBuffer(2);
const fp16View = new DataView(fp16Buffer);

function float32ToFloat16(val) {
  if (typeof fp16View.setFloat16 === 'function') {
    fp16View.setFloat16(0, val, true);
    return fp16View.getUint16(0, true);
  }
  // フォールバック（標準のFloat16Arrayが存在する場合）
  if (typeof Float16Array !== 'undefined') {
    return new Uint16Array(new Float16Array([val]).buffer)[0];
  }
  return 0;
}

function float16ToFloat32(u16) {
  if (typeof fp16View.getFloat16 === 'function') {
    fp16View.setUint16(0, u16, true);
    return fp16View.getFloat16(0, true);
  }
  if (typeof Float16Array !== 'undefined') {
    return new Float16Array(new Uint16Array([u16]).buffer)[0];
  }
  return 0;
}

// 前処理
async function preprocess(img, targetWidth, targetHeight) {
  const offscreenCanvas = document.createElement('canvas');
  offscreenCanvas.width = targetWidth;
  offscreenCanvas.height = targetHeight;
  const offCtx = offscreenCanvas.getContext('2d');

  offCtx.drawImage(img, 0, 0, targetWidth, targetHeight);
  const imageData = offCtx.getImageData(0, 0, targetWidth, targetHeight);
  const { data } = imageData;

  const float16Data = new Uint16Array(3 * targetWidth * targetHeight);

  for (let i = 0; i < targetWidth * targetHeight; i++) {
    float16Data[i] = float32ToFloat16(data[i * 4] / 255.0);
    float16Data[targetWidth * targetHeight + i] = float32ToFloat16(data[i * 4 + 1] / 255.0);
    float16Data[2 * targetWidth * targetHeight + i] = float32ToFloat16(data[i * 4 + 2] / 255.0);
  }

  return new ort.Tensor('float16', float16Data, [1, 3, targetHeight, targetWidth]);
}

// 描画処理
function processAndDrawBoxes(outputTensor, origWidth, origHeight, ctx, threshold) {
  const dims = outputTensor.dims;
  const rawData = outputTensor.data;

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

  const getValue = (index) => {
    if (rawData instanceof Float32Array) {
      return rawData[index];
    }
    if (typeof Float16Array !== 'undefined' && rawData instanceof Float16Array) {
      return rawData[index];
    }
    return float16ToFloat32(rawData[index]);
  };

  // 座標が正規化（0.0〜1.0）されているか判定するための最大値確認
  let maxCoordVal = 0;
  const sampleCount = Math.min(100, numBoxes);
  for (let i = 0; i < sampleCount; i++) {
    const cx = !isTransposed ? getValue(0 * numBoxes + i) : getValue(i * numChannels + 0);
    const cy = !isTransposed ? getValue(1 * numBoxes + i) : getValue(i * numChannels + 1);
    if (cx > maxCoordVal) maxCoordVal = cx;
    if (cy > maxCoordVal) maxCoordVal = cy;
  }

  // 最大値が1.5以下の場合は正規化（0〜1）出力とみなしスケールを変更
  const isNormalized = maxCoordVal <= 1.5;
  const scaleX = isNormalized ? origWidth : origWidth / MODEL_WIDTH;
  const scaleY = isNormalized ? origHeight : origHeight / MODEL_HEIGHT;

  for (let i = 0; i < numBoxes; i++) {
    let cx, cy, w, h, personScore;

    if (!isTransposed) {
      cx = getValue(0 * numBoxes + i);
      cy = getValue(1 * numBoxes + i);
      w  = getValue(2 * numBoxes + i);
      h  = getValue(3 * numBoxes + i);
      personScore = getValue(4 * numBoxes + i);
    } else {
      const stride = numChannels;
      cx = getValue(i * stride + 0);
      cy = getValue(i * stride + 1);
      w  = getValue(i * stride + 2);
      h  = getValue(i * stride + 3);
      personScore = getValue(i * stride + 4);
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
