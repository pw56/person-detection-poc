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

      session = await ort.InferenceSession.create('./yolo12n-fp16.onnx', {
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

// Float32の数値をUint16（Float16ビットパターン）に変換するヘルパー関数
function float32ToFloat16(val) {
  const f32 = new Float32Array([val]);
  const u32 = new Uint32Array(f32.buffer)[0];

  const sign = (u32 >> 31) & 0x1;
  let exp = (u32 >> 23) & 0xff;
  let frac = u32 & 0x7fffff;

  if (exp === 0xff) {
    exp = 0x1f;
    frac = frac ? 0x200 : 0;
  } else if (exp === 0) {
    exp = 0;
    frac = 0;
  } else {
    exp = exp - 127 + 15;
    if (exp >= 0x1f) {
      exp = 0x1f;
      frac = 0;
    } else if (exp <= 0) {
      exp = 0;
      frac = 0;
    } else {
      frac = frac >> 13;
    }
  }

  return (sign << 15) | (exp << 10) | frac;
}

// Uint16（Float16ビットパターン）をFloat32の数値に変換するヘルパー関数
function float16ToFloat32(u16) {
  const sign = (u16 >> 15) & 0x1;
  const exp = (u16 >> 10) & 0x1f;
  const frac = u16 & 0x3ff;

  let f32Val = 0;
  if (exp === 0x1f) {
    f32Val = frac ? NaN : (sign ? -Infinity : Infinity);
  } else if (exp === 0) {
    if (frac !== 0) {
      f32Val = (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
    } else {
      f32Val = sign ? -0 : 0;
    }
  } else {
    f32Val = (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
  }

  return f32Val;
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
    return float16ToFloat32(rawData[index]);
  };

  const scaleX = origWidth / MODEL_WIDTH;
  const scaleY = origHeight / MODEL_HEIGHT;

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
