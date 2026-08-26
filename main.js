let session = null;

// COCOデータセットの入力サイズ（標準は640x640）
const MODEL_WIDTH = 640;
const MODEL_HEIGHT = 640;

// ONNXモデルの初期化
async function initModel() {
  try {
    // ONNX Runtime Webでモデルをロード
    // ※ 実際にご使用のYOLOv12-nano ONNXモデルのパスを指定してください
    session = await ort.InferenceSession.create('./yolo12n.onnx', {
      executionProviders: ['wasm']
    });
    console.log('Model loaded successfully');
  } catch (e) {
    console.error('Failed to load model:', e);
  }
}

document.getElementById('imageInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file || !session) return;

  const img = new Image();
  img.src = URL.createObjectURL(file);
  await img.decode();

  const canvas = document.getElementById('outputCanvas');
  const ctx = canvas.getContext('2d');

  // キャンバスサイズを画像サイズに合わせる
  canvas.width = img.width;
  canvas.height = img.height;

  // 元画像を元サイズで描画
  ctx.drawImage(img, 0, 0);

  // 1. 前処理：画像を640x640のFloat32Tensorに変換 (RGB, 0-1正規化, NCHW形式)
  const inputTensor = await preprocess(img, MODEL_WIDTH, MODEL_HEIGHT);

  // 2. 推論実行
  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;
  const results = await session.run(feeds);

  // 出力テンソルの取得
  const output = results[session.outputNames[0]]; // [1, 84, 8400] などの形状

  // 3. 後処理 & 描画（基準なしでClass 0: person を全て抽出）
  processAndDrawBoxes(output, img.width, img.height, ctx);
});

// 画像の前処理 (Resize -> Normalize -> Float32Array [1, 3, 640, 640])
async function preprocess(img, targetWidth, targetHeight) {
  const offscreenCanvas = document.createElement('canvas');
  offscreenCanvas.width = targetWidth;
  offscreenCanvas.height = targetHeight;
  const offCtx = offscreenCanvas.getContext('2d');

  // 640x640にリサイズして描画
  offCtx.drawImage(img, 0, 0, targetWidth, targetHeight);
  const imageData = offCtx.getImageData(0, 0, targetWidth, targetHeight);
  const { data } = imageData; // RGBA配列

  const float32Data = new Float32Array(3 * targetWidth * targetHeight);

  // HWC (Height, Width, Channel) から NCHW (Channel, Height, Width) への変換と正規化 (0-255 -> 0.0-1.0)
  for (let i = 0; i < targetWidth * targetHeight; i++) {
    float32Data[i] = data[i * 4] / 255.0;                         // R
    float32Data[targetWidth * targetHeight + i] = data[i * 4 + 1] / 255.0; // G
    float32Data[2 * targetWidth * targetHeight + i] = data[i * 4 + 2] / 255.0; // B
  }

  return new ort.Tensor('float32', float32Data, [1, 3, targetHeight, targetWidth]);
}

// 出力データの解析と描画
function processAndDrawBoxes(outputTensor, origWidth, origHeight, ctx) {
  const [batch, channels, numBoxes] = outputTensor.dims; // 例: [1, 84, 8400]
  const data = outputTensor.data;

  // YOLOモデルの出力形式:
  // [cx, cy, w, h, class0_score, class1_score, ...] が numBoxes分並んでいる
  const scaleX = origWidth / MODEL_WIDTH;
  const scaleY = origHeight / MODEL_HEIGHT;

  for (let i = 0; i < numBoxes; i++) {
    // データアクセスのインデックス計算
    const cx = data[0 * numBoxes + i];
    const cy = data[1 * numBoxes + i];
    const w  = data[2 * numBoxes + i];
    const h  = data[3 * numBoxes + i];

    // COCOデータセットの Class 0 (person) のスコア
    const personScore = data[4 * numBoxes + i];

    // 「基準なし」ですべて描画（0より大きいスコアがある場合）
    if (personScore > 0) {
      // 640x640座標系から中心座標を左上座標(x, y)に変換し、元画像サイズにスケール
      const x = (cx - w / 2) * scaleX;
      const y = (cy - h / 2) * scaleY;
      const boxWidth = w * scaleX;
      const boxHeight = h * scaleY;

      // バウンディングボックスの描画
      ctx.beginPath();
      ctx.strokeStyle = '#00FF00'; // 固定色の緑
      ctx.lineWidth = 3;           // 線の太さを固定
      ctx.strokeRect(x, y, boxWidth, boxHeight);

      // 信頼度テキストの描画
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

// モデル初期化の実行
initModel();