let tracker;

// モデルの初期化 (YOLOv12-nano / RTMOモデル)
async function initModel() {
  if (window.rtmlib) {
    // rtmlibの検出器を準備 (信頼度の閾値を0に設定)
    tracker = new rtmlib.YOLOv8({
      modelPath: 'https://huggingface.co/models/yolov12n.onnx', // YOLOv12-nano ONNXパス
      scoreThreshold: 0.0 // 信頼度の基準なし（全検出）
    });
    await tracker.init();
  }
}

document.getElementById('imageInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const img = new Image();
  img.src = URL.createObjectURL(file);
  await img.decode();

  const canvas = document.getElementById('outputCanvas');
  const ctx = canvas.getContext('2d');
  
  // 1. キャンバスの内部解像度を画像の元のサイズに合わせる
  canvas.width = img.width;
  canvas.height = img.height;

  // 2. 画像をキャンバスに描画（必ず線を描く前に実行）
  ctx.drawImage(img, 0, 0);

  // 3. 推論実行
  const predictions = await tracker.detect(img);

  // 4. 検出された全てのバウンディングボックスを描画
  predictions.forEach(pred => {
    // COCOデータセットにおける人物 (person) のクラスIDは 0
    if (pred.classId === 0 || pred.label === 'person') {
      const [x, y, width, height] = pred.bbox;

      // 描画状態の設定をリセットして明確に適用
      ctx.beginPath();
      ctx.strokeStyle = '#00FF00'; // 見えやすい蛍光グリーン
      ctx.lineWidth = 4;
      ctx.fillStyle = '#00FF00';
      ctx.font = '20px sans-serif';

      // 枠線の描画 (strokeRect実行で実際に描画される)
      ctx.strokeRect(x, y, width, height);

      // 信頼度テキストの描画
      ctx.fillText(
        `person: ${(pred.score * 100).toFixed(1)}%`, 
        x, 
        y > 25 ? y - 5 : y + 20
      );
    }
  });
});

initModel();