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
  
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);

  // 推論実行
  const predictions = await tracker.detect(img);

  // 検出された全てのバウンディングボックスを描画
  predictions.forEach(pred => {
    // COCOデータセットにおける人物 (person) のクラスIDは 0
    if (pred.classId === 0 || pred.label === 'person') {
      const [x, y, width, height] = pred.bbox;

      // 枠線の描画
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = Math.max(2, Math.round(canvas.width / 300));
      ctx.strokeRect(x, y, width, height);

      // 信頼度テキストの表示
      ctx.fillStyle = '#00FF00';
      ctx.font = '16px sans-serif';
      ctx.fillText(
        `person: ${(pred.score * 100).toFixed(1)}%`, 
        x, 
        y > 20 ? y - 5 : y + 15
      );
    }
  });
});

initModel();