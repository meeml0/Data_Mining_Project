## Leaf Segmentation API

### Kurulum

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Uç Noktaları

- `GET /health` — Servis sağlık kontrolü
- `POST /predict` — `multipart/form-data` ile `file` alanına görsel yükleme

### Yanıt Formatı

```json
{
  "success": true,
  "results": {
    "disease_ratio": 0.32,
    "healthy_ratio": 0.68,
    "segmentation_mask": "base64_string"
  }
}
```

### Notlar

- Bu repo, model çalışmasını sahte (mock) olarak uygular. Gerçek model entegrasyonu için `app/model.py` dosyasındaki `ModelService` sınıfına PyTorch/TensorFlow model yükleme ve çıkarım (inference) kodunu ekleyin.


istek deneme:
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@leaf.jpg" => url + photo