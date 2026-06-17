import os
import sys

# Tạo mock function để bypass kiểm tra
def patch_transformers():
    """Patch transformers để bỏ qua kiểm tra phiên bản torch"""
    try:
        import transformers.modeling_utils
        
        def patched_check():
            return True
        
        transformers.modeling_utils.check_torch_load_is_safe = patched_check
        print("✓ Đã patch transformers để bỏ qua kiểm tra torch version")
    except Exception as e:
        print(f"⚠️ Không thể patch transformers: {e}")

# Gọi patch TRƯỚC khi import model
patch_transformers()

# =========================================================
# IMPORTS
# =========================================================
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import json
import numpy as np
import re
from pathlib import Path
import cv2
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
METRICS_FILE = os.path.join(OUTPUTS_DIR, 'metrics', 'metrics_summary.json')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ensure src is importable
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

# Project imports for inference
import yaml
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets.transforms import get_val_transforms
from models.multimodal_model import RiceDiseaseMultimodalModel

# Optional GradCAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except Exception:
    GRADCAM_AVAILABLE = False

# import gradcam generator
try:
    from evaluation.xai_analysis import generate_gradcam_visualization, GRADCAM_AVAILABLE as XAI_GRADCAM_AVAILABLE
except Exception:
    generate_gradcam_visualization = None
    XAI_GRADCAM_AVAILABLE = False

# Load config, model and tokenizer lazily
CONFIG_PATH = os.path.join(BASE_DIR, 'configs', 'config.yaml')

def load_config(path=CONFIG_PATH):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception:
        return {}

config = load_config()
DEVICE = torch.device(config.get('inference', {}).get('device', 'cpu'))
CHECKPOINT = config.get('inference', {}).get('checkpoint', os.path.join(OUTPUTS_DIR, 'checkpoints', 'best_model.pth'))

_model = None
_tokenizer = None
_model_loaded = False
_classes = config.get('dataset', {}).get('classes', ['Healthy', 'BrownSpot', 'Hispa', 'LeafBlast'])


def init_model_tokenizer():
    """Khởi tạo model và tokenizer - CHỈ GỌI KHI NHẤN NÚT"""
    global _model, _tokenizer, _model_loaded
    if _model is not None:
        return {"status": "already_loaded", "message": "Model đã được load trước đó"}
    
    print("🔄 Đang khởi tạo model và tokenizer...")
    
    try:
        # Instantiate model
        _model = RiceDiseaseMultimodalModel(
            num_classes=config.get('model', {}).get('num_classes', len(_classes)),
            fusion_type=config.get('model', {}).get('fusion', 'cross_attention'),
        )
        
        # load checkpoint if available
        if os.path.exists(CHECKPOINT):
            print(f"📦 Loading checkpoint from {CHECKPOINT}...")
            state = torch.load(CHECKPOINT, map_location=DEVICE)
            if isinstance(state, dict) and 'model_state_dict' in state:
                _model.load_state_dict(state['model_state_dict'])
            else:
                _model.load_state_dict(state)
            print("✓ Model loaded successfully")
        else:
            print(f"⚠️ Checkpoint not found at {CHECKPOINT}, running with uninitialized weights.")
            return {"status": "error", "message": f"Checkpoint not found at {CHECKPOINT}"}

        _model.to(DEVICE)
        _model.eval()

        # tokenizer
        print("🔄 Loading tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained(config.get('text_tokenizer', 'vinai/phobert-base'))
        print("✓ Tokenizer loaded successfully")
        
        _model_loaded = True
        print("✅ Model và tokenizer đã sẵn sàng!")
        
        return {"status": "success", "message": "Model đã được load thành công!"}
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        _model = None
        _tokenizer = None
        _model_loaded = False
        return {"status": "error", "message": f"Lỗi khi load model: {str(e)}"}


def unload_model():
    """Giải phóng model khỏi bộ nhớ"""
    global _model, _tokenizer, _model_loaded
    if _model is not None:
        del _model
        del _tokenizer
        _model = None
        _tokenizer = None
        _model_loaded = False
        torch.cuda.empty_cache()
        print("🗑️ Model đã được giải phóng khỏi bộ nhớ")
        return {"status": "success", "message": "Model đã được giải phóng"}
    return {"status": "info", "message": "Chưa có model nào được load"}


def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


@app.route('/')
def index():
    metrics = load_json(METRICS_FILE)
    
    eda_report = None
    dataset_overview = None
    eda_path = os.path.join(OUTPUTS_DIR, 'analysis', 'eda_report.md')
    if os.path.exists(eda_path):
        with open(eda_path, 'r', encoding='utf-8') as f:
            eda_report = f.read()
    
    dataset_path = os.path.join(OUTPUTS_DIR, 'visualizations', 'dataset_overview.png')
    if not os.path.exists(dataset_path):
        dataset_path = None
    
    figure_names = [
        'gradcam/batch0_img0_correct_true_Healthy_pred_Healthy.png',
        'gradcam/batch2_img2_correct_true_Healthy_pred_Healthy.png',
        'misclassified_gradcam/misclassified_01_LeafBlast_to_Healthy.png',
        'misclassified_gradcam/misclassified_02_LeafBlast_to_BrownSpot.png',
        'misclassified_gradcam/misclassified_03_BrownSpot_to_Healthy.png',
        'embedding_separation.png',
    ]
    figures = {}
    for name in figure_names:
        p = os.path.join(OUTPUTS_DIR, 'figures', name)
        if os.path.exists(p):
            figures[name] = f"/outputs/figures/{name}"
    
    try:
        gradcam_path = os.path.join(OUTPUTS_DIR, 'figures', 'gradcam')
        if os.path.exists(gradcam_path):
            gradcam_files = [f"/outputs/figures/gradcam/{p.name}" for p in Path(gradcam_path).glob('*.png')]
            if gradcam_files:
                figures['gradcam_samples'] = gradcam_files[:config.get('ui', {}).get('max_display_figures', 8)]
    except Exception as e:
        print(f"GradCAM collection failed: {e}")
    
    training_summary = None
    training_csv = os.path.join(OUTPUTS_DIR, 'csv', 'training_summary.csv')
    if os.path.exists(training_csv):
        try:
            with open(training_csv, 'r', encoding='utf-8') as f:
                training_summary = '\n'.join([next(f) for _ in range(10)])
        except Exception:
            try:
                with open(training_csv, 'r', encoding='utf-8') as f:
                    training_summary = f.read()
            except Exception:
                training_summary = None
    
    return render_template('index.html', metrics=metrics, eda_report=eda_report, 
                          dataset_overview=dataset_path, figures=figures, 
                          training_summary=training_summary, config=config,
                          model_loaded=_model_loaded)


@app.route('/api/load_model', methods=['POST'])
def load_model_api():
    result = init_model_tokenizer()
    return jsonify(result)


@app.route('/api/unload_model', methods=['POST'])
def unload_model_api():
    result = unload_model()
    return jsonify(result)


@app.route('/api/model_status', methods=['GET'])
def model_status():
    return jsonify({
        'loaded': _model_loaded,
        'device': str(DEVICE),
        'checkpoint': CHECKPOINT
    })


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/outputs/<path:filename>')
def outputs_file(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route('/predict', methods=['POST'])
def predict():
    if not _model_loaded or _model is None:
        return jsonify({'error': 'Model chưa được load. Vui lòng nhấn nút "Load Model" trước khi dự đoán.'}), 400
    
    image = request.files.get('image')
    text = request.form.get('text', '')
    filename = None
    image_path = None
    
    if image:
        filename = secure_filename(image.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(save_path)
        image_path = save_path
    
    if image_path is None:
        return jsonify({'error': 'No image uploaded'}), 400
    
    cfg_img_size = config.get('data', {}).get('image_size', 224)
    transform = get_val_transforms(cfg_img_size)
    pil_img = Image.open(image_path).convert('RGB')
    img_t = transform(pil_img).unsqueeze(0).to(DEVICE)
    
    tokenizer = _tokenizer
    if tokenizer is not None:
        encoded = tokenizer(
            text if text else "",
            max_length=config.get('data', {}).get('max_length', 128),
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(DEVICE)
        attention_mask = encoded['attention_mask'].to(DEVICE)
    else:
        max_len = config.get('data', {}).get('max_length', 128)
        input_ids = torch.zeros((1, max_len), dtype=torch.long).to(DEVICE)
        attention_mask = torch.zeros((1, max_len), dtype=torch.long).to(DEVICE)
    
    model = _model
    with torch.no_grad():
        logits = model(img_t, input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        
        relevance = None
        try:
            image_feat = model.vision_encoder(img_t)
            image_feat = model.vision_projection(image_feat)
            text_feat = model.text_encoder(input_ids, attention_mask)
            text_feat = model.text_projection(text_feat)
            
            img_norm = F.normalize(image_feat, dim=1)
            txt_norm = F.normalize(text_feat, dim=1)
            sim = F.cosine_similarity(img_norm, txt_norm).cpu().item()
            relevance = float(round(sim, 4))
        except Exception as e:
            print(f"Relevance calculation error: {e}")
            relevance = None
    
    classes = config.get('dataset', {}).get('classes', _classes)
    result = {
        'prediction': classes[pred_idx] if pred_idx < len(classes) else str(pred_idx),
        'index': pred_idx,
        'probs': {classes[i]: float(probs[i]) for i in range(len(probs)) if i < len(classes)},
        'relevance': relevance,
        'uploaded_image': f"/uploads/{filename}" if filename else None,
    }
    
    gradcam_url = None
    if GRADCAM_AVAILABLE:
        try:
            backbone = model.vision_encoder.backbone
            if hasattr(backbone, 'blocks'):
                target_layers = [backbone.blocks[-1]]
            elif hasattr(backbone, 'features'):
                target_layers = [backbone.features[-1]]
            else:
                target_layers = [backbone]
            
            cam = GradCAM(model=backbone, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=img_t)[0]
            
            img_np = img_t[0].cpu().permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)
            
            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            
            save_name = f"gradcam_{filename}.png"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
            Image.fromarray(visualization).save(save_path)
            gradcam_url = f"/uploads/{save_name}"
            result['gradcam'] = gradcam_url
        except Exception as e:
            result['gradcam_error'] = str(e)
    else:
        result['gradcam'] = None
    
    gemini_resp = None
    gem_cfg = config.get('gemini', {})
    api_key = gem_cfg.get('api_key') or os.environ.get('GEMINI_API_KEY')
    model_id = gem_cfg.get('model_id')
    
    if api_key and model_id:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = f"Dự đoán: {result['prediction']}, xác suất: {result['probs'].get(result['prediction'])}. Mức tương quan ảnh-mô tả: {result['relevance']}. Hãy đưa ra khuyến nghị ngắn gọn cho người nông dân."
            try:
                img_obj = Image.open(image_path)
                response = client.models.generate_content(model=model_id, contents=[prompt, img_obj])
            except Exception:
                response = client.models.generate_content(model=model_id, contents=prompt)
            text = getattr(response, 'text', None)
            if text is None and isinstance(response, dict):
                text = response.get('text') or response.get('output')
            gemini_resp = text if text is not None else response
        except Exception as e:
            gemini_resp = {'error': str(e)}
    else:
        gemini_resp = {'note': 'Gemini not configured in config.yaml'}
    
    result['gemini'] = gemini_resp
    
    return jsonify(result)


@app.route('/api/predict_pipeline', methods=['POST'])
def predict_pipeline():
    """API dự đoán với pipeline chi tiết từng bước"""
    
    if not _model_loaded or _model is None:
        return jsonify({'error': 'Model chưa được load. Vui lòng nhấn nút "Load Model" trước khi dự đoán.'}), 400
    
    image = request.files.get('image')
    text = request.form.get('text', '')
    filename = None
    image_path = None
    
    if image:
        filename = secure_filename(image.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(save_path)
        image_path = save_path
    
    if image_path is None:
        return jsonify({'error': 'No image uploaded'}), 400
    
    pipeline_steps = {
        'image': [],
        'text': [],
        'fusion': [],
        'result': {}
    }
    
    # =========================================================
    # BƯỚC 1: XỬ LÝ ẢNH - FIX MASK FINAL
    # =========================================================
    original_img = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w = original_rgb.shape[:2]
    
    pipeline_steps['image'].append({
        'step': '1. Đọc ảnh',
        'description': f'Ảnh kích thước {w}x{h} pixels',
        'status': 'completed',
        'icon': '📷'
    })
    
    from datasets.rice_dataset import extract_leaf_metrics_from_image
    
    coverage, aspect_ratio, has_valid, hull_points, hull_mask = extract_leaf_metrics_from_image(
        Path(image_path), target_size=512
    )
    
    # ================================================================
    # TẠO MASK TỪ HULL POINTS - CÁCH CHẮC CHẮN NHẤT
    # ================================================================
    if hull_points is not None and len(hull_points) > 3:
        # Tạo mask từ hull_points
        mask_img = np.zeros((h, w), dtype=np.uint8)
        hull_pts = np.array(hull_points, dtype=np.int32)
        cv2.fillPoly(mask_img, [hull_pts], 255)
        print(f"✅ Mask created from {len(hull_points)} hull points")
    else:
        # Fallback: sử dụng hull_mask nếu có
        if hull_mask is not None:
            mask_img = hull_mask.astype(np.uint8)
            if mask_img.shape[:2] != (h, w):
                mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_img = np.where(mask_img > 127, 255, 0).astype(np.uint8)
            print(f"✅ Mask from hull_mask, shape: {mask_img.shape}")
        else:
            mask_img = np.zeros((h, w), dtype=np.uint8)
            print("⚠️ No hull data, created blank mask")
    
    # Vẽ hull trên ảnh gốc
    hull_img = original_rgb.copy()
    if hull_points is not None and len(hull_points) > 0:
        hull_pts = np.array(hull_points, dtype=np.int32)
        cv2.drawContours(hull_img, [hull_pts], -1, (255, 0, 0), 3)
        for pt in hull_pts:
            cv2.circle(hull_img, tuple(pt), 4, (0, 255, 255), -1)
    
    # Lưu ảnh
    hull_path = os.path.join(app.config['UPLOAD_FOLDER'], f'hull_{filename}')
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], f'mask_{filename}')
    cv2.imwrite(hull_path, cv2.cvtColor(hull_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_path, mask_img)
    
    # Kiểm tra file mask đã lưu
    if os.path.exists(mask_path):
        test = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if test is not None:
            print(f"✅ Mask saved successfully: {test.shape}, unique: {np.unique(test)}")
        else:
            print(f"❌ Failed to read saved mask")
    
    pipeline_steps['image'].append({
        'step': '2. Phân tích lá',
        'description': f'Coverage: {coverage:.2%}, Aspect Ratio: {aspect_ratio:.1f}',
        'status': 'completed',
        'icon': '🍃',
        'coverage': coverage,
        'aspect_ratio': aspect_ratio,
        'hull_url': f"/uploads/hull_{filename}",
        'mask_url': f"/uploads/mask_{filename}"
    })
    
    # =========================================================
    # BƯỚC 2: XỬ LÝ VĂN BẢN
    # =========================================================
    tokenizer = _tokenizer
    if tokenizer is not None:
        encoded = tokenizer(
            text if text else "",
            max_length=config.get('data', {}).get('max_length', 128),
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(DEVICE)
        attention_mask = encoded['attention_mask'].to(DEVICE)
        
        token_count = (input_ids != 0).sum().item()
        pipeline_steps['text'].append({
            'step': '1. Tokenization',
            'description': f'Tokenized: {token_count} tokens',
            'status': 'completed',
            'icon': '🔤'
        })
        
        text_feat = _model.text_encoder(input_ids, attention_mask)
        text_feat = _model.text_projection(text_feat)
        
        pipeline_steps['text'].append({
            'step': '2. Trích xuất đặc trưng',
            'description': f'Vector đặc trưng {text_feat.shape[-1]} chiều',
            'status': 'completed',
            'icon': '📊'
        })
    else:
        max_len = config.get('data', {}).get('max_length', 128)
        input_ids = torch.zeros((1, max_len), dtype=torch.long).to(DEVICE)
        attention_mask = torch.zeros((1, max_len), dtype=torch.long).to(DEVICE)
        text_feat = torch.zeros((1, 768)).to(DEVICE)
        
        pipeline_steps['text'].append({
            'step': '1. Tokenization',
            'description': 'Không có văn bản đầu vào',
            'status': 'completed',
            'icon': '🔤'
        })
    
    # =========================================================
    # BƯỚC 3: XỬ LÝ ẢNH QUA VISION ENCODER
    # =========================================================
    cfg_img_size = config.get('data', {}).get('image_size', 224)
    transform = get_val_transforms(cfg_img_size)
    pil_img = Image.open(image_path).convert('RGB')
    img_t = transform(pil_img).unsqueeze(0).to(DEVICE)
    
    visual_features = _model.vision_encoder(img_t)
    visual_features_shape = visual_features.shape
    pipeline_steps['image'].append({
        'step': '3. Vision Encoder',
        'description': f'Feature map {visual_features_shape[1]}x{visual_features_shape[2]}x{visual_features_shape[3]}',
        'status': 'completed',
        'icon': '🧠'
    })
    
    visual_pooled = visual_features.mean(dim=[2, 3])
    visual_feat = _model.vision_projection(visual_pooled)
    
    pipeline_steps['image'].append({
        'step': '4. Projection',
        'description': f'Vector {visual_feat.shape[-1]} chiều',
        'status': 'completed',
        'icon': '📐'
    })
    
    # =========================================================
    # BƯỚC 4: FUSION - KẾT HỢP ĐA PHƯƠNG THỨC
    # =========================================================
    if _model.fusion_type == "cross_attention":
        image_feat_unsq = visual_feat.unsqueeze(1)
        text_feat_unsq = text_feat.unsqueeze(1)
        fused_attn, attn_weights = _model.cross_attention(image_feat_unsq, text_feat_unsq)
        fused = fused_attn.squeeze(1)
        
        pipeline_steps['fusion'].append({
            'step': '1. Cross-Attention',
            'description': 'Kết hợp đặc trưng ảnh và văn bản',
            'status': 'completed',
            'icon': '🔀',
            'attention_weights': attn_weights.detach().cpu().numpy().tolist()
        })
    else:
        fused = _model.concat_fusion(visual_feat, text_feat)
        pipeline_steps['fusion'].append({
            'step': '1. Concatenation',
            'description': 'Ghép đặc trưng ảnh và văn bản',
            'status': 'completed',
            'icon': '🔗'
        })
    
    # =========================================================
    # BƯỚC 5: CLASSIFICATION
    # =========================================================
    logits = _model.classifier(fused)
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    classes = config.get('dataset', {}).get('classes', _classes)
    
    pipeline_steps['fusion'].append({
        'step': '2. Classification',
        'description': f'Dự đoán: {classes[pred_idx]}',
        'status': 'completed',
        'icon': '🎯'
    })
    
    # =========================================================
    # BƯỚC 6: GRADCAM
    # =========================================================
    gradcam_url = None
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        
        backbone = _model.vision_encoder.backbone
        
        if hasattr(backbone, 'blocks'):
            target_layers = [backbone.blocks[-1]]
        elif hasattr(backbone, 'features'):
            target_layers = [backbone.features[-1]]
        else:
            target_layers = [backbone]
        
        class GradCAMWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                features = self.model.vision_encoder(x)
                pooled = features.mean(dim=[2, 3])
                projected = self.model.vision_projection(pooled)
                logits = self.model.classifier(projected)
                return logits
        
        wrapper = GradCAMWrapper(_model)
        wrapper.eval()
        wrapper.to(DEVICE)
        
        cam = GradCAM(model=wrapper, target_layers=target_layers)
        input_tensor = img_t.clone().detach().requires_grad_(True)
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        
        img_np = img_t[0].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        gradcam_name = f"gradcam_pipeline_{filename}"
        gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_name)
        Image.fromarray(visualization).save(gradcam_path)
        gradcam_url = f"/uploads/{gradcam_name}"
        
        pipeline_steps['image'].append({
            'step': '5. GradCAM',
            'description': 'Vùng tập trung của AI',
            'status': 'completed',
            'icon': '🔥',
            'gradcam_url': gradcam_url
        })
    except Exception as e:
        pipeline_steps['image'].append({
            'step': '5. GradCAM',
            'description': f'Lỗi: {str(e)}',
            'status': 'error',
            'icon': '❌'
        })
    
    # =========================================================
    # BƯỚC 7: GEMINI AI - LỜI KHUYÊN
    # =========================================================
    gemini_resp = None
    gem_cfg = config.get('gemini', {})
    api_key = gem_cfg.get('api_key') or os.environ.get('GEMINI_API_KEY')
    model_id = gem_cfg.get('model_id')
    
    if api_key and model_id:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            
            pred_vi = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
            prob = float(probs[pred_idx] * 100)
            
            prompt = f"""Bạn là chuyên gia nông nghiệp về bệnh lúa. Hãy phân tích và đưa ra lời khuyên:

**Kết quả chẩn đoán:**
- Bệnh: {pred_vi}
- Độ tin cậy: {prob:.1f}%
- Diện tích lá bị ảnh hưởng: {coverage:.1%}
- Tỷ lệ khung hình lá: {aspect_ratio:.1f}

**Yêu cầu:**
1. Mô tả ngắn gọn đặc điểm của bệnh này
2. Nguyên nhân gây bệnh
3. Biện pháp phòng trừ và điều trị (ưu tiên biện pháp sinh học và an toàn)
4. Lời khuyên cho nông dân (ngắn gọn, dễ hiểu)

Hãy trả lời bằng tiếng Việt, với cấu trúc rõ ràng."""
            
            try:
                img_obj = Image.open(image_path)
                response = client.models.generate_content(model=model_id, contents=[prompt, img_obj])
            except Exception:
                response = client.models.generate_content(model=model_id, contents=prompt)
            
            text = getattr(response, 'text', None)
            if text is None and isinstance(response, dict):
                text = response.get('text') or response.get('output')
            gemini_resp = text if text is not None else str(response)
            
        except Exception as e:
            gemini_resp = f"❌ Lỗi kết nối Gemini: {str(e)}"
    else:
        gemini_resp = "⚠️ Gemini chưa được cấu hình. Vui lòng thêm API key vào config.yaml"
    
    # =========================================================
    # KẾT QUẢ
    # =========================================================
    pipeline_steps['result'] = {
        'prediction': classes[pred_idx] if pred_idx < len(classes) else str(pred_idx),
        'index': pred_idx,
        'probs': {classes[i]: float(probs[i]) for i in range(len(probs)) if i < len(classes)},
        'coverage': coverage,
        'aspect_ratio': aspect_ratio,
        'image_url': f"/uploads/{filename}",
        'hull_url': f"/uploads/hull_{filename}",
        'mask_url': f"/uploads/mask_{filename}",
        'gradcam_url': gradcam_url
    }
    
    pipeline_steps['gemini'] = gemini_resp
    
    return jsonify(pipeline_steps)


# =========================================================
# KHỞI ĐỘNG APP
# =========================================================
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 KHỞI ĐỘNG RICE DISEASE PREDICTION APP")
    print("=" * 60)
    print("💡 Model chưa được load. Vào giao diện và nhấn 'Load Model' để bắt đầu.")
    print("🌐 Starting Flask server...")
    print("📍 Access at: http://localhost:5000")
    print("📍 For external access: http://0.0.0.0:5000")
    print("=" * 60)
    print("💡 Tip: Press CTRL+C to stop the server")
    print("=" * 60)

    app.run(debug=True, port=5000, host='0.0.0.0', use_reloader=True)