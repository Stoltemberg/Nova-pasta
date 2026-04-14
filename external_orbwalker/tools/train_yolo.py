"""
tools/train_yolo.py — Script de treinamento do modelo YOLO para o orbwalker.
Treina um YOLOv8-nano para detectar entidades do League of Legends.
"""
import os
import sys
import shutil
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("YOLOTrainer")


def split_dataset(dataset_dir: str, val_ratio: float = 0.15):
    """
    Divide o dataset em train/val automaticamente.
    Move val_ratio% das imagens + labels para a pasta val/.
    
    Estrutura esperada ANTES:
        dataset/
        ├── images/
        │   └── train/      ← todas as imagens aqui
        └── labels/
            └── train/      ← todos os .txt de labels aqui
    
    Estrutura DEPOIS:
        dataset/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
    """
    img_train = os.path.join(dataset_dir, "images", "train")
    lbl_train = os.path.join(dataset_dir, "labels", "train")
    img_val = os.path.join(dataset_dir, "images", "val")
    lbl_val = os.path.join(dataset_dir, "labels", "val")
    
    os.makedirs(img_val, exist_ok=True)
    os.makedirs(lbl_val, exist_ok=True)
    
    # Listar imagens que têm label correspondente
    images = []
    for f in os.listdir(img_train):
        if f.endswith(('.jpg', '.png', '.jpeg')):
            label_name = os.path.splitext(f)[0] + ".txt"
            label_path = os.path.join(lbl_train, label_name)
            if os.path.exists(label_path):
                images.append(f)
    
    if not images:
        logger.error(f"Nenhuma imagem com label encontrada em {img_train}")
        logger.error(f"Certifique-se de que as labels estão em {lbl_train}")
        return False
    
    # Shuffle e split
    random.shuffle(images)
    val_count = max(1, int(len(images) * val_ratio))
    val_images = images[:val_count]
    
    logger.info(f"Dataset: {len(images)} imagens com labels")
    logger.info(f"Split: {len(images) - val_count} train / {val_count} val")
    
    # Mover para val
    for img_name in val_images:
        label_name = os.path.splitext(img_name)[0] + ".txt"
        
        shutil.move(os.path.join(img_train, img_name), os.path.join(img_val, img_name))
        shutil.move(os.path.join(lbl_train, label_name), os.path.join(lbl_val, label_name))
    
    logger.info("Split completo!")
    return True


def train(dataset_yaml: str, epochs: int = 100, img_size: int = 640, batch: int = 16):
    """
    Treina um YOLOv8-nano no dataset.
    
    Args:
        dataset_yaml: Caminho para o arquivo lol_dataset.yaml
        epochs: Número de épocas de treino
        img_size: Tamanho da imagem de entrada (640 recomendado)
        batch: Batch size (reduzir se GPU com pouca VRAM)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics não instalado! Execute: pip install ultralytics")
        return
    
    logger.info("=" * 60)
    logger.info("  YOLO TRAINING — League of Legends Entity Detection")
    logger.info("=" * 60)
    logger.info(f"  Dataset: {dataset_yaml}")
    logger.info(f"  Epochs:  {epochs}")
    logger.info(f"  ImgSize: {img_size}")
    logger.info(f"  Batch:   {batch}")
    logger.info(f"  Model:   YOLOv8-nano (pré-treinado)")
    logger.info("=" * 60)
    
    # Carregar modelo pré-treinado (transfer learning)
    model = YOLO("yolov8n.pt")
    
    # Treinar
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        name="lol_orbwalker",
        patience=20,          # Early stopping após 20 épocas sem melhora
        save=True,
        save_period=10,        # Salvar checkpoint a cada 10 épocas
        plots=True,
        verbose=True,
        # ── Augmentação otimizada para LoL ──
        hsv_h=0.01,           # Pouca variação de hue (health bars têm cor fixa)
        hsv_s=0.3,            # Variação moderada de saturação
        hsv_v=0.3,            # Variação moderada de brilho
        degrees=0.0,          # SEM rotação (o jogo é sempre top-down)
        translate=0.1,        # Pouca translação
        scale=0.3,            # Variação de escala (zoom in/out)
        fliplr=0.0,           # SEM flip horizontal (health bars viram ao contrário)
        flipud=0.0,           # SEM flip vertical
        mosaic=0.5,           # Mosaic moderado
        mixup=0.1,            # Pouco mixup
    )
    
    # ── Copiar melhor modelo para models/ ──
    best_model = os.path.join("runs", "detect", "lol_orbwalker", "weights", "best.pt")
    target_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "orbwalker_yolo.pt")
    
    if os.path.exists(best_model):
        shutil.copy2(best_model, target_path)
        logger.info(f"\n  ✓ Modelo salvo em: {target_path}")
        logger.info(f"  ✓ Ative no config.py: YOLO_ENABLED = True")
    else:
        logger.error(f"  ✗ Modelo best.pt não encontrado em {best_model}")
    
    return results


def validate(model_path: str, dataset_yaml: str):
    """Valida o modelo treinado."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics não instalado!")
        return
    
    model = YOLO(model_path)
    results = model.val(data=dataset_yaml)
    
    logger.info("\n  Resultados de Validação:")
    logger.info(f"  mAP50:    {results.box.map50:.4f}")
    logger.info(f"  mAP50-95: {results.box.map:.4f}")
    
    return results


# ─────────── CLI ───────────

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    dataset_yaml = os.path.join(dataset_dir, "lol_dataset.yaml")
    
    if len(sys.argv) < 2:
        print("""
  Uso:
    python tools/train_yolo.py split       → Divide dataset em train/val
    python tools/train_yolo.py train       → Treina o modelo
    python tools/train_yolo.py train 200   → Treina com 200 épocas
    python tools/train_yolo.py validate    → Valida o modelo treinado
        """)
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == "split":
        split_dataset(dataset_dir)
    
    elif command == "train":
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        
        # Auto-split se val não existir
        val_dir = os.path.join(dataset_dir, "images", "val")
        if not os.path.exists(val_dir) or len(os.listdir(val_dir)) == 0:
            logger.info("Val set vazio, fazendo split automático...")
            if not split_dataset(dataset_dir):
                sys.exit(1)
        
        train(dataset_yaml, epochs=epochs)
    
    elif command == "validate":
        model_path = os.path.join(base_dir, "models", "orbwalker_yolo.pt")
        if not os.path.exists(model_path):
            logger.error(f"Modelo não encontrado: {model_path}")
            logger.error("Treine primeiro com: python tools/train_yolo.py train")
            sys.exit(1)
        validate(model_path, dataset_yaml)
    
    else:
        print(f"Comando desconhecido: {command}")
