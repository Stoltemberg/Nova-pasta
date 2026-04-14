import os
from ultralytics import YOLO

def export_to_amd_format(pt_path="runs/detect/lol_orbwalker/weights/best.pt"):
    """
    Carrega o modelo treinado em PyTorch (best.pt) e o converte para
    ONNX (Open Neural Network Exchange).
    Esse formato roda de forma nativa e paralela em placas AMD (RX 580)
    usando a API do DirectML, quadruplicando o FPS da visão.
    """
    if not os.path.exists(pt_path):
        print(f"[ERRO] O modelo '{pt_path}' não foi encontrado.")
        print("Treine a sua IA primeiro antes de tentar exportá-la para AMD!")
        return
        
    print("="*50)
    print("Iniciando Triturador de Tensores para Padrão AMD (ONNX)...")
    print("="*50)
    
    # Carregar modelo PyTorch
    model = YOLO(pt_path)
    
    try:
        # Exporta setando o Ops de otimização estritamente para aceleração leve
        print("[INFO] Isso pode levar alguns minutos. Aguarde...")
        exported_path = model.export(
            format="onnx",
            opset=12,         # Estável e rápido
            simplify=True,    # Remove nós inúteis e mescla as camadas da CNN
            imgsz=416,        # Forçar tamanho fixo para velocidade de renderização da caixa
            half=False        # Garantir precisão na GPU AMD
        )
        print("\n" + "="*50)
        print(f"✓ SUCESSO! Cérebro Exportado em: {exported_path}")
        print("="*50)
        print("\n>> Próximo Passo:")
        print("   1. Vá no arquivo config.py")
        print("   2. Altere YOLO_MODEL_PATH para apontar para o novo arquivo '.onnx'.")
        print("   3. Certifique-se de instalar os pacotes rodando:")
        print("      pip install onnx onnxruntime onnxruntime-directml")
        
    except Exception as e:
        print(f"\n[ERRO FATAL] Algo quebrou na exportação: {e}")
        print("Você instalou as bibliotecas básicas? (pip install onnx onnxsim)")

if __name__ == "__main__":
    export_to_amd_format()
