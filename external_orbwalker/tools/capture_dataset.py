"""
tools/capture_dataset.py — Captura screenshots do jogo para treinar o modelo YOLO.
Pressione F9 para capturar um frame. Os frames são salvos em dataset/images/
"""
import os
import sys
import time
import logging
import keyboard
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("DatasetCapture")

# Adicionar o diretório pai ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from screen_capture import ScreenCapture


class DatasetCapturer:
    """
    Ferramenta para capturar screenshots do LoL para treinamento YOLO.
    
    Uso:
        1. Abra o League of Legends (Practice Tool recomendado)
        2. Execute este script
        3. Pressione F9 para capturar frames
        4. Varie as situações: lane, teamfight, jungle, diferentes campeões/skins
        5. Pressione ESC para sair
    
    Dicas para um bom dataset:
        - Capture pelo menos 2000 frames
        - Varie: zoom, ângulo, skins, efeitos visuais, fog of war
        - Capture com e sem minions, com e sem campeões
        - Capture em diferentes momentos do jogo (early/mid/late)
        - Capture com diferentes HUDs (minimap, itens)
    """

    CAPTURE_KEY = "f9"
    EXIT_KEY = "esc"
    BURST_KEY = "f8"  # Captura 10 frames seguidos (1 por segundo)

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            base = os.path.dirname(os.path.dirname(__file__))
            output_dir = os.path.join(base, "dataset", "images", "train")
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.capture = ScreenCapture(target_fps=60)
        self.frame_count = self._count_existing()
        self.total_captured = 0

    def _count_existing(self) -> int:
        """Conta frames já existentes para não sobrescrever."""
        existing = [f for f in os.listdir(self.output_dir) if f.endswith(('.png', '.jpg'))]
        return len(existing)

    def start(self):
        """Inicia o capturador."""
        self.capture.start()

        print("\n" + "=" * 60)
        print("  DATASET CAPTURE TOOL")
        print("=" * 60)
        print(f"  Output: {self.output_dir}")
        print(f"  Frames existentes: {self.frame_count}")
        print()
        print(f"  [F9]  = Capturar 1 frame")
        print(f"  [F8]  = Burst: capturar 10 frames (1/s)")
        print(f"  [ESC] = Sair")
        print("=" * 60)
        print()

        keyboard.on_press_key(self.CAPTURE_KEY, lambda _: self._capture_frame())
        keyboard.on_press_key(self.BURST_KEY, lambda _: self._burst_capture())

        try:
            keyboard.wait(self.EXIT_KEY)
        except KeyboardInterrupt:
            pass
        finally:
            self.capture.stop()
            print(f"\n  Total capturado nesta sessão: {self.total_captured}")
            print(f"  Total no dataset: {self.frame_count}")

    def _capture_frame(self):
        """Captura e salva um frame."""
        frame = self.capture.grab()
        if frame is None:
            logger.warning("Frame capture failed!")
            return

        self.frame_count += 1
        self.total_captured += 1

        filename = f"frame_{self.frame_count:05d}.jpg"
        filepath = os.path.join(self.output_dir, filename)

        import cv2
        # Salvar como JPG com qualidade 95 (bom balanço tamanho/qualidade)
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"  ✓ Captured: {filename} ({frame.shape[1]}x{frame.shape[0]}) | Total: {self.frame_count}")

    def _burst_capture(self):
        """Captura 10 frames com 1 segundo de intervalo."""
        logger.info("  ⚡ Burst capture: 10 frames...")
        for i in range(10):
            self._capture_frame()
            time.sleep(1.0)
        logger.info("  ⚡ Burst complete!")


if __name__ == "__main__":
    capturer = DatasetCapturer()
    capturer.start()
