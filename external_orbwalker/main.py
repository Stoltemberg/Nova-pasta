"""
main.py — External Orbwalker funcional.
Baseado diretamente na arquitetura do auto-kite-bot.

Funciona em 2 camadas:
    Camada 1 (obrigatória): Orbwalker cego (A+click / R-click com timing perfeito)
    Camada 2 (opcional):    Visão computacional para targeting inteligente
"""
import time
import os
import sys
import ctypes
import threading
import logging
import keyboard

from riot_api import RiotAPI
from champion_data import ChampionData
from orbwalker.engine import OrbwalkerEngine
from scripts import get_script

# ── Opcional: visão ──
VISION_AVAILABLE = False
try:
    from screen_capture import ScreenCapture
    from vision.health_bar_detector import HealthBarDetector
    from vision.entity_classifier import EntityClassifier, EntityType
    VISION_AVAILABLE = True
except ImportError as e:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ExternalOrbwalker")

import cv2
import random

def convert_entity_to_yolo_class(entity_type) -> int:
    """Mapeia EntityType para as classes da YOLO do nosso dataset"""
    if entity_type == EntityType.CHAMPION: return 0
    if entity_type in (EntityType.MINION, EntityType.MINION_SIEGE): return 1
    if entity_type == EntityType.TURRET: return 2
    if entity_type == EntityType.MONSTER: return 4
    return 1  # Default fallback

def save_training_data_in_background(frame, entities, width, height, base_dir=None):
    """Salva a imagem do jogo e a anotação .txt em formato YOLO sem bloquear a thread."""
    try:
        timestamp = time.time()
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
        img_name = os.path.join(base_dir, "dataset", "images", "train", f"play_{timestamp:.2f}.jpg")
        lbl_name = os.path.join(base_dir, "dataset", "labels", "train", f"play_{timestamp:.2f}.txt")
        
        os.makedirs(os.path.dirname(img_name), exist_ok=True)
        os.makedirs(os.path.dirname(lbl_name), exist_ok=True)
        
        # 1. Salvar imagem JPG (qualidade alta o suficiente, tamanho leve)
        cv2.imwrite(img_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # 2. Gerar Labels da YOLO
        with open(lbl_name, "w") as f:
            for e in entities:
                hb = e.health_bar
                if hb is None: continue
                # Filtro de qualidade: Apenas labels de altissima confiança
                if e.confidence < 0.90: continue
                
                # Estimativa do Bounding Box baseada na Health Bar
                body_w = hb.width * 1.5
                body_h = hb.width * 2.0
                
                # Coordenadas YOLO: centro_x, centro_y, largura, altura (Tudo de 0 a 1)
                norm_x = e.screen_x / width
                norm_y = (e.screen_y + hb.height) / height 
                norm_w = body_w / width
                norm_h = body_h / height
                
                # Limites absolutos de borda
                norm_x = max(0.0, min(1.0, norm_x))
                norm_y = max(0.0, min(1.0, norm_y))
                norm_w = max(0.0, min(1.0, norm_w))
                norm_h = max(0.0, min(1.0, norm_h))
                
                yolo_class = convert_entity_to_yolo_class(e.entity_type)
                f.write(f"{yolo_class} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
    except Exception as ex:
        # Silencia erros de I/O para não interferir na gameplay
        pass


BANNER = """
 ╔═══════════════════════════════════════════════╗
 ║   EXTERNAL ORBWALKER — No Memory Reading      ║
 ║   Based on Auto-Kite Bot architecture          ║
 ╚═══════════════════════════════════════════════╝
"""


class ExternalOrbwalker:
    def __init__(self):
        self.riot_api = RiotAPI()
        self.champion_data = ChampionData()
        self.engine = None
        self.running = False

        # ── Vision thread ──
        self.vision_enabled = VISION_AVAILABLE
        self.vision_thread = None

        # ── Key states ──
        self._combo_held = False
        self._lasthit_held = False
        self._laneclear_held = False

    def start(self, blocking=True):
        print(BANNER)

        # ═══ 1. Conectar à Riot API ═══
        self.riot_api.start()
        logger.info("Aguardando League of Legends...")

        while not self.riot_api.connected:
            time.sleep(0.5)
        logger.info("✓ Conectado à API do jogo!")

        # ═══ 2. Identificar campeão ═══
        logger.info("Identificando campeão...")
        while not self.riot_api.raw_champion_name:
            time.sleep(0.5)

        champ = self.riot_api.champion_name
        raw = self.riot_api.raw_champion_name
        logger.info(f"✓ Campeão: {champ} ({raw})")

        # ═══ 3. Carregar dados do CommunityDragon ═══
        if not self.champion_data.load(raw):
            logger.error(f"Falha ao carregar dados de {raw}. Usando defaults.")

        # ═══ 4. Criar e iniciar engine ═══
        self.engine = OrbwalkerEngine(self.riot_api, self.champion_data)
        
        # Ativar visão no engine e carregar script do campeão
        self.engine.use_vision = True 
        self.engine.script = get_script(raw, self.riot_api, self.champion_data)
        
        # Permitir que as threads rodem antes de iniciá-las!
        self.running = True
        
        self.engine.start()

        # ═══ 5. Iniciar visão (opcional, thread separada) ═══
        if self.vision_enabled:
            self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
            self.vision_thread.start()
            logger.info("✓ Visão computacional ativada (thread separada)")
        else:
            logger.info("⚠ Visão computacional não disponível (instale opencv-python + dxcam)")
            logger.info("  O orbwalker vai funcionar no modo cego (A+click no cursor)")

        # ═══ 6. Registrar hotkeys ═══
        try:
            keyboard.hook(self._on_key_event)
        except Exception as e:
            logger.error(f"Keyboard hook falhou: {e}")
            logger.error("Execute como Administrador!")
            self.running = False
            return


        # ═══ 7. Exibir controles ═══
        print()
        logger.info("═" * 50)
        logger.info("ORBWALKER PRONTO!")
        logger.info("  [SPACE] Segurar = Orbwalk (Combo)")
        logger.info("  [X]     Segurar = Last Hit")
        logger.info("  [V]     Segurar = Lane Clear")
        logger.info("  [Ctrl+C] = Sair")
        logger.info("═" * 50)
        print()

        # ═══ 8. Status loop ═══
        if blocking:
            self._status_loop()

    def _on_key_event(self, event):
        """Idêntico ao auto-kite-bot: ativa/desativa o orbwalker."""
        if not self.engine:
            return

        key = getattr(event, 'name', '').lower()

        # ── SPACE = Combo ──
        if key == 'space':
            if event.event_type == keyboard.KEY_DOWN:
                if not self._combo_held:
                    self._combo_held = True
                    self.engine.active = True
                    self.engine.mode = "combo"
            elif event.event_type == keyboard.KEY_UP:
                self._combo_held = False
                if self.engine.mode == "combo":
                    self.engine.active = False
                    self.engine.clear_vision_target()

        # ── X = Last Hit ──
        elif key == 'x':
            if event.event_type == keyboard.KEY_DOWN:
                if not self._lasthit_held:
                    self._lasthit_held = True
                    self.engine.active = True
                    self.engine.mode = "lasthit"
            elif event.event_type == keyboard.KEY_UP:
                self._lasthit_held = False
                if self.engine.mode == "lasthit":
                    self.engine.active = False
                    self.engine.clear_vision_target()

        # ── V = Lane Clear ──
        elif key == 'v':
            if event.event_type == keyboard.KEY_DOWN:
                if not self._laneclear_held:
                    self._laneclear_held = True
                    self.engine.active = True
                    self.engine.mode = "laneclear"
            elif event.event_type == keyboard.KEY_UP:
                self._laneclear_held = False
                if self.engine.mode == "laneclear":
                    self.engine.active = False
                    self.engine.clear_vision_target()

        # ── Q/W/E/R = Tracking de Cooldown + Reset attack timer (animation cancel) ──
        if key in ('q', 'w', 'e', 'r'):
            if event.name and self.engine.script:
                 self.engine.script.on_key_event(key, event.event_type)
            
            if event.event_type == keyboard.KEY_DOWN and self.engine.active:
                self.engine.reset_attack_timer()

    # ═══════════════════════════════════════════
    #  VISION LOOP — Thread separada, NÃO bloqueia o orbwalker
    # ═══════════════════════════════════════════

    def _vision_loop(self):
        """
        Thread de visão que detecta entidades e atualiza o target do engine.
        Se falhar ou não encontrar nada, o orbwalker continua funcionando cego.
        RESILIENTE: erros por frame são ignorados, thread só morre em erro fatal.
        """
        import traceback

        capture = None
        try:
            capture = ScreenCapture(target_fps=60)
            capture.start()

            if not capture._started:
                logger.error("Vision: Screen capture falhou ao iniciar!")
                logger.info("Orbwalker continua no modo cego (A+click no cursor)")
                return

            screen_w, screen_h = capture.get_screen_size()
            detector = HealthBarDetector(screen_w, screen_h)
            classifier = EntityClassifier(screen_w, screen_h)
            
            # ── Integrando a Mente Neural (YOLO) ──
            from vision.yolo_detector import YOLODetector
            from config import VisionConfig
            yolo_detector = YOLODetector()

            logger.info(f"Vision thread OK @ {screen_w}x{screen_h}")

            error_count = 0
            frame_count = 0
            fps_timer = time.perf_counter()

            while self.running:
                try:
                    # Só processa quando o orbwalker está ativo
                    if not self.engine or not self.engine.active:
                        if self.engine:
                            self.engine.clear_vision_target()
                        time.sleep(0.05)
                        continue

                    frame = capture.grab()
                    if frame is None:
                        time.sleep(0.01)
                        continue

                    frame_count += 1

                    # Pipeline Híbrido de Visão Computacional
                    entities = []
                    health_bars = []
                    
                    # 1. Tentativa 1: Depende se YOLO é principal ou fallback
                    if not VisionConfig.YOLO_FALLBACK_ONLY and yolo_detector.available and VisionConfig.YOLO_ENABLED:
                        entities = yolo_detector.detect(frame)
                        
                    # 2. Tentativa 2: OpenCV (Atua como principal se YOLO_FALLBACK_ONLY for True, ou como fallback seguro se o YOLO der blind-spot em algum frame)
                    if not entities:
                        health_bars = detector.detect(frame)
                        entities = classifier.classify(frame, health_bars)
                        
                        # 3. YOLO atuando estritamente como Fallback se OpenCV não viu nada
                        if not entities and VisionConfig.YOLO_FALLBACK_ONLY and yolo_detector.available and VisionConfig.YOLO_ENABLED:
                            entities = yolo_detector.detect(frame)

                    if not entities:
                        self.engine.clear_vision_target()
                        time.sleep(0.005)
                        continue

                    # ── SHADOW CAPTURE (Pseudo-labeling) ──
                    # Diminui a taxa de amostragem quando o YOLO já estiver treinado pra poupar desempenho.
                    prob = 0.01 if yolo_detector.available else 0.05
                    if self.engine.active and random.random() < prob:
                        save_thread = threading.Thread(
                            target=save_training_data_in_background,
                            args=(frame.copy(), entities, screen_w, screen_h),
                            daemon=True
                        )
                        save_thread.start()

                    # ── Selecionar alvo baseado no modo ──
                    mode = self.engine.mode
                    target = None
                    cx, cy = screen_w // 2, screen_h // 2

                    if mode == "combo":
                        champs = [e for e in entities if e.entity_type == EntityType.CHAMPION]
                        if champs:
                            target = min(champs, key=lambda e: ((e.screen_x - cx)**2 + (e.screen_y - cy)**2))

                    elif mode == "lasthit":
                        minions = [e for e in entities if e.entity_type in (EntityType.MINION, EntityType.MINION_SIEGE)]
                        low_hp = [m for m in minions if m.health_bar.fill_ratio < 0.40]
                        if low_hp:
                            target = min(low_hp, key=lambda e: e.health_bar.fill_ratio)

                    elif mode == "laneclear":
                        all_targets = [e for e in entities if e.entity_type != EntityType.UNKNOWN]
                        minions_low = [e for e in all_targets
                                       if e.entity_type in (EntityType.MINION, EntityType.MINION_SIEGE)
                                       and e.health_bar.fill_ratio < 0.30]
                        if minions_low:
                            target = min(minions_low, key=lambda e: e.health_bar.fill_ratio)
                        elif all_targets:
                            target = min(all_targets, key=lambda e: ((e.screen_x - cx)**2 + (e.screen_y - cy)**2))

                    # ── Atualizar target no engine ──
                    if target:
                        self.engine.set_vision_target(
                            target.screen_x, target.screen_y,
                            target.entity_type.name.lower()
                        )
                    else:
                        self.engine.clear_vision_target()

                    # Log detalhado a cada 3 segundos
                    now = time.perf_counter()
                    if now - fps_timer >= 3.0:
                        vfps = frame_count / (now - fps_timer)

                        # ── DEBUG: Mostrar cada entidade detectada ──
                        logger.info(f"─── Vision Debug ─── FPS:{vfps:.1f} | Bars:{len(health_bars)} | Entities:{len(entities)}")
                        for i, e in enumerate(entities):
                            hb = e.health_bar
                            logger.info(
                                f"  [{i}] {e.entity_type.name:15s} | "
                                f"w={hb.width:3d}px h={hb.height:2d}px | "
                                f"pos=({hb.center_x:4d},{hb.center_y:4d}) | "
                                f"fill={hb.fill_ratio:.0%} | "
                                f"lvl={'YES' if e.has_level else 'no ':3s} | "
                                f"conf={e.confidence:.0%}"
                            )

                        frame_count = 0
                        fps_timer = now
                        error_count = 0

                except Exception as frame_err:
                    error_count += 1
                    if error_count <= 3:
                        logger.warning(f"Vision frame error ({error_count}): {frame_err}")
                    if error_count > 100:
                        logger.error("Vision: muitos erros seguidos. Tentando reconectar (Backoff de 5s)...")
                        time.sleep(5)
                        try:
                            if capture:
                                try:
                                    capture.stop()
                                except: pass
                            time.sleep(1)
                            capture = ScreenCapture(target_fps=60)
                            capture.start()
                            if capture._started:
                                screen_w, screen_h = capture.get_screen_size()
                                detector = HealthBarDetector(screen_w, screen_h)
                                classifier = EntityClassifier(screen_w, screen_h)
                                error_count = 0
                                logger.info("Vision: Thread recuperada com sucesso.")
                            else:
                                logger.error("Vision: Falha ao recriar captura.")
                        except Exception as rec_err:
                            logger.error(f"Vision: Falha crítica na recuperação: {rec_err}")
                        continue

                time.sleep(0.008)

        except Exception as e:
            logger.error(f"Vision thread fatal error: {e}")
            logger.error(traceback.format_exc())
            logger.info("Orbwalker continua funcionando no modo cego.")
        finally:
            if capture:
                try:
                    capture.stop()
                except:
                    pass

    def _status_loop(self):
        """Exibe status no console."""
        try:
            while self.running:
                if self.engine:
                    mode = self.engine.mode if self.engine.active else "IDLE"
                    as_val = self.riot_api.attack_speed
                    hp = self.riot_api.health_percent * 100

                    target, ttype = self.engine.get_vision_target()
                    vision_str = f"→ {ttype} @({target[0]},{target[1]})" if target else "nenhum"

                    spa = self.engine.get_seconds_per_attack()
                    windup = self.engine.get_windup_duration()

                    print(
                        f"\r  [{mode:10s}] "
                        f"AS:{as_val:.2f} "
                        f"SecPerAtk:{spa:.3f}s "
                        f"Windup:{windup:.3f}s "
                        f"HP:{hp:.0f}% "
                        f"Atks:{self.engine.attacks} "
                        f"Moves:{self.engine.moves} "
                        f"Vision:{vision_str:30s}",
                        end="", flush=True
                    )

                time.sleep(0.5)
        except KeyboardInterrupt:
            self._shutdown()

    def _shutdown(self):
        logger.info("\nShutting down...")
        self.running = False
        if self.engine:
            self.engine.stop()
        self.riot_api.stop()
        try:
            keyboard.unhook_all()
        except:
            pass
        logger.info("Goodbye!")


if __name__ == "__main__":
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
    except:
        is_admin = False

    if not is_admin:
        logger.warning("⚠ Execute como Administrador para keyboard hooks!")

    app = ExternalOrbwalker()
    try:
        app.start()
    except KeyboardInterrupt:
        app._shutdown()
