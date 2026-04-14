"""
config.py — Configurações globais, hotkeys e constantes do External Orbwalker.
"""
import json
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ExternalOrbwalker")


# ─────────────────────────── Hotkeys ───────────────────────────
class Hotkeys:
    ORBWALK = "space"          # Combo mode (prioriza campeões)
    LASTHIT = "x"              # Last-hit mode (prioriza minions matáveis)
    LANECLEAR = "v"            # Lane clear (ataca tudo)
    HARASS = "c"               # Harass = ataca campeão, se não tiver move
    TOGGLE_PAUSE = "f10"       # Pausa/resume total


# ─────────────────────────── Vision ───────────────────────────
class VisionConfig:
    # Resolução de referência para calibração dos tamanhos de health bar
    REFERENCE_WIDTH = 1920
    REFERENCE_HEIGHT = 1080

    # FPS alvo do pipeline de captura
    TARGET_FPS = 120

    # ── HSV ranges para health bars inimigas (vermelho) ──
    # O vermelho no HSV "wrapa" no hue=0, então precisamos de 2 ranges
    ENEMY_RED_LOWER_1 = (0, 120, 100)
    ENEMY_RED_UPPER_1 = (8, 255, 255)
    ENEMY_RED_LOWER_2 = (170, 120, 100)
    ENEMY_RED_UPPER_2 = (180, 255, 255)

    # ── Health bar aspect ratio filter ──
    MIN_ASPECT_RATIO = 4.0   # barras são horizontais
    MAX_ASPECT_RATIO = 25.0

    # ── Tamanhos de health bar em pixels @1080p ──
    # Estes valores servem como thresholds para classificação
    CHAMPION_BAR_MIN_WIDTH = 90       # campeões têm barras >= 90px
    MINION_SIEGE_BAR_MIN_WIDTH = 62   # siege/cannon
    MINION_BAR_MIN_WIDTH = 25         # minions normais
    MINION_BAR_MAX_WIDTH = 89         # teto para minion (abaixo = campeão)

    # ── Offset do centro do corpo em relação à health bar ──
    # A health bar fica acima da cabeça; o corpo está ~50px abaixo @1080p
    BODY_OFFSET_Y = 50

    # ── Nível indicator detection ──
    # Campeões têm um pequeno texto de nível ao lado da barra
    LEVEL_CHECK_OFFSET_X = -15   # pixels à esquerda da barra
    LEVEL_CHECK_OFFSET_Y = 0
    LEVEL_CHECK_SIZE = 14        # tamanho da região de check

    # ── YOLO Config ──
    YOLO_MODEL_PATH = "runs/detect/lol_orbwalker/weights/best.pt"  # Apontando para o default file de output do pytorch
    YOLO_CONFIDENCE = 0.50
    YOLO_ENABLED = True   # O robô agora tentará ligar o "cérebro neural"
    YOLO_FALLBACK_ONLY = False  # Se False, o YOLO atuará como visão PRINCIPAL, não apenas fallback!

    # ── Classes do YOLO (quando treinado) ──
    YOLO_CLASSES = {
        0: "enemy_champion",
        1: "enemy_minion",
        2: "enemy_turret",
        3: "ally_minion",
        4: "jungle_monster",
    }


# ─────────────────────────── Orbwalker ───────────────────────────
class OrbwalkerConfig:
    # Buffer de segurança para evitar cancel de auto-attack (segundos)
    WINDUP_BUFFER = 1.0 / 15.0  # ~66ms

    # Delay mínimo entre inputs para não sobrecarregar o jogo
    MIN_INPUT_DELAY = 1.0 / 30.0  # ~33ms

    # Tick rate do loop principal
    TICK_RATE = 1.0 / 60.0  # ~16ms

    # Compensação de ping (ms) — ajustável pelo usuário
    PING_OFFSET_MS = 0

    # Humanizer range (segundos) — variação randômica no timing
    HUMANIZER_MIN = 0.005  # 5ms
    HUMANIZER_MAX = 0.020  # 20ms

    # Offset de compensação para o delay da captura de tela (segundos)
    SCREEN_CAPTURE_COMPENSATION = 0.008  # ~8ms


# ─────────────────────────── Riot API ───────────────────────────
class RiotAPIConfig:
    BASE_URL = "https://127.0.0.1:2999/liveclientdata"
    ACTIVE_PLAYER = f"{BASE_URL}/activeplayer"
    PLAYER_LIST = f"{BASE_URL}/playerlist"
    ALL_GAME_DATA = f"{BASE_URL}/allgamedata"
    EVENT_DATA = f"{BASE_URL}/eventdata"
    TIMEOUT = 1.5  # segundos
    POLL_INTERVAL = 1.0 / 30.0  # ~33ms polling rate


# ─────────────────────────── CommunityDragon ───────────────────────────
class CDragonConfig:
    BASE_URL = "https://raw.communitydragon.org/latest/game/data/characters"
    TIMEOUT = 5  # segundos


# ─────────────────────────── Settings Persistence ───────────────────────────
SETTINGS_DIR = "settings"
SETTINGS_FILE = os.path.join(SETTINGS_DIR, "settings.json")

DEFAULT_SETTINGS = {
    "hotkeys": {
        "orbwalk": Hotkeys.ORBWALK,
        "lasthit": Hotkeys.LASTHIT,
        "laneclear": Hotkeys.LANECLEAR,
        "harass": Hotkeys.HARASS,
        "toggle_pause": Hotkeys.TOGGLE_PAUSE,
    },
    "orbwalker": {
        "ping_offset_ms": OrbwalkerConfig.PING_OFFSET_MS,
        "humanizer_min": OrbwalkerConfig.HUMANIZER_MIN,
        "humanizer_max": OrbwalkerConfig.HUMANIZER_MAX,
        "windup_buffer": OrbwalkerConfig.WINDUP_BUFFER,
    },
    "vision": {
        "yolo_enabled": VisionConfig.YOLO_ENABLED,
        "target_fps": VisionConfig.TARGET_FPS,
    },
}


def load_settings() -> dict:
    """Carrega settings do disco ou cria defaults."""
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    else:
        save_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict):
    """Persiste settings no disco."""
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)
