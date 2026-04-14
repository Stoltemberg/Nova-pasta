"""
vision/entity_classifier.py — Classifica health bars detectadas em tipos de entidade.
Usa tamanho da barra, presença de level indicator, e contexto.
"""
import cv2
import numpy as np
import logging
from enum import Enum, auto
from dataclasses import dataclass

from config import VisionConfig
from vision.health_bar_detector import HealthBar

logger = logging.getLogger("ExternalOrbwalker.EntityClassifier")


class EntityType(Enum):
    CHAMPION = auto()
    MINION = auto()
    MINION_SIEGE = auto()
    TURRET = auto()
    MONSTER = auto()
    UNKNOWN = auto()


@dataclass
class DetectedEntity:
    """Entidade detectada e classificada na tela."""
    entity_type: EntityType
    screen_x: int          # Posição X estimada do corpo na tela
    screen_y: int          # Posição Y estimada do corpo na tela
    health_bar: HealthBar  # Health bar original
    confidence: float      # Confiança da classificação (0.0 a 1.0)
    has_level: bool        # Se foi detectado um indicador de nível


class EntityClassifier:
    """
    Classifica health bars em tipos de entidade usando:
    1. Tamanho da health bar (principal)
    2. Presença de level indicator (secundário)
    3. Contexto visual (terciário)
    """

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        # LoL UI escala pela ALTURA (mesma fix do HealthBarDetector)
        self.scale = screen_height / VisionConfig.REFERENCE_HEIGHT

        # ── Thresholds escalados ──
        self.champ_min_w = int(VisionConfig.CHAMPION_BAR_MIN_WIDTH * self.scale)
        self.siege_min_w = int(VisionConfig.MINION_SIEGE_BAR_MIN_WIDTH * self.scale)
        self.minion_max_w = int(VisionConfig.MINION_BAR_MAX_WIDTH * self.scale)
        self.body_offset = int(VisionConfig.BODY_OFFSET_Y * self.scale)

        # ── Level indicator detection params ──
        self.level_offset_x = int(VisionConfig.LEVEL_CHECK_OFFSET_X * self.scale)
        self.level_offset_y = int(VisionConfig.LEVEL_CHECK_OFFSET_Y * self.scale)
        self.level_check_size = max(8, int(VisionConfig.LEVEL_CHECK_SIZE * self.scale))

        logger.info(
            f"EntityClassifier initialized | "
            f"Champion bar >= {self.champ_min_w}px | "
            f"Minion bar <= {self.minion_max_w}px"
        )

    def classify(self, frame: np.ndarray, health_bars: list[HealthBar]) -> list[DetectedEntity]:
        """
        Classifica uma lista de health bars detectadas.

        Args:
            frame: Frame BGR original (para análise de level indicator)
            health_bars: Lista de HealthBar do detector

        Returns:
            Lista de DetectedEntity classificadas
        """
        entities = []

        for hb in health_bars:
            # ── Verificar presença de level indicator ──
            has_level = self._check_level_indicator(frame, hb)

            # ── Classificar pelo tamanho + level ──
            entity_type, confidence = self._classify_by_size_and_level(hb, has_level)

            # ── Estimar posição do corpo ──
            body_x = hb.center_x
            body_y = hb.center_y + self.body_offset

            entities.append(DetectedEntity(
                entity_type=entity_type,
                screen_x=body_x,
                screen_y=body_y,
                health_bar=hb,
                confidence=confidence,
                has_level=has_level
            ))

        return entities

    def _classify_by_size_and_level(self, hb: HealthBar, has_level: bool) -> tuple[EntityType, float]:
        """
        Classificação baseada no tamanho da barra + presença de nível.

        Returns:
            (EntityType, confidence)
        """
        w = hb.width

        # ── Campeão: barra larga OU barra média com level ──
        if w >= self.champ_min_w:
            return EntityType.CHAMPION, 0.95

        if has_level and w >= self.siege_min_w:
            # Barra de tamanho médio mas tem level = provavelmente champion
            return EntityType.CHAMPION, 0.80

        # ── Minion siege/cannon: barra de tamanho intermediário ──
        if w >= self.siege_min_w:
            if has_level:
                return EntityType.CHAMPION, 0.75
            return EntityType.MINION_SIEGE, 0.85

        # ── Minion normal ──
        if w > 0:
            if has_level:
                # Barra pequena mas tem level = edge case, favorecemos champion
                return EntityType.CHAMPION, 0.60
            return EntityType.MINION, 0.90

        return EntityType.UNKNOWN, 0.0

    def _check_level_indicator(self, frame: np.ndarray, hb: HealthBar) -> bool:
        """
        Verifica se existe um indicador de nível ao lado da health bar.
        Campeões inimigos têm um pequeno texto de nível (número branco/amarelo)
        à esquerda da health bar.

        Returns:
            True se detectou indicador de nível
        """
        # Região de verificação: à esquerda da health bar
        check_x = hb.x + self.level_offset_x
        check_y = hb.y + self.level_offset_y
        size = self.level_check_size

        # Boundary check
        if check_x < 0 or check_y < 0:
            return False
        if check_x + size >= frame.shape[1] or check_y + size >= frame.shape[0]:
            return False

        # Extrair região
        roi = frame[check_y:check_y + size, check_x:check_x + size]

        if roi.size == 0:
            return False

        # ── Detectar texto branco/amarelo (nível) ──
        # Converter para grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Threshold para pixels claros (texto branco/amarelo do nível)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Se há uma quantidade significativa de pixels claros, há texto
        white_ratio = cv2.countNonZero(thresh) / max(gray.size, 1)

        # Nível tem ~15-35% de pixels brancos na ROI
        return 0.10 < white_ratio < 0.50

    def get_champions(self, entities: list[DetectedEntity]) -> list[DetectedEntity]:
        """Filtra apenas campeões."""
        return [e for e in entities if e.entity_type == EntityType.CHAMPION]

    def get_minions(self, entities: list[DetectedEntity]) -> list[DetectedEntity]:
        """Filtra apenas minions normais e siege."""
        return [e for e in entities
                if e.entity_type in (EntityType.MINION, EntityType.MINION_SIEGE)]

    def get_all_enemies(self, entities: list[DetectedEntity]) -> list[DetectedEntity]:
        """Retorna todas as entidades inimigas detectadas."""
        return [e for e in entities if e.entity_type != EntityType.UNKNOWN]

    def draw_debug(self, frame: np.ndarray, entities: list[DetectedEntity]) -> np.ndarray:
        """Desenha entidades classificadas com cores por tipo."""
        debug = frame.copy()
        colors = {
            EntityType.CHAMPION: (0, 0, 255),     # Vermelho
            EntityType.MINION: (0, 200, 200),      # Amarelo
            EntityType.MINION_SIEGE: (0, 165, 255), # Laranja
            EntityType.TURRET: (255, 0, 255),      # Magenta
            EntityType.MONSTER: (255, 100, 0),     # Azul
            EntityType.UNKNOWN: (128, 128, 128),   # Cinza
        }

        for entity in entities:
            color = colors.get(entity.entity_type, (255, 255, 255))
            hb = entity.health_bar

            # Health bar outline
            cv2.rectangle(debug, (hb.x, hb.y),
                          (hb.x + hb.width, hb.y + hb.height), color, 2)

            # Body position crosshair
            cv2.drawMarker(debug, (entity.screen_x, entity.screen_y),
                           color, cv2.MARKER_CROSS, 15, 2)

            # Label
            label = f"{entity.entity_type.name} ({entity.confidence:.0%})"
            if entity.has_level:
                label += " [LVL]"
            cv2.putText(debug, label, (hb.x, hb.y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return debug
