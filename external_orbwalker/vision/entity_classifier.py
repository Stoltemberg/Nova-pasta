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
    Classifica health bars em tipos de entidade usando múltiplos sinais:
    1. Tamanho da health bar (principal)
    2. Presença de level indicator (forte)
    3. Presença de barra de mana azul abaixo (forte)
    4. Altura da barra (campeões têm barras ligeiramente mais altas)
    5. Posição na tela (contexto)
    """

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.scale = screen_height / VisionConfig.REFERENCE_HEIGHT

        # ── Thresholds escalados ──
        # NOTA: A largura da health bar VERMELHA encolhe com o HP perdido.
        # Campeão full HP @1080p ≈ 103px, mas com 50% HP a porção vermelha ≈ 52px.
        # Por isso, width sozinha NÃO é suficiente para classificar.
        self.champ_min_w = int(VisionConfig.CHAMPION_BAR_MIN_WIDTH * self.scale)
        self.siege_min_w = int(VisionConfig.MINION_SIEGE_BAR_MIN_WIDTH * self.scale)
        self.minion_max_w = int(VisionConfig.MINION_BAR_MAX_WIDTH * self.scale)
        self.body_offset = int(VisionConfig.BODY_OFFSET_Y * self.scale)

        # ── Level indicator detection params (multi-zone) ──
        self.level_check_size = max(10, int(16 * self.scale))

        # ── Mana bar detection ──
        self.mana_check_height = max(3, int(5 * self.scale))
        self.mana_check_gap = max(1, int(2 * self.scale))  # Gap entre HP bar e mana bar

        logger.info(
            f"EntityClassifier initialized | "
            f"Champion bar >= {self.champ_min_w}px | "
            f"Siege >= {self.siege_min_w}px | "
            f"Minion max = {self.minion_max_w}px"
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

            # ── Classificar pelo tamanho + level + mana ──
            entity_type, confidence = self._classify_by_size_and_level(frame, hb, has_level)

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

    def _classify_by_size_and_level(self, frame: np.ndarray, hb: HealthBar, has_level: bool) -> tuple[EntityType, float]:
        """
        Classificação multi-sinal robusta.
        
        Sinais usados (em ordem de peso):
        1. Largura da barra vermelha (mas CUIDADO: encolhe com HP perdido)
        2. Level indicator (forte: minions nunca têm)
        3. Barra de mana azul (forte: minions nunca têm)
        4. Altura da barra (campeões = 4-5px, minions = 2-3px @1080p)
        """
        w = hb.width
        h = hb.height
        has_mana = self._check_mana_bar(frame, hb)
        
        # ════════════ CAMPEÃO CERTO ════════════
        # Barra muito larga = campeão indiscutível
        if w >= self.champ_min_w:
            return EntityType.CHAMPION, 0.95
        
        # Level indicator = NUNCA é minion (minions não mostram level)
        if has_level:
            return EntityType.CHAMPION, 0.92
        
        # Mana bar azul abaixo = NUNCA é minion
        if has_mana:
            return EntityType.CHAMPION, 0.90
        
        # ════════════ ZONA CINZA (62-89px) ════════════
        # Pode ser campeão com HP parcial OU siege minion.
        # Usar altura da barra como desempate.
        if w >= self.siege_min_w:
            # Barras de campeão são ligeiramente mais altas (~4-5px vs ~3px dos minions)
            bar_height_threshold = max(3, int(4 * self.scale))
            if h >= bar_height_threshold:
                return EntityType.CHAMPION, 0.75
            return EntityType.MINION_SIEGE, 0.80
        
        # ════════════ MINION CERTO ════════════
        # Barra pequena, sem level, sem mana = minion
        if w > 0:
            return EntityType.MINION, 0.90
        
        return EntityType.UNKNOWN, 0.0

    def _check_level_indicator(self, frame: np.ndarray, hb: HealthBar) -> bool:
        """
        Verifica se existe um indicador de nível ao lado da health bar.
        Campeões inimigos têm um número de nível branco/amarelo à esquerda.
        Minions NUNCA têm isso.
        
        Usa múltiplas zonas de verificação para resistir a variações de
        posição e escala entre campeões diferentes.
        """
        # ── Zona 1: Imediatamente à esquerda da barra ──
        # O level text fica ~3-18px à esquerda da borda da barra
        size = self.level_check_size
        
        # Testar duas posições levemente diferentes (variação entre campeões)
        check_positions = [
            (hb.x - size - 2, hb.y - 2),      # Posição padrão
            (hb.x - size + 2, hb.y - 4),      # Ligeiramente mais perto/acima
        ]
        
        for check_x, check_y in check_positions:
            if check_x < 0 or check_y < 0:
                continue
            if check_x + size >= frame.shape[1] or check_y + size >= frame.shape[0]:
                continue
            
            roi = frame[check_y:check_y + size, check_x:check_x + size]
            if roi.size == 0:
                continue
            
            # ── Detectar texto branco/amarelo ──
            # Converter para HSV para diferenciar branco/amarelo de ruído
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Branco puro (alto brilho, baixa saturação)
            _, white_thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
            white_ratio = cv2.countNonZero(white_thresh) / max(gray.size, 1)
            
            # Amarelo (hue 20-35, alta saturação) — level enemies em alguns skins
            yellow_mask = cv2.inRange(hsv_roi, 
                                      np.array([18, 80, 150], dtype=np.uint8),
                                      np.array([38, 255, 255], dtype=np.uint8))
            yellow_ratio = cv2.countNonZero(yellow_mask) / max(gray.size, 1)
            
            combined_ratio = white_ratio + yellow_ratio
            
            # Level text = ~10-45% de pixels claros na ROI
            if 0.08 < combined_ratio < 0.55:
                return True
        
        return False
    
    def _check_mana_bar(self, frame: np.ndarray, hb: HealthBar) -> bool:
        """
        Verifica se existe uma barra de mana azul logo abaixo da health bar.
        Campeões têm mana bar; minions NÃO.
        Este é um dos sinais mais confiáveis para diferenciar.
        """
        # Região abaixo da health bar
        mana_x = hb.x
        mana_y = hb.y + hb.height + self.mana_check_gap
        mana_w = hb.width
        mana_h = self.mana_check_height
        
        # Boundary check
        if mana_y + mana_h >= frame.shape[0] or mana_x + mana_w >= frame.shape[1]:
            return False
        if mana_x < 0 or mana_y < 0:
            return False
            
        roi = frame[mana_y:mana_y + mana_h, mana_x:mana_x + mana_w]
        if roi.size == 0:
            return False
        
        # Detectar azul (mana) no HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Azul = hue 90-130, saturação alta, brilho médio-alto
        blue_mask = cv2.inRange(hsv_roi,
                                np.array([90, 60, 80], dtype=np.uint8),
                                np.array([130, 255, 255], dtype=np.uint8))
        
        blue_ratio = cv2.countNonZero(blue_mask) / max(roi.size // 3, 1)  # divido por 3 canais
        
        # Se >15% dos pixels são azuis, é mana bar
        return blue_ratio > 0.15

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
