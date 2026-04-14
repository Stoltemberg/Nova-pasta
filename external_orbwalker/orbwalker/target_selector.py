"""
orbwalker/target_selector.py — Seleção inteligente de alvo baseada no modo de jogo.
Prioriza campeões, minions lastáveis, ou qualquer inimigo dependendo da hotkey ativa.
"""
import math
import logging
from vision.entity_classifier import DetectedEntity, EntityType

logger = logging.getLogger("ExternalOrbwalker.TargetSelector")


class TargetSelector:
    """
    Seleciona o melhor alvo para orbwalking baseado no tipo
    de entidade detectada e no modo ativo.
    """

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_center_x = screen_width // 2
        self.screen_center_y = screen_height // 2

    def _distance_to_center(self, entity: DetectedEntity) -> float:
        """Distância do alvo ao centro da tela (onde o player está ~sempre)."""
        dx = entity.screen_x - self.screen_center_x
        dy = entity.screen_y - self.screen_center_y
        return math.hypot(dx, dy)

    def _distance_to_cursor(self, entity: DetectedEntity, cursor_x: int, cursor_y: int) -> float:
        """Distância do alvo ao cursor do mouse."""
        dx = entity.screen_x - cursor_x
        dy = entity.screen_y - cursor_y
        return math.hypot(dx, dy)

    # ─────────────────────────── Modos de Seleção ───────────────────────────

    def select_combo(self, entities: list[DetectedEntity],
                     cursor_x: int = None, cursor_y: int = None) -> DetectedEntity | None:
        """
        Modo COMBO (Orbwalk): Prioriza campeões inimigos.

        Prioridade:
        1. Campeão mais próximo do cursor (ou do centro da tela)
        2. Se não há campeão, retorna None (não ataca minions em combo)
        """
        champions = [e for e in entities if e.entity_type == EntityType.CHAMPION]

        if not champions:
            return None

        # Selecionar pelo mais próximo
        if cursor_x is not None and cursor_y is not None:
            return min(champions, key=lambda e: self._distance_to_cursor(e, cursor_x, cursor_y))
        return min(champions, key=self._distance_to_center)

    def select_lasthit(self, entities: list[DetectedEntity],
                       cursor_x: int = None, cursor_y: int = None) -> DetectedEntity | None:
        """
        Modo LASTHIT: Prioriza minions com baixa vida (barras com pouco fill).

        Prioridade:
        1. Minion com menor fill_ratio (menor vida = mais chance de matar)
        2. Se empate, o mais próximo do cursor
        """
        minions = [e for e in entities
                   if e.entity_type in (EntityType.MINION, EntityType.MINION_SIEGE)]

        if not minions:
            return None

        # Filtrar minions com baixa vida (fill_ratio < 40% = provável lastável)
        low_hp_minions = [m for m in minions if m.health_bar.fill_ratio < 0.40]

        # Se há minions com pouca vida, priorizar esses
        candidates = low_hp_minions if low_hp_minions else minions

        # Ordenar por fill_ratio (menos vida primeiro), depois por distância
        if cursor_x is not None and cursor_y is not None:
            return min(candidates,
                       key=lambda e: (e.health_bar.fill_ratio,
                                      self._distance_to_cursor(e, cursor_x, cursor_y)))
        return min(candidates, key=lambda e: e.health_bar.fill_ratio)

    def select_laneclear(self, entities: list[DetectedEntity],
                         cursor_x: int = None, cursor_y: int = None) -> DetectedEntity | None:
        """
        Modo LANECLEAR: Ataca qualquer inimigo, prioridade mista.

        Prioridade:
        1. Minions lastáveis (fill < 30%)
        2. Minion siege
        3. Minion normal mais próximo
        4. Campeão (se não há minions)
        """
        minions = [e for e in entities
                   if e.entity_type in (EntityType.MINION, EntityType.MINION_SIEGE)]
        champions = [e for e in entities if e.entity_type == EntityType.CHAMPION]

        # Prioridade 1: minions lastáveis
        lasthittable = [m for m in minions if m.health_bar.fill_ratio < 0.30]
        if lasthittable:
            return min(lasthittable, key=lambda e: e.health_bar.fill_ratio)

        # Prioridade 2: siege minions
        sieges = [m for m in minions if m.entity_type == EntityType.MINION_SIEGE]
        if sieges:
            if cursor_x is not None and cursor_y is not None:
                return min(sieges, key=lambda e: self._distance_to_cursor(e, cursor_x, cursor_y))
            return min(sieges, key=self._distance_to_center)

        # Prioridade 3: qualquer minion
        if minions:
            if cursor_x is not None and cursor_y is not None:
                return min(minions, key=lambda e: self._distance_to_cursor(e, cursor_x, cursor_y))
            return min(minions, key=self._distance_to_center)

        # Prioridade 4: campeão (fallback)
        if champions:
            return min(champions, key=self._distance_to_center)

        return None

    def select_harass(self, entities: list[DetectedEntity],
                      cursor_x: int = None, cursor_y: int = None) -> DetectedEntity | None:
        """
        Modo HARASS: Ataca campeão se disponível, senão não ataca.
        Igual ao combo mas não cancela windup para dar move.
        """
        return self.select_combo(entities, cursor_x, cursor_y)

    def select_nearest(self, entities: list[DetectedEntity],
                       cursor_x: int = None, cursor_y: int = None) -> DetectedEntity | None:
        """Seleciona a entidade inimiga mais próxima do cursor, independente do tipo."""
        if not entities:
            return None

        if cursor_x is not None and cursor_y is not None:
            return min(entities, key=lambda e: self._distance_to_cursor(e, cursor_x, cursor_y))
        return min(entities, key=self._distance_to_center)
