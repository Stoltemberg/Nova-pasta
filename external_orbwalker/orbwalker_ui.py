"""
orbwalker_ui.py — Painel de Controle Principal do Bot (GUI)
Substitui o terminal de texto por um painel de aviador (Dark Mode e TopMost)
permitindo alteração de configuração in-game.
"""
import os
import sys
import threading
import time
import logging
import customtkinter as ctk

# Importar o núcleo do bot original
from main import ExternalOrbwalker
from config import VisionConfig, OrbwalkerConfig, save_settings, load_settings

# Redirecionar logging do console para silenciar um pouco no painel se necessário,
# Mas vamos focar na UI extraindo os dados dos objetos do bot.

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class OrbwalkerUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("External Orbwalker — Vision AI")
        self.geometry("450x550")
        self.resizable(False, False)
        
        # Modo Flutuante (Sempre por cima do jogo)
        self.attributes("-topmost", True)
        
        # Objeto principal do BOT
        self.bot = None
        self.bot_thread = None
        
        # Variáveis de Interface
        self.var_yolo = ctk.BooleanVar(value=VisionConfig.YOLO_ENABLED)
        self.var_ping = ctk.DoubleVar(value=OrbwalkerConfig.PING_OFFSET_MS)
        
        self._build_ui()
        self._updater_loop()

    def _build_ui(self):
        # ── HEADER ──
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", padx=20, pady=15)
        
        self.lbl_title = ctk.CTkLabel(self.header_frame, text="VISÃO EXTERNA", font=ctk.CTkFont(size=22, weight="bold"))
        self.lbl_title.pack()
        
        self.lbl_status = ctk.CTkLabel(self.header_frame, text="Status: Aguardando League of Legends...", text_color="#f1c40f")
        self.lbl_status.pack()

        # ── STATUS CARD (API & Jogo) ──
        self.card_game = ctk.CTkFrame(self, corner_radius=10)
        self.card_game.pack(fill="x", padx=20, pady=5)
        
        self.lbl_champ = ctk.CTkLabel(self.card_game, text="Campeão: Desconhecido", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_champ.grid(row=0, column=0, padx=15, pady=5, sticky="w")
        
        self.lbl_mode = ctk.CTkLabel(self.card_game, text="Modo: IDLE", font=ctk.CTkFont(size=14, weight="bold"), text_color="gray")
        self.lbl_mode.grid(row=0, column=1, padx=15, pady=5, sticky="e")

        self.lbl_hp = ctk.CTkLabel(self.card_game, text="HP: 0%", font=ctk.CTkFont(size=12))
        self.lbl_hp.grid(row=1, column=0, padx=15, pady=5, sticky="w")
        
        self.lbl_as = ctk.CTkLabel(self.card_game, text="AS: 0.00", font=ctk.CTkFont(size=12))
        self.lbl_as.grid(row=1, column=1, padx=15, pady=5, sticky="e")

        self.card_game.grid_columnconfigure(0, weight=1)
        self.card_game.grid_columnconfigure(1, weight=1)

        # ── COMBAT CARD (Ataques e Visão) ──
        self.card_combat = ctk.CTkFrame(self, corner_radius=10)
        self.card_combat.pack(fill="x", padx=20, pady=10)
        
        self.lbl_target = ctk.CTkLabel(self.card_combat, text="Mira Atual: Nenhum Alvo (Cego)", font=ctk.CTkFont(size=13))
        self.lbl_target.pack(pady=10, padx=15, anchor="w")
        
        self.lbl_attacks = ctk.CTkLabel(self.card_combat, text="Ataques Realizados: 0 | Movimentos: 0", font=ctk.CTkFont(size=12), text_color="gray")
        self.lbl_attacks.pack(pady=(0, 10), padx=15, anchor="w")

        # ── SETTINGS CARD ──
        self.card_settings = ctk.CTkFrame(self, corner_radius=10)
        self.card_settings.pack(fill="both", expand=True, padx=20, pady=5)
        
        lbl_settings_tit = ctk.CTkLabel(self.card_settings, text="Painel de Configurações", font=ctk.CTkFont(size=14, weight="bold"))
        lbl_settings_tit.pack(pady=(10, 15))

        # Switch YOLO
        self.sw_yolo = ctk.CTkSwitch(
            self.card_settings, text="Habilitar Cérebro Artificial (YOLO)",
            variable=self.var_yolo, command=self._on_yolo_toggle,
            progress_color="#2ecc71"
        )
        self.sw_yolo.pack(pady=5, padx=20, anchor="w")

        # Slider Ping
        self.lbl_ping = ctk.CTkLabel(self.card_settings, text=f"Compensação de Ping: {int(self.var_ping.get())} ms")
        self.lbl_ping.pack(pady=(15, 0), padx=20, anchor="w")
        
        self.sl_ping = ctk.CTkSlider(
            self.card_settings, from_=0, to=150,
            variable=self.var_ping, command=self._on_ping_changed
        )
        self.sl_ping.pack(fill="x", padx=20, pady=5)

        # ── INICIAR BOTÃO ──
        self.btn_run = ctk.CTkButton(
            self, text="INJETAR ORBWALKER NO JOGO", height=45, fg_color="#e67e22", hover_color="#d35400",
            font=ctk.CTkFont(size=14, weight="bold"), command=self.toggle_bot
        )
        self.btn_run.pack(pady=20, padx=20, fill="x")

    def _on_yolo_toggle(self):
        """Atualiza a Flag global para a Visão YOLO desativar em tempo real."""
        VisionConfig.YOLO_ENABLED = self.var_yolo.get()
        # Poderiamos salvar_settings() aqui para persistir.

    def _on_ping_changed(self, value):
        ms = int(value)
        self.lbl_ping.configure(text=f"Compensação de Ping: {ms} ms")
        OrbwalkerConfig.PING_OFFSET_MS = ms

    def toggle_bot(self):
        """Inicia ou Para as threads do bot."""
        if self.bot is None or not self.bot.running:
            # Inicializar
            self.btn_run.configure(text="DESLIGAR BOT", fg_color="#e74c3c", hover_color="#c0392b")
            self.lbl_status.configure(text="Status: Bot Rodando", text_color="#2ecc71")
            
            self.bot = ExternalOrbwalker()
            self.bot.running = True
            
            # Start background thread (blocking=False para não rodar o _status_loop interno)
            self.bot_thread = threading.Thread(target=self.bot.start, args=(False,), daemon=True)
            self.bot_thread.start()
        else:
            # Desligar
            if self.bot:
                self.bot._shutdown()
            self.btn_run.configure(text="INJETAR ORBWALKER NO JOGO", fg_color="#e67e22", hover_color="#d35400")
            self.lbl_status.configure(text="Status: Desconectado", text_color="#e74c3c")
            self._reset_labels()

    def _updater_loop(self):
        """Atualiza as Labels Visuais da Tela 15x por segundo lendo o bot na memória."""
        if self.bot and self.bot.running:
            
            # Atualiza conexão com Riot API
            if self.bot.riot_api.connected:
                # Campeão
                if self.bot.riot_api.champion_name:
                    self.lbl_champ.configure(text=f"Campeão: {self.bot.riot_api.champion_name}")
                
                # HP & AS
                hp = self.bot.riot_api.health_percent * 100
                ass = self.bot.riot_api.attack_speed
                self.lbl_hp.configure(text=f"HP: {hp:.0f}%")
                self.lbl_as.configure(text=f"AS: {ass:.2f}")
            else:
                self.lbl_champ.configure(text="Campeão: Desconectado")
                
            # Atualiza Status do Engine de Combate
            if self.bot.engine:
                # Mode
                if self.bot.engine.active:
                    self.lbl_mode.configure(text=f"Modo: {self.bot.engine.mode.upper()}", text_color="#e74c3c")
                else:
                    self.lbl_mode.configure(text="Modo: IDLE", text_color="gray")
                
                # Target Vision
                target, ttype = self.bot.engine.get_vision_target()
                if target:
                    color = "#e74c3c" if ttype.name == "CHAMPION" else "#3498db"
                    self.lbl_target.configure(text=f"Mira Atual: 🎯 {ttype.name} em X:{target[0]} Y:{target[1]}", text_color=color)
                else:
                    self.lbl_target.configure(text="Mira Atual: Nenhum Alvo (Buscando...)", text_color="white")
                
                # Ataques / Cliques
                self.lbl_attacks.configure(text=f"Ataques Realizados: {self.bot.engine.attacks} | Movimentos: {self.bot.engine.moves}")
                
        # Re-agenda loop
        self.after(66, self._updater_loop) # ~15 FPS UI update

    def _reset_labels(self):
        self.lbl_champ.configure(text="Campeão: Desconhecido")
        self.lbl_mode.configure(text="Modo: IDLE", text_color="gray")
        self.lbl_hp.configure(text="HP: 0%")
        self.lbl_as.configure(text="AS: 0.00")
        self.lbl_target.configure(text="Mira Atual: Nenhum Alvo (Cego)", text_color="white")
        self.lbl_attacks.configure(text="Ataques Realizados: 0 | Movimentos: 0")

    def on_closing(self):
        """Ao fechar a janela, matar o bot"""
        if self.bot:
            self.bot._shutdown()
        self.destroy()

if __name__ == "__main__":
    app = OrbwalkerUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
