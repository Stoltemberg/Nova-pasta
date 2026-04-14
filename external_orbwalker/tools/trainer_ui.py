"""
tools/trainer_ui.py — MLOps Dashboard (GUI) para treinar o YOLO.
Usa customtkinter para uma interface rica, dark mode e moderna.
"""
import os
import sys
import threading
import traceback
import customtkinter as ctk

# Configuração da aparência do framework (Elegante)
ctk.set_appearance_mode("Dark")  # Dark, Light, System
ctk.set_default_color_theme("blue")  # Themes: blue, green, dark-blue

class TrainerUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Neural Trainer — External Orbwalker")
        self.geometry("600x480")
        self.resizable(False, False)
        
        self.is_training = False
        
        self._build_ui()

    def _build_ui(self):
        # Frame Principal
        self.main_frame = ctk.CTkFrame(self, corner_radius=15)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Título
        self.lbl_title = ctk.CTkLabel(
            self.main_frame, 
            text="YOLO MLOps Dashboard",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.lbl_title.pack(pady=(20, 10))
        
        self.lbl_subtitle = ctk.CTkLabel(
            self.main_frame, 
            text="Acompanhamento em Tempo Real do Aprendizado",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.lbl_subtitle.pack(pady=(0, 20))

        # ── Dashboard Metrics Row ──
        self.metrics_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.metrics_frame.pack(fill="x", padx=30, pady=10)
        self.metrics_frame.grid_columnconfigure(0, weight=1)
        self.metrics_frame.grid_columnconfigure(1, weight=1)
        
        # Card 1: Inteligência (% mAP50)
        self.card_intel = ctk.CTkFrame(self.metrics_frame, corner_radius=10)
        self.card_intel.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.lbl_intel_title = ctk.CTkLabel(self.card_intel, text="Inteligência (mAP)", font=ctk.CTkFont(size=12))
        self.lbl_intel_title.pack(pady=(10, 0))
        self.lbl_intel_val = ctk.CTkLabel(self.card_intel, text="0.0%", font=ctk.CTkFont(size=28, weight="bold"), text_color="#2ecc71")
        self.lbl_intel_val.pack(pady=(5, 15))

        # Card 2: Épocas Analisadas
        self.card_epochs = ctk.CTkFrame(self.metrics_frame, corner_radius=10)
        self.card_epochs.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        self.lbl_epochs_title = ctk.CTkLabel(self.card_epochs, text="Progresso (Épocas)", font=ctk.CTkFont(size=12))
        self.lbl_epochs_title.pack(pady=(10, 0))
        self.lbl_epochs_val = ctk.CTkLabel(self.card_epochs, text="0 / 100", font=ctk.CTkFont(size=28, weight="bold"), text_color="#3498db")
        self.lbl_epochs_val.pack(pady=(5, 15))

        # ── Barra de Progresso ──
        self.progress_bar = ctk.CTkProgressBar(self.main_frame, height=15)
        self.progress_bar.pack(fill="x", padx=40, pady=(20, 10))
        self.progress_bar.set(0.0)

        # ── Botão de Ação ──
        self.btn_start = ctk.CTkButton(
            self.main_frame,
            text="INICIAR TREINAMENTO NEURAL",
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.start_training
        )
        self.btn_start.pack(pady=(20, 20))

    def update_dashboard(self, current_epoch, total_epochs, map50):
        # map50 costuma ir de 0.0 a 1.0 (onde 1.0 = 100%)
        intel_pct = map50 * 100
        
        self.lbl_intel_val.configure(text=f"{intel_pct:.1f}%")
        self.lbl_epochs_val.configure(text=f"{current_epoch} / {total_epochs}")
        
        if total_epochs > 0:
            self.progress_bar.set(current_epoch / total_epochs)

    def start_training(self):
        if self.is_training:
            return
            
        self.is_training = True
        self.btn_start.configure(state="disabled", text="TREINANDO... (olhe o terminal)")
        
        # O Motor Neural roda em Thread Separada para não congelar esta janela
        self.thread = threading.Thread(target=self.run_yolo_engine, daemon=True)
        self.thread.start()

    def run_yolo_engine(self):
        try:
            from ultralytics import YOLO
            
            # Reset UI
            self.after(0, lambda: self.update_dashboard(0, 100, 0.0))
            
            # Carregar Arquitetura Base
            model = YOLO("yolov8n.pt")
            
            # ── Hook/Callback: Pegar dados ao final de cada Época ──
            def on_fit_epoch_end(trainer):
                epoch = trainer.epoch + 1
                total = trainer.epochs
                
                # trainer.metrics é um dicionário; o mAP50 em BBoxes costuma ficar na chave 'metrics/mAP50(B)' no YOLOv8
                map50 = trainer.metrics.get('metrics/mAP50(B)', 0.0)
                
                # Forçar atualização na Thread Principal do CustomTkinter
                self.after(0, lambda: self.update_dashboard(epoch, total, map50))
            
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

            # Iniciar Treinamento
            base_dir = os.path.dirname(os.path.dirname(__file__))
            dataset_yaml = os.path.join(base_dir, "dataset", "lol_dataset.yaml")
            
            # Passar os MESMOS parâmetros da train_yolo.py
            print("\n" + "="*50)
            print(" [UI] Motor YOLO acionado! Verifique aqui os logs detalhados.")
            print("="*50 + "\n")
            
            model.train(
                data=dataset_yaml,
                epochs=100,
                imgsz=640,
                batch=16,
                name="lol_orbwalker",
                patience=20,
                plots=True,
                verbose=True,
                hsv_h=0.01, hsv_s=0.3, hsv_v=0.3, 
                degrees=0.0, translate=0.1, scale=0.3, 
                fliplr=0.0, flipud=0.0, 
                mosaic=0.5, mixup=0.1
            )
            
            self.after(0, self.training_finished_success)

        except Exception as e:
            print("===========================")
            print(" ERRO NO TREINAMENTO")
            print("===========================")
            traceback.print_exc()
            self.after(0, self.training_failed)

    def training_finished_success(self):
        self.is_training = False
        self.btn_start.configure(state="normal", text="TREINAMENTO CONCLUÍDO (Novo Treino)", fg_color="#2ecc71", hover_color="#27ae60")
        self.lbl_subtitle.configure(text="Sucesso! O novo cerebro 'best.pt' foi salvo.", text_color="#2ecc71")

    def training_failed(self):
        self.is_training = False
        self.btn_start.configure(state="normal", text="ERRO NO TREINAMENTO (Ver Console)", fg_color="#e74c3c", hover_color="#c0392b")

if __name__ == "__main__":
    app = TrainerUI()
    app.mainloop()
