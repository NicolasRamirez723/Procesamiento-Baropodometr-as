import matplotlib.pyplot as plt
import numpy as np
import os


"Se extraen trayectoria de COP para x e y, COP_y vs COP_x y fuerza-t"
MAX_PLOT_FRAMES = 5000


def read_data():
    for paciente in [1,2]:

        for pasada in [1,3,5,7,9]:
            PATH = f"data/muestras/The nature of functional variability in plantar pressure during a range of controlled walking speeds/Subjects1_2/Subject{paciente}_1p{pasada}.txt"

            data = []
            time_r = [] #tiempos de apoyo de la pierna derecha
            time_f = [] #tiempos de apoyo de la pierna izquierda

            FP_frames_r = [] #Fuerzas puntuales aplicada en el centro de presión del pie derecho por frame

            FP_frames_l = [] #Fuerzas puntuales aplicada en el centro de presión del pie izquierdo por frame


            COP_frames_x = []

            COP_frames_y = []
            frames = []
            # Estado de cada pie: evita reiniciar el tiempo mientras el pie continúa apoyado.
            apoyo_derecho = False
            apoyo_izquierdo = False
            tr_i = None
            tl_i = None
            try:
                with open(PATH,"r") as file:

                    for line in file:

                        #FORMATO FRAME: [Time (ms),	Position (m),l force (N),	r force (N)	, Butterfly.force(N),	Butterfly.x (mm),	Butterfly.y (mm)]
                        
                        frame = line.strip().split()
                        if len(frame) >= 7:
                            try:
                                tiempo = float(frame[0])
                            except ValueError:
                                continue

                            data.append(frame)
                            frames.append(tiempo)

                            fuerza_izquierda = float(frame[2])
                            fuerza_derecha = float(frame[3])
                            cop_x = float(frame[5])
                            cop_y = float(frame[6])

                            FP_frames_l.append(fuerza_izquierda)
                            FP_frames_r.append(fuerza_derecha)

                            # NaN corta la línea cuando no existe COP válido.
                            COP_frames_x.append(cop_x if np.isfinite(cop_x) else np.nan)
                            COP_frames_y.append(cop_y if np.isfinite(cop_y) else np.nan)

                            # El apoyo comienza cuando la fuerza pasa de cero a positiva.
                            if fuerza_derecha > 0 and not apoyo_derecho:
                                tr_i = tiempo
                                apoyo_derecho = True

                            if fuerza_izquierda > 0 and not apoyo_izquierdo:
                                tl_i = tiempo
                                apoyo_izquierdo = True

                            # El apoyo termina cuando la fuerza vuelve a cero.
                            if fuerza_derecha <= 0 and apoyo_derecho:
                                time_r.append(tiempo - tr_i)
                                apoyo_derecho = False
                                tr_i = None

                            if fuerza_izquierda <= 0 and apoyo_izquierdo:
                                time_f.append(tiempo - tl_i)
                                apoyo_izquierdo = False
                                tl_i = None

                # Cierra un apoyo que eventualmente continúe hasta el final del registro.
                if data:
                    ultimo_tiempo = float(data[-1][0])
                    if apoyo_derecho:
                        time_r.append(ultimo_tiempo - tr_i)
                    if apoyo_izquierdo:
                        time_f.append(ultimo_tiempo - tl_i)
                            
                print(f"Se ha leído el registro numero {pasada} baropodométrico del paciente numero {paciente} correctamente")

                PATH_BARO_PROCESADA = f"Baropodometrias procesadas/"

                dir_destino = os.path.join(PATH_BARO_PROCESADA,f"Subject{paciente}")

                
                os.makedirs(dir_destino,exist_ok=True) 
                     
                with open (f"{dir_destino}/Pasada_{pasada}.txt","w",newline='', encoding='utf-8') as archive:
                    #"Tiempo de apoyo medio pie izquierdo y derecho"
                    t_a_m_r = np.mean(time_r)

                    t_a_m_l = np.mean(time_f)

                    FP_m_r = np.mean(FP_frames_r)

                    FP_m_l = np.mean(FP_frames_l)



                    #Guardamos registro de los tiempos de apoyo medio para posteriormente calcular el impulso dinámico
                    archive.write(f"Paciente {paciente} pasada {pasada}. \nTiempo de apoyo medio pie izquierdo: {t_a_m_l}, fuerza puntual media pie izquierdo {FP_m_l}\n Tiempo de apoyo medio pie derecho: {t_a_m_r}, fuerza puntual media pie derecho {FP_m_r}")


                # Las gráficas usan solo los primeros 5000 frames; los cálculos anteriores usan todo el registro.
                frames_grafica = frames[:MAX_PLOT_FRAMES]
                fuerzas_izquierda_grafica = FP_frames_l[:MAX_PLOT_FRAMES]
                fuerzas_derecha_grafica = FP_frames_r[:MAX_PLOT_FRAMES]
                cop_x_grafica = COP_frames_x[:MAX_PLOT_FRAMES]
                cop_y_grafica = COP_frames_y[:MAX_PLOT_FRAMES]

                # Gráfico de centro de presiones (x,y).

                fig_cop, ax_cop = plt.subplots(figsize=(8, 6))
                ax_cop.plot(cop_x_grafica,cop_y_grafica,color = "red",label = "Centro de Presiones")

                ax_cop.grid(True)

                ax_cop.set_xlabel(r"$COP_{x}(mm)$",loc="right")

                ax_cop.set_ylabel(r"$COP_{y}(mm)$",loc="top",rotation=0)

                ax_cop.set_title(f"Trayectoria COP paciente {paciente} pasada {pasada}")

                fig_cop.savefig(dir_destino +f"/COP_trayectoria_paciente_{paciente}_pasada_{pasada}.png")

                plt.close(fig_cop)

                # Gráficos del COP en función del tiempo.
                fig_cop_tiempo, (ax_cop_x, ax_cop_y) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

                ax_cop_x.plot(frames_grafica, cop_x_grafica, color="darkorange")
                ax_cop_x.set_title("COP x en función del tiempo")
                ax_cop_x.set_ylabel("COP x (mm)")
                ax_cop_x.grid(True)

                ax_cop_y.plot(frames_grafica, cop_y_grafica, color="seagreen")
                ax_cop_y.set_title("COP y en función del tiempo")
                ax_cop_y.set_xlabel("Tiempo (ms)")
                ax_cop_y.set_ylabel("COP y (mm)")
                ax_cop_y.grid(True)

                plt.tight_layout()
                fig_cop_tiempo.savefig(os.path.join(dir_destino, f"COP_vs_tiempo_paciente_{paciente}_pasada_{pasada}.png"))
                plt.close(fig_cop_tiempo)

                #Genero los gráficos Fuerza puntual - tiempo en cada pie 
                fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))

                ax1.plot(frames_grafica,fuerzas_izquierda_grafica,color="Blue")

                ax1.set_title("Fuerza pie izquierdo")

                ax1.set_ylabel("Fuerza (N)")

                ax1.set_xlabel("Tiempo (ms)")

                ax1.grid(True)

                ax2.plot(frames_grafica,fuerzas_derecha_grafica,color="Blue")

                ax2.set_title("Fuerza pie derecho")

                ax2.set_ylabel("Fuerza (N)")

                ax2.set_xlabel("Tiempo (ms)")

                ax2.grid(True)

                plt.tight_layout()

                plt.savefig(os.path.join(dir_destino,f"Fuerzas_paciente_{paciente}_pasada_{pasada}.png"))

                print("Registro baropodométrico procesado y guardado correctamente")


            except FileNotFoundError:   

                print("Error: Archivo sin encontrar")



if __name__ == "__main__":

    read_data()
    

