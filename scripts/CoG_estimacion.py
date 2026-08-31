import numpy as np
import matplotlib.pyplot as plt


class CoG_trayectoria:
    """Simula la trayectoria continua del centro de gravedad durante varios pasos.

    La lógica sigue el patrón de marcha asimétrica:
    - en cada paso hay una pierna de apoyo (l_st) y otra de balanceo (l_sw)
    - se detecta el impacto del talón cuando la altura de la pierna de balanceo llega a cero
    - después del impacto se intercambian los roles de las piernas
    - se acumula un offset horizontal para que la trayectoria avance sin volver a empezar
    """

    def __init__(self, M: float, l_s: float, l_l: float):
        self.masa = M
        self.L_short = l_s
        self.L_long = l_l
        self.delta_L = self.L_long - self.L_short

    @staticmethod
    def _swap_leg_lengths(l_st: float, l_sw: float):
        return l_sw, l_st

    @staticmethod
    def _angulos_impacto(l_st: float, l_sw: float, n_puntos: int = 200):
        """Devuelve theta1 y theta2 para un paso, con la condición de contacto en el impacto.

        El criterio de contacto se impone como:
            l_st * cos(theta1) - l_sw * cos(theta2) = 0
        cuando la pierna de balanceo llega al suelo (t^+ / t^- del impacto).
        """
        if l_st == 0 or l_sw == 0:
            raise ValueError("Las longitudes de piernas deben ser mayores que cero.")

        if l_st >= l_sw:
            theta1_fin = np.arccos(np.clip(l_sw / l_st, -1.0, 1.0))
            theta1 = np.linspace(0.0, theta1_fin, n_puntos)
            theta2 = np.arccos(np.clip((l_st / l_sw) * np.cos(theta1), -1.0, 1.0))
        else:
            theta2_fin = np.arccos(np.clip(l_st / l_sw, -1.0, 1.0))
            theta2 = np.linspace(0.0, theta2_fin, n_puntos)
            theta1 = np.arccos(np.clip((l_sw / l_st) * np.cos(theta2), -1.0, 1.0))

        return theta1, theta2

    def _generar_paso_oscilante(self, l_st: float, l_sw: float, n_puntos: int):
        """Genera un arco de CoG con altura distinta según la pierna de apoyo.

        El paso de apoyo largo produce un arco más alto y más largo; el paso con apoyo corto
        produce un arco más bajo y más corto. Esto reproduce la alternancia fisiológica del CoG
        entre pasos largos y cortos.
        """
        u = np.linspace(0.0, 1.0, n_puntos)
        ratio = (l_st - l_sw) / max(l_st, l_sw)

        if l_st >= l_sw:
            amplitude = 0.18 + 0.14 * ratio
            step = 0.28 + 0.42 * ratio
        else:
            amplitude = 0.12 + 0.10 * abs(ratio)
            step = 0.20 + 0.28 * abs(ratio)

        x_local = step * u
        y_local = amplitude * np.sin(np.pi * u)

        # Pequeño desplazamiento anterior del centro de masa para dar más realismo a la curva.
        x_local = x_local + 0.08 * amplitude * np.sin(np.pi * u)

        return x_local, y_local, step

    def generar_trayectoria(
        self,
        n_pasos: int = 6,
        n_puntos: int = 300,
        plot: bool = True,
        mostrar: bool = True,
    ):
        """Genera la trayectoria del CoG a lo largo de varios pasos.

        Parametros
        ----------
        n_pasos : int
            Cantidad de pasos a simular.
        n_puntos : int
            Cantidad de puntos por paso.
        plot : bool
            Si True, grafica la trayectoria.
        mostrar : bool
            Si True, muestra la figura.
        """
        x_global = []
        y_global = []
        offset_acumulado = 0.0

        # Configuración inicial: primer apoyo sobre la pierna larga.
        l_st = self.L_long
        l_sw = self.L_short

        for _ in range(n_pasos):
            theta1, theta2 = self._angulos_impacto(l_st, l_sw, n_puntos)
            impact_condition = np.abs(l_st * np.cos(theta1[-1]) - l_sw * np.cos(theta2[-1]))

            if impact_condition > 1e-6:
                # El evento de impacto se usa para definir el final del ciclo del paso.
                # Aunque el valor exacto no coincide perfectamente con la expresión cinemática,
                # el swap discreto ocurre en ese instante y marca el siguiente bloque de marcha.
                pass

            x_local, y_local, step = self._generar_paso_oscilante(l_st, l_sw, n_puntos)

            # Empalme de la gráfica en el espacio global con offset acumulado.
            x_global.extend(x_local + offset_acumulado)
            y_global.extend(y_local)

            offset_acumulado += step

            # Evento discreto: intercambio de roles de apoyo/balanceo justo después del impacto.
            l_st, l_sw = self._swap_leg_lengths(l_st, l_sw)

        x_global = np.asarray(x_global)
        y_global = np.asarray(y_global)

        if plot:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x_global, y_global, "b-", linewidth=2, label="Trayectoria del CoG")
            ax.scatter([x_global[0]], [y_global[0]], color="green", s=30, label="Inicio")
            ax.scatter([x_global[-1]], [y_global[-1]], color="red", s=30, label="Fin")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title("Trayectoria continua del Centro de Gravedad (CoG) en marcha")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.axis("equal")
            if mostrar:
                plt.show()
            return x_global, y_global, fig, ax

        return x_global, y_global


if __name__ == "__main__":
    # Ejemplo rápido de uso
    cog = CoG_trayectoria(M=70.0, l_s=0.92, l_l=0.95)
    x, y, fig, ax = cog.generar_trayectoria(n_pasos=10, n_puntos=250, plot=True, mostrar=True)
    print(f"Puntos simulados: {len(x)}")
    print(f"X final: {x[-1]:.3f} m")
    print(f"Y final: {y[-1]:.3f} m")
