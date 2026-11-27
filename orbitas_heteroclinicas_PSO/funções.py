import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Função que retorna as derivadas de um vetor de estado usando as equações diferencias do CRTBP
def f(t, estado, mu):
    x, y, vx, vy = estado

    r1_3 = ((x + mu)**2 + y**2)**1.5
    r2_3 = ((x + (mu - 1))**2 + y**2)**1.5

    dvxdt = 2*vy + x - ((1 - mu)*(x + mu) / r1_3) - (mu*(x + mu - 1) / r2_3)
    dvydt = -2*vx + y - (y*(1 - mu) / r1_3) - (mu*y / r2_3)

    return np.array([vx, vy, dvxdt, dvydt])


# 2. Condições iniciais ao redor de L4 (Gómez et al., 1988)
def ci_WuL4(theta, mu, r):
    # Autovalores e parâmetros lineares em L4
    lambda2 = 1.5 * (1 - np.sqrt(1 - 3*mu*(1 - mu)))
    abs_lambda = (27 * mu * (1 - mu))**0.25 / np.sqrt(2)

    alpha = np.sqrt(np.sqrt(27 * mu * (1 - mu)) - 1) / 2
    beta = np.sqrt(np.sqrt(27 * mu * (1 - mu)) + 1) / 2

    alpha_linha = alpha * (abs_lambda**2 - lambda2) / (2 * abs_lambda**2)
    beta_linha = beta * (abs_lambda**2 + lambda2) / (2 * abs_lambda**2)

    # Parametrização circular no plano da variedade instável
    a1 = r * np.cos(theta)
    a2 = (r * np.sin(theta) - r * alpha_linha * np.cos(theta)) / beta_linha

    # Estado local no sistema centrado em L4 e rotacionado por alpha_rot
    x0 = a1
    y0 = a1 * alpha_linha + a2 * beta_linha
    vx0 = a1 * alpha + a2 * beta
    vy0 = a1 * (alpha_linha * alpha - beta * beta_linha) + a2 * (alpha * beta_linha + alpha_linha * beta)

    condicao_local = np.array([x0, y0, vx0, vy0])

    # Ângulo de rotação para alinhar com os eixos locais de L4
    alpha_rot = 0.5 * np.arctan(np.sqrt(3) * (1 - 2*mu))

    cos_a = np.cos(alpha_rot)
    sen_a = np.sin(alpha_rot)

    # Matriz de rotação
    M = np.array([
        [cos_a, -sen_a],
        [sen_a,  cos_a]
    ])

    # Transforma posição e velocidade de volta para o sistema do baricentro
    pos_baricentro = M @ condicao_local[0:2]
    vel_baricentro = M @ condicao_local[2:4]

    # Translação do ponto L4: (0.5 - mu, sqrt(3)/2)
    pos_L4 = np.array([0.5 - mu, np.sqrt(3) / 2])
    pos_baricentro += pos_L4

    return np.concatenate((pos_baricentro, vel_baricentro))

def custo(theta, mu, r=1e-4):
    """
    Retorna o estado [x, y, vx, vy] no corte y = 0.
    """
    theta = float(theta) % (2 * np.pi)
    condicao_inicial = ci_WuL4(theta, mu, r)

    def evento_y0(t, estado, mu_param):
        if t < 0.1:
            return 1.0
        return estado[1]
    
    evento_y0.terminal = True
    evento_y0.direction = 0

    sol = solve_ivp(
        fun=f,
        t_span=[0, 50],
        y0=condicao_inicial,
        method='DOP853',
        events=evento_y0,
        args=(mu,),
        rtol=1e-12,
        atol=1e-12
    )

    if len(sol.y_events[0]) > 0:
        return sol.y_events[0][0]
    return np.array([np.nan, np.nan, np.nan, np.nan])

def plotar_orbitas(thetas, mu, r=1e-4, t_max=50):
    """
    Integra e plota as órbitas 1-heteroclínicas a partir dos ângulos theta encontrados,
    utilizando a simetria reversível R1 do CRTBP.
    """
    if len(thetas) == 0:
        print("Nenhuma órbita heteroclínica para plotar.")
        return
    
    cores = ['blue', 'crimson', 'darkgreen', 'purple', 'orange', 'brown', 'teal']

    for i, theta in enumerate(thetas):
        condicao_inicial = ci_WuL4(theta, mu, r)

        # Evento para interromper a integração no corte ortogonal y = 0
        def evento_y0(t, estado, mu_param):
            if t < 0.1:
                return 1.0
            return estado[1]
        
        evento_y0.terminal = True
        evento_y0.direction = 0

        sol = solve_ivp(
            fun=f,
            t_span=[0, t_max],
            y0=condicao_inicial,
            method='DOP853',
            events=evento_y0,
            args=(mu,),
            rtol=1e-12,
            atol=1e-12
        )

        if len(sol.y_events[0]) > 0:
            x1, y1 = sol.y[0], sol.y[1]
            
            # Simetria reversível R1: (x, y, vx, vy) -> (x, -y, -vx, vy)
            x2 = x1[::-1]
            y2 = -y1[::-1]

            x_completo = np.concatenate((x1, x2))
            y_completo = np.concatenate((y1, y2))
            cor = cores[i % len(cores)]

            if mu < 0.5:
                plt.plot(-x_completo, y_completo, color=cor, linewidth=1.5)
            else:
                plt.plot(x_completo, y_completo, color=cor, linewidth=1.5)