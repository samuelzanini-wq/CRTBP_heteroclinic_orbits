import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

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

def avaliar_vx_no_corte(theta, mu, r=1e-4):
    """
    Retorna o vetor de estado completo [x, y, vx, vy] no primeiro corte y = 0.
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


def encontrar_orbitas_heteroclinicas(mu, n_amostras=800, r=1e-4):
    """
    Encontra as órbitas 1-heteroclínicas filtrando tangências e duplicatas
    diretamente no plano de Poincaré (x, vy), conforme Gómez et al. (1988).
    """
    print(f"\nSearching for 1-Heteroclinic orbits")
    # endpoint=False evita duplicar 0 e 2*pi
    thetas_grid = np.linspace(0, 2 * np.pi, n_amostras, endpoint=False)
    
    estados_corte = np.array([avaliar_vx_no_corte(t, mu, r) for t in thetas_grid])
    vxs = estados_corte[:, 2]

    thetas_heteroclinicos = []
    pontos_poincare_unicos = []  # Armazena pares (x, vy) já aceitos

    for i in range(len(thetas_grid) - 1):
        v1, v2 = vxs[i], vxs[i+1]
        t1, t2 = thetas_grid[i], thetas_grid[i+1]

        if not np.isnan(v1) and not np.isnan(v2) and (v1 * v2 < 0) and abs(v1 - v2) < 1.5:
            try:
                raiz_theta = brentq(lambda t: avaliar_vx_no_corte(t, mu, r)[2], t1, t2, xtol=1e-11)
                estado_raiz = avaliar_vx_no_corte(raiz_theta, mu, r)
                
                x_corte, _, vx_res, vy_corte = estado_raiz

                # 1. Critério de Transversalidade de Gómez: descarta raspagens (vy ~ 0)
                if np.isnan(vx_res) or abs(vx_res) > 1e-9 or abs(vy_corte) < 0.05:
                    continue

                # 2. Unicidade no espaço de fase (x, vy): evita órbitas sobrepostas
                ponto_atual = np.array([x_corte, vy_corte])
                ja_existe = any(np.linalg.norm(ponto_atual - p_exist) < 1e-3 for p_exist in pontos_poincare_unicos)

                if not ja_existe:
                    thetas_heteroclinicos.append(raiz_theta)
                    pontos_poincare_unicos.append(ponto_atual)

            except Exception:
                continue

    thetas_heteroclinicos.sort()
    
    print(f"=== {len(thetas_heteroclinicos)} 1-Heteroclinic orbits found ===")
    for idx, t_h in enumerate(thetas_heteroclinicos):
        est = avaliar_vx_no_corte(t_h, mu, r)
        print(f"Orbit {idx+1}: theta = {t_h:.6f} rad | x = {est[0]:.6f} | vy = {est[3]:.6f} | res Vx = {est[2]:.2e}")

    return thetas_heteroclinicos