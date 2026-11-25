import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import funções as fu 

with open('parametros.txt', 'r', encoding='utf-8') as f:
    params = {}
    lendo_parametros = False
    for linha in f:
        linha = linha.strip()
        if "PARÂMETROS:" in linha:
            lendo_parametros = True
            continue
            
        if lendo_parametros and '=' in linha:
            chave, valor = linha.split('=')
            chave = chave.strip()
            valor = valor.strip()
            
            if chave == 'n_amostras':
                params[chave] = int(valor)
            else:
                params[chave] = float(valor)

mu = params['mu']
n_amostras = params['n_amostras']
r = params['r']

def gerar_secao_poincare_gomez(mu, n_amostras=500, r=1e-4):
    """
    Implementa o método de Gómez et al. (1988):
    1. Define condições iniciais em torno de L4 parametrizadas por um círculo de raio r e theta.
    2. Utiliza o integrador DOP853 (RK78) para propagar as trajetórias.
    3. Registra os pontos do primeiro corte na seção de Poincaré (y = 0) no plano (x, Vx).
    """
    
    # Conjunto de valores de theta cobrindo todo o círculo de parametrização local
    thetas = np.linspace(0, 2 * np.pi, n_amostras)
    
    x_cortes = []
    vx_cortes = []

    print(f"Iniciando integração")

    for theta in thetas:
        # 1. Obtenção da condição inicial da variedade instável W^u(L4) com raio r = 10^-4
        condicao_inicial = fu.ci_WuL4(theta, mu, r)

        # 2. Definição do evento de parada no primeiro cruzamento transversal com o eixo x (y = 0)
        def evento_y0(t, estado, mu_param):
            if t < 0.1:  # Ignora o instante inicial na vizinhança de L4
                return 1.0
            return estado[1]
        
        evento_y0.terminal = True
        evento_y0.direction = 0  # Detecta cruzamentos em ambas as direções

        # 3. Integração numérica utilizando o integrador DOP853 (RK78 do SciPy)
        sol = solve_ivp(
            fun=fu.f,
            t_span=[0, 60],
            y0=condicao_inicial,
            method='DOP853',  # Método Runge-Kutta de alta ordem (RK78)
            events=evento_y0,
            args=(mu,),
            rtol=1e-12,       # Tolerâncias estritas conforme o artigo de Gómez et al.
            atol=1e-12
        )

        # 4. Se a trajetória interceptou o eixo y = 0, armazena x e Vx
        if len(sol.y_events[0]) > 0:
            estado_corte = sol.y_events[0][0]
            x_corte = estado_corte[0]   # Posição x no corte
            vx_corte = estado_corte[2]  # Velocidade Vx no corte
            
            x_cortes.append(x_corte)
            vx_cortes.append(vx_corte)

    return np.array(x_cortes), np.array(vx_cortes)

def plot_poincare(x, Vx):
    plt.plot(-x, Vx, linestyle='', marker='.', color='black', markersize='3')
    plt.axhline(0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('x (Normalized)', fontsize=11)
    plt.ylabel(r'$\dot{x}$ (Normalized)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    #plt.legend()
    #plt.tight_layout()

# Cálculo e Plotagem da Seção de Poincaré L4 -> L5
x, Vx = gerar_secao_poincare_gomez(1 - mu, n_amostras, r)
plot_poincare(-x, Vx)
plt.title(r'Poincaré Section $\mu$ =' + f'{mu}\nL4 to L5', fontsize=12)
plt.show()
plt.close()

# Cálculo e Plotagem da Seção de Poincaré L5 -> L4
x, Vx = gerar_secao_poincare_gomez(mu, n_amostras, r)
plot_poincare(x, Vx)
plt.title(r'Poincaré Section $\mu$ =' + f'{mu}\nL5 to L4', fontsize=12)
plt.show()
plt.close()