import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import funções as fu 

# Lendo os parêmetros
with open('parametros.txt', 'r', encoding='utf-8') as f:
    params = {}
    lendo_parametros = False
    for linha in f:
        linha = linha.strip()
        if "PARÂMETROS:" in linha or "PARAMETROS:" in linha:
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

# Função que encontra órbitas heteroclínicas pelo método do Gómez
def gomez_heteroclinica(mu, n_amostras, r):
    # Identificação dos ângulos das Órbitas Heteroclínicas (onde Vx = 0)
    thetas_h = fu.encontrar_orbitas_heteroclinicas(mu, n_amostras, r)

    if len(thetas_h) > 0:
        
        orbitas_plotadas = 0
        orbitas = []

        for i, theta_val in enumerate(thetas_h):
            # Avalia o residual de Vx no corte y = 0 para este theta
            vx_res = fu.avaliar_vx_no_corte(theta_val, mu, r)

            # Precisão:
            # Filtra apenas as órbitas cujo módulo do residual de Vx seja menor que um erro


            # Condição inicial usando o theta refinado
            estado_inicial = fu.ci_WuL4(theta_val, mu, r)

            # Evento de corte ortogonal em y = 0
            def evento_y0(t, estado, mu_param):
                if t < 0.5:
                    return 1.0
                return estado[1]
            
            evento_y0.terminal = True
            evento_y0.direction = 0

            # Integração da primeira metade (L4 até o corte y = 0)
            sol_metade1 = solve_ivp(
                fun=fu.f,
                t_span=[0, 50],
                y0=estado_inicial,
                method='DOP853',
                events=evento_y0,
                args=(mu,),
                rtol=1e-12,
                atol=1e-12
            )

            if len(sol_metade1.y_events[0]) > 0:
                x1, y1 = sol_metade1.y[0], sol_metade1.y[1]
                
                # Simetria Reversível R1 para construir a segunda metade (até L5)
                x2 = x1[::-1]
                y2 = -y1[::-1]

                x_completo = np.concatenate((x1, x2))
                y_completo = np.concatenate((y1, y2))

                if np.any(x_completo <= -1):
                    continue
                elif mu < 0.5:
                    orbitas.append([-x_completo, y_completo])
                    orbitas_plotadas += 1
                else:
                    orbitas.append([x_completo, y_completo])
                    orbitas_plotadas += 1
    else:
        print("Nenhuma órbita heteroclínica foi encontrada.")

    return orbitas, orbitas_plotadas, thetas_h

def plot_orbitas(orbitas, orbitas_plotadas, theta_h):

    print(f"\nFiltrando e plotando órbita(s)")
            
    plt.figure(figsize=(9, 8))
    #cores_arco_iris = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
    cores_arco_iris = ['blue', 'indigo', 'violet', 'green', 'yellow', 'orange', 'red']
            
    for i, eixo in enumerate(orbitas):
        plt.plot(eixo[0], eixo[1], color=cores_arco_iris[i], linewidth=1.5,  alpha=1.0)

    # Elementos do CRTBP

    # Mass m1
    plt.plot(mu, 0, 'ko', markersize=(1-mu)*10, label=r'$m_1$ Antiope (90)')
    plt.annotate(rf'$m_1 \approx {1- mu:.1f}$', (mu, 0), xytext=(10, 5), textcoords='offset points', fontsize=10, weight='bold',)

    # Mass m2
    plt.plot(-(1 - mu), 0, 'ko', markersize=mu*10, label=r'$m_2$ S/200 (90) 1')
    plt.annotate(rf'$m_2 \approx {mu:.1f}$', (-(1 - mu), 0), xytext=(10, 5), textcoords='offset points', fontsize=10, weight='bold',)
    
    # Ponto L4 
    pos_L4 = [-(0.5 - mu), np.sqrt(3) / 2]
    plt.plot(pos_L4[0], pos_L4[1], 'kx', markersize=9, markeredgewidth=2)
    plt.annotate('L4', (pos_L4[0], pos_L4[1]), xytext=(10, 5), textcoords='offset points', fontsize=10, weight='bold',)

    # Ponto L5
    pos_L5 = [-(0.5 - mu), -np.sqrt(3) / 2]
    plt.plot(pos_L5[0], pos_L5[1], 'kx', markersize=9, markeredgewidth=2)
    plt.annotate('L5', (pos_L5[0], pos_L5[1]), xytext=(10, 5), textcoords='offset points', fontsize=10, weight='bold',)
    
    # Configurações básicas
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('x-axis (Normalized)', fontsize=11)
    plt.ylabel('y-axis (Normalized)', fontsize=11)
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', framealpha=0.9, fontsize=9)
    plt.tight_layout()


orbitas, n_orbitas, thetas_h = gomez_heteroclinica((1 - mu), n_amostras, r)
orbitas2, n_orbitas2, thetas_h2 = gomez_heteroclinica(mu, n_amostras, r)

plot_orbitas(orbitas + orbitas2, n_orbitas + n_orbitas2, thetas_h + thetas_h2)
plt.title(r'1-Heteroclinic Orbits $\mu$ = ' + f'{mu}\nAntiope (90) - S/2000 (90) 1')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()