import funções as fu
import numpy as np
import python_pso as pso
import matplotlib.pyplot as plt

# Leitura dos parametros
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

# Ângulos do anel de condições iniciais (endpoint=False evita duplicar 0 e 2pi)
thetas = np.linspace(0, 2*np.pi, n_amostras, endpoint=False)

# Wrapper da função custo compatível com python_pso.py (recebe array [theta] e retorna escalar |vx|)
def criar_custo_pso(mu_val, r_val, t_min, t_max):
    def custo_pso(theta_arr):
        theta = float(theta_arr[0])
        
        # Mantém a partícula dentro do subintervalo onde ocorreu a troca de sinal
        if theta < t_min or theta > t_max:
            return 1e6
            
        estado = fu.custo(theta, mu_val, r_val)
        x_corte, _, vx_corte, vy_corte = estado
        
        # Penaliza falhas de corte, proximidade das massas ou tangências (vy ~ 0)
        if np.isnan(vx_corte) or abs(vy_corte) < 0.05:
            return 1e6
            
        return abs(vx_corte)
    return custo_pso


# Função que calcula as órbitas heteroclínicas via Seção de Poincaré + PSO local
def heteroclinicas(thetas, mu, r):
    print(f"Mapeando Seção de Poincaré")

    # Estado de cada condição inicial no corte y = 0: [x, y, vx, vy]
    estados = np.array([fu.custo(theta, mu, r) for theta in thetas])
    xs = estados[:, 0]
    vxs = estados[:, 2]
    vys = estados[:, 3]

    # Posições das massas primárias
    x_m1 = -mu
    x_m2 = 1 - mu
    margem_massa = 0.005

    indices = []
    for j in range(len(thetas) - 1):
        x1, x2 = xs[j], xs[j+1]
        vx1, vx2 = vxs[j], vxs[j+1]
        vy1, vy2 = vys[j], vys[j+1]

        # Ignora pontos sem cruzamento
        if np.isnan(vx1) or np.isnan(vx2):
            continue

        # Exclui cruzamentos muito próximos às posições das massas primárias (-mu e 1-mu)
        perto_massa1 = (abs(x1 - x_m1) < margem_massa) or (abs(x2 - x_m1) < margem_massa)
        perto_massa2 = (abs(x1 - x_m2) < margem_massa) or (abs(x2 - x_m2) < margem_massa)
        if perto_massa1 or perto_massa2:
            continue

        # Exclui tangências/raspagens no eixo x (vy ~ 0)
        if abs(vy1) < 0.05 or abs(vy2) < 0.05:
            continue

        # Verifica troca de sinal em vx
        if (vx1 * vx2 <= 0):
            indices.append(j)

    print(f"Intervalos candidatos detectados: {len(indices)}\n")

    thetas_encontrados = []
    pontos_poincare_unicos = []

    # PSO simples aplicado a cada intervalo candidato para refinar theta
    for idx in indices:
        t_esq = thetas[idx]
        t_dir = thetas[idx+1]
        
        novos_thetas = np.linspace(t_esq, t_dir, 10)
        particulas = [[t] for t in novos_thetas]
        
        custo_fn = criar_custo_pso(mu, r, t_esq, t_dir)
        theta_best, vx_residual = pso.PSO(custo_fn, particulas, max_iter=80)
        
        theta_sol = float(theta_best[0])
        estado_final = fu.custo(theta_sol, mu, r)
        x_f, _, vx_f, vy_f = estado_final

        # Validação final e filtro de unicidade no plano de Poincaré (x, vy)
        if not np.isnan(vx_f) and abs(vx_f) < 1e-4 and abs(vy_f) >= 0.05:
            ponto_atual = np.array([x_f, vy_f])
            if not any(np.linalg.norm(ponto_atual - p_exist) < 1e-3 for p_exist in pontos_poincare_unicos):
                thetas_encontrados.append(theta_sol)
                pontos_poincare_unicos.append(ponto_atual)

    thetas_encontrados.sort()

    print(f"\n=== Órbitas Heteroclínicas Encontradas: {len(thetas_encontrados)} ===")
    for k, t_h in enumerate(thetas_encontrados):
        est = fu.custo(t_h, mu, r)
        print(f"Órbita {k+1}: theta = {t_h:.6f} rad ({np.degrees(t_h):.2f}°) | x = {est[0]:.6f} | vx res = {est[2]:.2e}")

    return thetas_encontrados

if __name__ == "__main__":
    thetas_het_l5 = heteroclinicas(thetas, mu, r)
    thetas_het_l4 = heteroclinicas(thetas, 1-mu, r)

    plt.figure(figsize=(9, 8))

    fu.plotar_orbitas(thetas_het_l5, mu, r)
    fu.plotar_orbitas(thetas_het_l4, 1-mu, r)

    # Elementos do CRTBP
    # Mass m1
    plt.plot(mu, 0, 'ko', markersize=(1-mu)*10, label=r'$m_1$ Antiope (90)')
    plt.annotate(rf'$m_1 \approx {1- mu:.1f}$', (mu, 0), xytext=(10, 5), textcoords='offset points', fontsize=10, weight='bold',)
       
    # Mass m2
    plt.plot(mu-1, 0, 'ko', markersize=mu*10, label=r'$m_2$ S/200 (90) 1')
    plt.annotate(rf'$m_2 \approx {mu:.1f}$', (-(1 - mu), 0), xytext=(10, 5), textcoords='offset points', fontsize=10, weight='bold',)
    
    # Ponto L4 
    pos_L4 = [-(0.5 - mu), np.sqrt(3) / 2]
    plt.plot(pos_L4[0], pos_L4[1], 'kx', markersize=9, markeredgewidth=2)
    plt.annotate('L4', (pos_L4[0], pos_L4[1]), xytext=(10, 5), textcoords='offset points', fontsize=10, weight='bold',)
        
    # Ponto L5
    pos_L5 = [-(0.5 - mu), -np.sqrt(3) / 2]
    plt.plot(pos_L5[0], pos_L5[1], 'kx', markersize=9, markeredgewidth=2)
    plt.annotate('L5', (pos_L5[0], pos_L5[1]), xytext=(10, 5), textcoords='offset points', fontsize=10, weight='bold',)

    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title(fr'Órbitas 1-Heteroclínicas no CRTBP ($\mu = {mu}$)', fontsize=12)
    plt.xlabel('x (Sistema Sinódico)', fontsize=11)
    plt.ylabel('y (Sistema Sinódico)', fontsize=11)
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()