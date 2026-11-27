# Implementação em python do algoritimo Particle Swarm Optimization (versão simples).

import numpy as np

def PSO(custo, particulas, max_iter=500):
    num_particulas = len(particulas)
    dimensao = len(particulas[0])

    # Garante que cada partícula seja um array NumPy de tipo float
    particulas = [np.array(p, dtype=float) for p in particulas]

    # Inicialização das velocidades
    velocidades = [np.zeros(dimensao) for _ in range(num_particulas)]

    # Avaliação Inicial
    avaliacao = [custo(p) for p in particulas]

    # Melhores posições individuais (pBest) e seus custos
    pbest_posicao = [np.copy(p) for p in particulas]
    pbest_custo = avaliacao[:]

    # Melhor posição global (gBest)
    custo_global = min(avaliacao)
    gbest_posicao = np.copy(particulas[avaliacao.index(custo_global)])

    # Parâmetros

    # As componentes sociail e cognitiva tem o mesmo peso para cada partícula.
    c1 = 2.0
    c2 = 2.0

    # Aqui, a componente de inércia vai diminuindo conforme vai passando o tempo. Isso é interessante pois simula o comportamento dos pássaros quando estão se aproximando da caça.
    w = 0.9
    w_min = 0.4
    iteracao = 0

    # Loop principal (apenas com critério de iteração)
    while iteracao < max_iter:
        iteracao += 1
        w = max(w_min, w - 0.001)

        for i in range(num_particulas):
            r1 = np.random.rand(dimensao)
            r2 = np.random.rand(dimensao)

            # Atualização da velocidade
            velocidades[i] = (w * velocidades[i] + 
                            c1 * r1 * (pbest_posicao[i] - particulas[i]) + 
                            c2 * r2 * (gbest_posicao - particulas[i]))
            
            # Atualização da posição
            particulas[i] = particulas[i] + velocidades[i]

            # Avaliação da nova posição
            nova_avaliacao = custo(particulas[i])

            # Atualização do pBest
            if nova_avaliacao < pbest_custo[i]:
                pbest_custo[i] = nova_avaliacao
                pbest_posicao[i] = np.copy(particulas[i])

                # Atualização do gBest
                if nova_avaliacao < custo_global:
                    custo_global = nova_avaliacao
                    gbest_posicao = np.copy(particulas[i])

    print(f'\nMelhor solução encontrada (gBest): {gbest_posicao}')
    print(f'Custo: {custo_global}')

    return gbest_posicao, custo_global

# Referências:

# https://youtu.be/JhgDMAm-imI?si=E7genQ3uqCxivG_a
# https://youtu.be/QvHl7qzbsU8?si=cg9josMchZKt-22n

# Caso queira aprofundar no comportamento dos pássaros: (pode ser útil caso queira fazer simulações mais complexas ou com comportamento mais parecido com dos animais)
# https://youtu.be/n8JeaBnV5EQ?si=Si6G7CUPFXofueN9