# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 21:25:33 2023

@author: GIDEAO OLIVEIRA  DOS SANTOS
"""

########################### BIBLIOTECAS #####################################
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

############################# FUNCOES #######################################

#F1 - Entrada:[dominio, funcoes de pertinencia de cada nivel(3),lista valores]
#     Saida: [Graus de pertinecias de cada elemento do dominio segundo funcao]
def graus_pertinencia(x, y_baixo, y_mediano, y_alto, lista_notas):  
    x_area = x
    y_area_baixo = y_baixo
    y_area_mediano = y_mediano
    y_area_alto = y_alto
    A = lista_notas
    n = len(A)
    PB = np.zeros(n)
    PM = np.zeros(n)
    PA = np.zeros(n)
    for i in range(n):
           PB[i] = fuzz.interp_membership(x_area,y_area_baixo, A[i])
           PM[i] = fuzz.interp_membership(x_area,y_area_mediano, A[i])
           PA[i] = fuzz.interp_membership(x_area,y_area_alto, A[i])
    return PB, PM, PA


#F2 - Entrada: [dominio, ativacoes(3 niveis), metodo]
#     Saida: valor defuzyficado, controle

def defuzzyfica(x, ativacao_baixo, ativacao_mediano, ativacao_alto, metodo):
    x_saida = x
    alto = ativacao_alto
    mediano = ativacao_mediano
    baixo = ativacao_baixo
    metodo = metodo
    controle = np.fmax(baixo, np.fmax(mediano,alto))
    performace = fuzz.defuzz(x_saida, controle, metodo)
    return performace, controle

#F3 - Entrada: [nome, dominio, funcoes de pertinencia (3 niveis), valor]
#     Saida: [grafico] 
def plota_grafico_fuzzy(nome, x, y_baixo, y_mediano, y_alto, nota):
    nome = nome 
    nota = nota
    x_area = x
    x_area0 = np.zeros_like(x_area)
    y_area_baixo = y_baixo   
    area_baixo = fuzz.interp_membership(x_area,y_area_baixo, nota)
    pert_baixo = np.fmin(area_baixo,y_area_baixo)  
    y_area_mediano = y_mediano
    area_mediano = fuzz.interp_membership(x_area,y_area_mediano, nota)
    pert_mediano = np.fmin(area_mediano,y_area_mediano)   
    y_area_alto = y_alto
    area_alto = fuzz.interp_membership(x_area,y_area_alto, nota)
    pert_alto = np.fmin(area_alto,y_area_alto)   
    fig, ax = plt.subplots(figsize = (8,5))
    ax.fill_between(x_area, x_area0, pert_baixo, facecolor = 'r')
    ax.plot(x_area, y_area_baixo, 'r', label = 'Baixo')
    ax.fill_between(x_area, x_area0, pert_mediano, facecolor = 'y') 
    ax.plot(x_area, y_area_mediano, 'y', label = 'Aceitável')
    ax.fill_between(x_area, x_area0, pert_alto, facecolor = 'g')
    ax.plot(x_area, y_area_alto, 'g', label = 'Superior')   
    ax.legend(loc ="center left")
    ax.set_title('Desempenho Profissional do ' + nome);
    plt.show()
    return plt


####################### DEFINICAO DO AMBIENTE ###############################

areas = ["Combate a Incêndio", "Salvamento aquático", "Salvamento em altura", 
         "Salvamento terrestre", "Atendimento pré-hospitalar", "Doutrina"]
provas = ['Prática', 'Teórica', 'Autoavaliacão', 'Lateral']
competencias = ['Conformidade','Eficiência', 'Perícia', 'Sinestesia', 'Vigor']

n_a = len(areas)
n_p = len(provas)
n_c = len(competencias)


############################## MATRIZES DO SISTEMA ##########################

matriz_notas = np.zeros((n_a,n_p))  #M1 - Matriz de notas
matriz_contribuicao = np.zeros((n_p,1)) #M2 - Matriz dos pesos


######################### DADOS DE ENTRADA ############################### 

nome = "Soldado A"

# Nota das disiciplinas: [teórica, prática]  
nota_ci   = [2.0, 3.0]
nota_saq  = [1.0, 10.0]
nota_salt = [5.0, 7.0]
nota_sat  = [7.0, 8.0]
nota_aph  = [10.0, 7.0]
nota_dou  = [10.0, 8.0]

#nota da autoavaliacao das competenicias nas disicplinas: [conformindade, 
#                                        eficiencia, pericia,sinestesia, vigor]

auto_comp_ci   = [5.0, 3.0, 4.0, 8.0, 7.0]
auto_comp_saq  = [5.0, 3.0, 4.0, 8.0, 7.0]
auto_comp_salt = [5.0, 3.0, 4.0, 8.0, 7.0]
auto_comp_sat  = [5.0, 3.0, 4.0, 8.0, 7.0]
auto_comp_aph  = [5.0, 3.0, 4.0, 8.0, 7.0]
auto_comp_dou  = [5.0, 3.0, 4.0, 8.0, 7.0]

#nota avaliacao lateral das competenicias nas disicplinas: [conformindade, 
#                                      eficiencia, pericia, sinestesia, vigor]

lat_comp_ci   = [5.0, 3.0, 4.0, 8.0, 7.0]
lat_comp_saq  = [5.0, 3.0, 4.0, 8.0, 7.0]
lat_comp_salt = [5.0, 3.0, 4.0, 8.0, 7.0]
lat_comp_sat  = [5.0, 3.0, 4.0, 8.0, 7.0]
lat_comp_aph  = [5.0, 3.0, 4.0, 8.0, 7.0]
lat_comp_dou  = [5.0, 3.0, 4.0, 8.0, 7.0]

###############################################################################
################### MONTADO A MATRIZ DE CONTRIBUIÇÃO ##########################

# dados obtido pelo questionario do grupo gestor
pratica = 0.24 
teorica = 0.33
auto = 0.20 
lateral = 0.23

matriz_contribuicao[0,0] = pratica
matriz_contribuicao[1,0] = teorica
matriz_contribuicao[2,0] = auto
matriz_contribuicao[3,0] = lateral


################## MOTANDO AS FUNCOES DE PERTINENCIAS ########################

# funcao para nota das areas: avaliaçoes difisas (auto e lateral)
#  (funcao triangular adotada por conveniencia)

x = np.arange(0,11,1) # limites das notas (0 a 10)

y_baixa = fuzz.trimf(x, [0,0,5.0]) # função triangula_ pertinecia_baixa
y_media = fuzz.trimf(x, [0,5.0,10.0]) # função triangula_ pertinencia_media
y_alta = fuzz.trimf(x, [5,10,10]) # função triangula_ pertinencia_alta

# funcao para o desempenho profissional (score)
# (funcao trapezoidal adota por melhor ajustes  obtido por dados do grupo
#                                                             gestor)

xt = np.arange(0,1001,100) # eixo da performace (0 a 1000)

yt_baixo = fuzz.trapmf(xt, [0,0,100, 600])
yt_aceitavel = fuzz.trapmf(xt,[400.0, 600.0, 700.0, 900.0])
yt_superior = fuzz.trapmf(xt,[700,900,1000,1001])


# metodo de defuzzyficacao

metodo = 'bisector'   #adotado por melhor ajuste
#metodo = 'centroid'
#metodo='som'
#metodo='mom'


########################### OPERACAO DO SISTEMA ##############################

# preenchendo a matreiz dads notas das areas

matriz_notas[0,:2] = nota_ci
matriz_notas[1,:2] = nota_saq
matriz_notas[2,:2] = nota_salt
matriz_notas[3,:2] = nota_sat
matriz_notas[4,:2] = nota_aph
matriz_notas[5,:2] = nota_dou


def main():
   
############################# SISTEMA FUZZY 1  ###############################

# Antecedentes:
#      competencias (valiaveis lingisticas: baixa, mediana, alta)
#    
# Consequente:
#      nota das avaliacoes (valiaveis linguistica: baixa, media alta)
#
# Regras logias:
#       Regra1: Se as competências forem baixas, então nota será baixa    
#       Regra2: Se as competencias forem medianas, então nota será media   
#       Regra3: Se as competencias forem altas, então nota será alta 

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# acha os graus de pretinencia das competencias da autoavaliação em cada area
    
    CI_au_B, CI_au_M, CI_au_A = graus_pertinencia(
                        x, y_baixa, y_media, y_alta, auto_comp_ci)
    SAQ_au_B, SAQ_au_M, SAQ_au_A = graus_pertinencia(
                        x, y_baixa, y_media, y_alta, auto_comp_saq)
    SALT_au_B, SALT_au_M, SALT_au_A = graus_pertinencia(
                        x, y_baixa, y_media, y_alta,auto_comp_salt)     
    SAT_au_B, SAT_au_M, SAT_au_A = graus_pertinencia(
                        x, y_baixa, y_media, y_alta, auto_comp_sat)   
    APH_au_B, APH_au_M, APH_au_A = graus_pertinencia(
                        x, y_baixa, y_media, y_alta, auto_comp_aph)
    DOU_au_B, DOU_au_M, DOU_au_A = graus_pertinencia(
                        x, y_baixa, y_media, y_alta, auto_comp_dou)  

# acha graus de pretinencia das competencias da avaliação lateral em cada area

    CI_l_B, CI_l_M, CI_l_A = graus_pertinencia(
                        x, y_baixa, y_media, y_alta, lat_comp_ci)
    SAQ_l_B, SAQ_l_M, SAQ_l_A = graus_pertinencia(
                        x, y_baixa, y_media, y_alta,lat_comp_saq)
    SALT_l_B, SALT_l_M, SALT_l_A = graus_pertinencia(
                        x, y_baixa, y_media, y_alta, lat_comp_salt)  
    SAT_l_B, SAT_l_M, SAT_l_A = graus_pertinencia(
                        x, y_baixa, y_media, y_alta,lat_comp_sat)   
    APH_l_B, APH_l_M, APH_l_A = graus_pertinencia(
                        x, y_baixa, y_media, y_alta,lat_comp_aph)
    DOU_l_B, DOU_l_M, DOU_l_A = graus_pertinencia(
                        x, y_baixa, y_media, y_alta,lat_comp_dou)


# ativando as regras para auto avaliação 
 
#[CI]
    ativacao0_regra1_ci = np.max(CI_au_B)
    ativacao0_regra2_ci = np.max(CI_au_M)
    ativacao0_regra3_ci = np.max(CI_au_A)

#[SAQ]
    ativacao0_regra1_saq = np.max(SAQ_au_B)
    ativacao0_regra2_saq = np.max(SAQ_au_M)
    ativacao0_regra3_saq = np.max(SAQ_au_A)

#[SALT]
    ativacao0_regra1_salt = np.max(SALT_au_B)
    ativacao0_regra2_salt = np.max(SALT_au_M)
    ativacao0_regra3_salt = np.max(SALT_au_A)

#[SAT]
    ativacao0_regra1_sat = np.max(SAT_au_B)
    ativacao0_regra2_sat = np.max(SAT_au_M)
    ativacao0_regra3_sat = np.max(SAT_au_A)

#[APH]
    ativacao0_regra1_aph = np.max(APH_au_B)
    ativacao0_regra2_aph = np.max(APH_au_M)
    ativacao0_regra3_aph = np.max(APH_au_A)

#[DOU]
    ativacao0_regra1_dou = np.max(DOU_au_B)
    ativacao0_regra2_dou = np.max(DOU_au_M)
    ativacao0_regra3_dou = np.max(DOU_au_A)


# ativando as regras na avaliação lateral

#[CI]
    ativacao1_regra1_ci = np.max(CI_l_B)
    ativacao1_regra2_ci = np.max(CI_l_M)
    ativacao1_regra3_ci = np.max(CI_l_A)

#[SAQ]
    ativacao1_regra1_saq = np.max(SAQ_l_B)
    ativacao1_regra2_saq = np.max(SAQ_l_M)
    ativacao1_regra3_saq = np.max(SAQ_l_A)

#[SALT]
    ativacao1_regra1_salt = np.max(SALT_l_B)
    ativacao1_regra2_salt = np.max(SALT_l_M)
    ativacao1_regra3_salt = np.max(SALT_l_A)

#[SAT]
    ativacao1_regra1_sat = np.max(SAT_l_B)
    ativacao1_regra2_sat = np.max(SAT_l_M)
    ativacao1_regra3_sat = np.max(SAT_l_A)

#[APH]
    ativacao1_regra1_aph = np.max(APH_l_B)
    ativacao1_regra2_aph = np.max(APH_l_M)
    ativacao1_regra3_aph = np.max(APH_l_A)

#[DOU]
    ativacao1_regra1_dou = np.max(DOU_l_B)
    ativacao1_regra2_dou = np.max(DOU_l_M)
    ativacao1_regra3_dou = np.max(DOU_l_A)


# ativando notas para autoavaliação

#[CI]
    ativacao0_nota_ci_baixo = np.fmin(ativacao0_regra1_ci,y_baixa)
    ativacao0_nota_ci_mediano = np.fmin(ativacao0_regra2_ci, y_media)
    ativacao0_nota_ci_alto = np.fmin(ativacao0_regra3_ci, y_alta)
  
#[SAQ]
    ativacao0_nota_saq_baixo = np.fmin(ativacao0_regra1_saq,y_baixa)
    ativacao0_nota_saq_mediano = np.fmin(ativacao0_regra2_saq, y_media)
    ativacao0_nota_saq_alto = np.fmin(ativacao0_regra3_saq, y_alta)

#[SALT]
    ativacao0_nota_salt_baixo = np.fmin(ativacao0_regra1_salt,y_baixa)
    ativacao0_nota_salt_mediano = np.fmin(ativacao0_regra2_salt, y_media)
    ativacao0_nota_salt_alto = np.fmin(ativacao0_regra3_salt, y_alta)    
   
#[SAT]
    ativacao0_nota_sat_baixo = np.fmin(ativacao0_regra1_sat,y_baixa)
    ativacao0_nota_sat_mediano = np.fmin(ativacao0_regra2_sat, y_media)
    ativacao0_nota_sat_alto = np.fmin(ativacao0_regra3_sat, y_alta)

#[APH]
    ativacao0_nota_aph_baixo = np.fmin(ativacao0_regra1_aph,y_baixa)
    ativacao0_nota_aph_mediano = np.fmin(ativacao0_regra2_aph, y_media)
    ativacao0_nota_aph_alto = np.fmin(ativacao0_regra3_aph, y_alta)

#[DOU]
    ativacao0_nota_dou_baixo = np.fmin(ativacao0_regra1_dou,y_baixa)
    ativacao0_nota_dou_mediano = np.fmin(ativacao0_regra2_dou, y_media)
    ativacao0_nota_dou_alto = np.fmin(ativacao0_regra3_dou, y_alta)

# ativando notas para avaliação lateral

#[CI]
    ativacao1_nota_ci_baixo = np.fmin(ativacao1_regra1_ci,y_baixa)
    ativacao1_nota_ci_mediano = np.fmin(ativacao1_regra2_ci, y_media)
    ativacao1_nota_ci_alto = np.fmin(ativacao1_regra3_ci, y_alta)
  
#[SAQ]
    ativacao1_nota_saq_baixo = np.fmin(ativacao1_regra1_saq,y_baixa)
    ativacao1_nota_saq_mediano = np.fmin(ativacao1_regra2_saq, y_media)
    ativacao1_nota_saq_alto = np.fmin(ativacao1_regra3_saq, y_alta)

#[SALT]
    ativacao1_nota_salt_baixo = np.fmin(ativacao1_regra1_salt,y_baixa)
    ativacao1_nota_salt_mediano = np.fmin(ativacao1_regra2_salt, y_media)
    ativacao1_nota_salt_alto = np.fmin(ativacao1_regra3_salt, y_alta)    
   
#[SAT]
    ativacao1_nota_sat_baixo = np.fmin(ativacao1_regra1_sat,y_baixa)
    ativacao1_nota_sat_mediano = np.fmin(ativacao1_regra2_sat, y_media)
    ativacao1_nota_sat_alto = np.fmin(ativacao1_regra3_sat, y_alta)

#[APH]
    ativacao1_nota_aph_baixo = np.fmin(ativacao1_regra1_aph,y_baixa)
    ativacao1_nota_aph_mediano = np.fmin(ativacao1_regra2_aph, y_media)
    ativacao1_nota_aph_alto = np.fmin(ativacao1_regra3_aph, y_alta)

#[DOU]
    ativacao1_nota_dou_baixo = np.fmin(ativacao1_regra1_dou,y_baixa)
    ativacao1_nota_dou_mediano = np.fmin(ativacao1_regra2_dou, y_media)
    ativacao1_nota_dou_alto = np.fmin(ativacao1_regra3_dou, y_alta)

#&&&&&&& Defuzzyficando para encontrar o valor das avaliacoes &&&&&&&&&&&&&&&

# achando os valores exados das da autoavaliacao (0)

#[CI]
    nota0_ci, controle0_ci = defuzzyfica(
        x, ativacao0_nota_ci_baixo,ativacao0_nota_ci_mediano, 
        ativacao0_nota_ci_alto, metodo)
    matriz_notas[0,2] =  nota0_ci
    
#[SAQ]
    nota0_saq, controle0_saq = defuzzyfica(
        x, ativacao0_nota_saq_baixo, ativacao0_nota_saq_mediano, 
        ativacao0_nota_saq_alto, metodo) 
    matriz_notas[1,2] = nota0_saq  
    
#[SALT]
    nota0_salt, controle0_salt = defuzzyfica(
        x, ativacao0_nota_salt_baixo, ativacao0_nota_salt_mediano,
        ativacao0_nota_salt_alto, metodo) 
    matriz_notas[2,2] = nota0_salt

#[SAT]
    nota0_sat, controle0_sat = defuzzyfica(
        x, ativacao0_nota_sat_baixo, ativacao0_nota_sat_mediano,
        ativacao0_nota_sat_alto, metodo) 
    matriz_notas[3,2] = nota0_sat
    
#[APH]
    
    nota0_aph, controle0_aph = defuzzyfica(
        x, ativacao0_nota_aph_baixo, ativacao0_nota_aph_mediano,
        ativacao0_nota_aph_alto, metodo)  
    matriz_notas[4,2] = nota0_aph
    
#[DOU]
    nota0_dou, controle0_dou = defuzzyfica(
        x, ativacao0_nota_dou_baixo,ativacao0_nota_dou_mediano,
        ativacao0_nota_dou_alto, metodo) 
    matriz_notas[5,2] = nota0_dou
    
# achando o representante exato das avaliações lateral (1)
    
#[CI]
    nota1_ci, controle1_ci = defuzzyfica(
        x, ativacao1_nota_ci_baixo, ativacao1_nota_ci_mediano,
        ativacao1_nota_ci_alto, metodo)  
    matriz_notas[0,3] = nota1_ci

#[SAQ]
    nota1_saq, controle1_saq = defuzzyfica(
        x, ativacao1_nota_saq_baixo, ativacao1_nota_saq_mediano, 
        ativacao1_nota_saq_alto, metodo) 
    matriz_notas[1,3] = nota1_saq
    
#[SALT]
    nota1_salt, controle1_salt = defuzzyfica(
        x, ativacao1_nota_salt_baixo,ativacao1_nota_salt_mediano,
        ativacao1_nota_salt_alto, metodo) 
    matriz_notas[2,3] = nota1_salt
    
#[SAT]
    nota1_sat, controle1_sat = defuzzyfica(
        x, ativacao1_nota_sat_baixo, ativacao1_nota_sat_mediano, 
        ativacao1_nota_sat_alto, metodo) 
    matriz_notas[3,3] = nota1_sat
        
#[APH]
    nota1_aph, controle1_aph = defuzzyfica(
        x, ativacao1_nota_aph_baixo, ativacao1_nota_aph_mediano, 
        ativacao1_nota_aph_alto, metodo)  
    matriz_notas[4,3] = nota1_aph
    
#[DOU]
    nota1_dou, controle1_dou = defuzzyfica(
        x, ativacao1_nota_dou_baixo, ativacao1_nota_dou_mediano,
        ativacao1_nota_dou_alto, metodo) 
    matriz_notas[5,3] = nota1_dou   

    
############################ SISTEMA LINEAR ##################################

# calculando a performace do Soldado nas áreas Y = AX

    PERFOR_AREAS = np.dot(matriz_notas, matriz_contribuicao)  
    S = np.ndarray.flatten(np.transpose(PERFOR_AREAS)) # transforma em lista


########################## SISTEMA FUZZY 2 ###################################

#Antecedentes:
#   nota nas areas(valiaveis lingisticas: baixa, esperada, superior)
#    
#Consequente:
#   Desempenho profissional (valiaveis linguistica: baixo, aceitavel, superior)
#
# Regras logicas:
#     Regra1: Se as notas forem baixas, então desempenho será baixo    
#     Regra2: Se as notas forem esperadas, então nota será aceitavel   
#     Regra3: Se as notas forem altas, então nota será superior 

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    
# Encontando os graus de pertinencia

    SB, SE, SA = graus_pertinencia(x, yt_baixo, yt_aceitavel, yt_superior, S)

# ativando as regras para auto avaliação 

    ativacao_regra1_prof = np.max(SB)
    ativacao_regra2_prof = np.max(SE)
    ativacao_regra3_prof = np.max(SA)

# ativando a nota

    ativacao_des_prof_baixo = np.fmin(ativacao_regra1_prof,yt_baixo)
    ativacao_des_prof_aceitavel = np.fmin(ativacao_regra2_prof, yt_aceitavel)
    ativacao_des_prof_superior = np.fmin(ativacao_regra3_prof, yt_superior)


# achando o representante SCORE da performace profissional

    PERFOR_PROF, controle_prof = defuzzyfica(xt, ativacao_des_prof_baixo, 
            ativacao_des_prof_aceitavel, ativacao_des_prof_superior, metodo)
    

#$$$$$$$$$$$$$$$$$$$$$$$$$ PLOTANDO O GRAFICO DO DESEMPENHO $$$$$$$$$$$$$$$$$$

    plota_grafico_fuzzy(nome, xt, yt_baixo,
                        yt_aceitavel, yt_superior,PERFOR_PROF)

    print("PERF de CI:", '{:.2f}'.format(S[0]))
    print("PERF de SAQ:", '{:.2f}'.format(S[1]))
    print("PERF de SALT:", '{:.2f}'.format(S[2]))
    print("PERF de SAT:", '{:.2f}'.format(S[3]))
    print("PERF de APH:", '{:.2f}'.format(S[4]))
    print("PERF de DOU", '{:.2f}'.format(S[5]))
    print("SCORE:"'{:.0f}'.format(PERFOR_PROF))
    

    

if __name__ == "__main__":
    main()

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$














