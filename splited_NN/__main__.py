from mc_neuro import *
from __include__ import *
from qb_neuro import *

# Я генерирую случайный quarterbeats

number_of_generated_data = 100 
mc:list[int] = []

duration_qb_training()
print(duration_qb_generate([2,6]))

# mc_training()



# for idx in range(number_of_generated_data):
#     mc.append(round(mc_next_generate(idx)))

# # print(mc)

