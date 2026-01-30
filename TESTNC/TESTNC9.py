



import numpy as np
from pulp import *
import time
# ============= داده‌های از قبل تولید شده ============= #
np.random.seed(42)

num_customers = 100
num_midpoints = 20
num_warehouses = 10
p_exist_ij1=0.7
p_exist_j1j2=0.8
# هزینه ساخت نقاط میانی j1
f_j1 = np.random.randint(50, 200, size=num_midpoints)

# هزینه ساخت انبارها j2
f_j2 = np.random.randint(80, 300, size=num_warehouses)

d_i_j1 = np.where(
    np.random.rand(num_customers, num_midpoints) < p_exist_ij1,
    np.random.randint(10, 50, size=(num_customers, num_midpoints)),
    None
)

# نقطه میانی → انبار (ناقص)
d_j1_j2 = np.where(
    np.random.rand(num_midpoints, num_warehouses ) < p_exist_j1j2,
    np.random.randint(50, 100, size=(num_midpoints,num_warehouses )),
    None
)

# ============= شاخص‌ها ============= #
I = range(num_customers)
V1 = range(num_midpoints)
V2 = range(num_warehouses)
# ============= تعریف مدل ============= #
model = LpProblem("Two_Echelon_Location", LpMinimize)

# متغیرها
w1 = LpVariable.dicts(
    "w1",
    [(i,j1) for i in I for j1 in V1 if d_i_j1[i][j1] is not None],
    0, 1, LpBinary
)

w2 = LpVariable.dicts(
    "w2",
    [(j1,j2) for j1 in V1 for j2 in V2 if d_j1_j2[j1][j2] is not None],
    0, 1, LpBinary
)

Y1 = LpVariable.dicts("Y1", V1, 0, 1, LpBinary)
Y2 = LpVariable.dicts("Y2", V2, 0, 1, LpBinary)


# تابع هدف
model += (
    lpSum(d_i_j1[i][j1] * w1[(i,j1)] for i in I for j1 in V1 if (i,j1) in w1) +
    lpSum(d_j1_j2[j1][j2] * w2[(j1,j2)] for j1 in V1 for j2 in V2 if (j1,j2) in w2) +
    lpSum(f_j1[j1] * Y1[j1] for j1 in V1) +
    lpSum(f_j2[j2] * Y2[j2] for j2 in V2)
)

for i in I:
    model += lpSum(w1[(i,j1)] for j1 in V1 if (i,j1) in w1) == 1

for (i,j1) in w1:
    model += w1[(i,j1)] <= Y1[j1]

for (j1,j2) in w2:
    model += w2[(j1,j2)] <= Y1[j1]

for (j1,j2) in w2:    
    model += w2[(j1,j2)] <= Y2[j2]

for i in I:
    for j1 in V1:
        if (i,j1) in w1:
            model += (w1[(i,j1)] <= lpSum(w2[(j1,j2)] for j2 in V2 if (j1,j2) in w2))

# ============= حل مدل ============= #
start_time = time.perf_counter()
model.solve(PULP_CBC_CMD(msg=0))
end_time = time.perf_counter()

print("Status:", LpStatus[model.status])
print("Total Cost:", value(model.objective))
print(f"time optimal: {end_time - start_time:.4f} s")

#for j2 in V2:
    #if Y2[j2].varValue > 0.5:
       # print(f"Warehouse {j2}")

#for j1 in V1:
    #if Y1[j1].varValue > 0.5:
        #print(f"Midpoint {j1}")

#print("\nتخصیص مشتری → مرکز → انبار:")
#for i in I:
    #for j1 in V1:
        #if w1[(i,j1)].varValue > 0.5:
            #for j2 in V2:
               ## if w2[(j1,j2)].varValue > 0.5:
                  #  print(f"Customer {i} → Midpoint {j1} → Warehouse {j2}")

import random

# ---------------- توابع GA ----------------

def generate_random_chromosome():
    while True:
        y_j1 = [random.randint(0, 2) % 2 for _ in V1]
        y_j2 = [random.randint(0, 2) % 2 for _ in V2]
        if sum(y_j1) >= 1 and sum(y_j2) >= 1:
            return y_j1 + y_j2
        


def evaluate_chromosome(chromosome):
    y_j1 = chromosome[:len(V1)]
    y_j2 = chromosome[len(V1):]

    total_cost = sum(f_j1[j] for j, a in enumerate(y_j1) if a == 1) \
               + sum(f_j2[j] for j, a in enumerate(y_j2) if a == 1)

    if sum(y_j1) == 0 or sum(y_j2) == 0:
        return total_cost + 10000, "نامعتبر"

    v1_to_v2 = {}
    warehouse_connections = {j2: 0 for j2 in range(len(V2)) if y_j2[j2] == 1}

    for j1 in range(len(V1)):  
        if y_j1[j1] == 1:
            open_j2 = [j2 for j2 in range(len(V2)) if y_j2[j2] == 1]
            if not open_j2:
                return total_cost + 10000, "نامعتبر"
            valid_j2 = [j2 for j2 in open_j2 if d_j1_j2[j1][j2] is not None]
            if not valid_j2:
                return total_cost + 10000, "نامعتبر"
            best_j2 = min(valid_j2, key=lambda j2: d_j1_j2[j1][j2])           
            v1_to_v2[j1] = best_j2
            warehouse_connections[best_j2] += 1
            total_cost += d_j1_j2[j1][best_j2]
    for count in warehouse_connections.values():
        if count == 0 :
            return total_cost + 10000, "نامعتبر"
        
    midpoint_usage = {j1: 0 for j1 in v1_to_v2.keys()}
   
    for i in range(len(I)):
        valid_midpoints = list(v1_to_v2.keys())
        if not valid_midpoints:
            return total_cost + 10000, "نامعتبر"
        valid_j1 = [j1 for j1 in valid_midpoints if d_i_j1[i][j1] is not None]
        if not valid_j1:
            return total_cost + 10000, "نامعتبر"
        best_j1 = min(valid_j1, key=lambda j1: d_i_j1[i][j1])
        midpoint_usage[best_j1] += 1 
        total_cost += d_i_j1[i][best_j1]
    for j1, count in midpoint_usage.items():
        if count == 0:
            return total_cost + 10000, "نامعتبر"
        
    return total_cost, "معتبر"

def generate_initial_population(size=30):
    pop = []
    while len(pop) < size: 
        chrom = generate_random_chromosome()
        c, s = evaluate_chromosome(chrom)
        if s == "معتبر":
            pop.append({
                "chromosome": chrom, "cost": c,"status": s})
    return pop

def roulette(pop, k=2):
    valids = [p for p in pop if p["status"] == "معتبر"]
    if not valids:
        valids = pop
    costs = [p["cost"] for p in valids]
    max_cost = max(costs)
    epsilon = 1
    fitnesses = [max_cost - cost + epsilon for cost in costs]
    total_fitness = sum(fitnesses)
    weights = [f / total_fitness for f in fitnesses]
    return random.choices(valids, weights=weights, k=k)

def crossover(p1, p2, crossover_rate=0.8):
    if random.random() < crossover_rate:
        cut = random.randint(1, len(p1)-1)
        c1 = p1[:cut] + p2[cut:]
        c2 = p2[:cut] + p1[cut:]
    else:
        c1 = p1.copy()
        c2 = p2.copy()
    
    return c1, c2

def mutate(c, rate=0.1):
    child = c[:] 
    for i in range(len(child)):
        if random.random() < rate:
            child[i] = 1 - child[i]
    return child



# ---------------- اجرای GA ----------------
start_time = time.perf_counter()
pop = generate_initial_population(30)

for g in range(50000):
    p1, p2 = [p["chromosome"] for p in roulette(pop)]
    c1, c2 = crossover(p1, p2)
    c1, c2 = mutate(c1), mutate(c2)
    c1_cost, c1_stat = evaluate_chromosome(c1)
    c2_cost, c2_stat = evaluate_chromosome(c2)
    pop.extend([
        {"chromosome": c1, "cost": c1_cost, "status": c1_stat},
        {"chromosome": c2, "cost": c2_cost, "status": c2_stat}
    ])
    pop = sorted(pop, key=lambda x: x["cost"])[:30]
end_time = time.perf_counter()    

best = min(pop, key=lambda x: x["cost"])
print("\n===resolts GA ===")
print("کروموزوم:", best["chromosome"])
print("Total Cost GA:", best["cost"])
#print("وضعیت:", best["status"])
print(f"time GA: {end_time - start_time:.4f} s")