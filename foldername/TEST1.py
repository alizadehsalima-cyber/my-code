
import numpy as np
from pulp import *

# ============= داده‌های از قبل تولید شده ============= #
np.random.seed(42)

num_customers = 10
num_midpoints = 7
num_warehouses = 5

# هزینه ساخت نقاط میانی j1
f_j1 = np.random.randint(50, 200, size=num_midpoints)

# هزینه ساخت انبارها j2
f_j2 = np.random.randint(80, 300, size=num_warehouses)

# هزینه مشتری → نقطه میانی
d_i_j1 = np.random.randint(10, 100, size=(num_customers, num_midpoints))

# هزینه نقطه میانی → انبار
d_j1_j2 = np.random.randint(10, 100, size=(num_midpoints, num_warehouses))

# ============= شاخص‌ها ============= #
I  = range(1, num_customers+1)
V1 = range(1, num_midpoints+1)
V2 = range(1, num_warehouses+1)

# ============= تعریف مدل ============= #
model = LpProblem("Two_Echelon_Location", LpMinimize)

# متغیرها
w1 = LpVariable.dicts("w1", [(i,j1) for i in I for j1 in V1], 0, 1, LpBinary)
w2 = LpVariable.dicts("w2", [(j1,j2) for j1 in V1 for j2 in V2], 0, 1, LpBinary)
Y1 = LpVariable.dicts("Y1", V1, 0, 1, LpBinary)
Y2 = LpVariable.dicts("Y2", V2, 0, 1, LpBinary)

# تابع هدف
model += (
    lpSum(d_i_j1[i-1][j1-1] * w1[(i,j1)] for i in I for j1 in V1) +
    lpSum(d_j1_j2[j1-1][j2-1] * w2[(j1,j2)] for j1 in V1 for j2 in V2) +
    lpSum(f_j1[j1-1] * Y1[j1] for j1 in V1) +
    lpSum(f_j2[j2-1] * Y2[j2] for j2 in V2)
)

# محدودیت‌ها
for i in I:
    model += lpSum(w1[(i,j1)] for j1 in V1) == 1

for i in I:
    for j1 in V1:
        model += w1[(i,j1)] <= Y1[j1]

for j1 in V1:
    for j2 in V2:
        model += w2[(j1,j2)] <= Y2[j2]

for j1 in V1:
    model += lpSum(w2[(j1,j2)] for j2 in V2) >= Y1[j1]

# ============= حل مدل ============= #
model.solve(PULP_CBC_CMD(msg=0))

print("Status:", LpStatus[model.status])
print("Total Cost:", value(model.objective))

print("هزینه ساخت نقاط میانی (f_j1):")
print(f_j1)

print("\nهزینه ساخت انبارها (f_j2):")
print(f_j2)

print("\nهزینه مشتری → نقطه میانی (d_i_j1):")
print(d_i_j1)

print("\nهزینه نقطه میانی → انبار (d_j1_j2):")
print(d_j1_j2)

print("\nانبارهای فعال:")
for j2 in V2:
    if Y2[j2].varValue > 0.5:
        print(f"Warehouse {j2}")

print("\nمراکز میانی فعال:")
for j1 in V1:
    if Y1[j1].varValue > 0.5:
        print(f"Midpoint {j1}")

print("\nتخصیص مشتری → مرکز → انبار:")
for i in I:
    for j1 in V1:
        if w1[(i,j1)].varValue > 0.5:
            for j2 in V2:
                if w2[(j1,j2)].varValue > 0.5:
                    print(f"Customer {i} → Midpoint {j1} → Warehouse {j2}")


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
    for j1 in range(len(V1)):  # 0..6
        if y_j1[j1] == 1:
            open_j2 = [j2 for j2 in range(len(V2)) if y_j2[j2] == 1]
            if not open_j2:
                return total_cost + 10000, "نامعتبر"
            best_j2 = min(open_j2, key=lambda j2: d_j1_j2[j1][j2])
            v1_to_v2[j1] = best_j2
            total_cost += d_j1_j2[j1][best_j2]

    for i in range(len(I)):
        valid_midpoints = list(v1_to_v2.keys())
        if not valid_midpoints:
            return total_cost + 10000, "نامعتبر"
        best_j1 = min(valid_midpoints, key=lambda j1: d_i_j1[i][j1])
        total_cost += d_i_j1[i][best_j1]

    return total_cost, "معتبر"

def generate_initial_population(size=30):
    pop = []
    for _ in range(size):
        chrom = generate_random_chromosome()
        c, s = evaluate_chromosome(chrom)
        pop.append({"chromosome": chrom, "cost": c, "status": s})
    return pop

def roulette(pop, k=2):
    valids = [p for p in pop if p["status"] == "معتبر"]
    if not valids:
        valids = pop
    weights = [1/p["cost"] for p in valids]
    return random.choices(valids, weights=weights, k=k)

def crossover(p1, p2):
    cut = random.randint(1, len(p1)-1)
    return p1[:cut] + p2[cut:], p2[:cut] + p1[cut:]

def mutate(c, rate=0.1):
    out = c[:]
    for i in range(len(out)):
        if random.random() < rate:
            out[i] = 1 - out[i]
    if sum(out[:len(V1)]) < 1 or sum(out[len(V1):]) < 1:
        return generate_random_chromosome()
    return out

# ---------------- اجرای GA ----------------
pop = generate_initial_population(30)

for g in range(1000):
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

best = min(pop, key=lambda x: x["cost"])
print("\n=== نتیجه الگوریتم ژنتیک ===")
print("کروموزوم:", best["chromosome"])
print("تابع هدف GA:", best["cost"])
print("وضعیت:", best["status"])



import numpy as np
from pulp import *
import random
import time

# ============= داده‌ها (غیر از زمان اجرا) ============= #
np.random.seed(42)

num_customers = 10
num_midpoints = 7
num_warehouses = 5

# هزینه تولید نقاط میانی j1
f_j1 = np.random.randint(50, 200, size=num_midpoints)

# هزینه تولید انبارها j2
f_j2 = np.random.randint(80, 300, size=num_warehouses)

# هزینه مشتری → نقطه میانی
d_i_j1 = np.random.randint(10, 100, size=(num_customers, num_midpoints))

# هزینه نقطه میانی → انبار
d_j1_j2 = np.random.randint(10, 100, size=(num_midpoints, num_warehouses))

# شاخص‌ها
I  = range(1, num_customers+1)
V1 = range(1, num_midpoints+1)
V2 = range(1, num_warehouses+1)

# ================= حل مدل دقیق با Pulp ================= #
model = LpProblem("Two_Echelon_Location", LpMinimize)

# متغیرها
w1 = LpVariable.dicts("w1", [(i,j1) for i in I for j1 in V1], 0, 1, LpBinary)
w2 = LpVariable.dicts("w2", [(j1,j2) for j1 in V1 for j2 in V2], 0, 1, LpBinary)
Y1 = LpVariable.dicts("Y1", V1, 0, 1, LpBinary)
Y2 = LpVariable.dicts("Y2", V2, 0, 1, LpBinary)

# تابع هدف
model += (
    lpSum(d_i_j1[i-1][j1-1] * w1[(i,j1)] for i in I for j1 in V1) +
    lpSum(d_j1_j2[j1-1][j2-1] * w2[(j1,j2)] for j1 in V1 for j2 in V2) +
    lpSum(f_j1[j1-1] * Y1[j1] for j1 in V1) +
    lpSum(f_j2[j2-1] * Y2[j2] for j2 in V2)
)

# محدودیت‌ها
for i in I:
    model += lpSum(w1[(i,j1)] for j1 in V1) == 1

for i in I:
    for j1 in V1:
        model += w1[(i,j1)] <= Y1[j1]

for j1 in V1:
    for j2 in V2:
        model += w2[(j1,j2)] <= Y2[j2]

for j1 in V1:
    model += lpSum(w2[(j1,j2)] for j2 in V2) >= Y1[j1]

# اندازه‌گیری زمان حل مدل دقیق
start_time = time.perf_counter()
model.solve(PULP_CBC_CMD(msg=0))
end_time = time.perf_counter()

print("=== مدل دقیق با Pulp ===")
print("Status:", LpStatus[model.status])
print("Total Cost:", value(model.objective))
print(f"زمان اجرای مدل دقیق: {end_time - start_time:.4f} ثانیه\n")

# ================= توابع GA ================= #
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
    for j1 in range(len(V1)):
        if y_j1[j1] == 1:
            open_j2 = [j2 for j2 in range(len(V2)) if y_j2[j2] == 1]
            if not open_j2:
                return total_cost + 10000, "نامعتبر"
            best_j2 = min(open_j2, key=lambda j2: d_j1_j2[j1][j2])
            v1_to_v2[j1] = best_j2
            total_cost += d_j1_j2[j1][best_j2]

    for i in range(len(I)):
        valid_midpoints = list(v1_to_v2.keys())
        if not valid_midpoints:
            return total_cost + 10000, "نامعتبر"
        best_j1 = min(valid_midpoints, key=lambda j1: d_i_j1[i][j1])
        total_cost += d_i_j1[i][best_j1]

    return total_cost, "معتبر"

def generate_initial_population(size=30):
    pop = []
    for _ in range(size):
        chrom = generate_random_chromosome()
        c, s = evaluate_chromosome(chrom)
        pop.append({"chromosome": chrom, "cost": c, "status": s})
    return pop

def roulette(pop, k=2):
    valids = [p for p in pop if p["status"] == "معتبر"]
    if not valids:
        valids = pop
    weights = [1/p["cost"] for p in valids]
    return random.choices(valids, weights=weights, k=k)

def crossover(p1, p2):
    cut = random.randint(1, len(p1)-1)
    return p1[:cut] + p2[cut:], p2[:cut] + p1[cut:]

def mutate(c, rate=0.1):
    out = c[:]
    for i in range(len(out)):
        if random.random() < rate:
            out[i] = 1 - out[i]
    if sum(out[:len(V1)]) < 1 or sum(out[len(V1):]) < 1:
        return generate_random_chromosome()
    return out

# ================= اجرای GA و اندازه‌گیری زمان ================= #
start_time = time.perf_counter()

pop = generate_initial_population(30)
for g in range(300):
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

best = min(pop, key=lambda x: x["cost"])

end_time = time.perf_counter()

print("=== الگوریتم ژنتیک (GA) ===")
print("کروموزوم:", best["chromosome"])
print("تابع هدف GA:", best["cost"])
print("وضعیت:", best["status"])
print(f"زمان اجرای GA: {end_time - start_time:.4f} ثانیه")
