import pulp
import time
# مدل
model = pulp.LpProblem("Facility_Location_2Layer", pulp.LpMinimize)

# مجموعه‌ها
I = range(1, 29)        # مشتری‌ها
V1 = range(1, 13)      # تسهیلات لایه ۱
V2 = range(1, 5)       # تسهیلات لایه ۲

# متغیرها
w1 = pulp.LpVariable.dicts("w1", (I, V1), cat='Binary')      # مشتری ↔ لایه۱
w2 = pulp.LpVariable.dicts("w2", (V1, V2), cat='Binary')    # لایه۱ ↔ لایه۲
y1 = pulp.LpVariable.dicts("y1", V1, cat='Binary')                   # بازبودن لایه۱
y2 = pulp.LpVariable.dicts("y2", V2, cat='Binary')                   # بازبودن لایه۲
T = pulp.LpVariable.dicts("T", (V1, V2), lowBound=0, cat='Integer')

# لیست اتصال مجاز مشتری ↔ لایه۱
allowed_assignments = {
    1: [2, 3, 4],
    2: [3, 4, 8],
    3: [3, 5],
    4: [3, 5, 8],
    5: [3, 4, 5, 8],
    6: [5, 11],
    7: [5, 9, 11],
    8: [5, 9, 10, 11, 12],
    9: [5, 8, 9, 11],
    10: [3, 4, 5, 7, 8, 9],
    11: [6, 7, 8, 9, 10],
    12: [6, 7, 8, 9, 10],
    13: [1, 4, 7, 8],
    14: [1, 2, 3, 4, 7, 8],
    15: [1, 2, 4],
    16: [1, 2, 4, 7, 8],
    17: [1, 2, 4],
    18: [1, 2, 7],
    19: [1, 7, 8],
    20: [1, 6],
    21: [6, 10],
    22: [6, 7, 10],
    23: [6, 9, 10, 12],
    24: [9, 10, 11, 12],
    25: [10, 11, 12],
    26: [11, 12],
    27: [11, 12],
    28: [11, 12]
}

# Layer 1 to Layer 2 connection costs (w2_{j1_j2}) based on the provided matrix
#d = {[j1, j2] for j1 in facilities1 for j2 in facilities2}
d = {
    (1, 1): 200, (1, 2): 100, (1, 3): 300, (1, 4): 150,
    (2, 1): 100, (2, 2): 150, (2, 3): 100, (2, 4): 300,
    (3, 1): 400, (3, 2): 400, (3, 3): 200, (3, 4): 400,
    (4, 1): 150, (4, 2): 200, (4, 3): 100, (4, 4): 250,
    (5, 1): 550, (5, 2): 450, (5, 3): 400, (5, 4): 550,
    (6, 1): 300, (6, 2): 100, (6, 3): 300, (6, 4): 150,
    (7, 1): 250, (7, 2): 100, (7, 3): 200, (7, 4): 150,
    (8, 1): 250, (8, 2): 200, (8, 3): 200, (8, 4): 300,
    (9, 1): 400, (9, 2): 300, (9, 3): 300, (9, 4): 350,
    (10, 1): 450, (10, 2): 200, (10, 3): 300, (10, 4): 200,
    (11, 1): 600, (11, 2): 400, (11, 3): 500, (11, 4): 400,
    (12, 1): 650, (12, 2): 450, (12, 3): 600, (12, 4): 500
}
c = 700
v = {
    1: 100, 2: 250, 3: 300, 4: 300, 5: 700, 6: 600,
    7: 500, 8: 500, 9: 500, 10: 550, 11: 500, 12: 800,
    13: 600, 14: 450, 15: 300, 16: 350, 17: 200, 18: 100,
    19: 700, 20: 500, 21: 600, 22: 550, 23: 900, 24: 400,
    25: 700, 26: 800, 27: 1000, 28: 1000
}

# Layer 1 facility opening costs (y1_{j1})
f1 = {
    1: 700000, 2: 500000, 3: 1000000, 4: 800000, 5: 700000, 6: 900000,
    7: 750000, 8: 800000, 9: 1000000, 10: 900000, 11: 1000000, 12: 900000
}

# Layer 2 facility opening costs (y2_{j2})
f2 = {
    1: 3500000, 2: 2600000, 3: 2500000, 4: 3000000
}

# Objective function (minimize total cost, only w2 and y1, y2)
model += pulp.lpSum(T[j1][j2]*d[j1, j2] for j1 in V1 for j2 in V2) + \
         pulp.lpSum(f1[j1] * y1[j1] for j1 in V1) + \
         pulp.lpSum(f2[j2] * y2[j2] for j2 in V2)

# --- محدودیت اول: هر مشتری فقط به یکی از تسهیلات مجاز لایه۱ وصل شود ---
for i in I:
    model += pulp.lpSum(w1[i][j1] for j1 in allowed_assignments[i]) == 1

# --- محدودیت دوم: لینک‌دهی مشتری ↔ لایه۱ ---
for i in I:
    for j1 in allowed_assignments[i]:
        model += w1[i][j1] <= y1[j1]

# --- محدودیت سوم: لینک‌دهی لایه۱ ↔ لایه۲ (همه به همه) ---
for j1 in V1:
    for j2 in V2:
        model += w2[j1][j2] <= y1[j1]
        model += w2[j1][j2] <= y2[j2]

for i in I:
  for j1 in allowed_assignments[i]:
    model += w1[i][j1] <= pulp.lpSum(w2[j1][j2] for j2 in V2)

for j1 in V1:
    model += pulp.lpSum(v[i] * w1[i][j1] for i in I if j1 in allowed_assignments[i]) <= \
             c*pulp.lpSum( T[j1][j2] for j2 in V2)
    
#for j1 in facilities1:
    #for j2 in facilities2:
        #model += pulp.lpSum(v[i] * w1[i][j1] for i in customers if j1 in allowed_assignments[i]) <= c * T[j1][j2]    

for j1 in V1:
    for j2 in V2:
        model += T[j1][j2] <= 1000 * w2[j1][j2]


# حل مدل
start_time = time.perf_counter()
model.solve()
end_time = time.perf_counter()
# بررسی وضعیت حل
print("Status:", pulp.LpStatus[model.status])

print("Open Layer 1 Facilities:")
for j1 in V1:
    if y1[j1].varValue == 1:
        print(f"Facility {j1}")
print("Open Layer 2 Facilities:")
for j2 in V2:
    if y2[j2].varValue == 1:
        print(f"Facility {j2}")
print("\n--- مسیرهای اتصال ---")
for i in I:
    for j1 in allowed_assignments[i]:
        if w1[i][j1].varValue == 1:
            # یافتن تسهیلات لایه ۲ که این تسهیلات لایه ۱ به آن متصل شده
            connected_j2 = None
            for j2 in V2:
                if w2[j1][j2].varValue == 1:
                    connected_j2 = j2
                    break
            
            print(f"Customer {i} → Facility {j1} Layer1 → Facility {connected_j2} Layer2 (Connection cost: {d[j1,connected_j2]})")

print("\n--- تعداد کامیون‌های ارسال شده ---")
for j1 in V1:
    for j2 in V2:
        if T[j1][j2].varValue > 0:
            print(f"Facility {j2} to Facility {j1} : {T[j1][j2].varValue} کامیون")

print(f"time: {end_time - start_time:.4f} s")
