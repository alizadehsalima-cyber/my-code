import pulp

# مدل
model = pulp.LpProblem("Facility_Location_2Layer", pulp.LpMinimize)

# مجموعه‌ها
customers = range(1, 29)        # مشتری‌ها
facilities1 = range(1, 13)      # تسهیلات لایه ۱
facilities2 = range(1, 5)       # تسهیلات لایه ۲

# متغیرها
w1 = pulp.LpVariable.dicts("w1", (customers, facilities1), cat='Binary')      # مشتری ↔ لایه۱
w2 = pulp.LpVariable.dicts("w2", (facilities1, facilities2), cat='Binary')    # لایه۱ ↔ لایه۲
y1 = pulp.LpVariable.dicts("y1", facilities1, cat='Binary')                   # بازبودن لایه۱
y2 = pulp.LpVariable.dicts("y2", facilities2, cat='Binary')                   # بازبودن لایه۲

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
d={
    (1, 1): 3, (1, 2): 2, (1, 3): 2, (1, 4): 1,
    (2, 1): 4, (2, 2): 1, (2, 3): 4, (2, 4): 1,
    (3, 1): 1, (3, 2): 4, (3, 3): 2, (3, 4): 2,
    (4, 1): 3, (4, 2): 1, (4, 3): 1, (4, 4): 1,
    (5, 1): 2, (5, 2): 3, (5, 3): 1, (5, 4): 1,
    (6, 1): 2, (6, 2): 2, (6, 3): 3, (6, 4): 1,
    (7, 1): 2, (7, 2): 4, (7, 3): 3, (7, 4): 1,
    (8, 1): 1, (8, 2): 3, (8, 3): 3, (8, 4): 2,
    (9, 1): 2, (9, 2): 1, (9, 3): 3, (9, 4): 2,
    (10, 1): 1, (10, 2): 2, (10, 3): 1, (10, 4): 1,
    (11, 1): 2, (11, 2): 2, (11, 3): 4, (11, 4): 1,
    (12, 1): 1, (12, 2): 4, (12, 3): 3, (12, 4): 1
    #(13, 1): 1, (13, 2): 4, (13, 3): 2, (13, 4): 3
}

# Layer 1 facility opening costs (y1_{j1})
f1 = {
    1: 700, 2: 500, 3: 1000, 4: 800, 5: 700, 6: 900,
    7: 750, 8: 800, 9: 1000, 10: 600, 11: 700, 12: 700
}

# Layer 2 facility opening costs (y2_{j2})
f2 = {
    1: 3000, 2: 2200, 3: 2200, 4: 2500
}

# Objective function (minimize total cost, only w2 and y1, y2)
model += pulp.lpSum(d[j1, j2] * w2[j1][j2] for j1 in facilities1 for j2 in facilities2) + \
         pulp.lpSum(f1[j1] * y1[j1] for j1 in facilities1) + \
         pulp.lpSum(f2[j2] * y2[j2] for j2 in facilities2)

# --- محدودیت اول: هر مشتری فقط به یکی از تسهیلات مجاز لایه۱ وصل شود ---
for i in customers:
    model += pulp.lpSum(w1[i][j1] for j1 in allowed_assignments[i]) == 1

# --- محدودیت دوم: لینک‌دهی مشتری ↔ لایه۱ ---
for i in customers:
    for j1 in allowed_assignments[i]:
        model += w1[i][j1] <= y1[j1]

# --- محدودیت سوم: لینک‌دهی لایه۱ ↔ لایه۲ (همه به همه) ---
for j1 in facilities1:
    for j2 in facilities2:
        model += w2[j1][j2] <= y1[j1]
        model += w2[j1][j2] <= y2[j2]

for i in customers:
  for j1 in allowed_assignments[i]:
    model += w1[i][j1] <= pulp.lpSum(w2[j1][j2] for j2 in facilities2)


# حل مدل
model.solve()

# بررسی وضعیت حل
print("Status:", pulp.LpStatus[model.status])

print("Open Layer 1 Facilities:")
for j1 in facilities1:
    if y1[j1].varValue == 1:
        print(f"Facility {j1}")
print("Open Layer 2 Facilities:")
for j2 in facilities2:
    if y2[j2].varValue == 1:
        print(f"Facility {j2}")
print("\n--- مسیرهای اتصال ---")
for i in customers:
    for j1 in allowed_assignments[i]:
        if w1[i][j1].varValue == 1:
            # یافتن تسهیلات لایه ۲ که این تسهیلات لایه ۱ به آن متصل شده
            connected_j2 = None
            for j2 in facilities2:
                if w2[j1][j2].varValue == 1:
                    connected_j2 = j2
                    break
            
            print(f"Customer {i} → Facility {j1} Layer1 → Facility {connected_j2} Layer2 (Connection cost: {d[j1,connected_j2]})")