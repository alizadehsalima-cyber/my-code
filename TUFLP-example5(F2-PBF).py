from pulp import *
model = LpProblem("Multi_Echelon_Location_Problem", LpMinimize)
I = [1, 2, 3, 4]
J1=[1, 2, 3]
J2 = [1, 2]
F1 = {1:100, 2:80, 3:90}
F2= {1:30, 2:40}
d1= {(1,1):1, (1,2):2, (1,3):2,
       (2,1):2, (2,2):3, (2,3):2,
       (3,1):4, (3,2):2, (3,3):1.5,
       (4,1):2, (4,2):3, (4,3):4}
d2= {(1,1):4, (1,2):3,
        (2,1):5, (2,2):3,
        (3,1):2, (3,2):3}
#w,1 = LpVariable.dicts("w1", [(i,j1)for i in I for j1 in J1], cat='Binary')
#w2 = LpVariable.dicts("w2", [(j1,j2)for j1 in J1 for j2 in J2], cat='Binary')
Y1 = LpVariable.dicts("Y1", J1, cat='Binary')
Y2 = LpVariable.dicts("Y2", J2, cat='Binary')
x = LpVariable.dicts("x", [(i,j1,j2)for i in I for j1 in J1 for j2 in J2], cat='Binary')

# مدل
model = LpProblem("3_Index_Location_Problem", LpMinimize)

# تابع هدف
#model+=(
#   lpSum((d1[(i,j1)] + d2[(j1,j2)]) * x[(i,j1,j2)] for i in I for j1 in J1 for j2 in J2) +
   # lpSum(F1[j1] * Y1[j1] for j1 in J1) +
  #  lpSum(F2[j2] * Y2[j2] for j2 in J2)
#)
model += (
    lpSum(d1[(i,j1)]* x[(i,j1,j2)] for i in I for j1 in J1 for j2 in J2) +
    lpSum(d2[(j1,j2)]* x[(i,j1,j2)] for i in I for j1 in J1 for j2 in J2) +
    lpSum(F1[j1] * Y1[j1] for j1 in J1) +
    lpSum(F2[j2] * Y2[j2] for j2 in J2)
)

# محدودیت‌ها
for i in I:
    model += lpSum(x[(i,j1,j2)] for j1 in J1 for j2 in J2) == 1

for i in I:
    for j2 in J2:
        model += lpSum(x[(i,j1,j2)] for j1 in J1) <= Y2[j2]

for i in I:
    for j1 in J1:
        model += lpSum(x[(i,j1,j2)] for j2 in J2) <= Y1[j1]

# حل مدل
model.solve()

# چاپ نتایج
print("Status:", LpStatus[model.status])
print("Total Cost:", value(model.objective))
print("\nSelected x[i,j1,j2]:")
for i in I:
    for j1 in J1:
        for j2 in J2:
            if value(x[(i,j1,j2)]) > 0.5:
                print(f"x_{i}{j1}{j2} = 1  → Customer {i} is served via depot {j1} and hub {j2}")

print("\nOpened Facilities:")
for j1 in J1:
    if value(Y1[j1]) > 0.5:
        print(f"Y1_{j1} = 1  → Open depot {j1}")
for j2 in J2:
    if value(Y2[j2]) > 0.5:
        print(f"Y2_{j2} = 1  → Open hub {j2}")
