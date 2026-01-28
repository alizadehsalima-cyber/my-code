from pulp import *
import time
model = LpProblem("Multi_Echelon_Location_Problem", LpMinimize)
I = [1, 2, 3, 4]
V1=[1, 2, 3]
V2 = [1, 2]
#f_j1 = [100, 80, 90]
f_j1 = {1:100, 2:80, 3:90}
f_j2= {1:30, 2:40}
d1= {(1,1):1, (1,2):2, (1,3):2,
       (2,1):2, (2,2):3, (2,3):2,
       (3,1):4, (3,2):2, (3,3):1.5,
       (4,1):2, (4,2):3, (4,3):4}
d2= {(1,1):4, (1,2):3,
        (2,1):5, (2,2):3,
        (3,1):2, (3,2):3}
w1 = LpVariable.dicts("w1", [(i,j1)for i in I for j1 in V1], cat='Binary')
w2 = LpVariable.dicts("w2", [(j1,j2)for j1 in V1 for j2 in V2], cat='Binary')
Y1 = LpVariable.dicts("Y1", V1, cat='Binary')
Y2 = LpVariable.dicts("Y2", V2, cat='Binary')
 
model += (
    lpSum(d1[(i,j1)]*w1[(i,j1)]for i in I for j1 in V1)+
    lpSum(d2[(j1,j2)]*w2[(j1,j2)]for j1 in V1 for j2 in V2)+
    lpSum(f_j1[j1]*Y1[j1]for j1 in V1)+
    lpSum(f_j2[j2]*Y2[j2]for j2 in V2)
)

for i in I:
    model+=lpSum(w1[(i,j1)] for j1 in V1)==1

for i in I: 
    for j1 in V1:
        model+=w1[(i,j1)]<= Y1[j1]

for j1 in V1: 
    for j2 in V2:
        model+=w2[(j1,j2)]<= Y1[j1]

for j1 in V1: 
    for j2 in V2:
        model+=w2[(j1,j2)]<= Y2[j2]

for i in I: 
    for j1 in V1:
        model+=w1[(i,j1)]<= lpSum(w2[(j1,j2)] for j2 in V2)

# Solve the model
start_time = time.perf_counter()
model.solve()
end_time = time.perf_counter()

# Print results
print("Status:", LpStatus[model.status])
print("Total Cost:", value(model.objective))
for v in model.variables():
    if v.varValue == 1:
        print(v.name, "=", v.varValue)

print("\nAssignment Results:")
for i in I:
    for j1 in V1:
        if w1[(i, j1)].varValue is not None and w1[(i, j1)].varValue > 0.5:
            for j2 in V2:
                if w2[(j1, j2)].varValue is not None and w2[(j1, j2)].varValue > 0.5:
                    print(f"Customer {i} is served by facility level-1 {j1} through facility level-2 {j2}.")
print(f"time: {end_time - start_time:.4f} s")