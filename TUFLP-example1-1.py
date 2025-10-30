# کامل‌ترین پیاده‌سازی بر اساس مدل داده‌شده با استفاده از pulp
import pulp

# مدل
model = pulp.LpProblem("Supply_Chain_Model", pulp.LpMinimize)

# مجموعه‌ها
I = [1, 2, 3]         # Facility index
J = [1, 2, 3, 4]      # Satellite index
K = [1, 2, 3, 4, 5, 6, 7]  # Product index

# --- تعریف متغیرها ---
X = pulp.LpVariable.dicts("X", [(i,j,k) for i in I for j in J for k in K], 0, 1, pulp.LpBinary)
T = pulp.LpVariable.dicts("T", [(i,j) for i in I for j in J], 0, 1, pulp.LpBinary)
Y = pulp.LpVariable.dicts("Y", I, 0, 1, pulp.LpBinary)
U = pulp.LpVariable.dicts("U", J, 0, None, pulp.LpInteger)
V = pulp.LpVariable.dicts("V", I, 0, None, pulp.LpInteger)
H = pulp.LpVariable.dicts("H", [(i,j) for i in I for j in J], 0, None, pulp.LpInteger)

# --- تابع هدف (در صورت نیاز) ---
model += (
    150*Y[1] + 135*Y[2] + 170*Y[3]
    + 16*V[1] + 23*V[2] + 12*V[3]
    + 20*U[1] + 15*U[2] + 25*U[3] + 25*U[4]
    + 5*H[1,1] + 7*H[1,2] + 7*H[1,3] + 8*H[2,1] + 5*H[2,2] + 8.5*H[2,4] + 4*H[3,2] + 9*H[3,3] + 7*H[3,4]
    + 1.5*X[1,1,1] + 4*X[1,1,3] + 3*X[1,1,4]
    + 2.5*X[1,2,1] + 2.5*X[1,2,2] + 4.5*X[1,2,5] + 1*X[1,2,6] + 4*X[1,2,7]
    + 1.5*X[1,3,2] + 3.5*X[1,3,4] + 4*X[1,3,7]
    + 1.5*X[2,1,1] + 4*X[2,1,3] + 3*X[2,1,4]
    + 2.5*X[2,2,1] + 2.5*X[2,2,2] + 4.5*X[2,2,5] + 1*X[2,2,6] + 4*X[2,2,7]
    + 3*X[2,4,3] + 1.5*X[2,4,5] + 1.5*X[2,4,6]
    + 2.5*X[3,1,1] + 2.5*X[3,2,2] + 4.5*X[3,2,5] + 1*X[3,2,6] + 4*X[3,2,7]
    + 1.5*X[3,3,2] + 3.5*X[3,3,4] + 4*X[3,3,7]
    + 3*X[3,4,3] + 1.5*X[3,4,5] + 1.5*X[3,4,6]
)

# --- محدودیت‌ها ---
# پوشش سفارش‌ها
model += X[1,1,1] + X[1,2,1] + X[2,1,1] + X[2,2,1] == 1
model += X[1,2,2] + X[1,3,2] + X[2,2,2] + X[3,2,2] + X[3,3,2] == 1
model += X[1,1,3] + X[2,1,3] + X[2,4,3] + X[3,4,3] == 1
model += X[1,1,4] + X[1,3,4] + X[2,1,4] + X[3,3,4] == 1
model += X[1,2,5] + X[2,2,5] + X[2,4,5] + X[3,2,5] + X[3,4,5] == 1
model += X[1,2,6] + X[2,2,6] + X[2,4,6] + X[3,2,6] + X[3,4,6] == 1
model += X[1,2,7] + X[1,3,7] + X[2,2,7] + X[3,2,7] + X[3,3,7] == 1

# T محدودیت
model += T[1,1] + T[2,1] <= 1
model += T[1,2] + T[2,2] + T[3,2] <= 1
model += T[1,3] + T[3,3] <= 1
model += T[2,4] + T[3,4] <= 1

# ظرفیت Y_i
model += 3*X[1,1,1] + 5*X[1,1,3] + 2*X[1,1,4] + 3*X[1,2,1] + 5*X[1,2,2] + 9*X[1,2,5] + X[1,2,6] +2*X[1,2,7]+ 5*X[1,3,2] + 2*X[1,3,4] + 2*X[1,3,7] <= 27*Y[1]
model += 3*X[2,1,1] + 5*X[2,1,3] + 2*X[2,1,4] + 3*X[2,2,1] + 5*X[2,2,2] + 9*X[2,2,5] + X[2,2,6] + 2*X[2,2,7] + 5*X[2,4,3] + 9*X[2,4,5] + X[2,4,6] <= 27*Y[2]
model += 3*X[3,2,1] + 5*X[3,2,2] + 9*X[3,2,5] + X[3,2,6] + 2*X[3,2,7] + 5*X[3,3,2] + 2*X[3,3,4] + 2*X[3,3,7] + 5*X[3,4,3] + 9*X[3,4,5] + X[3,4,6] <= 27*Y[3]

# X ≤ T
# فرض بر این که X و T قبلا به صورت دیکشنری تعریف شدن مثل:
# X = pulp.LpVariable.dicts("X", [(i,j,k) for i in I for j in J for k in K], 0, 1, pulp.LpBinary)
# T = pulp.LpVariable.dicts("T", [(i,j) for i in I for j in J], 0, 1, pulp.LpBinary)

# حالا محدودیت‌ها:
model += X[1,1,1] <= T[1,1]
model += X[1,1,3] <= T[1,1]
model += X[1,1,4] <= T[1,1]

model += X[1,2,1] <= T[1,2]
model += X[1,2,2] <= T[1,2]
model += X[1,2,5] <= T[1,2]
model += X[1,2,6] <= T[1,2]
model += X[1,2,7] <= T[1,2]

model += X[1,3,2] <= T[1,3]
model += X[1,3,4] <= T[1,3]
model += X[1,3,7] <= T[1,3]

model += X[2,1,1] <= T[2,1]
model += X[2,1,3] <= T[2,1]
model += X[2,1,4] <= T[2,1]

model += X[2,2,1] <= T[2,2]
model += X[2,2,2] <= T[2,2]
model += X[2,2,5] <= T[2,2]
model += X[2,2,6] <= T[2,2]
model += X[2,2,7] <= T[2,2]

model += X[2,4,3] <= T[2,4]
model += X[2,4,5] <= T[2,4]
model += X[2,4,6] <= T[2,4]

model += X[3,2,1] <= T[3,2]
model += X[3,2,2] <= T[3,2]
model += X[3,2,5] <= T[3,2]
model += X[3,2,6] <= T[3,2]
model += X[3,2,7] <= T[3,2]

model += X[3,3,2] <= T[3,3]
model += X[3,3,4] <= T[3,3]
model += X[3,3,7] <= T[3,3]

model += X[3,4,3] <= T[3,4]
model += X[3,4,5] <= T[3,4]
model += X[3,4,6] <= T[3,4]

# T ≤ Y
model += T[1,1] <= Y[1]
model += T[1,2] <= Y[1]
model += T[1,3] <= Y[1]

model += T[2,1] <= Y[2]
model += T[2,2] <= Y[2]
model += T[2,4] <= Y[2]

model += T[3,2] <= Y[3]
model += T[3,3] <= Y[3]
model += T[3,4] <= Y[3]


# محدودیت U_j
model += 7*X[1,1,1] + 7*X[1,1,4] + 8*X[1,1,3] + 7*X[2,1,1] + 8*X[2,1,3] + 7*X[2,1,4] <= 10*U[1]
model += 7*X[1,2,1] + 4*X[1,2,2] + 3*X[1,2,5] + 2*X[1,2,6] + 4*X[1,2,7] + 7*X[2,2,1] + 4*X[2,2,2] + 3*X[2,2,5] + 2*X[2,2,6] + 4*X[2,2,7] + 7*X[3,2,1] + 4*X[3,2,2] + 3*X[3,2,5] + 2*X[3,2,6] + 4*X[3,2,7] <= 10*U[2]
model += 4*X[1,3,2] + 7*X[1,3,4] + 4*X[1,3,7] + 4*X[3,3,2] + 7*X[3,3,4] + 4*X[3,3,7] <= 10*U[3]
model += 8*X[2,4,3] + 3*X[2,4,5] + 2*X[2,4,6] + 8*X[3,4,3] + 3*X[3,4,5] + 2*X[3,4,6] <= 10*U[4]

# محدودیت V_i
model += 3*X[1,1,1] + 5*X[1,1,3] + 2*X[1,1,4] + 3*X[1,2,1] + 5*X[1,2,2] + 9*X[1,2,5] + X[1,2,6]+ 2*X[1,2,7] + 5*X[1,3,2] + 2*X[1,3,4] + 2*X[1,3,7] <= 12*V[1]
model += 3*X[2,1,1] + 5*X[2,1,3] + 2*X[2,1,4] + 3*X[2,2,1] + 5*X[2,2,2] + 9*X[2,2,5] + X[2,2,6] + 2*X[2,2,7] + 5*X[2,4,3] + 9*X[2,4,5] + X[2,4,6] <= 12*V[2]
model += 3*X[3,2,1] + 5*X[3,2,2] + 9*X[3,2,5] + X[3,2,6] + 2*X[3,2,7] + 5*X[3,3,2] + 2*X[3,3,4] + 2*X[3,3,7] + 5*X[3,4,3] + 9*X[3,4,5] + X[3,4,6] <= 12*V[3]

# محدودیت H_ij (نمونه‌هایی)
model += 3*X[1,1,1] + 5*X[1,1,3] + 2*X[1,1,4] <= 6*H[1,1]
model += 3*X[1,2,1] + 5*X[1,2,2] + 9*X[1,2,5] + X[1,2,6] + 2*X[1,2,7] <= 6*H[1,2]
model += 5*X[1,3,2] + 2*X[1,3,4] + 2*X[1,3,7] <= 6*H[1,3]
model += 3*X[2,1,1] + 5*X[2,1,3] + 2*X[2,1,4] <= 6*H[2,1]
model += 3*X[2,2,1] + 5*X[2,2,2] + 9*X[2,2,5] + X[2,2,6] + 2*X[2,2,7] <= 6*H[2,2]
model += 5*X[2,4,3] + 9*X[2,4,5] + X[2,4,6] <= 6*H[2,4]
model += 3*X[3,2,1] + 5*X[3,2,2] + 9*X[3,2,5] + X[3,2,6] + 2*X[3,2,7] <= 6*H[3,2]
model += 5*X[3,3,2] + 2*X[3,3,4] + 2*X[3,3,7] <= 6*H[3,3]
model += 5*X[3,4,3] + 9*X[3,4,5] + X[3,4,6] <= 6*H[3,4]

# حل مدل
model.solve()
print("Status:", pulp.LpStatus[model.status])
for v in model.variables():
    if v.varValue not in [0, None]:
        print(v.name, "=", v.varValue)
import os
from pulp import value, LpStatus


output_dir = r"C:\Users\User\Downloads\Compressed\HSU-Thesis-V1.5\HSU-Thesis - 1.5"

# اطمینان از وجود پوشه خروجی (اگر وجود نداشت بسازش)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

full_output_path = os.path.join(output_dir, 'results_full.tex')

with open(full_output_path, 'w', encoding='utf-8') as f:
    f.write('\\subsection{نتایج کامل اجرای مدل - فایل خروجی تجمیعی}\n')
    f.write('\\begin{latin}\n')
    f.write('\\begin{tabular}{l}\n')

    # وضعیت مدل
    f.write(f'Status: {LpStatus[model.status]} \\\\\n')
    if model.status == 1:
        f.write(f'Total Cost: {value(model.objective)} \\\\\n')
    else:
        f.write('Total Cost: N/A \\\\\n')

    # دپوها
    f.write('Depot openings: \\\\\n')
    for i in I:
        f.write(f'Y\\lr{{[}}{i}\\lr{{]}} = {value(Y[i])} \\\\\n')

    f.write('Trucks entering depots (V): \\\\\n')
    for i in I:
        f.write(f'V\\lr{{[}}{i}\\lr{{]}} = {value(V[i])} \\\\\n')
    f.write('\\end{tabular}\n')
    f.write('\\clearpage\n')
    f.write('\\begin{tabular}{l}\n')
    # مراکز میانی
    f.write('Satellite utilizations (U): \\\\\n')
    for j in J:
        f.write(f'U\\lr{{[}}{j}\\lr{{]}} = {value(U[j]):.2f} \\\\\n')
    
    f.write('Truck movements between depots and satellites (H): \\\\\n')
    for i in I:
        for j in J:
            f.write(f'H\\lr{{[(}}{i}, {j}\\lr{{)]}} = {value(H[(i,j)])} \\\\\n')
    f.write('\\end{tabular}\n')
    f.write('\\begin{tabular}{H}\n')
    # تخصیص مشتری
    f.write('Customer assignments: \\\\\n')
    for k in K:
        for i in I:
            for j in J:
                 val = pulp.value(X[i, j, k])
                 if val is not None and val > 0.5:
                    f.write(f'Customer {k} served by depot {i} via satellite {j} \\\\\n')

    f.write('\\end{tabular}\n')
    f.write('\\end{latin}\n')

print(f"✅ Full output saved to: {full_output_path}") 
