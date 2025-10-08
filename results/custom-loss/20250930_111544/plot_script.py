import re
import matplotlib.pyplot as plt

log_file = '/proj/aurora/Smartness/checkpoints-mixed-training/Custom-CQI-12-epochs/20250930_111544/20250930_111544.log'

# Listas para armazenar valores
loss = []
student_loss_sum = []
teacher_loss_sum = []
student_entropy = []
CQI = []

# Regex para capturar apenas os campos desejados
pattern = re.compile(
    r'loss:\s*(\d+\.\d+).*?student_loss_sum:\s*(\d+\.\d+).*?teacher_loss_sum:\s*(\d+\.\d+).*?student_entropy:\s*(\d+\.\d+).*?CQI:\s*(\d+\.\d+)'
)

# Ler e extrair valores do log
with open(log_file, 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            loss.append(float(match.group(1)))
            student_loss_sum.append(float(match.group(2)))
            teacher_loss_sum.append(float(match.group(3)))
            student_entropy.append(float(match.group(4)))
            CQI.append(float(match.group(5)))

# Downsampling - a cada 10 passos
step = 10
loss = loss[::step]
student_loss_sum = student_loss_sum[::step]
teacher_loss_sum = teacher_loss_sum[::step]
student_entropy = student_entropy[::step]
CQI = CQI[::step]

# FUncao de suavizacao
def moving_average(x, w=5):
    """Retorna a média móvel da lista x com janela w"""
    return [sum(x[i-w:i])/w if i >= w else sum(x[:i+1])/(i+1) for i in range(len(x))]

# Suavizar curvas
loss_smooth = moving_average(loss, w=5)
student_loss_sum_smooth = moving_average(student_loss_sum, w=5)
teacher_loss_sum_smooth = moving_average(teacher_loss_sum, w=5)
student_entropy_smooth = moving_average(student_entropy, w=5)
CQI_smooth = moving_average(CQI, w=5)

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

data = [loss_smooth, student_loss_sum_smooth, teacher_loss_sum_smooth, student_entropy_smooth, CQI_smooth]
titles = ['loss', 'student_loss_sum', 'teacher_loss_sum', 'student_entropy', 'CQI']

for i, (d, title) in enumerate(zip(data, titles)):
    axes[i].plot(d, label=title, color='tab:blue')
    axes[i].set_title(title)
    axes[i].set_xlabel('Iterações (downsampled)')
    axes[i].set_ylabel('Valor')
    axes[i].grid(True)
    axes[i].legend()

# Remover subplot vazio
fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()
