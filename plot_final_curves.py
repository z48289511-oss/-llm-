import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ==========================================
# 1. 提取自你真实日志的验证集准确率数据
# ==========================================
# 验证集数据：在 Epoch 1.0, 2.0, 3.0(接近3) 时的记录
val_epochs = [1.0, 2.0, 2.992]
val_accs = [0.9238, 0.9246, 0.9248]

# ==========================================
# 2. 全局样式设置 (打造学术顶刊质感)
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5 # 边框加粗一点，更有质感

# ==========================================
# 3. 绘制专属 Validation Accuracy 曲线
# ==========================================
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300) # 黄金比例

# 画线：使用学术绿，线条加粗，配合方形标记点(marker='s')
ax.plot(val_epochs, val_accs, color='#2ca02c', linewidth=2.5, marker='s', markersize=8, label='Validation Accuracy')



# 设置轴标签和标题
ax.set_xlabel('迭代轮次 (Epochs)', fontweight='bold', labelpad=10)
ax.set_ylabel('准确率 (Accuracy)', fontweight='bold', labelpad=10)
ax.set_title('图 4.5 验证集准确率', pad=15, fontweight='bold', fontsize=14)

# 细节优化：横向网格线设为极淡的虚线
ax.grid(True, axis='y', linestyle='--', linewidth=0.5, color='#CBD5E0')

# 设置严谨的坐标轴范围
ax.set_xlim(0.8, 3.2) # 让数据点居中显示
# 聚焦高分段，让 92.38% 到 92.48% 的上升趋势清晰可见
ax.set_ylim(0.92, 0.93) 

# 将 Y 轴转换为百分比格式显示 (如 92.0%, 92.5%)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))

# 保存图片
plt.tight_layout()
output_filename = 'Data/Validation_Accuracy_Final.png'
plt.savefig(output_filename)
print(f"✅ 极简风学术级 Validation Accuracy 图已保存为: {output_filename}")