import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 提取自你真实日志的训练集数据
# ==========================================
train_epochs = [
    0.003, 0.017, 0.035, 0.053, 0.071, 0.088, 0.106, 0.124, 0.142, 0.160,
    0.302, 0.444, 0.604, 0.817, 1.010, 1.206, 1.401, 1.615, 1.810, 2.039,
    2.252, 2.448, 2.643, 2.803, 2.981
]
train_losses = [
    1.921, 1.900, 1.793, 1.471, 1.189, 0.900, 0.637, 0.506, 0.421, 0.381,
    0.298, 0.294, 0.275, 0.256, 0.257, 0.254, 0.236, 0.249, 0.227, 0.208,
    0.186, 0.210, 0.215, 0.206, 0.201
]

# ==========================================
# 2. 全局样式设置 (打造学术顶刊质感)
# ==========================================
# 启用中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5 # 边框加粗一点，更有质感

# ==========================================
# 3. 绘制专属 Training Loss 曲线
# ==========================================
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300) # 黄金比例，高度稍扁，适合插在论文文字之间

# 画线：使用深邃的科技蓝，稍微加粗，配合淡淡的阴影填充效果
ax.plot(train_epochs, train_losses, color='#2b6cb0', linewidth=2.5, alpha=0.9, label='Training Loss')

# 在曲线下方增加极淡的填充色，这是当前顶级 AI 论文最爱用的高级排版手法！
ax.fill_between(train_epochs, train_losses, 0, color='#2b6cb0', alpha=0.1)

# 设置轴标签和标题
ax.set_xlabel('迭代轮次 (Epochs)', fontweight='bold', labelpad=10)
ax.set_ylabel('交叉熵损失值', fontweight='bold', labelpad=10)
ax.set_title('图 4.2 训练集损失函数', pad=15, fontweight='bold', fontsize=14)

# 细节优化：网格线设为极淡的虚线，不喧宾夺主
ax.grid(True, linestyle='--', linewidth=0.5, color='#CBD5E0')

# 设置严谨的坐标轴范围
ax.set_xlim(0, 3.0)
ax.set_ylim(0, 2.0)

# ==========================================
# 4. 增加“学术亮点”注释框 (逼格拉满)
# ==========================================
# 标注那个断崖式下降的拐点，引导答辩老师看懂图


# 保存图片
plt.tight_layout()
output_filename = 'Training_Loss_Only_Pro.png'
plt.savefig(output_filename)
print(f"✅ 极简风学术级 Training Loss 图已保存为: {output_filename}")