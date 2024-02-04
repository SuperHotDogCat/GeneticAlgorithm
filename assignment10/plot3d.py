import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return (x**6-6*x**4+9*x**2+x) + (y**6-6*y**4+9*y**2+y)

# グリッドの作成
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 3Dプロット
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# 軸ラベルの設定
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x, y)')

# LaTeX表記のタイトルの設定
plt.title(r'3D Plot of $f(x, y) = x^6 - 6x^4 + 9x^2 + x + y^6 - 6y^4 + 9y^2 + y$')

# 画像ファイルとして保存
plt.savefig('3d_plot.png', dpi=300)

plt.show()