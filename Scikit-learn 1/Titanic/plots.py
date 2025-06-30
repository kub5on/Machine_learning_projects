from main import train_y
import matplotlib.pyplot as plt

values = train_y.value_counts()
labels = ['not survived', 'survived']
plt.bar([0, 1], values)
plt.xticks([0, 1], labels)
plt.suptitle('Survivors')
plt.savefig('survivors_plot')
plt.show()
