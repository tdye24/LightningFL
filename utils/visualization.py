import torch
import time
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patheffects as PathEffects
from mpl_toolkits.mplot3d import Axes3D
from models.fedmc.cifar10.CIFAR10 import CIFAR10
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from data.cifar10.cifar10 import get_cifar10_dataLoaders


def plot_scatter(x, colors, filename='fedmc'):
    colors = np.array(colors).flatten()
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))
    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    # if filename == 'fedmc':
    # ax.set_title('FedMC')
    # elif filename == 'fedmc_woat':
    # ax.set_title('FedMC w/o. ET')
    sc = ax.scatter(x[:, 0], x[:, 1], s=8, c=palette[colors.astype(np.int)], cmap=plt.cm.get_cmap('Paired'))

    # plt.xlim(-100, 100)
    # plt.ylim(-100, 100)
    ax.axis('off')
    ax.axis('tight')  # add the labels for each digit corresponding to the label
    txts = []
    # for i in range(num_classes):
    #     # Position of each label at median of data points.
    #     class_text = str(i)
    #     xtext, ytext = np.median(x[colors == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, str(class_text), fontsize=24)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)
    plt.savefig(f'./{filename}.pdf')
    plt.show()

    return f, ax, sc, txts


class Model(CIFAR10):
    def __init__(self):
        super(Model, self).__init__(dropout=[0.5, 0.5, 0.9, 0.9, 0.9, 0.5])

    def get_embedding(self, x):
        # 50x3x32x32
        shared_embedding = self.shared_encoder(x)
        return shared_embedding


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model = Model()
model = model.cuda()

model.load_state_dict(torch.load('../saved_models/fedmc_woat/cifar10_cifar10/model.pkl'))
model.eval()

users, trainLoaders, testLoaders = get_cifar10_dataLoaders(batch_size=100,
                                                           train_transform=train_transform,
                                                           test_transform=test_transform)
EMBEDDINGS = []
LABELS = []
with torch.no_grad():
    for user, loader in testLoaders.items():
        for step, (data, labels) in enumerate(loader):
            data, labels = data.cuda(), labels.cuda()
            embedding = model.get_embedding(data)
            EMBEDDINGS.extend(embedding.cpu().numpy().tolist())
            LABELS.extend(labels.cpu().numpy().tolist())
        print(f"user: {user}, # of samples: {len(EMBEDDINGS)}")
# EMBEDDINGS = pd.DataFrame(EMBEDDINGS).sample(frac=1.0, random_state=10).reset_index(drop=True)
# LABELS = pd.DataFrame(LABELS).sample(frac=1.0, random_state=10).reset_index(drop=True)
# EMBEDDINGS = EMBEDDINGS.values
# LABELS = LABELS.values
# with open('./fedmc_woat-data.tsv', 'w') as f:
#     for i in range(len(EMBEDDINGS)):
#         for element in EMBEDDINGS[i]:
#             f.write(f'{element}\t')
#         f.write('\n')
#     f.close()
# with open('./fedmc_woat-meta.tsv', 'w') as f:
#     for i in range(len(LABELS)):
#         f.write(f'{LABELS[i][0]}\n')
#     f.close()
# X = pd.DataFrame(EMBEDDINGS)
# Y = pd.DataFrame(LABELS)
# X = X.sample(frac=0.01, random_state=10).reset_index(drop=True)
# Y = Y.sample(frac=0.01, random_state=10).reset_index(drop=True)
# pca = PCA(n_components=50)
# pca.fit(X=X)
# X = pca.fit_transform(X)
# print(X.shape)
# time_start = time.time()
# tsne = TSNE(perplexity=25, n_iter=1000, learning_rate=10)
# tsne_results = tsne.fit_transform(X.values)
# print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
#
# plot_scatter(tsne_results, Y, filename='fedmc_woat-final')


# time_start = time.time()
# tsne = TSNE(n_components=3)
# X = pd.DataFrame(EMBEDDINGS)
# Y = pd.DataFrame(LABELS)
# X = X.sample(frac=0.01, random_state=10).reset_index(drop=True)
# Y = Y.sample(frac=0.01, random_state=10).reset_index(drop=True)
# tsne_results = tsne.fit_transform(X.values)
# print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
# fig = plt.figure()
# ax = Axes3D(fig)
# colors = np.array(Y).flatten()
# # choose a color palette with seaborn.
# num_classes = len(np.unique(colors))
# palette = np.array(sns.color_palette("hls", num_classes))
# ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], s=4, c=palette[colors.astype(np.int)], cmap=plt.cm.get_cmap('Paired'))
# plt.xticks([])
# plt.yticks([])
# # plt.axis('off')
# # ax.set_title('FedMC')
# # ax.set_title('FedMC w/o. ET')
# plt.savefig('./FedMC-3D.pdf')
# plt.show()

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

X = EMBEDDINGS
y = np.array(LABELS).flatten()
# first reduce dimensionality before feeding to t-sne
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)
# randomly sample data to run quickly
rows = np.arange(10000)
np.random.shuffle(rows)
n_select = 10000
# reduce dimensionality with t-sne
tsne = TSNE(n_components=2, verbose=1, perplexity=25, n_iter=1000, learning_rate=10)
tsne_results = tsne.fit_transform(X_pca[rows[:n_select], :])
# visualize
df_tsne = pd.DataFrame(tsne_results, columns=['X', 'Y'])
df_tsne['label'] = y[rows[:n_select]]
sns.lmplot(x='X', y='Y', data=df_tsne, hue='label', fit_reg=False, legend_out=True, scatter_kws={"s": 10}, legend=False)
plt.show()
