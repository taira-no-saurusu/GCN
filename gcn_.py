import time
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
import matplotlib.pyplot as plt
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn_extra.cluster import KMedoids  # K-Medoids
from sklearn.cluster import KMeans # Kmeans
from sklearn.metrics.cluster import adjusted_rand_score  # ARI
from statistics import stdev #標準偏差
import random #train_mask生成
import sys
import csv
import torch_geometric.utils.convert
from sklearn.decomposition import PCA


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        #torch.manual_seed(1234)
        self.conv1 = GCNConv(115, 70)
        self.conv2 = GCNConv(70, 4)
        self.conv3 = GCNConv(4, 4)
        self.classifier = Linear(4,12)

    """
    h   : 埋め込み結果
    out : 分類結果
    """
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()

        out = self.classifier(h)

        return out, h
"""
Tensor単位行列を得る
"""
def get_identity_matrix(size:int):
    matrix = []
    for i in range(size):
        l = [0.]*size
        l[i] = 1.
        matrix.append(l)
    return torch.Tensor(matrix)


"""
networkxのグラフインスタンスを生成
"""
def generate_Graph(name):
    G = nx.Graph()

    if name == "football":
        f = open('football.txt', 'r')
        datalist = f.readlines()

        for l in datalist:
            list = l.strip('\n').split(",")
            if list[2] == '0':
                continue
            else:
                G.add_edge(list[0], list[1], weight=int(list[2]))
                G.add_edge(list[1], list[0], weight=int(list[2]))

    elif name == "polbooks":
        f = open('polbooks.txt', 'r')
        datalist = f.readlines()

        for l in datalist:
            list = l.strip('\n').split(",")
            if list[2] == '0':
                continue
            else:
                G.add_edge(list[0], list[1], weight=int(list[2]))
                G.add_edge(list[1], list[0], weight=int(list[2]))

        pass
    else:
        sys.exit("グラフ生成の引数の名前がおかしいです")
    G = from_networkx(G)
    G.x = get_identity_matrix(G.num_nodes)

    return G


"""
label_listをテキストファイルから読み込んで返す
"""
def get_label_list(name):
    label_list = []
    if name == "football":
        f = open('label_football.txt', 'r')
        datalist = f.readlines()

        for l in datalist:
           label_list.append(int(l.rstrip("\n")))

    elif name == "polbooks":
        f = open('label_polbooks.txt', 'r')
        datalist = f.readlines()

        for l in datalist:
           label_list.append(int(l.rstrip("\n")))

    else:
        sys.exit("グラフ生成の引数の名前がおかしいです")

    return label_list


"""
csvファイルを書き出す
"""
def read_csv(filename):
    with open(filename, encoding='utf8', newline='') as f:
        csvreader = csv.reader(f)
        list = []
        for row in csvreader:
            for i in row:
                list.append(int(i))

    return torch.Tensor(list).to(torch.long)

"""
学習状況を表示
h     : 埋め込み空間([ノード数,2次元固定])
color : 正解ラベルのリスト
epoch : 呼び出し時のエポック数
loss  : 呼び出し時の損失
"""
def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch},Loss:{loss.item():.4f}', fontsize=16)
    plt.show()

"""
networkxのGraphクラスからkarateclubを描画すると共に正解ラベルとしてラベルのテンソルを返す
G: nwtworkグラフインスタンス
draw : 描画するかどうか
"""
def draw_karateclub(draw=False):
    G = nx.karate_club_graph()
    pos = nx.spring_layout(G)

    color_list = torch.Tensor([0 if G.nodes[i]["club"] ==
                               "Mr. Hi" else 1 for i in G.nodes()]).to(torch.long)
    # 色別に描画
    if draw:
        nx.draw_networkx(G, pos, node_color=color_list, cmap=plt.cm.RdYlBu)
        plt.show()
    return color_list

"""
colorlist に基づいてベクトルデータを可視化
colorlist が与えられない場合、karateclubの正解ラベルに基づいて可視化
"""
def draw_embedded_vector(Y, colorlist):

    fig, ax = plt.subplots()
    for i in range(len(colorlist)):
        ax.annotate(str(i), (Y[i, 0], Y[i, 1]))
        if colorlist[i] == 0:
            ax.scatter(Y[i, 0], Y[i, 1], c="r")
            pass
        elif colorlist[i] == 1:
            ax.scatter(Y[i, 0], Y[i, 1], c="b")
            pass
        elif colorlist[i] == 2:
            ax.scatter(Y[i, 0], Y[i, 1], c="y")

        elif colorlist[i] == 3:
            ax.scatter(Y[i, 0], Y[i, 1], c="g")

        elif colorlist[i] == 4:
            ax.scatter(Y[i, 0], Y[i, 1], c="c")

        elif colorlist[i] == 5:
            ax.scatter(Y[i, 0], Y[i, 1], c="m")

        elif colorlist[i] == 6:
            ax.scatter(Y[i, 0], Y[i, 1], c="coral")

        elif colorlist[i] == 7:
            ax.scatter(Y[i, 0], Y[i, 1], c="lightpink")

        elif colorlist[i] == 8:
            ax.scatter(Y[i, 0], Y[i, 1], c="skyblue")

        elif colorlist[i] == 9:
            ax.scatter(Y[i, 0], Y[i, 1], c="darkgray")

        elif colorlist[i] == 10:
            ax.scatter(Y[i, 0], Y[i, 1], c="goldenrod")

        elif colorlist[i] == 11:
            ax.scatter(Y[i, 0], Y[i, 1], c="lightgreen")

    plt.show()


"""
ランダムに選択された一部のノード情報をのみを使って学習
data : グラフ情報
model : GCNモデル
train_mask : 学習に使うノードリスト 
"""
def train_at_partial(data, model, criterion, optimizer,TRUE_LABEL):
   
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], torch.Tensor(TRUE_LABEL).to(torch.long)[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, h

"""
num_trainの数だけTrue、それ以外はFalseの配列を返す。
arraySize : 配列の大きさ
num_train : Trueを格納する数
"""
def get_random_train_mask(arraySize,num_train,TRUE_LABEL):
    mask = [False]*arraySize
    # 例外処理
    if (num_train > arraySize):
        print("指定した配列の大きさより、1の格納数の方が大きいです")
        sys.exit("ERROR!!def get_tarin_mask : num_train>arraySize!!!")
    if 11 in TRUE_LABEL:  # footballのラベル生成
        list = [[i for i, x in enumerate(TRUE_LABEL) if x == 0], [i for i, x in enumerate(TRUE_LABEL) if x == 1], [
            i for i, x in enumerate(TRUE_LABEL) if x == 2], [i for i, x in enumerate(TRUE_LABEL) if x == 3], [i for i, x in enumerate(TRUE_LABEL) if x == 4], [i for i, x in enumerate(TRUE_LABEL) if x == 5], [i for i, x in enumerate(TRUE_LABEL) if x == 6], [i for i, x in enumerate(TRUE_LABEL) if x == 7], [i for i, x in enumerate(TRUE_LABEL) if x == 8], [i for i, x in enumerate(TRUE_LABEL) if x == 9], [i for i, x in enumerate(TRUE_LABEL) if x == 10], [i for i, x in enumerate(TRUE_LABEL) if x == 11]]
        for i in range(num_train):
            if len(list[i % 12]) == 0:
                continue
            index = random.choice(list[i % 12])
            mask[index] = True
            list[i % 12].remove(index)

    else:  # Karateclubの正解データによってクラスタリング
        list = [[i for i, x in enumerate(TRUE_LABEL) if x == 0], [
            i for i, x in enumerate(TRUE_LABEL) if x == 1], [i for i, x in enumerate(TRUE_LABEL) if x == 2]]
        for i in range(num_train):
            if len(list[i % 3]) == 0:
                continue
            index = random.choice(list[i % 3])
            mask[index] = True
            list[i % 3].remove(index)

    #print(f"Trained_node_number_is : {get_index(mask,True)}")
    mask = torch.BoolTensor(mask)
    
    return mask


"""
リストの中に含まれるTrueの添字をリストで返す
"""
def get_index(mask,tar,READ=False):
    if READ:
        return [i+1 for i, x in enumerate(mask) if x == tar]
    else :
        return [i for i, x in enumerate(mask) if x == tar]

    
"""
情報出力
"""
def info(model,data):
    print("ーーーーGCNモデルーーーー")
    print(model)
    print("ーーーーーーーーーーーーー")
    _, h = model(data.x, data.edge_index)
    print(f'Embedding shape:{list(h.shape)}')
    print(f"ノード数 : {list(h.shape)[0]} ノード")
    print(f"埋め込み次元数 : {list(h.shape)[1]} 次元")
    print(f"教師データ数(損失算出に使用) : {len(data.train_mask[data.train_mask])} ノード")


def execution( NUM_TRAIN,EPOCH, VIEW_TRAIN,TRUE_LABEL,DATA):

    # GCNモデルの初期化
    model = GCN()

    #学習情報出力
    info(model,DATA)

    #初期化
    h = None
    loss = None
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  
    train_mask = get_random_train_mask(len(DATA.x),NUM_TRAIN,TRUE_LABEL)
    DATA.train_mask = train_mask

    for epoch in range(EPOCH+1):
       
        loss, h = train_at_partial(DATA, model, criterion, optimizer,TRUE_LABEL)
            
        # 学習状況の表示
        if VIEW_TRAIN:
            if epoch % 10 == 0:
                print()
                visualize_embedding(h, color=TRUE_LABEL, epoch=epoch, loss=loss)
                time.sleep(0.1)
    return h, loss

"""
GCNによる埋め込みからkmedoidsによるクラスタリング、ARI算出までを任意の回数行う

    times        : 実施回数
    TRAIN_ALL    : 全てのノード情報を用いて学習するか（通常は一部のノード情報を用いて学習）(bool)
    DEFAULT      : karateclubインスタンスに設定されたデフォルトの一部のノード情報で学習するか(bool)
    NUM_TRAIN    : 学習に利用するノード数(int)
    EPOCH        : エポック(int)
    VIEW_TRAIN   : 学習状況を表示するか(bool)
    N_CLUSTER    : Kmedoidsのクラスタ数(bool)
    TRUE_LABEL   : 全てのノードの正解ラベル情報(list)   
"""
def exec_to_kmedoids(times, NUM_TRAIN, EPOCH, VIEW_TRAIN,VIEW_CLUSTERING, N_CLUSTER, TRUE_LABEL,METHOD,DATA=None):
    true_label = draw_karateclub(False)
    ARI_list = []
    max_EVM = None
    min_EVM = None
    max_pred = None
    min_pred = None
    max_ari = -100
    min_ari = 100

    #print(f"{METHOD}で実行しまず")

    for i in range(times):

        """
        下の表示は一時的に削除
        """
        #print(f"==========================={i+1}回目============================")
        #GCN実行
        h, _ = execution( NUM_TRAIN, EPOCH, VIEW_TRAIN, TRUE_LABEL,DATA)

        #ndarray化
        embedded_vector_matrix = h.detach().numpy()

        pred = None
        #クラスタリング実行
        if METHOD=="kmedoids":
            pred = KMedoids(n_clusters=N_CLUSTER, random_state=0).fit_predict(embedded_vector_matrix)

        elif METHOD=="kmeans":
            pred = KMeans(n_clusters = N_CLUSTER,random_state=0).fit_predict(embedded_vector_matrix)
        else :
            sys.exit("kmedoidsでもkmeansでもありません")
        #ari算出
        ari = adjusted_rand_score(TRUE_LABEL, pred)
        ARI_list.append(ari)
        if VIEW_CLUSTERING:
            print(
                f"==========================={i+1}回目============================")
            # 正解ラベルによる埋め込みベクトルの可視化
            print(f"埋め込み結果")
            draw_embedded_vector(embedded_vector_matrix, TRUE_LABEL)

            # kmedoidsによるクラスタ結果による埋め込みベクトルの可視化
            print(f"クラスタリング結果")
            draw_embedded_vector(embedded_vector_matrix, colorlist=pred)

            print(f"{i+1}回目 ARI : {ari}")


        if ari>max_ari :
            max_ari = ari
            max_EVM = h
            max_pred = pred
           
            
        if min_ari>ari :
            min_ari = ari
            min_EVM = h
            min_pred = pred

    print(f"最大ARI({get_index(ARI_list,max_ari,READ = True)}回目実行) : {max_ari}")
    print(f"最小ARI({get_index(ARI_list,min_ari,READ = True)}回目実行) : {min_ari}")
    print(f"平均ARI : {np.mean(ARI_list)}")
    print(f"標準偏差 : {stdev(ARI_list)}")

    return ARI_list, max_EVM, min_EVM, max_pred, min_pred

#グラフ埋め込みを実行して、埋め込みベクトルを主成分分析を行い、固有値を返す
def exec_to_pca(NUM_TRAIN, EPOCH, VIEW_TRAIN, TRUE_LABEL,DATA=None):
    #GCN実行
    h, _ = execution( NUM_TRAIN, EPOCH, VIEW_TRAIN, TRUE_LABEL,DATA)
    #ndarray化
    embedded_vector_matrix = h.detach().numpy()
    #主成分分析
    pca = PCA()
    pca.fit(embedded_vector_matrix)
    #固有値
    eigenvalues = pca.explained_variance_
    return eigenvalues
    

     
