import time
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.datasets import KarateClub
import matplotlib.pyplot as plt
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn_extra.cluster import KMedoids  # K-Medoids
from sklearn.metrics.cluster import adjusted_rand_score  # ARI
import random #train_mask生成
import sys



class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(34, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, 2)

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
def draw_karateclub(draw=True):
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
def draw_embedded_vector(Y, colorlist=None):
    if colorlist is None:
        colorlist = draw_karateclub(False)

    fig, ax = plt.subplots()
    for i in range(len(colorlist)):
        ax.annotate(str(i), (Y[i, 0], Y[i, 1]))
        if colorlist[i] == 0:
            ax.scatter(Y[i, 0], Y[i, 1], c="b")
            pass
        elif colorlist[i] == 1:
            ax.scatter(Y[i, 0], Y[i, 1], c="r")
            pass

    plt.show()

"""
全てのノードを使って学習
"""
def train_all(data, model, criterion, optimizer):
    #正解ラベル取得
    true_label = draw_karateclub(draw=False)
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    loss = criterion(out, true_label)
    loss.backward()
    optimizer.step()
    return loss, h


"""
ランダムに選択された一部のノード情報をのみを使って学習
data : グラフ情報
model : GCNモデル
train_mask : 学習に使うノードリスト 
"""
def train_at_partial(data, model, criterion, optimizer):
    #正解ラベル取得
    true_label = draw_karateclub(draw=False)
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], true_label[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, h

"""
num_trainの数だけ1、それ以外は0の配列を返す。
arraySize : 配列の大きさ
num_train : 1を格納する数
"""
def get_train_mask(arraySize:int,num_train=4):

    #例外処理
    if(num_train>arraySize):
        print("指定した配列の大きさより、1の格納数の方が大きいです")
        sys.exit("ERROR!!def get_tarin_mask : num_train>arraySize!!!")

    false_size = arraySize-num_train
    mask = [False]*false_size + [True]*num_train
    random.shuffle(mask)
    print(f"Trained_node_number_is : {get_index(mask,True)}")
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


def execution(TRAIN_ALL=False,DEFAULT = True, NUM_TRAIN = 4,EPOCH = 30, VIEW_TRAIN= False):

    #Karateclub正解ラベル
    TRUE_LABEL = draw_karateclub(False).detach().numpy()

    # geometric.datasetのkarateclubのグラフインスタンスを取得
    DATA = KarateClub()[0]

    # GCNモデルの初期化
    model = GCN()

    #学習情報出力
    #info(model,DATA)

    #初期化
    h = None
    loss = None
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if not(TRAIN_ALL) and not(DEFAULT):
        train_mask = get_train_mask(arraySize=len(DATA.x),num_train=NUM_TRAIN)
        DATA.train_mask = train_mask

    for epoch in range(EPOCH+1):
        # 全てのラベルで学習させる
        if TRAIN_ALL:
            loss, h = train_all(DATA, model, criterion, optimizer)
        #一部のラベルでで学習  
        else:
            loss, h = train_at_partial(DATA, model, criterion, optimizer)
            
        # 学習状況の表示
        if VIEW_TRAIN:
            if epoch % 10 == 0:
                print()
                visualize_embedding(h, color=TRUE_LABEL, epoch=epoch, loss=loss)
                time.sleep(0.1)
    return h, loss


def exec_to_kmedoids(times=50, TRAIN_ALL=False, DEFAULT=True, NUM_TRAIN=4, EPOCH=30, VIEW_TRAIN=False, N_CLUSTER=2):
    true_label = draw_karateclub(False)
    ARI_list = []
    max_EVM = None
    min_EVM = None
    max_pred = None
    four_cluss = None
    min_pred = None
    max_ari = -100
    min_ari = 100

    for i in range(times):
        print(f"==========================={i+1}回目============================")
        #GCN実行
        h,_ = execution(TRAIN_ALL, DEFAULT, NUM_TRAIN, EPOCH, VIEW_TRAIN)

        #ndarray化
        embedded_vector_matrix = h.detach().numpy()

        #正解ラベルによる埋め込みベクトルの可視化
        draw_embedded_vector(embedded_vector_matrix)

        #kmedoids実行
        pred = KMedoids(n_clusters=2, random_state=0).fit_predict(embedded_vector_matrix)

        #kmedoidsによるクラスタ結果による埋め込みベクトルの可視化
        draw_embedded_vector(embedded_vector_matrix, colorlist=pred)

        #ari算出
        ari = adjusted_rand_score(true_label, pred)
        ARI_list.append(ari)
        print(f"{i+1}回目 ARI : {ari}")
        print("")

        if ari>max_ari :
            max_ari = ari
            max_EVM = h
            max_pred = pred
            four_cluss = pred = KMedoids(n_clusters=N_CLUSTER, random_state=0).fit_predict(embedded_vector_matrix)

            
        elif min_ari>ari :
            min_ari = ari
            min_EVM = h
            min_pred = pred



    print(f"最大ARI({get_index(ARI_list,max_ari,READ = True)}回目実行) : {max_ari}")
    print(f"最小ARI({get_index(ARI_list,min_ari,READ = True)}回目実行) : {min_ari}")

    return ARI_list, max_EVM, min_EVM, max_pred, min_pred,four_cluss

    

     
