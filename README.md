# corporationClassifier
この `README.md` には、各コードの実行結果、各コードの説明を記載する。
実行ファイルは `test_corpolation_classifier.py`で、 データ作成は `corporation_bert.py`、 結果の画像作成・出力は `plot_solutions.py` で、行う。

## 項目 [Contents]

1. [データの作成（csvファイルから文章特徴量に変換するまでの流れ） : `corporation_bert.py`](#ID_1)
	1. [コードの説明](#ID_1-1)
	2. [コードの実行結果](#ID_1-2)

2. [実験方法 : `test_corpolation_classifier.py`](#ID_2)
	1. [コードの説明](#ID_2-1)
	2. [コードの実行結果](#ID_2-2)

3. [実験結果の確認方法：`plot_solutions.py`](#ID_3)
	1. [コードの説明](#ID_3-1)
	2. [コードの実行結果](#ID_3-2)



<a id="ID_1"></a>

## データの作成：`corporation_bert.py`

<a id="ID_1-1"></a>
### コードの説明
事前に用意しているcsvファイル(bert/data/~.csv)を読み込み、文章特徴量に変換する。<br>

はじめに、csvファイルを読み込み、title,description,classに分ける。
ここで、京都大学の黒橋・河原研究室が公開している「BERT日本語Pretrainedモデル」を利用する。
<br>
[modelの作成方法](https://github.com/yagays/pytorch_bert_japanese)
```preprocess.py
if __name__ ==  '__main__':

    df = pd.read_csv("../../corporationClassifier/data/data_all.csv") #load csv file

    bert = BertWithJumanModel("../model/Japanese_L-12_H-768_A-12_E-30_BPE") #モデルの読み込み

    title = df['title'] #タイトルの全データ
    description = df['description'] #説明の全データ
    y = df['class'] #0 or 1
    data_len = title.shape[0]

    out_title = [] #タイトルの全データの特徴量が入る
    out_description = [] #内容の全データの特徴量が入る
```

<br>

次にtitle,descriptionそれぞれに対してbertを用いて特徴量を作成する。この時の特徴ベクトルは(データ数、特徴ベクトル(768))となっている。
```preprocess.py
#データ数分forループ
  for i in range(data_len):

      parts_title = [] #タイトルごとの特徴量が入る
      parts_description = [] #内容ごとの特徴量が入る

      #-------------------------
      #タイトルに関して特徴量を抽出
      #-------------------------
      parts_title = bert.get_sentence_embedding(title[i])
      #-------------------------
      #説明に関して特徴量を抽出
      #-------------------------
      parts_description = bert.get_sentence_embedding(description[i])
      #-------------------------

      out_title.append(parts_title) #タイトルごとに配列に追加
      out_description.append(parts_description) #説明ごとに配列に追加

```
<br>
最後にtitle,descriptionの特徴量を結合することにより、1536次元のベクトル（変数X）を作成する。

```preprocess.py
for i in range(len(out_title)):

      feature_stack = np.concatenate([np.array(out_title[i]),np.array(out_description[i])])

      if i == 0:
          X = feature_stack[np.newaxis]
      else:
          X = np.append(X,feature_stack[np.newaxis],axis=0)
Y = np.array(y) #labelをnumpyに変換

#-------------------------
#-------------------------
#------データをpickleで保存
with open('../data/out/data_x.pickle','wb') as f:
    pickle.dump(X,f)
with open('../data/out/data_y.pickle','wb') as f:
    pickle.dump(Y,f)
```
<a id="ID_1-2"></a>

### コードの実験結果
corporationClassifier/data/outに変数X,Yがそれぞれdata_x.pickle,data_y.pickleとして保存される。

<a id="ID_2"></a>

## 実験方法 : `test_corpolation_classifier.py`

<a id="ID_2-1"></a>

### コードの説明
Epoch数とバッチサイズを決め、先ほど用意したデータを読み込む。
```test_corpolation_classifier.py
#Epoch数
nEpo = 1000
# バッチデータ数
batchSize = 30

#======================================
# データ読み込み
X = pickle.load(open("../data/out/data_x.pickle","rb"))
Y = pickle.load(open("../data/out/data_y.pickle","rb"))
```
読み込んだデータをtrain,testに分割する。ここでは、train:test = 8:2としている。引数のtest_sizeによって割合が変化する。
```test_corpolation_classifier.py
(train_x,test_x,train_y,test_y) = train_test_split(X,Y,test_size = 0.2,random_state=0)
```
学習とテストに使用するプレイスホルダーを用意し、今回用いるネットワークを作成する。baseNNの引数であるratesをもとにfc層でdropoutを行っている。
```test_corpolation_classifier.py
x_train = tf.placeholder(tf.float32,shape=[None,400])
x_label = tf.placeholder(tf.float32,shape=[None,1])


x_test = tf.placeholder(tf.float32,shape=[None,400])
x_test_label = tf.placeholder(tf.float32,shape=[None,1])

## build model
train_pred = baseNN(x_train,rates=[0.2,0.5])

test_preds = baseNN(x_test,reuse=True,isTrain=False)

```
#### ニューラルネットワークのプログラム
```test_corpolation_classifier.py
def baseNN(x,reuse=False,isTrain=True,rates=[0.0,0.0]):
    node = [1536,768,300,100,1]
    layerNum = len(node)-1
    f_size = 3

    with tf.variable_scope('baseCNN') as scope:
        if reuse:
            scope.reuse_variables()

        W = [weight_variable("convW{}".format(i),[node[i],node[i+1]]) for i in range(layerNum)]
        B = [bias_variable("convB{}".format(i),[node[i+1]]) for i in range(layerNum)]
        fc1 = fc_relu(x,W[0],B[0],rates[1])
        #fc = [fc_relu(fc,W[i+1],B[i+1]) for i in range(layerNum-1)]
        fc2 = fc_relu(fc1,W[1],B[1])
        fc3 = fc_relu(fc2,W[2],B[2])
        fc3 = tf.matmul(fc3,W[3]) + B[3]
    return fc3
```
<a id="ID_2-2"></a>
### コードの実行結果
出力の最後にtrain,testのconfusion matrixが表示される。
<br>
[confusion matrixの参考](https://qiita.com/TsutomuNakamura/items/a1a6a02cb9bb0dcbb37f)
<br>
```
train confusion matrix:
[[18  0]
 [ 0 12]]
test confusion matrix :
[[18  3]
 [ 3 21]]
```
test confusion matrix
![プレゼンテーション1＿_page-0001](https://user-images.githubusercontent.com/44080085/63784830-36fefa00-c92a-11e9-9b72-1c3ff815748f.jpg)
corpolationClassifier/data/outにtestしたデータが[test_result.csv](https://github.com/hhachiya/corporationClassifier/blob/master/data/out/test_result.csv)として保存される。
```test_result.csv
title	description	true class	predict class
176	簡単にオリジナルステッカー印刷 | 1000枚3,550円～低価格で製作	ステッカー・シール・ラベル印刷！自分だけのカスタムステッカー作成はステッカージャパンで！ 24時間365日注文受付・送料無料・低価格保証・豊富な素材・製品: アート紙ステッカー, ユポステッカー, 透明ステッカー, 屋外用ステッカー。	1	1
177	ステッカー・シールラベル印刷が激安 | 印刷通販【メガプリント】	メガプリントのステッカー印刷・シール印刷は写真も綺麗な高品質オフセット印刷なのに格安・激安で作成することが可能です。 ... カットの仕方も格安で制作することが出来る四角ステッカーから台紙までカットできる全抜きステッカー、シート状にハーフカットを配置 …	0	0
178	カッティングステッカー、切り文字ステッカー作成の激安専門店！	カッティングステッカー、切り文字ステッカー作成の事ならお任せ下さい！1枚から激安で作成します！自作では難しい、細かいデザインも最新機械でオーダー作成可能です！用途に合わせたシートも豊富にご用意！当店のシートは簡単に貼り付け、剥がせます！	0	1
179	ステッカー印刷-小ロット対応｜印刷通販【デジタ】	ステッカー印刷のネット通販デジタは驚きの激安価格で高品質なステッカー印刷を実現。屋外用フルカラーステッカーを、より自由に激安価格で制作することが可能になりました。長期間の使用にも耐えられる耐候インクを使用し、車やバイクに貼っても使える …	0	0
```
また、実験で得られたtrain loss,test_loss,train auc,test aucがcorpolationClassifier/data/out/logにtest_corpolation_classifier_log.pickleとして保存される。これは、最後に結果をplotするときに用いる。

<a id="ID_3"></a>
## 実験結果の確認方法：`plot_solutions.py`
<a id="ID_3-1"></a>
### コードの説明
`test_corpolation_classifier.py`で作成されたtest_corpolation_classifier_log.pickleを読み込み変数dataに格納する。
```plot_solutions.py
with open("../data/out/log/test_corpolation_classifier_log.pickle","rb") as f:
      for i in range(dataN*dataType):
          data.append(pickle.load(f))

```
変数dataをplotする。ここで、train loss,test lossのｙ軸は`plt.ylim([0,2])`で統一化している。同様に、aucをplotする際には`plt.ylim([0.5,1.1])`で軸の大きさを定めている。

```plot_solutions.py
for i in range(len(data_name)):
    plt.close()
    if i == 1 or i == 4:
        continue
    plt.plot(range(ite),data[i])
    if data_name[i] == "train loss" or data_name[i] == "test loss":
        plt.ylim([0,2])
    else:
        plt.ylim([0.5,1.1])
    plt.xlabel("iteration")
    plt.ylabel(data_name[i])
    plt.savefig("../data/out/{0}.png".format(data_name[i]))
```
<a id="ID_3-2"></a>
### コードの実行結果
以下のようにtrain loss,test loss,train auc,test auc,train precision,test precision,train recall,test recallがcorpolationClassifier/data/out保存される。
<br>
<img src ="https://user-images.githubusercontent.com/44080085/71329767-0cd1e780-256c-11ea-9709-f25f1274a10a.png" width="300">

<img src ="https://user-images.githubusercontent.com/44080085/71329783-23783e80-256c-11ea-9856-fb88ebb8336e.png" width="300">

<img src ="https://user-images.githubusercontent.com/44080085/71329790-33901e00-256c-11ea-972e-9f095c2d2738.png" width="300">

<img src ="https://user-images.githubusercontent.com/44080085/71329797-3f7be000-256c-11ea-9349-9e29ad722108.png" width="300">

<img src ="https://user-images.githubusercontent.com/44080085/71329806-4dc9fc00-256c-11ea-879a-fad571703a3e.png" width="300">

<img src ="https://user-images.githubusercontent.com/44080085/71329817-5de1db80-256c-11ea-9631-c911ed447e03.png" width="300">

<img src ="https://user-images.githubusercontent.com/44080085/71329825-676b4380-256c-11ea-929b-ff7f6b5988b6.png" width="300">
<img src ="https://user-images.githubusercontent.com/44080085/71329832-718d4200-256c-11ea-85c4-3171a67f18d4.png" width="300">

<br>

[auc,precision,recallの参考](http://kurora-shumpei.hatenablog.com/entry/2019/06/01/%E4%BA%8C%E5%80%A4%E5%88%86%E9%A1%9E%E3%81%AB%E3%81%8A%E3%81%91%E3%82%8B%E8%A9%95%E4%BE%A1%E6%8C%87%E6%A8%99%E3%81%A8pAUC%E6%9C%80%E5%A4%A7%E5%8C%96(%E5%89%8D%E7%B7%A8))
