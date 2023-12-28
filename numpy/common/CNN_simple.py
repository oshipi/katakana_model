import numpy as np
from collections import OrderedDict
from layers import Convolution, MaxPooling, ReLU, Affine, SoftmaxWithLoss, BatchNormalization, Dropout

class ConvNet:
    """
    搭載オプション
    Batch Normalization(常時オン)
    Dropout(デフォルト : オフ)
    Weight Decay(デフォルト : 0)
    Label Smoothing(デフォルト : 0)
    
    (重みの初期値はデフォルトで0.01)
    
    Parameters
    ----------
    input_dim : tuple, 入力の配列形状(チャンネル数、画像の高さ、画像の幅)
    conv_param : dict, 畳み込みの条件
    pool_param : dict, プーリングの条件
    filter_num_list : list, filterの数のリスト
    hidden_size_list : list, 隠れ層のノードの数のリスト
    output_size : int, 出力層のノード数（15）
    """
    def __init__(self, input_dim, conv_param, pool_param, filter_num_list, hidden_size_list,
                 use_dropout = False, weight_decay_lambda=0., label_smoothing_eps = 0., weight_init_std=0.01, output_size=15):
        
        self.label_smoothing_eps = label_smoothing_eps
        self.use_dropout = use_dropout
        self.hidden_size_list = hidden_size_list
        self.weight_decay_lambda = weight_decay_lambda
        self.filter_num_list = filter_num_list
        
        pool_size = pool_param['pool_size']
        pool_pad = pool_param['pad']
        pool_stride = pool_param['stride']
        
        input_size = input_dim[1]
        input_after = input_dim[0]
        
        cv_list = filter_num_list

        # 重みの初期化
        self.params = {}
        # 畳み込み層(Conv)
        std = weight_init_std
        for idx in range(1, len(cv_list)+1):
            filter_size = conv_param['filter_size' + str(idx)]
            filter_pad = conv_param['pad' + str(idx)]
            filter_stride = conv_param['stride' + str(idx)]
            
            filter_num = cv_list[idx - 1]
            conv_output_size = (input_size + 2*filter_pad - filter_size) // filter_stride + 1 # 畳み込み後のサイズ(H,W共通)
            pool_output_size = (conv_output_size + 2*pool_pad - pool_size) // pool_stride + 1 # プーリング後のサイズ(H,W共通)

            self.params['W'+ str(idx)] = std * np.random.randn(filter_num, input_after, filter_size, filter_size)
            self.params['b'+ str(idx)] = np.zeros(filter_num) #畳み込みフィルターのバイアス
            self.params['gamma'+ str(idx)] = np.ones(filter_num)
            self.params['beta'+ str(idx)] = np.zeros(filter_num)

            input_size = pool_output_size
            input_after = filter_num
        
        # 全結合(Affine)
        hidden_size = hidden_size_list[-1]
        pool_output_pixel = filter_num * pool_output_size * pool_output_size # プーリング後のピクセル総数
        
        self.params['W_hidden'] = std * np.random.randn(pool_output_pixel, hidden_size)
        self.params['b_hidden'] = np.zeros(hidden_size)
        
        self.params['gamma_last'] = np.ones(hidden_size)
        self.params['beta_last'] = np.zeros(hidden_size)
        
        self.params['W_last'] = std * np.random.randn(hidden_size, output_size)
        self.params['b_last'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        # 畳み込み層
        for idx in range(1, len(cv_list)+1):
            self.layers['Conv'+ str(idx)] = Convolution(self.params['W'+ str(idx)], self.params['b'+ str(idx)],conv_param['stride' + str(idx)], conv_param['pad' + str(idx)])
            self.layers['BatchNorm'+ str(idx)] = BatchNormalization(self.params['gamma'+ str(idx)], self.params['beta'+ str(idx)])
            self.layers['ReLU'+ str(idx)] = ReLU()
            self.layers['Pool'+ str(idx)] = MaxPooling(pool_h=pool_size, pool_w=pool_size, stride=pool_stride, pad=pool_pad) 
            if self.use_dropout:
                self.layers['Dropout'+ str(idx)] = Dropout(dropout_ratio = 0.1*idx)
        # 全結合
        self.layers['Affine_hidden'] = Affine(self.params['W_hidden'], self.params['b_hidden'])
        self.layers['BatchNorm_last'] = BatchNormalization(self.params['gamma_last'], self.params['beta_last'])
        self.layers['ReLU_last'] = ReLU()
        if self.use_dropout:
            self.layers['Dropout_last'] = Dropout(dropout_ratio = 0.4)
        self.layers['Affine_last'] = Affine(self.params['W_last'], self.params['b_last'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x,train_flg=False):
        for key, layer in self.layers.items():
            if  "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        
        return x

    def loss(self, x, t,train_flg=False):
        """
        損失関数
        x : 入力データ
        t : 教師データ
        """
        y = self.predict(x,train_flg)
        
        # 荷重減衰を考慮した損失を求める
        lmd = self.weight_decay_lambda     
        eps = self.label_smoothing_eps # ラベル平滑化の程度も考慮
        weight_decay = 0
        for idx in range(1, len(self.filter_num_list) + 1):
            W = self.params['W' + str(idx)]
            
            # 全ての行列Wについて、1/2* lambda * Σwij^2を求め、積算していく
            weight_decay += 0.5 * lmd * np.sum(W**2)

        return self.last_layer.forward(y, t, eps) + weight_decay
        

    def accuracy(self, x, t, batch_size=500):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0] 

    def gradient(self, x, t):
        """勾配を求める（誤差逆伝播法）
        Parameters
        ----------
        x : 入力データ
        t : 教師データ
        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        lmd = self.weight_decay_lambda
        grads = {}
        # 畳み込み層
        for idx in range(1, len(self.filter_num_list)+1):
            grads['W'+ str(idx)], grads['b'+ str(idx)] = self.layers['Conv'+ str(idx)].dW + lmd * self.layers['Conv' + str(idx)].W, self.layers['Conv'+ str(idx)].db
            grads['gamma'+ str(idx)] = self.layers['BatchNorm'+ str(idx)].dgamma
            grads['beta'+ str(idx)] = self.layers['BatchNorm'+ str(idx)].dbeta
        # 全結合
        grads['W_hidden'], grads['b_hidden'] = self.layers['Affine_hidden'].dW, self.layers['Affine_hidden'].db
        grads['W_last'], grads['b_last'] = self.layers['Affine_last'].dW, self.layers['Affine_last'].db
        grads['gamma_last'] = self.layers['BatchNorm_last'].dgamma
        grads['beta_last'] = self.layers['BatchNorm_last'].dbeta

        return grads