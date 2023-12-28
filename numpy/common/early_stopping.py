import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, verbose=1):
        """
        Parameters:
            patience(int): 監視するエポック数
            verbose(int): 早期終了の出力フラグ
                          出力(1),出力しない(0)        
        """

        self.epoch = 0 # 監視中のエポック数のカウンターを初期
        self.pre_accuracy = 0. # 比較対象を0で初期化
        self.patience = patience # 監視対象のエポック数をパラメーターで初期化
        self.verbose = verbose # 早期終了メッセージの出力フラグをパラメーターで初期化
        
    def __call__(self, current_accuracy):
        """
        Parameters:
            current_accuracy(float): 1エポック終了後の検証データの正解率
        Return:
            True:監視回数の上限までに前エポックの正解率を超えない場合
            False:監視回数の上限までに前エポックの正解率を超えた場合
        """
        
        if self.pre_accuracy >= current_accuracy:
            self.epoch += 1 # カウンターを1増やす

            if self.epoch > self.patience: # 監視回数の上限に達した場合
                if self.verbose:  # 早期終了のフラグが1の場合
                    print('early stopping')
                return True # 学習を終了するTrueを返す
            
        else:
            self.epoch = 0 # カウンターを0に戻す
            self.pre_accuracy = current_accuracy # 正解率の値を更新する
        
        return False