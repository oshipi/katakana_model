import numpy as np

def mean_squared_error(y, t):
    """
    y : 出力値
    t : 正解値
    """    
    if y.ndim==1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)
        
    batch_size = y.shape[0]    
    return 0.5 * np.sum((y - t)**2) / batch_size

def label_smoothed_cross_entropy_error(y, t, eps):
    """
    ラベル平滑化が適用された交差エントロピー誤差を計算
    ----------
    y : 出力の確率分布 (shape: [batch_size, num_classes])
    t : 正解ラベル (shape: [batch_size, num_classes])
    eps : ラベル平滑化の程度を調整するパラメータ

    """
    if y.ndim == 1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)

    batch_size = y.shape[0]
    num_classes = y.shape[1]

    # ラベル平滑化
    t_smooth = (1 - eps) * t + eps / num_classes

    # クロスエントロピー誤差の計算
    delta = 1e-7
    cross_entropy_error = -np.sum(t_smooth * np.log(y + delta)) / batch_size

    return cross_entropy_error