"""
stability of Attention+FNN against Magic Number ±7
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.94.058102

"""
import numpy as np
import torch
import torch.autograd.functional as AF
import matplotlib.pyplot as plt
import argparse
import src.plots_spectrum as ps

def dprint(s,fp):
    print(s)
    if(type(fp)==list):
        for f in fp:
            print(s,file=f)
    else:   
        print(s,file=fp)

def r01(shape):
    return torch.rand(shape, dtype=torch.float64)

def suf(params):
    return "_".join([ f"{k}{p}" for k,p in vars(params).items()])

def plothist(xs,i,j,params,bins=25):
    plt.figure()
    a=np.array([x[i,j] for x in xs])
    hist,_=np.histogram(a,bins)
    plt.plot(hist)
    plt.title(f"{i}_{j}_{suf(params)}")
    plt.savefig(f"histx_{i}_{j}_{suf(params)}.png")
    plt.clf()

def plot_all(x,i=0,j=0):
    plt.figure()
    for i in range(x.shape[1]):
        for j in range(x.shape[0]):
            plt.plot(x[j,i])
    plt.savefig(f"plotx_{i}_{j}.png")

def showmat(x,i):
    plt.figure()
    plt.imshow(x)
    plt.savefig(f"x_{i}.png")


def softmax(x):
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum()

# W_K, W_Q, W_V are eye matrix
def selfattention(x, rate=1.0):
    return rate * softmax(x @ x.t()) @ x + (1 - rate) * x

def FNN(W, x, beta=1, th=0):
    return torch.tanh(W @ x * beta + th)

def Resnet(W, x, beta=1, th=0):
    return torch.tanh(W @ x * beta + th) + x

def calcf(W, x, func, M=7, N=3, L=20, attentionLnum=10, FNNnum=1, beta=2, show=False):
    NM = N * M
    xs = []
    for l in range(L):
        for i in range(FNNnum):
            x = func(W, x, beta)
        for i in range(attentionLnum):
            x = selfattention(x)
        xs.append(x)
    if show:
        print(f"M={M},N={N},L={L},attentionL={attentionLnum}")
        print("last x", x)
    return torch.stack(xs)

def calc(W, x, M=7, N=3, L=20, attentionLnum=10, FNNnum=1, beta=2, show=False):
    return calcf(W, x, FNN, M, N, L, attentionLnum, FNNnum, beta, show)
def calcres(W, x, M=7, N=3, L=20, attentionLnum=10, FNNnum=1, beta=2, show=False):
    return calcf(W, x, Resnet, M, N, L, attentionLnum, FNNnum, beta, show)

def calcxs(params, func, num=100, W=None, showhist=True):
    N = params.N
    M = params.M
    if W is None:
        W = r01((N, N))
    xss = []
    for i in range(num):
        x = r01((N, M))
        xs = calc(W, x, func, params.M, params.N, params.L,
                  params.attentionLnum, params.FNNnum, params.beta, params.eps)
        xss.append(xs[-1, :, :])
        print(xs.shape)
        if i < 20:
            xs_np = xs.detach().cpu().numpy()
            plt.plot(xs_np[:, 0, 0], xs_np[:, 0, 1], marker="o")
    plt.title(f"{suf(params)}")
    plt.savefig(f"traj_{suf(params)}.png")
    plt.clf()
    if showhist:
        plothist(xss, 0, 1, params)
    return xss

# ============================================================
# autograd版 ヤコビアン計算
# ============================================================
def calcJ_autograd(f, x):
    """
    torch.autograd.functional.jacobian を使ってヤコビアンを計算する。

    f: (N, M) テンソルを受け取り (N, M) テンソルを返す関数
    x: 入力テンソル (N, M)

    戻り値: (NM, NM) のヤコビアン行列
    """
    N, M = x.shape
    NM = N * M
    # jacobian は flat なテンソルを扱う方が安定するため、
    # f を flatten 入出力に変換するラッパーを作る
    def f_flat(x_flat):
        return f(x_flat.reshape(N, M)).reshape(NM)

    J = AF.jacobian(f_flat, x.reshape(NM).detach().requires_grad_(True))
    # J の shape: (NM, NM)
    return J.detach()

# ============================================================
# リアプノフ指数計算 (autograd版)
# ============================================================
def calc_lyap(W, x, func=FNN, calcJ=calcJ_autograd, M=7, N=3, L=20, attentionLnum=10, FNNnum=1,
              beta=2, eps=1e-4, show=False, th=0, tiny=1e-300):
    print("calcJ",calcJ)
    NM = N * M
    Q = torch.eye(NM, dtype=x.dtype)
    lyap_sum = torch.zeros(NM, dtype=x.dtype)

    for l in range(L):
        for i in range(FNNnum):
            # autograd でヤコビアンを計算
            f_fnn = lambda xx: func(W, xx, beta, th)
            J = calcJ(f_fnn, x)
            x = func(W, x, beta, th)
            Q, R = torch.linalg.qr(J @ Q, mode='reduced')
            lyap_sum += torch.log(torch.clamp(torch.abs(torch.diag(R)), min=tiny))

        for i in range(attentionLnum):
            J = calcJ(selfattention, x)
            x = selfattention(x)
            Q, R = torch.linalg.qr(J @ Q, mode='reduced')
            lyap_sum += torch.log(torch.clamp(torch.abs(torch.diag(R)), min=tiny))

    lyap_sum = lyap_sum / (L * (FNNnum + attentionLnum))
    if show:
        print(f"M={M},N={N},L={L},attentionL={attentionLnum}")
        print(lyap_sum)
        print("last x", x)
        plot_all(x)
    return x, lyap_sum

def calc_lyap_autograd(W, x, func=FNN, M=7, N=3, L=20, attentionLnum=10, FNNnum=1,
              beta=2, eps=1e-4, show=False, th=0, tiny=1e-300):
    return calc_lyap(W, x, func,calcJ_autograd, M, N, L, attentionLnum, FNNnum,
                    beta, eps, show ,th, tiny)

# ============================================================
# 数値微分版（旧版・比較用に残す）
# ============================================================
def xd(x,N,M,eps):
    def p(i,j):
        d=x.detach()
        d[i,j]+=eps
        return d
    return [[p(i,j) for i in range(N) ] for j in range(M)]

def calcJ_numerical(xds, x, f, eps):
    # for xd_row in xds:
    #     for d in xd_row:
    #         print(d)
    # f(d)
    return torch.tensor([[(f(d) - x).numpy() / eps for d in xd_row] for xd_row in xds])

# ============================================================
# リアプノフ指数計算 (数値微分版・比較用)
# ============================================================
def calc_lyap_numerical(W, x, func=FNN, M=7, N=3, L=20, attentionLnum=10, FNNnum=1,
                        beta=2, eps=1e-4, show=False, th=0, tiny=1e-300):
    NM=N*M
    def calcJ(f,x):
            xds_list = xd(x, N, M, eps)
            x1 = f(x)
            return calcJ_numerical(xds_list, x1, f, eps).reshape(NM, NM)
    return calc_lyap(W, x, func,calcJ,
                    M, N, L, attentionLnum, FNNnum,
                    beta, eps, show ,th, tiny)

def plot_lyaps(filename="lyap.csv",outfilename='lyaps_pair.png'):
    import pandas as pd
    import seaborn as sns
    df=pd.read_csv(filename)
    df["NM"]=df["N"]*df["M"]
    df["Attention/FNN"]=df["attentionLnum"]/df["FNNnum"]
    df=df[["NM","attentionLnum","FNNnum","Attention/FNN","max lyap","min lyap"]]
    pg = sns.pairplot(df)    
    pg.savefig(outfilename)

def calc_lyaps(num=1,func=FNN,filename="lyap",beta=2):
    L=100
    filename=filename+f"_beta{beta}"
    with open(filename+"_spectrum.csv","w") as fpp:
        with open(filename+".csv","w") as fp:
            dprint("N,M,attentionLnum,FNNnum,max lyap,min lyap",fp)        
            dprint("N,M,attentionLnum,FNNnum,max lyaps",fpp)        
            for N in [2,3,5,10]:
                for M in [3,5,10]:
                    W=r01((N,N))
                    for attentionLnum in [0,3,10]:
                        for FNNnum in [1,5,10,15,20]:
                            for n in range(num):
                                th=r01(M)
                                x=r01((N,M))
                                x,lyap=calc_lyap(W,x,func,calcJ_autograd,M,N,L,attentionLnum,FNNnum,beta=beta,th=th)
                                dprint(f"{N},{M},{attentionLnum},{FNNnum},{torch.max(lyap)},{torch.min(lyap)}",fp)
                                lyap=" ".join([ f"{lyap[i]}"for i in range(lyap.shape[0]) ])
                                print(f"{N},{M},{attentionLnum},{FNNnum},[{lyap}]",file=fpp)
    plot_lyaps(filename+".csv", filename+".png")
    ps.plot(filename+".csv")

# ============================================================
# 動作確認
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stability of Attention+FNN against Magic Number ±7")
    parser.add_argument("--num", type=int, default=100, help="num. of trials")
    parser.add_argument("--M", type=int, default=7, help="row of input matrix")
    parser.add_argument("--N", type=int, default=3, help="column of input matrix")
    parser.add_argument("--L", type=int, default=5, help="layer numner")
    parser.add_argument("--attentionLnum", type=int, default=5, help="attention layer numner between FNNs")
    parser.add_argument("--FNNnum", type=int, default=5, help="FNN numner between attention layers")
    parser.add_argument("--beta", type=float, default=1.4, help="coef of FNN tanh")
    parser.add_argument("--eps", type=float, default=1e-4, help="noise to calculate ")
    parser.add_argument("--lyap", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--resnet", action="store_true")
    parser.add_argument("--rate", type=float, default=1.,help="rate of attention") 
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if(args.test):
        N, M ,L= 3, 7, 5
        W = torch.rand(N, N, dtype=torch.float64)
        x = torch.rand(N, M, dtype=torch.float64)
        print("=== autograd版 ===")
        x_ag, lyap_ag = calc_lyap_autograd(W, x.clone(), FNN, M, N, L, attentionLnum=2, FNNnum=1, beta=2, show=True)
        print("\n=== 数値微分版 ===")
        x_nm, lyap_nm = calc_lyap_numerical(W, x.clone(), FNN, M, N, L, attentionLnum=2, FNNnum=1, beta=2, eps=1e-6, show=True)
        print("\n=== 差分 (autograd - numerical) ===")
        print("lyap diff:", (lyap_ag - lyap_nm).abs().max().item())
    elif(args.lyap):
        if(args.resnet):        
            calc_lyaps(func=Resnet,filename="lyap_Resnet",beta=args.beta)
        else:
            calc_lyaps(func=FNN,beta=args.beta)
    elif(args.plot):
        if(args.resnet):        
            plot_lyaps(filename="lyaps_Resnet.csv")
        else:
            plot_lyaps(filename=f"lyaps.csv")
    elif(args.resnet):
        xs=calcxs(args,Resnet,args.num)
    else:
        xs=calcxs(args,FNN,args.num)

