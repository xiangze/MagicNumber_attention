"""
stability of Attention+FNN against Magic Number ±7
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.94.058102

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import argparse

rnd = np.random.default_rng(1234)

def dprint(s,fp):
    print(s)
    print(s,file=fp)

def r01(size):
    return (rnd.random(size)-0.5)*2

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

def plot_all(x,i,j):
    plt.figure()
    for i in range(x.shape[1]):
        for j in range(x.shape[0]):
            plt.plot(x[i,j])
    plt.savefig(f"plotx_{i}_{j}.png")

def showmat(x,i):
    plt.figure()
    plt.imshow(x)
    plt.savefig(f"x_{i}.png")

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

#W_K,W_Q,W_V are eye matrix
def selfattention(x):
    return softmax(x@x.transpose())@x

def FNN(W,x,beta=1,th=0):
    return np.tanh(W@x*beta+th)


def calc(W,x,M=7,N=3,L=20,attentionLnum=10,FNNnum=1,beta=2,eps=1e-4,show=False):
    NM=N*M
    xs=[]
    for l in range(L):
        for i in range(FNNnum):
            x=FNN(W,x,beta)
        for i in range(attentionLnum):
            x=selfattention(x)
        xs.append(x)
    if(show):
        print(f"M={M},N={N},L={L},attentionL={attentionLnum}")
        print("last x",x)
    return np.array(xs)

def calcxs(params,num=100,W=None,showhist=True):
    N=params.N
    M=params.M
    if(W is None):
        W=r01((N,N))
    xss=[]
    for i in range(num):
        x=r01((N,M))
        xs=calc(W,x,params.M,params.N,params.L,params.attentionLnum,params.FNNnum,params.beta,params.eps)
        xss.append(xs[-1,:,:])
        print(xs.shape)
        if(i<20):
            plt.plot(xs[:,0,0],xs[:,0,1],marker="o")
    plt.title(f"{suf(params)}")
    plt.savefig(f"traj_{suf(params)}.png")
    plt.clf()

    if(showhist):
        plothist(xss,0,1,params)
    return xss

def xd(x,N,M,eps):
    xds=[]
    for j in range(M):
        xd=[]
        for i in range(N):
            d=x.copy()
            d[i,j]+=eps
            xd.append(d)
        xds.append(xd)
    return xds

def calcJ(xds,x,f,eps):
    return np.array([[(f(d)-x)/eps for d in xd ] for xd in xds])

def calc_lyap(W,x,M=7,N=3,L=20,attentionLnum=10,FNNnum=1,beta=2,eps=1e-4,show=False,th=0,tiny=1e-300):
    NM=N*M
    Q = np.eye(NM)
    lyap_sum = np.zeros(NM)
    #log|δ0|=|log√∑^{NM}ε^2=|log√∑^{NM}ε^2|=0.5*|log(NM)+log(ε^2)|=0.5*|log(NM)|+2*|log(ε)|
    for l in range(L):
        for i in range(FNNnum):
            xds=xd(x,N,M,eps)
            #xds=xd(x,N,M,-eps)
            x1=FNN(W,x,beta,th)
            J=calcJ(xds,x1,lambda xx:FNN(W,xx,beta,th),eps).reshape(NM,NM)
            #log(|J|/|δ|)=log(|J|)-log|δ|
            Q, R = linalg.qr(J @ Q, mode='economic')
            lyap_sum += np.log(np.maximum(np.abs(np.diag(R)), tiny))
            x=x1
        for i in range(attentionLnum):
            xds=xd(x,N,M,eps)
            x1=selfattention(x)
            J=calcJ(xds,x,selfattention,eps).reshape(NM,NM)
            Q, R = linalg.qr(J @ Q, mode='economic')
            lyap_sum += np.log(np.maximum(np.abs(np.diag(R)), tiny))
            x=x1
    lyap_sum=lyap_sum/(L*(FNNnum+attentionLnum))
    if(show):
        print(f"M={M},N={N},L={L},attentionL={attentionLnum}")
        print(lyap_sum)
        print("last x",x)
        plot_all(x)
    return x,lyap_sum

def calc_lyaps(num=1,filename="lyap.csv"):
    L=100
    with open(filename,"w") as fp:
        dprint("N,M,attentionLnum,FNNnum,max lyap,min lyap",fp)        
        for N in [2,3,5,10]:
            for M in [3,5,10]:
                W=r01((N,N))
                for attentionLnum in [0,3,10]:
                    for FNNnum in [3,10]:
                        for n in range(num):
                            th=r01(M)
                            x=r01((N,M))
                            x,lyap=calc_lyap(W,x,M,N,L,attentionLnum,FNNnum,th)
                            dprint(f"{N},{M},{attentionLnum},{FNNnum},{np.max(lyap)},{np.min(lyap)}",fp)

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
    args = parser.parse_args()

    if(args.lyap):
        calc_lyaps()
    else:
        xs=calcxs(args,args.num)
    