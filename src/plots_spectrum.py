import matplotlib.pyplot as plt
import sys

def setspectrum(data,spectrum,k,kk):
    if not k in data:
        data[k]={}
        data[k][kk]=[spectrum]
    else:
        if not kk in data[k]:
            data[k][kk]=[spectrum]
        else:
            data[k][kk].append(spectrum)    
    return data

def feed(lines,tr=False):
    data={}
    for l in lines[1:]:
        l=l.split(",")
        N, M, attnum, FNNnum=l[:4]
        if(len(l)==5):
            spectrum=[float(i) for i in l[-1].replace("[","").replace("]","").split()]
        else:
            spectrum=[ float(s.replace("[","").replace("]","")) for s in l[4:]]
        
        k=f"N{N},M{M}"
        kk=f"att{attnum}_FNN{FNNnum}"
        if(tr):
            data=setspectrum(data,spectrum,kk,k)
        else:
            data=setspectrum(data,spectrum,k,kk)
    return data

def plot(filename):
    beta=filename.split("_")[1]
    with open(filename) as fp:
        lines = fp.read().splitlines()

    for tr in [False,True]:
        data=feed(lines,tr)
        for k,datak in data.items():
            plt.figure()
            for kk,specs in datak.items():
               # print(kk,len(specs),len(specs[0]))
                for s in  specs:
                    plt.plot(s,label=kk)
            plt.legend()
            plt.grid()
            plt.title(f"lyapunov spectrum {k}, {beta}")
            fname=f"img/spectrum_{k}_{beta}"
            if(tr):
                fname+="_tr"
            plt.savefig(fname+".png")
            plt.clf()

if __name__ == "__main__":
    filename=sys.argv[1]
    plot(filename)