import matplotlib.pyplot as plt
import sys
import matplotlib.cm as cm

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

def _plot(data,beta,fname_tr="img/spectrum",debug=False):
    for k,datak in data.items():
        fname=f"{fname_tr}_{k}"
        plt.figure()
        for j,(kk,specs) in enumerate(datak.items()):
            for s in specs:
                plt.plot(s,label=kk,color=cm.hsv(j/len(specs[0])))
        plt.legend()
        plt.grid()
        plt.title(f"lyapunov spectrum {k}, beta {beta}")
        plt.savefig(fname+".png")
        plt.close()

def plot(filename,beta=0,debug=False):
    if(beta==0):
        beta=filename.split("_")[1]
    with open(filename) as fp:
        lines = fp.read().splitlines()

    data=feed(lines,tr=False)
    fname=f"img/spectrum_{beta}"
    _plot(data,beta,fname,debug)

def plot_all():
    for beta in [1.4,2.0]:
        filename=f"result/lyap_beta{beta}_spectrum.csv"
        plot(filename,beta,True)

if __name__ == "__main__":
    #filename=sys.argv[1]
    plot_all()
    #plot(filename)