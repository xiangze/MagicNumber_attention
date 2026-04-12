# -------------------------------------------------------------------
# 共通ユーティリティ
# -------------------------------------------------------------------
def dprint(s,fp):
    print(s)
    if(type(fp)==list):
        for f in fp:
            print(s,file=f)
    elif(fp==None):
        print(s)
    else:   
        print(s,file=fp)

def banner(s,num=60):
    print("\n" + "="*num)
    print(s)
    print("="*num)
