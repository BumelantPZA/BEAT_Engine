import sys
for f in sys.argv[1:]:
    L=[l.split(' ',3) for l in open(f).read().splitlines()]
    t0=float(L[0][0]); print('==',f, L[0][3] if len(L[0])>3 else L[0][2])
    seen=set(); lw=0; ln=0; lwin={}
    phase_idx=0; submitted=0
    for t,c,*rest in L:
        lab=' '.join(rest)
        if lab.startswith('launch1'):
            ww,nm=rest[1].split(" ",1); w=float(ww); lw+=w; ln+=1; lwin[nm]=w; continue
        key=lab.split(' overlap')[0]
        if lab.startswith('phase submitted'): submitted+=1
        if submitted>1: break
        if lab.startswith('phase iterend') or lab.startswith('phase iter ') and not lab.endswith((' 1',' 2')) : continue
        if lab.startswith('phase ext_iter') and not lab.endswith((' 1',' 2',' 3')): continue
        if lab.startswith('phase overlap'): lab='phase overlap_plan'
        print(f"{float(t)-t0:8.2f} s  compile {float(c):6.2f}  launch1 sum {lw:6.2f} (n={ln})  {lab[:60]}")
    print('top first launches:', sorted(lwin.items(), key=lambda x:-x[1])[:5])
