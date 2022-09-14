

ll = []
f = open("filelists/ljs_mel_text_filelist.txt")
for l in f.readlines():
    if "LJ001-0014" not in l:
        ll.append(l)
f.close()


import random
random.seed(61112)

random.shuffle(ll)


l1 = ll[:3000]
l2 = ll[3000:6000]
l3 = ll[6000:9000]
l4 = ll[9000:12000]


print(len(l1),len(l2),len(l3),len(l4))

lll = [l1,l2,l3,l4]
for i,lset in enumerate(lll):
    t = lset[:2500]
    v = lset[2500:2900]
    ts = lset[2900:3000]
    tfile = open("./data/ar_train_%s.txt"%str(i),"w")
    for lt in t:
        tfile.write(lt)
    tfile.close()


    tv = open("./data/ar_val_%s.txt"%str(i),"w")
    for lv in v:
        tv.write(lv)
    tv.close()


    tsfile = open("./data/ar_test_%s.txt"%str(i),"w")
    for ss in ts:
        tsfile.write(ss)
    tsfile.close()
