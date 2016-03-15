import IPython, numpy as np, scipy as sp, matplotlib.pyplot as plt, matplotlib, sklearn, librosa, cmath,math
from IPython.display import Audio
import basicFuncs as bf
import repet as rt
import sys
from scipy import stats  
from scipy import spatial
import datetime
import csv


def readAudio(fileString):
    signal, sr = librosa.load(fileString)
    return signal, sr

def createXV(signal, sr):
    X = bf.stft(signal, 2048, 1024)
    V = bf.plt_spectrogram(X, 2048, 1024, sr)
    return X,V

def createDecomposition(V, transIn):
    if np.min(V) < 0:
        V = V - np.min(V)
    comp, act = librosa.decompose.decompose(V, transformer = transIn)
    return comp, act

def createMasks(X, V, comp, act):
    nc1 = np.zeros(comp.shape)
    nc1[:,0] = comp[:,0]
    na1 = np.zeros(act.shape)
    na1[0,:] = act[0,:]

    nc2 = np.zeros(comp.shape)
    nc2[:,1] = comp[:,1]
    na2 = np.zeros(act.shape)
    na2[1,:] = act[1,:]

    newthing1 = nc1.dot(na1)
    newthing2 = nc2.dot(na2)

    mask1 = newthing1/(newthing1 + newthing2)
    mask2 = newthing2/(newthing1 + newthing2)

    fullmask1 = np.zeros(X.shape)
    fullmask2 = np.zeros(X.shape)

    fullmask1[:V.shape[0], :] = mask1
    fullmask1[V.shape[0]:, :] = np.flipud(mask1)

    fullmask2[:V.shape[0], :] = mask2
    fullmask2[V.shape[0]:, :] = np.flipud(mask2)

    return fullmask1, fullmask2

def applyMask(X, m1, m2):
    X1 = X * m1
    X2 = X * m2

    return X1, X2

def transformSignal(X1, X2):
    fg = bf.istft(X1, 1024)
    bg = bf.istft(X2, 1024)
    return fg, bg

def writeAudio(fg,bg, sr, tStr, n):
    fn = "{0}/fg-{0}-{1}.wav".format(n,tStr)
    bn = "{0}/bg-{0}-{1}.wav".format(n,tStr)
    librosa.output.write_wav(fn, fg.real, sr)
    librosa.output.write_wav(bn, bg.real, sr)

def separateSources(fileString, t, tStr, n):
    signal, sr = readAudio(fileString)
    X, V = createXV(signal, sr)
    comp, act = createDecomposition(V, t)
    compMask, actMask = createMasks(X, V, comp, act)
    X_fg, X_bg = applyMask(X, compMask, actMask)
    fg, bg = transformSignal(X_fg, X_bg)
    writeAudio(fg, bg, sr, tStr, n)
    return fg, bg

def computeStats():
    #fd = open('accuracyData.csv','a')
    with open("accuracyData.csv", 'a') as fp:
        wr = csv.writer(fp, dialect='excel')
        for g in ["c", "r"]:
            print "In Genre: {0}".format(g)
            for i in range(1,6):
                print "In File: {0}".format(i)
                n = "{0}{1}".format(g,i)
                nv = "{0}{1}v".format(g,i)
                print "Read A"
                bsignal, sr = readAudio("{0}/{0}.wav".format(n))
                print "Read B"
                vsignal, sr = readAudio("{0}/{1}.wav".format(n,nv))
                print "Read C"
                voiceOnly = vsignal - bsignal
                #librosa.output.write_wav('test/voiceOnly.wav', vv, sr)
                #for transformer in [sklearn.decomposition.NMF(n_components=2), sklearn.decomposition.TruncatedSVD(n_components=2), sklearn.decomposition.PCA(n_components=2)]:
                for transformer in []:
                    print "With Transformer: {0}".format(str(transformer)[:3])
                    fg, bg = separateSources("{0}/{1}.wav".format(n, nv), transformer, str(transformer)[:3], n)
                    bg_a = spatial.distance.cosine(bsignal[:len(bg)], bg[:len(bsignal)]) 
                    bg_a =  1-bg_a.real
                    fg_a = spatial.distance.cosine(voiceOnly[:len(fg)], fg[:len(voiceOnly)])  
                    fg_a = 1- fg_a.real
                    dt = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
                    print "Pre Row"
                    data = [n, g, str(transformer)[:3], fg_a, bg_a, (fg_a + bg_a)/ 2, dt]
                    wr.writerow(data)
                    print data
                    print "Wrote Row"

                print "With Transformer: REP"
                bg, fg = rt.performRepet("{0}/{1}.wav".format(n, nv), 200, .6, n)
                bg = bg[::2]
                fg = fg[::2]
                print bg.shape
                print fg.shape
                bg_a = np.linalg.norm(bsignal[:len(bg)] - bg[:len(bsignal)]) 
                bg_a =  1-bg_a.real
                fg_a = np.linalg.norm(voiceOnly[:len(fg)] - fg[:len(voiceOnly)])  
                fg_a = 1- fg_a.real
                dt = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
                print "Pre Row"
                data = [n, g, "REP", fg_a, bg_a, (fg_a + bg_a)/ 2, dt]
                wr.writerow(data)
                print data
                print "Wrote Row"

if __name__ == "__main__":

    # b, f = repet('beetrecord10secsVoice.wav', 200, .6)
    # librosa.output.write_wav('bRepet.wav', b, 44100)
    # librosa.output.write_wav('fRepet.wav', f, 44100)
    print sys.argv[1]
    if sys.argv[1] == "0":
        computeStats()
    else:
        transformer = sys.argv[1]
        if transformer == "NMF":
            separateSources(sys.argv[2], sklearn.decomposition.NMF(n_components=2), "NMF", "")
        elif transformer == "SVD":
            separateSources(sys.argv[2], sklearn.decomposition.TruncatedSVD(n_components=2), "SVD", "")
        elif transformer == "PCA":
            separateSources(sys.argv[2], sklearn.decomposition.PCA(n_components=2), "PCA", "")
        elif transformer == "REPET":
            rt.performRepet(sys.argv[2], 200, .6, "")
        else:
            sys.exit('Usage: python audio.py (NMF/SVD/PCA/REPET) fileString')
