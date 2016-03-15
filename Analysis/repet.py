import IPython, numpy as np, scipy as sp, matplotlib.pyplot as plt, matplotlib, sklearn, librosa, cmath,math
from IPython.display import Audio
from scipy.fftpack import ifft
from scipy.fftpack import fft

import basicFuncs as bf
import sys

def acorr(x):
    """
    Takes a 1D numpy array and returns the autocorrelation. 
    
    Input Parameter:
    ----------------
    x: 1D numpy array of length Lx
    
    Ouput Parameter:
    ----------------
    x_acorr: 1D numpy array of length x_len containing the values of the autocorrelation function of x
    
    Note: the actual length of autocorrelation function is (2*x_len)-1, but since this function is symmetric
          we can cut off (x_len)-1 samples and return one side of length x_len without losing information. 
    """
    
    x_len=np.size(x)
    x_pad=np.concatenate([x,np.zeros(x_len-1)])
    x_acorr=ifft(np.abs(fft(x_pad))**2).real
    x_acorr=x_acorr[0:x_len]
    
    x_acorr=x_acorr/(np.arange(x_len)[::-1]+1)  # normalize by the number of nonzero additions
    return x_acorr

def beat_spectrum(S, plotBeat= True):
    """
    Computes the beat spectrum of a music mixture by applyting the autocorrelation to all frequency channels (rows)
    of the spectrogram and taking the average. Note that the assumption is that the mixture is composed of a repeating
    background and a non-repeating foreground. Therefore, the spikes in the beat spectrum represent the repetitive 
    events in the spectrogram.  This function must PLOT the beat spectrum for viewing, since you'll be looking at the
    resulting plot to pick the repeating period used by REPET. 
    
    Input Parameter:
    ----------------
    S: 2D numpy array containng the spectrogram of a signal
    
    Output Parameter:
    ------------------
    beat_spec: 1D numpy array containing the beat spectrum 
    
    """
    # your code goes here
    beat_spec = np.zeros(acorr(S[0]).shape)
    for row in S:
        beat_spec += acorr(row)
    beat_spec /= S.shape[0]
    if plotBeat:
        plt.plot(beat_spec)
    return beat_spec

def repeating_segment(V,p):
    """
    Computes the repeating-segment model using the period obtained from the beat spectrum. The repeating segment
    is that median background we'll be using to compare to the spectrum. Its time duration will be the length of
    one period that you learned from the beat spectrum. 
    
    Input Parameters:
    -----------------
    V: 2D numpy array containing the magnitude spectrogram of the mixture
    p: scalar, repeating period (in number of samples) found from the beat spectrum 
    
    Output Parameter:
    S: 2D numpy array containing the repeating segment 
    """  
    # your code goes here
    x = 0
    count = 0
#     print math.ceil(V.shape[1]/p)
    segments = np.zeros((int((math.ceil(V.shape[1]/p) + 1)), V.shape[0], p))
    #print segments.shape
    while x < V.shape[1]:
#         print x
        segment = np.zeros((V.shape[0], p))
        if (x + p) < V.shape[1]:
            segment = V[:, x:x+p]
        else:
            segment[:, 0:V.shape[1]-x] = V[:, x:V.shape[1]]
        segments[count, :, :] = segment
        #print segments.shape
        x += p
        count += 1
    return np.median(segments, axis=0)

# def repeating_segment(V,p):
#     """
#     Computes the repeating-segment model using the period obtained from the beat spectrum. The repeating segment
#     is that median background we'll be using to compare to the spectrum. Its time duration will be the length of
#     one period that you learned from the beat spectrum. 
    
#     Input Parameters:
#     -----------------
#     V: 2D numpy array containing the magnitude spectrogram of the mixture
#     p: scalar, repeating period (in number of samples) found from the beat spectrum 
    
#     Output Parameter:
#     S: 2D numpy array containing the repeating segment 
#     """  
#     # your code goes here
    
#     # Take p-sized snippets and add them together into S
#     S = np.zeros([V.shape[0], p]).astype(float)
#     last_full = 0
#     cnt = 0
#     for i in range(p, V.shape[1], p):
#         S += V[:,last_full:i]
#         last_full = i
#         cnt += 1
#     print S
    
#     # Added Case for if last segment is shorter
#     S_index = 0
#     for i in range(last_full, V.shape[1]):
#         S[:, S_index] += V[:,i]
#         S_index += 1
#     print S
    
#     # Divide by respective counts to get median
#     S[:,:S_index] /= (cnt+1)
#     S[:,S_index:] /= (cnt)
#     return S

def repeating_spectrogram(V,S):
    """
    Builds the repeating-spectrogram model using the repeating segment model computed in the previous question.
    This is composed of multiple copies of the repeating segment, tiled together in the time-dimension so that it
    is the exact same size as the input signal's spectrogram.
    
    Input Parameters:
    -----------------
    V: 2D numpy array containing the magnitude spectrogram of the mixture
    S: 2D numpy array containing the repeating segment model 
    
    Output Parameter:
    ------------------
    W: 2D numpy array containing the repeating-spectrogram 
    """
    # your code goes here
    
    # Create a deep copy of V
    W = np.copy(V)
    
    W_i = 0
    S_i = 0
    
    # for all columns in W
    while W_i < W.shape[1]:
        # if S is valid
        if S_i < S.shape[1]:
            # Take the minimum over the entire column element wise
            W[:,W_i] = np.minimum(V[:,W_i], S[:, S_i])
            
            # Move to the next column in W and the repeating segment
            W_i += 1
            S_i += 1
        else:
            # Begin a new repeating segment to compare
            S_i = 0
    return W

def repeating_mask(V,W):
    """
    Before we do our separation, we need to compare our repeating_spectrogram to the original spectrogram of the signal
    to determine how to assign the energy in each time-frequency point (each element in your matrix). This function
    does that and makes a mask that will be applied to the spectrum. Note that unlike the denoising case in the first 
    question, here we are computing a "soft mask" with values ranging between 0 and 1.
    
    Input Parameters:
    -----------------
    V: numpy 2D array containing the magnitude spectrogram 
    W: numpy 2D array containing the repeating spectrogram
    
    Output Parameter:
    M: numpy 2D array containing the repeating mask 
    """
    # your code goes here
    return (W+1e-16)/(V+1e-16)

def repet(file_path, rep_period, mask_thr):
    """
    Runs the REPET algorithm using functions implemented in qustions 3 through 8. NOTE: In the real REPET algorithm
    it also determines the repeating period from the beat spectrum by building a peak picker to find the best
    period from the beat spectrum. This seemed like too much for a homework, so we're making YOU the peak picker. 
    You'll have to run the beat_spectrum function before calling repet and input the value for rep_period.
    
    Input Parameters:
    ------------------
    file_path: string, path to the audio file to separate
    rep_period: scalar, repeating period in number of samples
    mask_thr: scalar, a value between 0 and 1 for thresholding the soft mask
    
    Output:
    --------
    Separated background and foreground magnitude spectrogram plots 
    
    Returns: 
    ---------
    Separated background and foreground time signals
    """
    
    # Load the noisy mixture
    signal, sample_rate = librosa.load(file_path, 44100) 

    # Comput the bf.stft of the audio signal 
    win_length_sec=0.04 # 40 msec window
    win_length_samp=int(2**np.ceil(np.log2(win_length_sec*sample_rate))) # next power2 of winow length in no. of samples
    n_fft=win_length_samp
    hop_size=win_length_samp/2 # 50% overlap
    win_type=sp.signal.hanning
    X = bf.stft(signal, win_length_samp, hop_size, window_type = 'hann')

    # Compute the magnitude spectrogram (half spectrum)
    Vm = np.abs(X[0:win_length_samp/2+1,:])
    Nf,Nt=Vm.shape
    
    
    # Compute and plot the beat spectrum
    beat_spec = beat_spectrum(Vm**2)
    beat_spec = beat_spec/beat_spec[0] # normalization
    
    # plt.figure()
    # plt.plot(beat_spec)
    # plt.grid('on')
    # plt.axis('tight')
    # plt.xlabel('Lag (sample number)')
    # plt.title('Beat spectrum of the noisy mixture')
    
    ### REPET Algorithm ###
    
    # Compute the repeating soft mask
    Sm=repeating_segment(Vm,rep_period)
    Wm=repeating_spectrogram(Vm,Sm)
    soft_mask=repeating_mask(Vm,Wm)
    
    #plt.figure()
    #plt.pcolormesh(soft_mask)
    
    # Compute the repeating binary mask
    binary_mask=np.zeros(soft_mask.shape)
    binary_mask[soft_mask>=mask_thr]=1
    
    
    # Estimate the background and foreground via masking
    binary_mask=np.vstack([binary_mask,np.flipud(binary_mask[1:-1,:])])
    
    X_bg = binary_mask*X+1e-16  # background STFT
    signal_bg = np.real(bf.istft(X_bg, hop_size)) # background time signal
    
    X_fg = (1-binary_mask)*X+1e-16  # foreground STFT
    signal_fg = np.real(bf.istft(X_fg, hop_size)) # foreground time signal
            
    # Plot the log magnitude spectrograms and the binary mask
    # plt.figure()    
    # plt.subplot(211)
    # _, _, _ = bf.plt_spectrogram(X,win_length_samp, hop_size, sample_rate,tick_labels='time-freq')
    # plt.title('Mixture')
        
    # plt.subplot(212)
    # plt.pcolormesh(binary_mask[0:Nf,:])
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Frequency (Hz)')
    # plt.title('Binary mask')
    # plt.axis('tight')
    
    # plt.figure()
    # plt.subplot(211)
    # _, _, _ = bf.plt_spectrogram(X_bg,win_length_samp, hop_size, sample_rate,tick_labels='time-freq')
    # plt.title('Background')
    
    # plt.subplot(212)
    # _, _, _ = bf.plt_spectrogram(X_fg,win_length_samp, hop_size, sample_rate,tick_labels='time-freq')
    # plt.title('Foreground')
    
    return signal_bg,signal_fg

def performRepet(file_path, rep_period, mask_thr, n):
    b, f = repet(file_path, 200, .6)
    librosa.output.write_wav('{0}/{0}-bg-REP.wav'.format(n), b, 44100)
    librosa.output.write_wav('{0}/{0}-fg-REP.wav'.format(n), f, 44100)
    return f, b


