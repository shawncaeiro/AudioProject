import librosa
from scipy.fftpack import fft
from scipy.signal import hann
import numpy as np
import scipy as sp
from scipy.fftpack import ifft

def getsonglength(path_to_audio):
    song, sr = librosa.load(path_to_audio)
    return len(song) / float(sr)

def combineandsplitsongs(path_to_audio, path_to_voice, path_of_output, path_to_output1, path_to_output2):
    song, sr = librosa.load(path_to_audio)
    voice, sr = librosa.load(path_to_voice)
    song[0:len(voice)] += voice
    librosa.output.write_wav(path_of_output, song, sr)
    split1len = splitty(song, path_to_output1, path_to_output2, sr)
    return split1len

def combinesongs(path_to_audio, path_to_voice, path_of_output):
    song, sr = librosa.load(path_to_audio)
    voice, sr = librosa.load(path_to_voice)
    song[0:len(voice)] += voice
    librosa.output.write_wav(path_of_output, song, sr)
    return path_of_output

def splitty(song, path_to_output1, path_to_output2, sr):
    window_size = 2048
    hop_size = 1024
    combined = song

    X_voice = stft(combined, window_size, hop_size)

    V_voice = plt_spectrogram(X_voice, window_size, hop_size, sr)
    if np.min(V_voice) < 0:
        newV_voice = V_voice - np.min(V_voice)
    else:
        newV_voice = V_voice

    comp, act = librosa.decompose.decompose(newV_voice, n_components=2)
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

    fullmask1 = np.zeros(X_voice.shape)
    fullmask2 = np.zeros(X_voice.shape)

    fullmask1[:V_voice.shape[0], :] = mask1
    fullmask1[V_voice.shape[0]:, :] = np.flipud(mask1)

    fullmask2[:V_voice.shape[0], :] = mask2
    fullmask2[V_voice.shape[0]:, :] = np.flipud(mask2)

    part1X = X_voice * fullmask1
    part2X = X_voice * fullmask2

    part1 = istft(part1X, hop_size)
    part2 = istft(part2X, hop_size)

    librosa.output.write_wav(path_to_output1, part1.real, sr)
    librosa.output.write_wav(path_to_output2, part2.real, sr)
    return len(part1)

def stft(signal, window_size, hop_size, window_type = 'hann'):
    """
    Computes the short term fourier transform of a 1-D numpy array, where the array
    is windowed into a set of subarrays, each of length window_size. The distance between
    window centers (in samples) is given by hop_size. The type of window applied is
    determined by window_type. This returns a 2-D numpy array where the ith column
    is the FFT of the ith window. Each column contains an array of complex values.

    Input Parameters
    ----------------
    signal: The 1-d (complex or real) numpy array containing the signal
    window_size: an integer scalar specifying the number of samples in a window
    hop_size: an integer specifying the number of samples between the start of adjacent windows
    window_type: a string specifying one of two "hann" or "rectangular"

    Returns
    -------
    a 2D numpy array of complex numbers where the array column is the FFT of the ith window,
    and the jth element in the ith column is the jth frequency of analysis.
    """

    # figure out how many hops
    length_to_cover_with_hops = len(signal) - window_size;
    assert (length_to_cover_with_hops >= 0), "window_size cannot be longer than the signal to be windowed"
    num_hops = 1 + length_to_cover_with_hops/hop_size;

    # make our window function
    if (window_type == 'hann'):
        window = sp.signal.hann(window_size, sym=False)
    else:
        window = np.ones(window_size)

    stft = [0]*num_hops
    # fill the array with values
    for hop in range(num_hops):
        start = hop*hop_size
        end = start + window_size
        unwindowed_sound = signal[start:end]
        windowed_sound =  unwindowed_sound * window
        stft[hop]= fft(windowed_sound, window_size)
    return np.array(stft).T

def plt_spectrogram(X,win_length, hop_size, sample_rate, zoom_x=None, zoom_y=None,tick_labels='time-freq'):
    """
    Plots the log magnitude spectrogram.

    Input Parameters:
    ------------------
    X: 2D complex numpy array containing the stft values. Rows correspond to frequency bins and columns to time frames.
    win_length: the length of the analysis window
    hop_size: the hop size between adjacent windows
    sample_rate: sampling frequency
    tick_labels: the type of x and y tick labels, there are two options:
                 'time-freq': shows times (sec) on the x-axis and frequency (Hz) on the y-axis (default)
                 'bin-frame': shows time frame number on the x-axis and frequency bin number on the y-axis

    zoom_x: 1 by 2 numpy array containing the range of values on the x-axis, e.g. zoom_t=np.array([x_start,x_end])
    zoom_y: 1 by 2 numpy array containing the range of values on the y-axis, e.g. zoom_f=np.array([y_start,y_end])


    Returns:
    ---------
    times: 1D real numpy array containing time instances corresponding to stft frames
    freqs: 1D real numpy array containing frequencies of analyasis up to Nyquist rate
    2D plot of the magnitude spectrogram
    """

    # Find the size of stft
    Nf,Nt=np.shape(X)

    # Compute the log magnitude spectrogram
    X=20*np.log10(np.abs(X))

    # Extract the first half of the spectrum for each time frame
    X=X[0:Nf/2]
    # Nf=np.shape(X)[0]
    #
    # # Generate time vector for plotting
    # times=(hop_size/float(sample_rate))*np.arange(Nt)
    #
    # # Generate frequency vector for plotting
    # freqs=(float(sample_rate)/win_length)*np.arange(Nf)
    #
    # # Generate time and frequency matrices for pcolormesh
    # times_matrix,freqs_matrix=np.meshgrid(times,freqs)
    # #
    # # Plot the log magnitude spectrogram
    # plt.title('Log magnitude spectrogram')
    # if tick_labels == 'bin-frame':
    #     plt.pcolormesh(X)
    #     plt.xlabel('Time-frame Number')
    #     plt.ylabel('Frequency-bin Number')
    # else:
    #     plt.pcolormesh(times_matrix,freqs_matrix,X)
    #     plt.xlabel('Time (sec)')
    #     plt.ylabel('Frequency (Hz)')
    #
    # # Zoom in on the plot if specified
    # if zoom_x is None and zoom_y is None:
    #     plt.axis('tight')
    #
    # if zoom_x is not None:
    #     plt.xlim(zoom_x)
    #
    # if zoom_y is not None:
    #     plt.ylim(zoom_y)
    #
    return X

def istft(X, hop_size):
    """
    Takes a 2-D numpy array representing an STFT of some signal, where stft[i]
    is the FFT of the ith window as input and stft[i,k] is the kth frequency of analysis.
    Performs an inverse FFT on each window and then does overlap & add resynthesis to rebuild
    the original signal the STFT was built from.

    Input Parameters
    ----------------
    X: a 2-D numpy array of complex numbers representing an STFT, where the ith
    column is the FFT of the ith window, and the jth row is the jth frequency of analysis.

    hop_size: an integer specifying the number of samples between the start of adjacent windows.

    Returns
    -------
    a 1-d numpy array of (possibly complex) values representing the original signal used to make X
    """

    # make an empty signal of the appropriate length
    window_size,num_hops = X.shape
    signal_length = (num_hops-1)*hop_size + window_size
    signal = np.zeros(signal_length,dtype='complex');

    #fill the signal
    for n in range(num_hops):
        start = n * hop_size
        end = start + window_size
        signal[start:end] = signal[start:end] + ifft(X[:,n])
    return signal
