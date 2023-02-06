#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""
import argparse
import queue
import sys
import traceback

import numpy as np
import sounddevice as sd
from time import sleep


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument('-c',
    '--channel', type=int, default=1,  metavar='CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-t', '--time-integration', type=float, default=125, metavar='DURATION',
    help='time integration window in samples')
parser.add_argument(
    '-b', '--blocksize', type=int, default=1024, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
args = parser.parse_args(remaining)
if args.channel < 1 :
    parser.error('argument CHANNEL: must be >= 1')
mapping = args.channel - 1
q = queue.Queue()


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print('Error:')
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[:, mapping])

def stft(x, wind=np.hanning(1024), nhop=512, nfft=None):
    nwind = len(wind)
    if nfft is None:
        nfft = len(wind)
    nfr = (len(x)-nwind)//nhop+1
    s = np.zeros((nwind//2+1, nfr))
    for ii, ist in enumerate(range(0, len(x)-nwind, nhop)):
        xx = x[ist:ist+nwind]*wind
        xf = np.fft.fft(xx)
        s[:,ii] = np.abs(xf[:nfft//2+1])**2
    return s

def log_center_and_lim_freqs(fst=31.25, fmax=8000., oct_frac=1.):
    allfreqs = 2**np.arange(np.log2(fst)-oct_frac/2,np.log2(fmax)+oct_frac, oct_frac/2)
    centfreqs = allfreqs[1::2]
    bandlims = np.vstack((allfreqs[0:-2:2], allfreqs[2::2]))
    return centfreqs, bandlims


def log_rect_filter(sr, nfft, center_freqs):
    f_orig = np.arange(nfft//2+1) * sr/nfft
    f_orig_lims = np.vstack(((f_orig - sr/nfft/2), (f_orig + sr/nfft/2)))
    
    center_logs = np.log2(center_freqs)
    band_lim_logs = (center_logs[:-1] + center_logs[1:])/2
    band_lim_logs = np.concatenate(([2*center_logs[0]-band_lim_logs[0]],
                                    band_lim_logs,
                                    [2*center_logs[-1] - band_lim_logs[-1]]))
    band_lims = 2**band_lim_logs

    mx = np.zeros((len(f_orig),len(center_freqs)))
    for ii in range(len(center_freqs)):
        # ones in bins fully contained in band
        idx = (f_orig >= band_lims[ii]) & (f_orig < band_lims[ii+1])
        mx[idx,ii] = 1
        
    # ToDo
    # Fix band boundaries (interpolate)
    return mx

def process_frames():
    """process frames for each integration window    
    """
    global window 
    global mx
    
    alldat = []
    while True:
        try:
            data = q.get_nowait()
            alldat.extend(data)
        except queue.Empty:
            break
    alldat = np.array(alldat)
    n = len(alldat)
    if n > 0: 
        nfr = int(np.floor((n-len(window))/(len(window)//2)))
        nhop = n//nfr
        s = stft(alldat, wind=window, nhop=nhop)
        avs = np.mean(s,axis=1)
        obs = mx.T@avs
        obstr = ""
        for oo in obs:
            obstr += "{0:.2f}, ".format(10*np.log10(oo))
        
        print("{0:.2f} :: {1}".format(10*np.log10((np.sum(alldat**2)/len(alldat))), obstr ))
    return 


try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']

    length = int(args.time_integration * args.samplerate / (1000))
    cf,_ = log_center_and_lim_freqs()
    mx = log_rect_filter(sr=args.samplerate, nfft=args.blocksize, center_freqs=cf)
    window = np.hanning(args.blocksize)

    stream = sd.InputStream(
        device=args.device, channels=(args.channel),
        samplerate=args.samplerate, callback=audio_callback)
    with stream:
        while True:
            process_frames()
            sleep(args.time_integration/1000)
except Exception as e:
    print("Exception!!")
    traceback.print_exc()
    parser.exit(type(e).__name__ + ': ' + str(e))
