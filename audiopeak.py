import librosa
import scipy.interpolate as interp
import numpy as np
        
class AudioPeak:
    def __init__(self, audiofilepath, fps, video_duration=None, offset=0.0):
        self.audiofilepath = audiofilepath
        self.fps = fps
        self.video_duration = video_duration
        self.offset = offset
            
        self.analyze()
        
    def analyze(self):
        def get_spec_norm(wav, sr, n_mels, hop_length):
            spec_raw = librosa.feature.melspectrogram(y=wav, sr=sr,
                                                       n_mels=n_mels,
                                                       hop_length=hop_length)
            spec_max = np.amax(spec_raw,axis=0)
            spec_norm = (spec_max - np.min(spec_max))/np.ptp(spec_max)
            return spec_norm
        
        def interp_(arr, new_size):
            arr1_interp = interp.interp1d(np.arange(arr.size),arr)
            return arr1_interp(np.linspace(0,arr.size-1, new_size))

        duration = self.video_duration
        y, sr = librosa.load(self.audiofilepath, offset=self.offset, duration=duration)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        self.duration = len(y) / sr
        self.total_frame_count = int(self.duration * self.fps)
        
        y_harm, y_perc = librosa.effects.hpss(y, margin=(1.0,5.0))

        hop_length = 512
        n_mels = 4 # 5

        strength = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
        times = librosa.times_like(strength, sr=sr, hop_length=hop_length)
        perc_s = get_spec_norm(y_perc, sr, n_mels, hop_length)
        harm_s = get_spec_norm(y_harm, sr, n_mels, hop_length)
        
        librosa_frame_count = len(strength)
              
        self.orig_times = times
        self.orig_strength = strength
        self.orig_perc = perc_s
        self.orig_harm = harm_s

        strength = interp_(strength, self.total_frame_count)
        strength = librosa.util.normalize(strength)
        
        perc_s = interp_(perc_s, self.total_frame_count)
        perc_s = librosa.util.normalize(perc_s)
        
        harm_s = interp_(harm_s, self.total_frame_count)
        harm_s = librosa.util.normalize(harm_s)

        times = interp_(times, self.total_frame_count)
#         beats_time = librosa.frames_to_time(beats, sr=sr)
        
        beats_array = np.zeros(self.total_frame_count)
        for i in beats:
            ii = int (i * self.total_frame_count / librosa_frame_count)
            beats_array[ii] = 1.0

        self.times = times
        self.strength = strength
        self.perc = perc_s
        self.harm = harm_s
        self.beats = beats_array
        self.beat_indices = np.where(self.beats == 1.0)[0]
        self.tempo = tempo
            
    def __str__(self):
        def short(vec):
            if len(vec) <= 10:
                return '[' + ' '.join([f'{v:4.2f}' for v in vec]) + ']'
            return '[' + ' '.join([f'{v:3.1f}' for v in vec[:5]])  \
                + ' ... ' + ' '.join([f'{v:4.2f}' for v in vec[-5:]])  \
                + ']'

        str =  f'{self.audiofilepath}\n'
        str += f'self.duration = {self.duration}\n'
        str += f'self.total_frame_count = {self.total_frame_count}\n'
        str += f'self.tempo = {self.tempo}\n'
        str += f'len(self.times) = {len(self.times)}\n'
        str += f'self.times = {short(self.times)}\n'
        str += f'self.strength = {short(self.strength)}\n'
        str += f'self.beats = {short(self.beats)}\n'
        str += f'self.perc = {short(self.perc)}\n'
        str += f'self.harm = {short(self.harm)}\n'

        return str

    def plot(self, start_frame=None, end_frame=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=4, sharex=True)
        
        times = self.times
        strength = self.strength
        perc = self.perc
        harm = self.harm
        beats = self.beats
        if start_frame is not None:
            if end_frame is not None:
                times = times[start_frame:end_frame]
                strength = strength[start_frame:end_frame]
                perc = perc[start_frame:end_frame]
                harm = harm[start_frame:end_frame]
                beats = beats[start_frame:end_frame]
            else:
                times = times[start_frame:]
                strength = strength[start_frame:]
                perc = perc[start_frame:]
                harm = harm[start_frame:]
                beats = beats[start_frame:]
        elif end_frame is not None:
            times = times[:end_frame]
            strength = strength[:end_frame]
            perc = perc[:end_frame]
            harm = harm[:end_frame]
            beats = beats[:end_frame]
                

        ax[0].plot(times, strength, label='strength', color='black')
        ax[1].plot(times, perc, label='perc', color='blue')
        ax[2].plot(times, harm, label='harm', color='orange')
        ax[3].plot(times, beats, label='beats', color='red')
        fig.legend()
        return fig

    def dump(self, vec, fn=lambda x: x):
        return ", ".join([f'{i}: ({fn(v):.2f})' for i, v in enumerate(vec)])

if __name__ == '__main__':
    r = AudioPeak('./audio.mp3', 15)
    print(r)
    r.plot().savefig('./result.png')
    print('-----------')
    print(r.dump(r.strength, lambda x: x**2))
    print('-----------')
    print(r.dump(r.strength, lambda x: 1+x**2))
