import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import ipytone
import ipywidgets as wg
import numpy as np
from scipy.signal import square, sawtooth
from scipy.io import wavfile
import time
from IPython.display import display

def sine_function(t,frequency,amp):
    return amp * np.sin(2 * np.pi * frequency * t) # this is the equation written in the cell above!

def interactive_sine(time_interval,
                    initial_amplitude,
                    initial_frequency,
                    amp_max,
                    freq_max):
    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots(figsize=(8,6)) # creating the figure
    ax.set_ylim([-amp_max,amp_max]) # setting the minimum/maximum values for the y axis
    t = time_interval
    initial_sine = sine_function(t, initial_frequency, initial_amplitude) # evaluating the sine function at the points
    line, = ax.plot(t, initial_sine, lw=2) # plotting the sine function
    
    #######################################################################
    #### Everything below this line is just technical details         #####
    #### Feel free to read it if you want, but no need to understand! #####
    #######################################################################
    
    plt.grid() # overlaying a grid on the axes
    ax.set_xlabel('Time t [s]')
    ax.set_ylabel(r"A$\sin(2\pi ft)$")
    
    # Make sliders to control freq/amplitude   
    freq_slider = wg.FloatSlider(value=initial_frequency,
                                 min=1,
                                 max=freq_max,
                                 valstep=1,
                                description=r'Frequency $f$ [Hz]')
    amp_slider = wg.FloatSlider(value=initial_amplitude,
                                min=0.1,
                                max=amp_max,
                                description=r"Amplitude")
    
    # The function to be called anytime a slider's value changes
    def update(freq,amp):
        line.set_ydata(sine_function(t,freq,amp))
        fig.canvas.draw_idle()
    
    act = wg.interactive(update,freq=freq_slider,amp=amp_slider)
    
    resetAll = wg.Button(description=f"Reset")
    def reset_all(click):
        amp_slider.value = initial_amplitude
        freq_slider.value = initial_frequency
    resetAll.on_click(reset_all)

    plt.show()
    display(wg.HBox(act.children[:-1]))
    display(act.children[-1])#Show the output
    display(wg.HBox([resetAll]))

def interactive_wave(time_interval,
                    initial_amplitude,
                    initial_frequency,
                    amp_max,
                    freq_max):
    from scipy.signal import square, sawtooth
    
    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots(figsize=(6,4)) # creating the figure
    ax.set_ylim([-amp_max,amp_max]) # setting the minimum/maximum values for the y axis
    t = time_interval
    y_values = sine_function(t, initial_frequency, initial_amplitude) # evaluating the sine function at the points    
    line, = ax.plot(t, y_values, lw=2) # plotting the sine function
    plt.grid() # overlaying a grid on the axes
    ax.set_xlabel('Time t [s]')
    ax.set_ylabel(r"Signal")
    plt.show()
    
    # Make sliders to control freq/amplitude   
    freq_slider = wg.FloatSlider(value=initial_frequency,
                                 min=1,
                                 max=freq_max,
                                 valstep=1,
                                description=r'Frequency $f$ [Hz]')
    amp_slider = wg.FloatSlider(value=initial_amplitude,
                                min=0.1,
                                max=amp_max,
                                description=r"Amplitude")

    def change_wave(wave_type,amp,freq):
        if wave_type == 'sine':
            line.set_ydata(sine_function(t,freq,amp))
        elif wave_type == 'square':
            line.set_ydata(amp*square(2*np.pi*freq*t))
        elif wave_type == 'sawtooth':
            line.set_ydata(amp*sawtooth(2*np.pi*freq*t))
        elif wave_type == 'triangle':
            line.set_ydata(amp*sawtooth(2*np.pi*freq*t,width=0.5))
        fig.canvas.draw_idle()

    type_dropdown = wg.Dropdown(
        options=['sine', 'square', 'sawtooth', 'triangle'],
        value='sine',
        description="Wave Type")
    act = wg.interactive(change_wave,wave_type=type_dropdown,amp=amp_slider,freq=freq_slider)

    resetAll = wg.Button(description=f"Reset")
    def reset_all(click):
        amp_slider.value = initial_amplitude
        freq_slider.value = initial_frequency
        type_dropdown.value = 'sine'
    resetAll.on_click(reset_all)

    display(wg.HBox(act.children[:-1]))
    display(act.children[-1])#Show the output
    display(wg.HBox([resetAll]))
    
def oscillator_sampler():
    osc = ipytone.Oscillator(volume=-5)
    osc.connect(ipytone.destination)
    
    freq_slider = wg.FloatSlider(
        value=440,
        min=100,
        max=1000,
        step=1,
        description="Frequency"
    )
    
    type_dropdown = wg.Dropdown(
        options=['sine', 'square', 'sawtooth', 'triangle'],
        value='sine',
        description="Wave Type"
    )
    
    toggle_play_button = wg.ToggleButton(
        value=False,
        description="Start/Stop"
    )

    stopAll = wg.ToggleButton(value=False,description=f"Kill Process")
    def stop_all(change):
        if change['new']:
            for osc in oscillators:
                osc.stop()
                osc.disconnect(ipytone.destination)
                osc.dispose()
                del osc
    
    wg.jslink((freq_slider, 'value'), (osc.frequency, 'value'))
    wg.link((type_dropdown, 'value'), (osc, 'type'))
    
    def start_stop_osc(change):
        if change['new']:
            osc.start()
        else:
            osc.stop()
    def stop_full(change):
        if change['new']:
            osc.stop()
            osc.disconnect(ipytone.destination)
            osc.dispose()
            del osc
            
    toggle_play_button.observe(start_stop_osc, names='value')
    stopAll.observe(stop_all, names='value')
    
    display(wg.VBox([freq_slider, type_dropdown, toggle_play_button,stopAll]))

def quad_oscillator():
    notes = [440,554.37,659.25,880]
    oscillators = []
    vbox_f = []
    vbox_typ = []
    vbox_vol = []
    vbox_stop = []
    def toggle(change):
        index = vbox_stop.index(change.owner)
        if change['new']:
            oscillators[index].start()
        else:
            oscillators[index].stop()
    for i,note in enumerate(notes):
        oscillators.append(ipytone.Oscillator(volume=-5))
        oscillators[i].connect(ipytone.destination)
        vbox_f.append(wg.FloatSlider(value=note,min=100,max=1000,step=0.01,description=f"Frequency {i+1}"))
        vbox_typ.append(wg.Dropdown(options=['sine', 'square', 'sawtooth', 'triangle'],value='sine'))
        vbox_stop.append(wg.ToggleButton(value=False,description=f"Start/Stop {i+1}"))
        vbox_vol.append(wg.FloatSlider(value=-5,min=-15,max=5,step=0.1,description=f"Volume {i+1}"))
    for i in range(len(notes)):
        wg.jslink((vbox_f[i], 'value'), (oscillators[i].frequency, 'value'))
        wg.link((vbox_typ[i], 'value'), (oscillators[i], 'type'))
        vbox_stop[i].observe(toggle,names='value')
        wg.jslink((vbox_vol[i], 'value'), (oscillators[i].volume, 'value'))
    
    resetAll = wg.Button(description=f"Reset Notes")
    def reset_all(click):
        for i in range(len(notes)):
            oscillators[i].frequency.value = notes[i]
            oscillators[i].volume.value = -5
    resetAll.on_click(reset_all)
    
    stopAll = wg.ToggleButton(value=False,description=f"Stop Everything")
    def stop_all(change):
        if change['new']:
            for osc in oscillators:
                osc.stop()
                osc.disconnect(ipytone.destination)
                osc.dispose()
                del osc
    stopAll.observe(stop_all, names='value')
    
    display(wg.VBox([wg.HBox([wg.VBox(vbox_f),wg.VBox(vbox_typ),wg.VBox(vbox_stop),wg.VBox(vbox_vol)]),
                     wg.HBox([resetAll,stopAll])]))

def triangle_coeffs(n):
    if n % 2 == 0:
        return 0
    else:
        return (8/np.pi**2)*((-1)**((n-1)/2)/n**2)

def square_coeffs(n):
    if n % 2 == 0:
        return 0
    else:
        return 4/(n*np.pi)

def saw_coeffs(n):
    return -1/(n*np.pi)

def sine_coeffs(n):
    return 0 if n>1 else 1

def fourier_sandbox():
    nmax = 50
    plt.figure(figsize=(6,4))
    t = np.linspace(0,4,1000)
    sines = np.concatenate([np.sin(n*np.pi*t).reshape(1,-1) for n in range(1,nmax)],axis=0)
    coeffs_triangle = np.array([triangle_coeffs(n) for n in range(1,nmax)]).reshape(-1,1)
    coeffs_square = np.array([square_coeffs(n) for n in range(1,nmax)]).reshape(-1,1)
    coeffs_saw = np.array([saw_coeffs(n) for n in range(1,nmax)]).reshape(-1,1)
    coeffs_sine = np.array([sine_coeffs(n) for n in range(1,nmax)]).reshape(-1,1)

    triangle = sines[coeffs_triangle[:,0]!=0]*coeffs_triangle[coeffs_triangle[:,0]!=0]
    square = sines[coeffs_square[:,0]!=0]*coeffs_square[coeffs_square[:,0]!=0]
    saw = sines[coeffs_saw[:,0]!=0]*coeffs_saw[coeffs_saw[:,0]!=0]
    sine = sines[coeffs_sine[:,0]!=0]*coeffs_sine[coeffs_sine[:,0]!=0]

    ncurr = 1
    y = np.sum((coeffs_sine*sines)[:ncurr],axis=0)
    line, = plt.plot(t,y)
    plt.ylim([-1.5,1.5])

    style = {'description_width': 'initial'}
    n_terms_slider = wg.IntSlider(value=ncurr,
                                 min=1,
                                 max=nmax/2,
                                description=r'Number of terms',
                                 style=style)
    type_dropdown = wg.Dropdown(
        options=['sine', 'square', 'sawtooth', 'triangle'],
        value='sine',
        description="Wave Type",
        style=style
    )
    def change_wave(wtype,n_sines):
        if wtype=='sine':
            line.set_ydata(np.sum(sine[:n_sines],axis=0))
        elif wtype=='square':
            line.set_ydata(np.sum(square[:n_sines],axis=0))
        elif wtype=='triangle':
            line.set_ydata(np.sum(triangle[:n_sines],axis=0))
        elif wtype=='sawtooth':
            line.set_ydata(np.sum(saw[:n_sines],axis=0))
        fig.canvas.draw_idle()
    act = wg.interactive(change_wave,wtype=type_dropdown,n_sines=n_terms_slider)
    display(wg.HBox([type_dropdown,n_terms_slider]))
    
    plt.show()

def solve_sho(beta,t):
    from scipy.integrate import odeint
    k = 4*np.pi**2
    m = 1
    w0 = np.sqrt(k/m)
    def sho(y,t):
        x, xdot = y
        dydt = [xdot, -x*w0**2 - 2*beta*xdot]
        return dydt
    y0 = [1,0]
    sol = odeint(sho,y0,t)
    return sol[:,0]

def interactive_sho():
    beta0 = 0
    t = np.linspace(0,5,1000)
    sol0 = solve_sho(beta0,t)

    plt.figure(figsize=(6,4))
    ax = plt.gca()
    line, = plt.plot(t,sol0)
    ax.set_title(f"Damping = {beta0}")
    plt.xlabel("Time [seconds]")
    plt.ylabel("Oscillator Position")

    def change_sol(beta):
        sol = solve_sho(beta,t)
        line.set_ydata(sol)
        ax.set_title(f"Damping = {beta}")
    damping = wg.FloatSlider(value=beta0,min=0,max=10,step=0.05,description=r'Damping')
    act = wg.interactive(change_sol,beta=damping)
    display(wg.HBox([damping]))
    plt.show()

def interactive_sampling():
    t = 2
    freq = 1
    samp_rate = 30
    times = np.linspace(0,t,t*samp_rate+1)
    fine_times = np.linspace(0,t,1000)
    waveform = np.sin(2*np.pi*freq*times)
    fine_waveform = np.sin(2*np.pi*freq*fine_times)
    
    fig = plt.figure(figsize=(10,4))
    line, = plt.plot(times,waveform,label="Sampled Waveform",marker='.',markersize=8)
    line2, = plt.plot(fine_times,fine_waveform,linestyle='--',color='forestgreen',label="True Waveform")
    plt.ylim([-1.2,1.5])
    plt.legend(ncol=2,loc='upper center')
    plt.xlabel("Time [seconds]")
    plt.ylabel("Amplitude")
    
    style = {'description_width': 'initial'}
    fslider = wg.FloatSlider(value=freq,min=1,max=10,step=0.1,description=f"Frequency",style=style)
    rslider = wg.IntSlider(value=samp_rate,min=1,max=100,step=1,description=f"Sampling Rate",style=style)
    
    def update(freq,rate):
        times = np.linspace(0,t,t*rate+1)
        waveform = np.sin(2*np.pi*freq*times)
        line.set_xdata(times)
        line.set_ydata(waveform)
        line2.set_ydata(np.sin(2*np.pi*freq*fine_times))
        fig.canvas.draw_idle()
    act = wg.interactive(update,freq=fslider,rate=rslider)
    plt.show()
    display(wg.HBox([fslider,rslider]))

def minmax_scale(x):
    xmin = np.min(x)
    xmax = np.max(x)
    x = (x-xmin)/(xmax-xmin) # scale to [0,1]
    x = x-0.5 # scale to [-0.5,0.5]
    x = 2*x # scale to [-1,1]
    return x

def digitize(x,bit_depth=8):
    x = (x - x.min())/(x.max()-x.min())
    x = np.rint(x*(2**bit_depth-1))
    steps = np.linspace(0,2**bit_depth,2**bit_depth,endpoint=False).reshape(-1,1)
    digi = steps[np.argmin(np.abs(steps-x),axis=0),0]
    digi = minmax_scale(digi)
    return digi

def interactive_bitdepth():
    t = 2
    freq = 1
    samp_rate = 30
    bd = 8
    times = np.linspace(0,t,t*samp_rate+1)
    fine_times = np.linspace(0,t,1000)
    waveform = np.sin(2*np.pi*freq*times)
    fine_waveform = np.sin(2*np.pi*freq*fine_times)
    
    fig = plt.figure(figsize=(10,4))
    line, = plt.plot(times,waveform,label="Sampled Waveform",marker='.',markersize=8)
    line2, = plt.plot(fine_times,fine_waveform,linestyle='--',color='forestgreen',label="True Waveform")
    plt.ylim([-1.2,1.5])
    plt.legend(ncol=2,loc='upper center')
    plt.xlabel("Time [seconds]")
    plt.ylabel("Amplitude")
    
    style = {'description_width': 'initial'}
    fslider = wg.FloatSlider(value=freq,min=1,max=10,step=0.1,description=f"Frequency",style=style)
    rslider = wg.IntSlider(value=samp_rate,min=1,max=100,step=1,description=f"Sampling Rate",style=style)
    bdslider = wg.IntSlider(value=bd,min=1,max=16,step=1,description=f"Bit Depth",style=style)
    
    def update(freq,rate,bitdepth):
        times = np.linspace(0,t,t*rate+1)
        waveform = np.sin(2*np.pi*freq*times)
        line.set_xdata(times)
        line.set_ydata(digitize(waveform,bit_depth=bitdepth))
        line2.set_ydata(np.sin(2*np.pi*freq*fine_times))
        fig.canvas.draw_idle()
    act = wg.interactive(update,freq=fslider,rate=rslider,bitdepth=bdslider)
    plt.show()
    display(wg.HBox([fslider,rslider,bdslider]))


def ibd2():
    t = 0.01
    freq = 440
    samp_rate = 10000
    bd = 8
    times = np.linspace(0,t,int(t*samp_rate)+1)
    fine_times = np.linspace(0,t,1000)
    waveform = np.sin(2*np.pi*freq*times)
    fine_waveform = np.sin(2*np.pi*freq*fine_times)
    
    fig = plt.figure(figsize=(10,4))
    line, = plt.plot(times,waveform,label="Sampled Waveform",marker='.',markersize=8)
    line2, = plt.plot(fine_times,fine_waveform,linestyle='--',color='forestgreen',label="True Waveform")
    plt.ylim([-1.2,1.5])
    plt.legend(ncol=2,loc='upper center')
    plt.xlabel("Time [seconds]")
    plt.ylabel("Amplitude")
    
    style = {'description_width': 'initial'}
    fslider = wg.FloatSlider(value=freq,min=100,max=1000,step=1,description=f"Frequency",style=style)
    rslider = wg.IntSlider(value=samp_rate,min=1000,max=10000,step=1,description=f"Sampling Rate",style=style)
    bdslider = wg.IntSlider(value=bd,min=1,max=16,step=1,description=f"Bit Depth",style=style)
    
    def update(freq,rate,bitdepth):
        times = np.linspace(0,t,int(t*rate)+1)
        waveform = np.sin(2*np.pi*freq*times)
        line.set_xdata(times)
        line.set_ydata(digitize(waveform,bit_depth=bitdepth))
        line2.set_ydata(np.sin(2*np.pi*freq*fine_times))
        fig.canvas.draw_idle()
    act = wg.interactive(update,freq=fslider,rate=rslider,bitdepth=bdslider)
    plt.show()
    display(wg.HBox([fslider,rslider,bdslider]))

def interactive_bitdepth_sampling_player():
    t = 0.01
    freq = 440
    samp_rate = 5000
    bd = 8
    times = np.linspace(0,t,int(t*samp_rate)+1)
    fine_times = np.linspace(0,t,int(44100*t)+1)
    waveform = np.sin(2*np.pi*freq*times)
    fine_waveform = np.sin(2*np.pi*freq*fine_times)
    
    fig = plt.figure(figsize=(10,4))
    line, = plt.plot(times,waveform,label="Sampled Waveform",marker='.',markersize=8)
    line2, = plt.plot(fine_times,fine_waveform,linestyle='--',color='forestgreen',label="True Waveform")
    ax = plt.gca()
    plt.ylim([-1.2,1.5])
    plt.legend(ncol=2,loc='upper center')
    plt.xlabel("Time [seconds]")
    plt.ylabel("Amplitude")
    
    style = {'description_width': 'initial'}
    fslider = wg.FloatSlider(value=freq,min=100,max=2000,step=1,description=f"Frequency",style=style)
    rslider = wg.IntSlider(value=samp_rate,min=500,max=10000,step=1,description=f"Sampling Rate",style=style)
    bdslider = wg.IntSlider(value=bd,min=1,max=16,step=1,description=f"Bit Depth",style=style)
    def update(freq,rate,bitdepth):
        times = np.linspace(0,t,int(t*rate)+1)
        waveform = np.sin(2*np.pi*freq*times)
        line.set_xdata(times)
        line.set_ydata(digitize(waveform,bit_depth=bitdepth))
        line2.set_ydata(np.sin(2*np.pi*freq*fine_times))
        ax.set_title(f"Freq = {freq}")
        fig.canvas.draw_idle()
    act = wg.interactive(update,freq=fslider,rate=rslider,bitdepth=bdslider)

    playNote = wg.Button(description=f"Play Sampled Note", style=style)
    def play_note(click):
        duration = 1
        size = int(rslider.value * duration)
        factor = fslider.value * np.pi * 2 / rslider.value
        waveform = digitize(np.sin(np.arange(size) * factor),bit_depth=bdslider.value)
        down_rate = 44100/rslider.value
        buff = ipytone.AudioBuffer(url_or_array=waveform)
        player = ipytone.Player(buff).to_destination()
        player.playback_rate = 1/down_rate
        player.volume.value = -5
        player.start().stop(f"+{duration}")
    playNote.on_click(play_note)

    osc = ipytone.Oscillator(volume=-5).to_destination()
    playTrue = wg.Button(description=f"Play True Note", style=style)
    def play_true(click):
        duration = 1
        osc.frequency.value = fslider.value
        print(osc.frequency)
        osc.start().stop(f"+1")
    playTrue.on_click(play_true)

    plt.show()
    #display(wg.HBox([fslider,rslider,bdslider]))
    display(wg.VBox([wg.HBox([fslider,rslider,bdslider]),wg.HBox([playNote,playTrue])]))
