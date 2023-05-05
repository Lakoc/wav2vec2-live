import PySimpleGUI as sg
from live_asr import LiveWav2Vec2


font_base = font=('Ubuntu Font Family', 10)
# GUI layout
layout = [
    [sg.Button("START", key="-RECORD-", font=font_base),
     sg.Button("PAUSE", key="-PAUSE-", disabled=True, font=font_base),
     sg.Button("CLEAR", key="-CLEAR-", font=font_base),
     ],
    [sg.Text("Transcript:", font=font_base)],
    [sg.Multiline("", size=(115, 25), font=('Ubuntu Font Familys', 20), key="-TRANSCRIPT-")],
]

# GUI window
window = sg.Window("ASR Demo", layout,  font=font_base)

asr = LiveWav2Vec2("checkpoint-115000")

# Event loop
while True:
    event, values = window.read(timeout=100)

    if event == sg.WIN_CLOSED:
        break

    if event == "-RECORD-":
        window["-RECORD-"].update(disabled=True)
        window["-PAUSE-"].update(disabled=False)
        asr.start()

    if event == "-PAUSE-":
        window["-PAUSE-"].update(disabled=True)
        asr.pause()
        window["-RECORD-"].update(disabled=False)

    if event == "-CLEAR-":
        window["-TRANSCRIPT-"].update("")

    if asr.asr_output_queue.qsize() > 0:
        text, sample_length, inference_time, confidence = asr.get_last_text()
        window['-TRANSCRIPT-'].update(text + '\n', append=True)

asr.stop()
window.close()
