import PySimpleGUIQt as sg

col = [[sg.Button('Button 1'),] for i in range(10)]
col2 = [[sg.Slider((1,10)) for i in range(10)]]
sg.theme('DarkBrown1')

num_buttons = 2
layout = [[sg.Text('Your typed chars appear here:'), sg.Text('', key='_OUTPUT_')],
            [sg.Input(do_not_clear=True, key='_IN_')],
            *[[sg.Button('Button'),] for i in range(num_buttons)],
            [sg.Slider(range=(1,100), text_color='white', orientation='h', key='SLIDER'),
            sg.Drop(('Choice 1', 'choice 2'), key='DROP'), sg.Stretch()],
            [sg.Button('Detailed Info'), sg.Button('Delete Rows' ), sg.Button('Disappear') ,sg.Button('Reappear')],
            [sg.Button('Show Sliders'), sg.Button('Show Buttons'), sg.Button('Hide Sliders') ,sg.Button('Hide Buttons') , sg.Button('Exit')],
            [sg.Column(col, key='COL', visible=False), sg.Column(col2, key='COL2', visible=False)]
          ]

window = sg.Window('Window Title', resizable=True).Layout(layout).Finalize()

window.Element('SLIDER').Update(visible=False)
window.Element('SLIDER').Update(visible=False)
window.Refresh()
window.Refresh()

window.Size = window.Size

num_buttons = 2
while True:             # Event Loop
    event, values = window.Read()
    print(event, values, window.Size)
    if event is None or event == 'Exit':
        break
    if event == 'Detailed Info':
        layout = [[sg.Button('Option %s'%i) for i in range(num_buttons)],
                  [sg.Button('Add Rows'), sg.Button('Delete Rows'), sg.Button('Exit')]]
        window1 = sg.Window('Window Title', no_titlebar=True,).Layout(layout)
        # window.Close()
        # window = window1
        window1.Read(timeout=0)
    if event == 'Hide Buttons':
        window.Element('SLIDER').Update(visible=False)
        window.Element('COL').Update(visible=False)
    elif event == 'Show Buttons':
        window.Element('SLIDER').Update(visible=True)
        window.Element('COL').Update(visible=True)
    elif event == 'Hide Sliders':
        window.Element('COL2').Update(visible=False)
        window.Element('DROP').Update(visible=False)
    elif event == 'Show Sliders':
        window.Element('COL2').Update(visible=True)
        window.Element('DROP').Update(visible=True)

    window.Refresh()
    window.Refresh()
    window.Size =  window.Size

    print(window.Size[0])
window.Close()