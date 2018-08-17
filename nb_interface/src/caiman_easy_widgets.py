import ipywidgets as widgets

out = widgets.Output(layout={'border': '1px solid black', 'height':'700px', 'overflow-y':'scroll'})

out_accordion = widgets.Accordion(children=[out])
out_accordion.set_title(0, 'Console Output')
out_accordion

status_bar_widget = widgets.HTML(
    value="Idle",
    #placeholder='Status',
    description='<b>Status:</b>',
	layout=widgets.Layout(width="90%")
)
