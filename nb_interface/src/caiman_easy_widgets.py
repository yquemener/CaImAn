import ipywidgets as widgets

out = widgets.Output(layout={'border': '1px solid black', 'height':'700px', 'overflow-y':'scroll'})
clear_console_btn = widgets.Button(
	description='Clear Console',
	disabled=False,
	button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='Clear Console',
	layout=widgets.Layout(width="30%")
)
out_txt = widgets.Textarea(layout={'border': '1px solid black', 'height':'700px', 'width':'900px','overflow-y':'scroll'})
out_box = widgets.VBox([clear_console_btn, out_txt])
out_accordion = widgets.Accordion(children=[out_box])
out_accordion.set_title(0, 'Console Output')

status_bar_widget = widgets.HTML(
    value="Idle",
    #placeholder='Status',
    description='<b>Status:</b>',
	layout=widgets.Layout(width="90%")
)
