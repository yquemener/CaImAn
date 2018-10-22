from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, IntSlider, Play, jslink, Tab
import ipywidgets as widgets
import os

#---------------- EDIT PANEL
# need: min_SNR, r_values_min, r_values_lowest, thresh_cnn_min,
# thresh_cnn_lowest,thresh_fitness_delta, min_std_reject
# SAME: min_SNR=2, r_values_min=0.9, r_values_lowest=-1, thresh_cnn_min=0.95, thresh_cnn_lowest=0.1,
# thresh_fitness_delta=-20., min_std_reject=0.5,
min_snr_edit_widget = widgets.FloatRangeSlider(
	value=[0.5,2.5],
	min=0.0,
	max=10.0,
	step=0.05,
	description='SNR Range:',
	#tooltip='Number of global background components',
	disabled=False,
	layout=widgets.Layout(margin='50px 0px 0px 0px')
)

rvalmin_edit_widget_ = widgets.FloatRangeSlider(
	value=[-1, 0.9],
	min=0.0,
	max=5.0,
	step=0.05,
	#description='R Values Min:',
	disabled=False,
	#layout=widgets.Layout(width="35%")
)
rvalmin_edit_widget = widgets.HBox([widgets.Label(value="R Values:"),rvalmin_edit_widget_])

'''rvallowest_edit_widget_ = widgets.BoundedFloatText(
	value= -1.,
	min= -5.0,
	max= 5.0,
	step=1,
	#description='R Values Lowest:',
	disabled=False,
	layout=widgets.Layout(width="35%")
)
rvallowest_edit_widget = widgets.HBox([widgets.Label(value="R Values Lowest:"),rvallowest_edit_widget_])
'''
cnnmin_edit_widget_ = widgets.FloatRangeSlider(
	value=[0.1, 0.95],
	min=0.0,
	max=1.0,
	step=0.05,
	#description='Thresh CNN Min:',
	disabled=False,
	#layout=widgets.Layout(width="35%")
)
cnnmin_edit_widget = widgets.HBox([widgets.Label(value="CNN:"),cnnmin_edit_widget_])

'''cnnlowest_edit_widget_ = widgets.BoundedFloatText(
	value=0.1,
	min=0.0,
	max=1.0,
	step=0.1,
	#description='Thresh CNN Lowest:',
	disabled=False,
	layout=widgets.Layout(width="30%")
)
cnnlowest_edit_widget = widgets.HBox([widgets.Label(value="Thresh CNN Lowest:"),cnnlowest_edit_widget_])
'''
'''fitness_delta_edit_widget_ = widgets.BoundedFloatText(
	value= -20.,
	min= -100.,
	max= 100.0,
	step= 1,
	#description='Thresh Fitness Delta:',
	disabled=False,
	layout=widgets.Layout(width="35%")
)
fitness_delta_edit_widget = widgets.HBox([widgets.Label(value="Thresh Fitness Δ:"),fitness_delta_edit_widget_])
'''
'''minstdreject_edit_widget_ = widgets.BoundedFloatText(
	value=0.5,
	min=0.0,
	max=50.0,
	step=0.1,
	#description='Min Std Reject:',
	disabled=False,
	layout=widgets.Layout(width="35%")
)
minstdreject_edit_widget = widgets.HBox([widgets.Label(value="Min SNR Reject:"),minstdreject_edit_widget_])
'''
gSig_edit_widget_ = widgets.IntRangeSlider(
	value=[4,10],
	min=0,
	max=100,
	step=1,
	description='gSig Range:',
	disabled=False,
	#layout=widgets.Layout(width="90%")
)
reset_edit_btn = widgets.Button(
	description='Reset',
	disabled=False,
	button_style='info', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='Reset ROIs',
	layout=widgets.Layout(width="45%")
)
update_edit_btn = widgets.Button(
	description='Update',
	disabled=False,
	button_style='info', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='Refine ROIs',
	layout=widgets.Layout(width="45%")
)

edit_panel_widget = widgets.VBox([min_snr_edit_widget, rvalmin_edit_widget, \
			cnnmin_edit_widget, reset_edit_btn, update_edit_btn], layout=widgets.Layout(display='None'))

#--------------- END EDIT PANEL
